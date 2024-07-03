import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers import DefaultDataCollator
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy, 
)
from .utils import load_corpus_file # normal use case
# from utils import load_corpus_file # for unit test

class ContextQADataset(Dataset):

    def __init__(
        self, 
        data_file, 
        n_max_segments=10,
        n_max_candidates=50,
        depth=10,
        budget=5,
        corpus_file=None,
        retrieval_file=None,
        quick_test=None
    ):
        """
        params
        ------
        n_max_segment: the max number of segments (exclude the initial q)
        n_max_candidates: the max depth of considered intialized retrieval
        depth: the size of psg for re-ranking as candidates may change
        budget: the size of psg for later generator to read
        """
        data = []
        with open(data_file, 'r') as f:
            for line in tqdm(f):
                data.append(json.loads(line.strip()))

        if quick_test is not None:
            data = data[:9]

        self.length = len(data)
        self.n_max_segments = n_max_segments
        self.n_max_candidates = n_max_candidates

        ## fixed attributes 
        self.qids = [None] * self.length
        self.questions = [None] * self.length
        self.answer_list = [[] for _ in range(self.length)]
        self.proof = [[] for _ in range(self.length)] 
        self.gold_context = [[] for _ in range(self.length)]
        self.corpus = {-1: {'text': "", 'title': ""}}

        ## dynamic attributes
        # self.context_pids = [[-1] for _ in range(self.length)] # will be empty in the begining
        self.actions = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]
        self._build(data)

        ## results for context candidates
        if retrieval_file is not None:
            self._load_retrieval(retrieval_file)
        else:
            self.candidate_pids = None

        ## corpus for mapping
        if corpus_file is not None:
            self._load_corpus(corpus_file)

        ## add context buffer count
        ## None means everything
        self.depth = depth
        self.budget = budget

    def _load_retrieval(self, file):
        """
        [NOTE] Here we will do additional preprocess. 
        We only keep the discard the passage
        """
        self.candidate_pids = defaultdict(list)
        with open(file, 'r') as f:
            for line in tqdm(f):
                qid, _, docid, rank, score, _ = line.strip().split()
                if int(rank) <= self.n_max_candidates:
                    self.candidate_pids[qid].append(docid)
                    self.corpus[docid] = {'text': "", 'title': ""}
                    # remove this if `_load_corpus` doesn't need to predefine

    def _load_corpus(self, dir):
        from multiprocessing import Pool
        files = glob(f'{dir}/*jsonl')
        with Pool(16) as pool:
            corpora = pool.map(load_corpus_file, files)

        for corpus in tqdm(corpora):
            for docid, docdict in corpus.items():
                try:
                    self.corpus[docid].update(docdict)
                except:
                    continue
                    # raiseKeyError as it's not in the retrieved budget

    def _build(self, data):
        for idx in range(self.length):
            self.qids[idx] = data[idx]['qid']
            self.questions[idx] = data[idx]['question_text']

            for answer in data[idx]['answer_list']:
                self.answer_list[idx].append(answer['answer_text'])

                for proof in answer['proof']:
                    self.proof[idx].append(proof['proof_text'])
                    self.gold_context[idx].append(proof['pid']) 
                    # this is not actually the chunk_id for corpus

    def __len__(self):
        return len(self.questions)

    def add_action(self, idx, act):
        try:
            i = self.actions[idx].index("") # empty strings
            self.actions[idx][i] = act
        except: # means it's full
            self.actions[idx].pop(0) # remove the first one
            self.actions[idx].append(act)
        # self.actions[idx] += [act]
        # self.actions[idx] = self.actions[idx][-(self.n_max_segments):]

    def adjust_context(self, idx, act, context):
        # act: the choices of action for adjusting context. ['add', 'remove', ]
        # self.context[idx] += [context]
        raise NotImplementedError

    def __getitem__(self, idx):
        qid = self.qids[idx]
        # ctx_pids = self.context_pids[idx][:self.budget] # can be qid or idx
        cand_pids = self.candidate_pids[qid][:self.budget] # can be qid or idx
        return {'index': idx,
                'question': self.questions[idx], 
                'actions': self.actions[idx],
                'answers': self.answer_list[idx],
                'candidate_pids': cand_pids, 
                'candidates': [self.corpus[pid] for pid in cand_pids],} # this is for reranker 

@dataclass
class ContextQACollator(DefaultDataCollator):
    tokenizer_r: Union[PreTrainedTokenizerBase] = None
    tokenizer_g: Union[PreTrainedTokenizerBase] = None
    truncation: Union[bool, str] = True
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_src_length: Union[int] = 256
    max_tgt_length: Union[int] = 16
    pad_to_multiple_of: Optional[int] = None
    num_mem_tokens: Union[int] = 16

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        params
        ------
        features: {
            'questions': [q1, q2, q3, ..., qB], 
            'actions': [ [i11, i12, ..., i1n], ..., [iB1, iB2, ..., iBn] ], 
                --> [ [i11, i21, ..., iB1], ..., [i1n, i2n, ..., iBn] ], 
            'answers': [[a11, a12, ...] , [a21, a22, ...]], 
                --> 'labels': [y1, y2, ..., yn]
        }

        Returns
        -------
        list-of-segs: [ features_of_segment1, features_of_segment2, ...]
        """
        batch = {}
        batch_r = {}
        batch_size = len(features)
        n_max_segments = len(features[0]['actions'])

        # encode the initial request
        initial_q = self.tokenizer_r.batch_encode_plus(
            [f['question'] for f in features],
            add_special_tokens=True,
            max_length=self.max_src_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        )
        # batch_r['q_texts'] = [[f['question'] for f in features]]
        batch_r['q_tokens'] = [initial_q['input_ids']]
        batch_r['q_mask'] = [initial_q['attention_mask']]

        # encode the action/followup text segment-by-segment
        ### this can stop appending when no additional segments (actions)
        for seg_num in range(n_max_segments): 
            ### collect action for every batch first
            ### [NOTE] the empty string will return empty list
            batch_action_q = [ features[b]['actions'][seg_num] for b in range(batch_size) ]
            action_q = self.tokenizer_r.batch_encode_plus(
                batch_action_q,
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            )
            # batch_r['q_texts'].append(batch_action_q)
            batch_r['q_tokens'].append(action_q['input_ids'])
            batch_r['q_mask'].append(action_q['attention_mask'])

        # encode the candidate texts
        candidate_size = len(features[0]['candidates'])
        batch_r['d_tokens'] = []
        batch_r['d_mask'] = []
        for ctx in range(candidate_size):
            # [NOTE] Could be possible to separate title and text
            candidate = self.tokenizer_r.batch_encode_plus(
                [f"{features[b]['candidates'][ctx]['title']} {features[b]['candidates'][ctx]['text']}" for b in range(batch_size)],
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            )
            batch_r['d_tokens'].append(candidate['input_ids'])
            batch_r['d_mask'].append(candidate['attention_mask'])

        ## rewarding: (1) token adjustment (2) mask for calculating likelihood
        batch['index'] = [f['index'] for f in features] # we record it 
        batch['question'] = [f['question'] for f in features] 
        batch['targets'] = ["#".join(f['answers']) for f in features] 
        batch['candidates'] = [f['candidates'] for f in features] 
        batch['inputs_for_retriever'] = batch_r
        # evid-R, evid-Rprec
        batch['answers'] = [f['answers'] for f in features] 
        # evid-R, evid-Rprec
        batch['candidate_pids'] = [f['candidate_pids'] for f in features] 
        return batch



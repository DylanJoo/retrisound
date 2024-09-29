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
from .utils import load_corpus_file, batch_iterator # normal use case
# from utils import load_corpus_file, batch_iterator # normal use case
import sys

class ContextQADataset(Dataset):

    def __init__(
        self, 
        data_file, 
        n_max_segments=10,
        n_max_candidates=10,
        depth=30,
        corpus_file=None,
        retrieval_file=None,
        quick_test=None,
        half_with_bottom=False
    ):
        with open(data_file, 'r') as f:
            raw_data = json.load(f)['train']

        data = []
        for key_id in raw_data:
            data.append({
                "sample_id": key_id,
                "question": raw_data[key_id]['ambiguous_question'],
                "last_answer": raw_data[key_id]['annotations'][0]['long_answer'],
                "sub_questions": [i['question'] for i in raw_data[key_id]['qa_pairs']],
                "sub_answers": [i['short_answers'][0] for i in raw_data[key_id]['qa_pairs']],
            })

        self.quick_test = quick_test 
        self.length = len(data)
        self.n_max_segments = n_max_segments
        self.n_max_candidates = n_max_candidates
        self.depth = depth
        self.half_with_bottom = half_with_bottom

        ## fixed attributes 
        self.qids = [None] * self.length
        self.questions = [None] * self.length
        self.answers = [None] * self.length
        self.sub_questions = [None] * self.length
        self.sub_answers = [None] * self.length
        self.corpus = {-1: {'text': "", 'title': ""}}
        self.n_feedbacks = [0] * self.length

        ## dynamic attributes
        # self.context_pids = [[-1] for _ in range(self.length)] # will be empty in the begining
        self.feedbacks = [ ["" for _ in range(self.n_max_segments)]  for _ in range(self.length)]
        self._build(data)

        ## results for context candidates
        if retrieval_file is not None:
            self._load_retrieval(retrieval_file)
        else:
            self.candidate_pids = None

        ## corpus for mapping
        if corpus_file is not None:
            self._load_corpus(corpus_file)


    def _build(self, data):
        for idx in range(self.length):
            self.qids[idx] = data[idx]['sample_id']
            self.questions[idx] = data[idx]['question']
            self.answers[idx] = data[idx]['last_answer']
            self.sub_questions[idx] = data[idx]['sub_questions']
            self.sub_answers[idx] = data[idx]['sub_answers']

    def _load_retrieval(self, file):
        """
        [NOTE] Here we will do additional preprocess. 
        We only keep the discard the passage
        """
        self.candidate_pids = defaultdict(list)
        with open(file, 'r') as f:
            for line in tqdm(f):
                qid, _, docid, rank, score, _ = line.strip().split()
                if int(rank) <= self.depth:
                    self.candidate_pids[qid].append(docid)
                    self.corpus[docid] = {'text': "", 'title': ""}
                    # remove this if `_load_corpus` doesn't need to predefine

        if self.half_with_bottom:
            for qid in self.candidate_pids:
                n_half = (self.n_max_candidates // 2)
                first_half = self.candidate_pids[qid][:n_half]
                second_half = self.candidate_pids[qid][-n_half:]
                self.candidate_pids[qid] = (first_half + second_half)

    def _load_corpus(self, dir):
        from multiprocessing import Pool
        files = glob(f'{dir}/*jsonl*')
        if self.quick_test is not None:
            files = files[:10]
        for batch_files in tqdm(batch_iterator(files, 1000), 'load wiki files', total=1+len(files)//1000):
            with Pool(processes=16) as pool:
                corpora = pool.map(load_corpus_file, batch_files)

            for corpus in corpora:
                for docid, docdict in corpus.items():
                    try:
                        self.corpus[docid].update(docdict)
                    except:
                        self.corpus[docid] = docdict
            del corpora

    def __len__(self):
        return len(self.questions)

    def add_feedback(self, idx, act):
        try:
            i = self.feedbacks[idx].index("") # empty strings
            self.feedbacks[idx][i] = act
            self.n_feedbacks[idx] += 1
        except: # means it's full
            self.feedbacks[idx] = [act] + ["" for _ in range(self.n_max_segments-1)] 
            self.n_feedbacks[idx] = 1

    def __getitem__(self, idx):
        qid = self.qids[idx]
        cand_pids = self.candidate_pids[qid][:self.n_max_candidates] # can be qid or idx

        if len(cand_pids) < self.n_max_candidates:
            cand_pids += [-1] * max(0, self.n_max_candidates - len(cand_pids))

        return {'index': idx,
                'questions': self.questions[idx], 
                'feedbacks': self.feedbacks[idx],
                'n_feedbacks': self.n_feedbacks[idx],
                'answers': self.answers[idx],
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
            'feedbacks': [ [i11, i12, ..., i1n], ..., [iB1, iB2, ..., iBn] ], 
                --> [ [i11, i21, ..., iB1], ..., [i1n, i2n, ..., iBn] ], 
            'answers': [[a11, a12, ...] , [a21, a22, ...]], 
                --> 'labels': [y1, y2, ..., yn]
        }

        Returns
        -------
        list-of-segs: [ features_of_segment1, features_of_segment2, ...]
        """
        batch = {}
        batch_r = self.get_inputs_for_retriever(features)

        batch['index'] = [f['index'] for f in features] # we record it 
        batch['questions'] = [f['questions'] for f in features] 
        batch['targets'] = [f['answers'] for f in features] 
        batch['candidates'] = [f['candidates'] for f in features] 
        batch['inputs_for_retriever'] = batch_r
        batch['candidate_pids'] = [f['candidate_pids'] for f in features] 
        batch['n_feedbacks'] = [f['n_feedbacks'] for f in features] 
        return batch

    def get_inputs_for_retriever(
        self, 
        features: List[Dict[str, Any]], 
        device="cpu"
    ):
        batch_r = {}
        batch_size = len(features)
        n_max_segments = len(features[0]['feedbacks'])

        # encode the initial request
        initial_q = self.tokenizer_r.batch_encode_plus(
            [f['questions'] for f in features],
            add_special_tokens=True,
            max_length=self.max_src_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        ).to(device)
        batch_r['q_tokens'] = [initial_q['input_ids']]
        batch_r['q_masks'] = [initial_q['attention_mask']]

        # encode the action/followup text segment-by-segment
        for seg_num in range(n_max_segments): 
            batch_action_q = [ features[b]['feedbacks'][seg_num] for b in range(batch_size) ]
            action_q = self.tokenizer_r.batch_encode_plus(
                batch_action_q,
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            ).to(device)
            batch_r['q_tokens'].append(action_q['input_ids'])
            batch_r['q_masks'].append(action_q['attention_mask'])

        # encode the candidate texts
        candidate_size = len(features[0]['candidates'])
        batch_r['d_tokens'] = []
        batch_r['d_masks'] = []
        for ctx in range(candidate_size):
            candidate = self.tokenizer_r.batch_encode_plus(
                [f"{features[b]['candidates'][ctx]['title']} {features[b]['candidates'][ctx]['text']}" for b in range(batch_size)],
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            ).to(device)
            batch_r['d_tokens'].append(candidate['input_ids'])
            batch_r['d_masks'].append(candidate['attention_mask'])

        return batch_r

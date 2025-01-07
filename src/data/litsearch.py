import datetime
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers import DefaultDataCollator
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy, 
)
import sys

from datasets import load_dataset

class PRFDataset(Dataset):

    def __init__(
        self, 
        dataset_dir, 
        split='test',
        n_max_segments=10, # n_max_feedback
        n_max_candidates=2,
        retrieval_file=None,
        judgement_file=None,
        quick_test=None,
        **kwargs
    ):
        query = load_from_disk(f"{dataset_dir}/query")
        query = {}
        corpus_clean_data  corpus_s2orc_data  query
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
        self.depth = depth
        self.half_with_bottom = half_with_bottom

        ## fixed attributes 
        self.qids = [None] * self.length
        self.questions = [None] * self.length
        self.answers = [None] * self.length
        self.sub_questions = [None] * self.length
        self.sub_answers = [None] * self.length
        self.corpus = {-1: {'text': "", 'title': ""}}
        if quick_test is not None:
            self.corpus = defaultdict(lambda: {'text': "this is a testing doc.", 'title': "this is a testing title."})

        ## dynamic attributes
        self.feedbacks = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]
        self.n_feedbacks = [0] * self.length
        self._build(data)
        self.positives = defaultdict(list)
        self.negatives = defaultdict(list)

        ## corpus for mapping
        if corpus_file is not None:
            self._load_corpus(corpus_file)

        ## load prerun judgements
        self.judgements = {}
        for qid in self.qids:
            self.judgements[qid] = defaultdict(int)
        self._load_judgement(judgement_file)

    def _build(self, data):
        for idx in range(self.length):
            self.qids[idx] = data[idx]['sample_id']
            self.questions[idx] = data[idx]['question']
            self.answers[idx] = data[idx]['last_answer']
            self.sub_questions[idx] = data[idx]['sub_questions']
            self.sub_answers[idx] = data[idx]['sub_answers']

    def _load_judgement(self, file):
        file = (file or f'/home/dju/temp/judge-{datetime.datetime.now().strftime("%b%d-%I%m")}.txt')
        self.judgement_file = file
        try:
            with open(file, 'r') as f:
                for line in tqdm(f):
                    qid, psgid, judge = line.strip().split()[:3]
                    self.judgements[qid][psgid] = judge
        except:
            with open(file, 'w') as f:
                f.write("qid\tpid\tj\tinfo\n")

    def _load_corpus(self):
        corpus = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
        self.corpus = corpus

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

    def add_judgements(self, idx, judgements, info=None):
        qid = self.qids[idx]
        with open(self.judgement_file, 'a') as f:
            for pid in judgements:
                j = judgements[pid]
                # in memory
                self.judgements[qid][pid] = j
                # to file
                if info is not None:
                    f.write(f"{qid}\t{pid}\t{j}\t{info}\n")
                else:
                    f.write(f"{qid}\t{pid}\t{j}\n")

    def update_candidates(self, idx, pids, scores=None):
        qid = self.qids[idx]
        for i, pid in enumerate(pids):
            if scores[i] < 3:
                self.negatives[qid].append(pid)
            elif scores[i] > 5:
                self.positives[qid].append(pid)

    def __getitem__(self, idx):
        qid = self.qids[idx]

        n_feedbacks = self.n_feedbacks[idx]
        if n_feedbacks >= 1:
            positive_context = [{"text": self.feedbacks[idx][n_feedbacks - 1], "title": ""}]
            negative_context = [self.corpus[pid] for pid in (self.negatives[qid]+[-1])[:1]]
            return {'index': idx,
                    'questions': self.questions[idx], 
                    'feedbacks': self.feedbacks[idx],
                    'n_feedbacks': self.n_feedbacks[idx],
                    'answers': self.answers[idx],
                    'sub_answers': self.sub_answers[idx],
                    'candidates': positive_context + negative_context}
        else:
            return {'index': idx,
                    'questions': self.questions[idx], 
                    'feedbacks': self.feedbacks[idx],
                    'n_feedbacks': self.n_feedbacks[idx],
                    'answers': self.answers[idx],
                    'sub_answers': self.sub_answers[idx],
                    'candidates': [{"text": "", "title": ""}, {"text": "", "title": ""}]}

@dataclass
class ContextQACollator(DefaultDataCollator):
    tokenizer_r: Union[PreTrainedTokenizerBase] = None
    tokenizer_g: Union[PreTrainedTokenizerBase] = None
    truncation: Union[bool, str] = True
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_src_length: Union[int] = 256
    pad_to_multiple_of: Optional[int] = None

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
        batch['answers'] = [f['sub_answers'] for f in features] 
        batch['candidates'] = [f['candidates'] for f in features] 
        batch['inputs_for_retriever'] = batch_r
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
        # original_pad_token = self.tokenizer_r.pad_token
        # self.tokenizer_r.pad_token = self.tokenizer_r.mask_token
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
        # self.tokenizer_r.pad_token = original_pad_token

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

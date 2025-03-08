import random
import json
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
import csv

# from beir.datasets.data_loader import GenericDataLoader
from data.ir_dataloader import IRDataLoader

class PRFDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        split='test',
        n_max_segments=10,
        n_negative_samples=2,
        quick_test=None,
        max_examples=None,
        **kwargs
    ):
        # nq has separated set
        if ('nq' in dataset_dir) and (split == 'train'):
            dataset_dir = dataset_dir.replace('nq', 'nq-train')

        corpus, self.queries, self.qrels = IRDataLoader(data_folder=dataset_dir).load(split=split)
        self.dataset_dir = dataset_dir
        self.corpus = corpus
        self.split = split

        # remove qrels without positive
        for qid in self.qrels:
            scores = list(self.qrels[qid].values())
            if not any([int(score) >= 1 for score in scores]):
                del self.queries[qid]
        self.length = len(self.queries)
        self.ids = list(self.queries.keys())
        self.corpus_ids = list(self.corpus.keys())

        # else:
        #     max_qrels = random.sample(self.qrels.keys(), len(self.qrels))[:max_examples]
        #     self.qrels = {k: self.qrels[k] for k in max_qrels}
        #     self.length = len(self.qrels)
        #     self.ids = list(self.qrels.keys())
        #     judged_docids = []
        #     for qid in self.qrels:
        #         judged_docids += [docid for docid in self.qrels[qid]]
        #     self.corpus = {id: passage for id, passage in self.corpus.items() if id in judged_docids}
        #     self.corpus_ids = list(self.corpus.keys())

        ## training attributes
        self.n_max_segments = n_max_segments
        self.n_negative_samples = n_negative_samples

        ## dynamic attributes
        self.n_feedbacks = [0] * self.length
        self.feedbacks = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]

    def __len__(self):
        return self.length

    def add_feedback(self, idx, fbk):
        if self.n_feedbacks[idx] == len(self.feedbacks[idx]):
            self.n_feedbacks[idx] = 1
            self.feedbacks[idx]= ([fbk] + self.feedbacks[idx])[:self.n_max_segments]
        else:
            n = self.n_feedbacks[idx]
            self.feedbacks[idx][n] = fbk 
            self.n_feedbacks[idx] += 1

    def __getitem__(self, idx):
        id = self.ids[idx]

        n = self.n_feedbacks[idx]
        query = self.queries[id]

        # positive
        judged_positive_ids = [pid for pid, score in self.qrels[id].items() if int(score) >= 1]
        positive_id = random.sample(judged_positive_ids, 1)[0]
        positive = self.corpus[positive_id]

        # negative (use judged if it has)
        try:
            judged_negative_ids = [pid for pid, score in self.qrels[id].items() if score < 1]
        except:
            judged_negative_ids = []

        negative_ids = random.sample(
                self.corpus_ids, max(0, self.n_negative_samples - len(judged_negative_ids))
            ) + judged_negative_ids[:self.n_negative_samples]
        negatives = [self.corpus[pid] for pid in negative_ids]

        # outputs
        return {'index': idx,
                'query': query,
                'feedbacks': self.feedbacks[idx],
                'n_feedbacks': n, 
                'contexts': [positive] + negatives }

    # [TODO] make it for unsueprvised learning
    # def get_random_crop(self):
    #     crops = {}
    #     for id, passage in self.corpus.items():
    #         passage = passage['text'].split('. ')
    #         random.shuffle(passage)
    #         n = 1 + len(passage) // 2
    #         crops[id] = ". ".join(passage[:n])
    #     return crops


@dataclass
class PRFCollator(DefaultDataCollator):
    tokenizer: Union[PreTrainedTokenizerBase] = None
    truncation: Union[bool, str] = True
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_src_length: Union[int] = 256
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        batch_r = self.get_inputs_for_retriever(features)
        batch['index'] = [f['index'] for f in features] # we record it 
        batch['query'] = [f['query'] for f in features] 
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

        # Query
        ## Initial query
        initial_q = self.tokenizer(
            [f['query'] for f in features],
            add_special_tokens=True,
            max_length=64,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        ).to(device)
        batch_r['q_tokens'] = [initial_q['input_ids']]
        batch_r['q_masks'] = [initial_q['attention_mask']]
        batch_r['q_types'] = [initial_q['token_type_ids']]

        ## Feedbacks as followup query
        for seg_num in range(n_max_segments): 
            batch_feedback_q = [ features[b]['feedbacks'][seg_num] for b in range(batch_size) ]
            feedback_q = self.tokenizer(
                [f['query'] for f in features], [fbk for fbk in batch_feedback_q],
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            ).to(device)
            batch_r['q_tokens'].append(feedback_q['input_ids'])
            batch_r['q_masks'].append(feedback_q['attention_mask'])
            batch_r['q_types'].append(feedback_q['token_type_ids'])

        # Document # positive + (negative if it has)
        candidate_size = len(features[0]['contexts'])
        batch_r['d_tokens'] = []
        batch_r['d_masks'] = []

        for i in range(candidate_size):
            candidate = self.tokenizer(
                [f"{features[b]['contexts'][i]['title']} {features[b]['contexts'][i]['text']}".strip() 
                    for b in range(batch_size)],
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            ).to(device)
            batch_r['d_tokens'].append(candidate['input_ids'])
            batch_r['d_masks'].append(candidate['attention_mask'])

        return batch_r

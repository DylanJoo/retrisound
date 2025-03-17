import os
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
from datasets import load_dataset
from data.ir_dataloader import IRDataLoader

class PRFDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        split='test',
        n_max_segments=10,
        n_negative_samples=2,
        quick_test=None,
        **kwargs
    ):
        # nq has separated set
        if ('nq' in dataset_dir) and (split == 'train'):
            dataset_dir = dataset_dir.replace('nq', 'nq-train')

        # load from local
        if os.path.exists(dataset_dir):
            corpus, self.queries, self.qrels = IRDataLoader(data_folder=dataset_dir).load(split=split)
        # load from ir_datasets
        else: 
            corpus, self.queries, self.qrels = IRDataLoader(prefix=dataset_dir).load_from_ir_datasets(
                ignore_corpus=('trec-dl' in dataset_dir)
            )

        self.dataset_dir = dataset_dir
        self.corpus = corpus
        self.split = split

        # remove queries that have only negative qrels
        for qid in self.qrels:
            scores = [s for docid, s in self.qrels.get(qid, {'dummy': -1}).items()]
            if not any([int(score) >= 1 for score in scores]):
                del self.queries[qid]

        # remove queries that have no qrels
        self.queries = {k: v for k, v in self.queries.items() if k in self.qrels}

        self.length = len(self.queries)
        self.ids = list(self.queries.keys())
        self.corpus_ids = list(self.corpus.keys()) if self.corpus else []

        ## training attributes
        self.n_max_segments = n_max_segments
        self.n_negative_samples = n_negative_samples

        ## dynamic attributes
        self.n_feedbacks = [0] * self.length
        self.feedbacks = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]

    def load_prebuilt_feedback(self, feedback_file='feedbacks.jsonl'):
        if os.path.exists(self.dataset_dir):
            feedbacks = load_dataset('json', data_files=os.path.join(self.dataset_dir, feedback_file))
            feedbacks = {data['id']: data['text'] for data in feedbacks}
        elif 'msmarco-passage' in self.dataset_dir:
            feedbacks = load_dataset('intfloat/query2doc_msmarco', split='train')
            feedbacks = {data['query_id']: data['pseudo_doc'] for data in feedbacks}
        elif 'trec-dl-2019' in self.dataset_dir:
            feedbacks = load_dataset('intfloat/query2doc_msmarco', split='trec_dl2019')
            feedbacks = {data['query_id']: data['pseudo_doc'] for data in feedbacks}
        elif 'trec-dl-2020' in self.dataset_dir:
            feedbacks = load_dataset('intfloat/query2doc_msmarco', split='trec_dl2020')
            feedbacks = {data['query_id']: data['pseudo_doc'] for data in feedbacks}

        for idx, qid in enumerate(self.ids):
            try:
                feedback = feedbacks.pop(qid, None)
            except:
                judged_positive_ids = [pid for pid, score in self.qrels[idx].items() if int(score) >= 1]
                positive_ids = random.sample(judged_positive_ids, 1)
                feedback = self.corpus[positive_ids[0]] if len(positive_ids) > 0 else ""

            self.feedbacks[idx][0] = feedback

    def __len__(self):
        return self.length

    def add_feedback(self, idx, fbk):
        if self.n_feedbacks[idx] == len(self.feedbacks[idx]):
            self.n_feedbacks[idx] = 1
            # self.feedbacks[idx]= ([fbk] + self.feedbacks[idx])[:self.n_max_segments]
            self.feedbacks[idx]= [fbk] + ["" for _ in range(self.n_max_segments-1)] # remove after reach the max
        else:
            n = self.n_feedbacks[idx]
            self.feedbacks[idx][n] = fbk 
            self.n_feedbacks[idx] += 1

    def __getitem__(self, idx):
        id = self.ids[idx]

        n = self.n_feedbacks[idx]
        query = self.queries[id]

        if self.split == 'test':
            return {'index': idx,
                    'query': query,
                    'feedbacks': self.feedbacks[idx],
                    'n_feedbacks': n, 
                    'contexts': [{"title": "", "text": ""}] * (1 + self.n_negative_samples) }

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


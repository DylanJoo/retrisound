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
from .utils import load_corpus_file, batch_iterator 

class PRFQADataset(Dataset):
    def __init__(
        self, 
        data_file, 
        split='test',
        n_max_segments=10,
        n_negative_samples=2,
        another_split_for_eval=None,
        corpus_dir=None,
        run_file=None,
        quick_test=None,
        **kwargs
    ):
        with open(data_file, 'r') as f:
            raw_data = json.load(f)[split]

        self.queries = {}
        self.answers = {}
        for qid in raw_data:
            self.question[qid] = raw_data[key_id]['ambiguous_question']
            self.answers[qid] = raw_data[key_id]['annotations'][0]['long_answer']

        self.corpus = self._load_corpus(corpus_dir)
        self.corpus_ids = list(self.corpus.keys())
        self.qrels = self._load_run(run_file)
        self.length = len(self.queries)
        self.ids = list(self.queries.keys())
        self.split = split
        self.quick_test = quick_test

        if run_file is not None:
            self._load_run(run_file)

        ## training attributes
        self.n_max_segments = n_max_segments
        self.n_negative_samples = n_negative_samples

        ## dynamic attributes
        self.n_feedbacks = [0] * self.length
        self.feedbacks = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]

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
                    self.corpus[docid] = docdict
            del corpora

    def _load_runs(self, file, negative_threshold=50):
        with open(file, 'r') as f:
            for line in f:
                qid, _, docid, rank, score, _ = line.strip().split()
                if rank == 1:
                    self.qrels[qid] = {docid: 1}
                if rank >= 50:
                    self.qrels[qid].update({docid: 0})

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
        positives = self.corpus[id]
        if self.split == 'test':
            query = self.queries[id]
            positives = self.corpus[id]
        else:
            query = self.queries[id]
            candidate_positive_ids = [pid for pid, score in self.qrels[id].items() if int(score) >= 1]
            positive_id = random.sample(candidate_positive_ids, 1)[0]
            positive = self.corpus[positive_id]

        try:
            candidate_negative_ids = [pid for pid, score in self.qrels[id].items() if score < 1]
            negative_ids = random.sample(candidate_negative_ids, self.n_negative_samples)
            negatives = [self.corpus[pid] for pid in negative_ids]
        except:
            negative_ids = random.sample(self.corpus_ids, self.n_negative_samples)
            negatives = [self.corpus[pid] for pid in negative_ids]

        # outputs
        return {'index': idx,
                'query': query,
                'feedbacks': self.feedbacks[idx],
                'n_feedbacks': n, 
                'contexts': [positive] + negatives }


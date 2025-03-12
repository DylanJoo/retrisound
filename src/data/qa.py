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
from data.qa_dataloader import QADataLoader

class PRFQADataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        split='test',
        n_max_segments=10,
        n_negative_samples=2,
        run_file=None,
        quick_test=None,
        **kwargs
    ):
        # as QA datasetsa are task-specific. Use the customized QAloader in QADataLoader
        self.corpus, self.queries, self.answers = QADataLoader(
            dataset_dir=dataset_dir, 
            split=split,
            corpus_path=kwargs.pop('corpus_path', None)
        ).load(split=split)

        self.dataset_dir = dataset_dir

        if split != 'test':
            self.length = len(self.queries)
            self.ids = list(self.queries.keys())
            self.corpus_ids = list(self.corpus.keys())
        else:
            raise NotImplementedError

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

        # positive == answer itself
        positive = self.answers[id]

        # negative (use judged if it has)
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

import random
import datetime
import torch
import numpy as np
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

from beir.datasets.data_loader import GenericDataLoader

class PRFDataset(Dataset):

    def __init__(
        self, 
        dataset_dir, 
        split='test',
        n_max_segments=10, 
        n_negative_samples=2,
        another_split_for_eval=None,
        judgement_file=None,
        quick_test=None,
        **kwargs
    ):
        # nq has separated set
        if ('nq' in dataset_dir) and (split == 'train'):
            dataset_dir = dataset_dir.replace('nq', 'nq-train')

        corpus, self.queries, self.qrels = GenericDataLoader(data_folder=dataset_dir).load(split=split)
        self.dataset_dir = dataset_dir
        self.corpus = corpus
        self.split = split

        # load another additional qrels if needed.
        if (another_split_for_eval is not None) and (split == 'train'):
            qrels_file = os.path.join(self.qrels_folder, another_split_for_eval + ".tsv")
            reader = csv.reader(open(qrels_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            next(reader)
            for id, row in enumerate(reader):
                query_id, corpus_id, score = row[0], row[1], int(row[2])
                if query_id not in self.eval_qrels:
                    self.eval_qrels[query_id] = {corpus_id: score}
                else:
                    self.eval_qrels[query_id][corpus_id] = score

        if split != 'test':
            self.length = len(self.queries)
            self.ids = list(self.queries.keys())
            self.corpus_ids = list(self.corpus.keys())

        if split == 'test':
            self.length = len(self.corpus)
            self.ids = list(self.corpus.keys())
            self.corpus_ids = list(self.corpus.keys())
            self.pseudo_queries = self.get_random_crop()

        ## training attributes
        self.n_max_segments = n_max_segments
        self.n_negative_samples = n_negative_samples

        ## dynamic attributes
        self.n_feedbacks = [0] * self.length
        self.feedbacks = [["" for _ in range(self.n_max_segments)] for _ in range(self.length)]

        # self.judgements = {}
        # for id in self.ids:
        #     self.judgements[id] = defaultdict(int)
        # self._load_judgement(judgement_file)

    def __len__(self):
        return self.length

    def add_feedback(self, idx, fbk):
        n = self.n_feedbacks[idx]
        self.feedbacks[idx][n] = fbk 
        self.n_feedbacks[idx] += 1

    def get_random_crop(self):
        crops = {}
        for id, passage in self.corpus.items():
            passage = passage['text'].split('. ')
            random.shuffle(passage)
            n = 1 + len(passage) // 2
            crops[id] = ". ".join(passage[:n])
        return crops

    def __getitem__(self, idx):
        id = self.ids[idx]

        n = self.n_feedbacks[idx]
        if self.split == 'test':
            query = self.queries[id] if n==0 else self.feedbacks[idx][n-1] # here maybe needs perturbation
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
            negative_ids = random.sample(self.corpus_ids, self.n_negative_samples)[0]
            negatives = [self.corpus[pid] for pid in negative_ids]

        # outputs
        return {'index': idx,
                'query': query,
                'feedbacks': self.feedbacks[idx],
                'n_feedbacks': n, 
                'contexts': [positive] + negatives }

    # def _load_judgement(self, file):
    #     file = (file or f'/home/dju/temp/judge-{datetime.datetime.now().strftime("%b%d-%I%m")}.txt')
    #     self.judgement_file = file
    #     try:
    #         with open(file, 'r') as f:
    #             for line in tqdm(f):
    #                 id, psgid, judge = line.strip().split()[:3]
    #                 self.judgements[id][psgid] = judge
    #     except:
    #         with open(file, 'w') as f:
    #             f.write("id\tpid\tj\tinfo\n")

    # def add_judgements(self, idx, judgements, info=None):
    #     id = self.ids[idx]
    #     with open(self.judgement_file, 'a') as f:
    #         for pid in judgements:
    #             j = judgements[pid]
    #
    #             if pid in self.qrels[id]:
    #                 judgements[pid] = 1
    #                 continue
    #
    #             self.judgements[id][pid] = j
    #             f.write(f"{id}\t{pid}\t{j}\t{info}\n")


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
        initial_q = self.tokenizer.batch_encode_plus(
            [f['query'] for f in features],
            add_special_tokens=True,
            max_length=64,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        ).to(device)
        batch_r['q_tokens'] = [initial_q['input_ids']]
        batch_r['q_masks'] = [initial_q['attention_mask']]

        ## Feedbacks as followup query
        for seg_num in range(n_max_segments): 
            batch_feedback_q = [ features[b]['feedbacks'][seg_num] for b in range(batch_size) ]
            feedback_q = self.tokenizer.batch_encode_plus(
                [fbk for fbk in batch_feedback_q],
                add_special_tokens=True,
                max_length=self.max_src_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors='pt'
            ).to(device)
            batch_r['q_tokens'].append(feedback_q['input_ids'])
            batch_r['q_masks'].append(feedback_q['attention_mask'])

        # Document # positive + (negative if it has)
        candidate_size = len(features[0]['contexts'])
        batch_r['d_tokens'] = []
        batch_r['d_masks'] = []

        for i in range(candidate_size):
            candidate = self.tokenizer.batch_encode_plus(
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

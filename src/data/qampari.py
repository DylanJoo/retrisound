import json
import numpy as np
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

class ContextQADataset(Dataset):

    def __init__(
        self, 
        data_file, 
        n_max_segments=10,
        corpus_file=None,
        budget=None,
        quick_test=None
    ):
        data = []
        with open(data_file, 'r') as f:
            for line in tqdm(f):
                data.append(json.loads(line.strip()))

        if quick_test is not None:
            data = data[:16]

        self.length = len(data)
        self.n_max_segments = n_max_segments

        ## fixed attributes 
        self.questions = [None] * self.length
        self.answer_list = [[] for _ in range(self.length)]
        self.proof = [[] for _ in range(self.length)] 
        self.gold_context = [[] for _ in range(self.length)]

        ## dynamic attributes
        self.context_pids = [[-1] for _ in range(self.length)] # will be empty in the begining
        self.actions = [["" for _ in range(self.n_max_segments - 1)] for _ in range(self.length)]
        self._build(data)

        if corpus_file:
            self._load_corpus(corpus_file)
        else:
            empty = {'text': "", 'title': ""}
            self.corpus = defaultdict(lambda: empty)

        ## add context buffer count
        ## None means everything
        self.budget = budget

    def _load_corpus(self, file):
        """ under-construction """
        self.corpus = defaultdict(dict)
        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.corpus[docid]['text'] = item['text']
                self.corpus[docid]['title'] = item['title']

    def _build(self, data):
        for idx in range(self.length):
            self.questions[idx] = data[idx]['question_text']

            for answer in data[idx]['answer_list']:
                self.answer_list[idx].append(answer['answer_text'])

                for proof in answer['proof']:
                    self.proof[idx].append(proof['proof_text'])
                    self.gold_context[idx].append(proof['pid']) # this is not actually the chunk_id for corpus

    def __len__(self):
        return len(self.questions)

    def add_action(self, idx, act):
        try:
            i = self.actions[idx].index("") # empty strings
            self.actions[idx][i] = act
        except: # means it's full
            self.actions[idx].pop(0) # remove the first one
            self.actions[idx].append(act)

    def adjust_context(self, idx, act, context):
        # act: the choices of action for adjusting context. ['add', 'remove', ]
        # self.context[idx] += [context]
        raise NotImplementedError

    def __getitem__(self, idx):
        pids = self.context_pids[idx][:self.budget]
        return {'question': self.questions[idx], 
                'actions': self.actions[idx],
                'answers': self.answer_list[idx],
                'contexts': [self.corpus[pid] for pid in pids],
                'pids': pids}


@dataclass
class ContextQACollator(DefaultDataCollator):
    tokenizer_r: Union[PreTrainedTokenizerBase] = None
    tokenizer_g: Union[PreTrainedTokenizerBase] = None
    truncation: Union[bool, str] = True
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_src_length: Union[int] = 256
    max_tgt_length: Union[int] = 16
    pad_to_multiple_of: Optional[int] = None
    # n_max_segments: Union[int] = 16 # should be done in dataset
    # from model options
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
        n_max_segments = len(features[0]['actions']) + 1 

        # encode the initial request
        initial_q = self.tokenizer_r.batch_encode_plus(
            [f['question'] for f in features],
            add_special_tokens=True,
            max_length=self.max_src_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        )
        batch_r['input_ids'] = [initial_q['input_ids']]
        batch_r['attention_mask'] = [initial_q['attention_mask']]

        # encode the action/followup text
        ## encode the action segment-by-segment
        for seg_num in range(n_max_segments - 1): # this should iterate over items
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
            batch_r['input_ids'].append(action_q['input_ids'])
            batch_r['attention_mask'].append(action_q['attention_mask'])

        ## rewarding: (1) token adjustment (2) mask for calculating likelihood
        ## Here should consider the decoder's input and output template
        # targets = self.tokenizer_g.batch_encode_plus(
        #     [", ".join(f['answers']) for f in features],  # maybe here can do sth, like ordering
        #     max_length=self.max_tgt_length,
        #     truncation=self.truncation,
        #     padding=self.padding,
        #     return_tensors='pt'
        # )
        batch['question'] = [f['question'] for f in features] # R, Rprec
        batch['contexts'] = [f['contexts'] for f in features] # R, Rprec
        batch['answers'] = [f['answers'] for f in features] # R, Rprec
        batch['pids'] = [f['pids'] for f in features] # evid-R, evid-Rprec
        batch['inputs_for_retriever'] = batch_r
        # batch['labels'] = targets.input_ids  # done in model loop
        # batch['decoder_attention_mask'] = targets.attention_mask 
        return batch



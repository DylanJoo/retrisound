import os
import json
import datetime
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers import DefaultDataCollator
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy, 
)

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

        # Initial query
        initial_q = self.tokenizer(
            [f['query'] for f in features],
            add_special_tokens=True,
            max_length=128,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors='pt'
        ).to(device)
        batch_r['q_tokens'] = [initial_q['input_ids']]
        batch_r['q_masks'] = [initial_q['attention_mask']]
        batch_r['q_types'] = [initial_q['token_type_ids']]

        # Feedback from LLM or pre-generated 
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

        # Document (one positive with multiple negative)
        candidate_size = len(features[0]['contexts'])
        batch_r['d_tokens'] = []
        batch_r['d_masks'] = []

        ## d_tokens: [ positive: [B, L], negative: [B, L], ... ]
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
import math

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from transformers import PreTrainedModel, AutoTokenizer
from modeling.utils import SubsetOperator

class SelectionHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.encoder = encoder
        self.gumbel_topk = SubsetOperator(k=1000, hard=True)

    def forward(self, input_ids, attention_mask, query_rep=None):
        logits = self.encoder(input_ids, attention_mask).logits

        # regression
        values = torch.sigmoid(logits.max(1).values) 

        # Gumbel top-k
        logprobs = F.log_softmax(logits, dim=-1)
        actions = self.gumbel_topk(logits) # B V
        logprobs = all_logprobs * actions # B V

        # monte-carlo
        # dist = torch.distributions.bernoulli(values) 
        # actions = dist.sample()
        # logprobs = dist.log_prob(actions)
        return values, logprobs, actions

class RegularizationHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, query_rep=None):
        reps = self.encoder(input_ids, attention_mask).reps

        # regression
        # values = torch.sigmoid(logits.max(1).values) # aggregate at sequence dim

        # monte-carlo
        # dist = torch.distributions.bernoulli(values) 
        # actions = dist.sample()
        # logprobs = dist.log_prob(actions)
        return values, logprobs, actions

class AnsweringHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.encoder = encoder
        self.gumbel_topk = SubsetOperator(k=1000, hard=True)

    def forward(self, input_ids, attention_mask, query_rep=None):
        reps = self.encoder(input_ids, attention_mask).reps
        all_logprobs = F.log_softmax(reps, dim=-1)             # B L V
        selections = self.gumbel_topk(reps, hard=True, dim=-1) # B L V
        logprobs = (all_logprobs * selections).sum(1)          # B V

        actions = reps

        return values, logprobs, actions
        # logits = self.encoder(input_ids, attention_mask).logits
        # values = torch.sigmoid(logits.max(1).values) 
        # logprobs = F.log_softmax(logits, dim=-1)
        # actions = self.gumbel_topk(logits) # B V
        # logprobs = all_logprobs * actions # B V
        # return values, logprobs, actions

@dataclass
class EncodersOutput(BaseModelOutput):
    q_reps: torch.FloatTensor = None
    q_logprobs: torch.FloatTensor = None
    q_values: torch.FloatTensor = None
    q_actions: torch.FloatTensor = None
    d_reps: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class SparseAdaptiveEncoders(nn.Module):
    def __init__(
        self, 
        opt, 
        encoder, 
        modifier=None,
        d_encoder=None,
        n_candidates=None
    ):
        super().__init__()
        self.opt = opt
        
        # modeling
        self.q_encoder = encoder
        self.d_encoder = encoder
        self.modifier = modifier
        self.tau = opt.tau
        self.n_candidates = n_candidates

        self.tokenizer = AutoTokenizer.from_pretrained('naver/splade-v3')

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            elif 'q_encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, q_tokens, q_masks, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)

        # encode query request and query feedback
        q_logprobs = []
        q_reps = []
        q_values = []
        q_actions = []
        for i in range(max_num_steps):
            if i == 0:
                q_rep = self.q_encoder(q_tokens[0], q_masks[0]).rep
                q_value, q_logprob, q_action = self.modifier(q_tokens[0], q_masks[0]) 
            else:
                q_value, q_logprob, q_action = self.modifier(q_tokens[i], q_masks[i])
                q_rep = q_action
                # q_rep =  q_reps[0] * q_action
                # q_rep = q_reps[0] + q_action

            q_reps.append(q_rep)
            q_values.append(q_value)
            q_actions.append(q_action)
            q_logprobs.append(q_logprob.sum(-1))

        q_reps = torch.stack(q_reps, dim=1) # B N_seg V
        q_values = torch.stack(q_values, dim=1) # B N_seg V
        q_actions = torch.stack(q_actions, dim=1) # B N_seg V
        q_logprobs = torch.stack(q_logprobs, dim=1) # B N_seg

        # loss calculation
        scores, loss_r = None, 0.0
        CELoss = nn.CrossEntropyLoss()

        # encode document if using contrastive signals
        d_reps = []
        if (d_tokens is not None):
            n_candidates = (self.n_candidates or len(d_tokens))
            for i in range(n_candidates):
                d_rep = self.d_encoder(d_tokens[i], d_masks[i]).rep  # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=1) # B N_cand H

            scores = (q_reps[:, 0, :]/self.tau) @ (d_reps[:, 0, :]).T
            labels = torch.arange(0, batch_size, dtype=torch.long, device=q_reps.device)
            loss_r = CELoss(scores, labels) # first query and document

        return EncodersOutput(
            q_reps=q_reps,
            q_logprobs=q_logprobs,
            q_values=q_values,
            q_actions=q_actions,
            d_reps=d_reps,
            loss=loss_r,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)

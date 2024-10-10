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
from transformers import PreTrainedModel

class RegularizationHead(nn.Module):
    def __init__(self, opt, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        logits = self.encoder(input_ids, attention_mask).logits
        values = torch.sigmoid(logits[:, 0, :])
        return values

@dataclass
class EncodersOutput(BaseModelOutput):
    q_reps: torch.FloatTensor = None
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

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            elif 'q_encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, q_tokens, q_masks, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('max_num_steps', n_segments)
        batch_size = q_tokens[0].size(0)

        # encode query request and query feedback
        q_reps = []
        for i in range(max_num_steps):
            if i == 0:
                q_rep = self.q_encoder(q_tokens[0], q_masks[0]).rep
            else:
                # print(q_masks[i])
                q_transform = self.modifier(q_tokens[i], q_masks[i]) 
                q_rep = q_reps[-1] * q_transform
            q_reps.append(q_rep)
        q_reps = torch.stack(q_reps, dim=1) # B N_seg H

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
            d_reps=d_reps,
            loss=loss_r,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.model.gradient_checkpointing_enable(**kwargs)

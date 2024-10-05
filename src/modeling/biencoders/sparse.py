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
# from .dist_utils import gather

@dataclass
class DualencoderOutput(BaseModelOutput):
    qembs: torch.FloatTensor = None
    dembs: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class PlusModifierHead(nn.Module):
    def __init__(self, input_size, output_size=None, **kwargs):
        super().__init__()
        output_size = (output_size or input_size)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, query, feedback):
        if self.fc is None:
            return (query + feedback)/2
        else:
            query = self.fc(query)
            return query + feedback

class DualSparseEncoder(PreTrainedModel):
    def __init__(
        self, 
        opt, 
        encoder, 
        f_encoder,
        d_encoder=None,
    ):
        super().__init__()
        self.opt = opt
        
        # modeling
        self.f_encoder = f_encoder
        if self.opt.shared:
            self.q_encoder = encoder
        else:
            self.q_encoder = encoder
            self.d_encoder = d_encoder

        self.modifier = PlusModifierHead(
            encoder.config.hidden_size,
            encoder.config.hidden_size,
        )
        self.tau = opt.tau
        self.n_candidates = n_candidates

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            elif 'q_encoder' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def encoder_queries(self, to_dense=True, **queries):
        if to_dense:
            return self.q_encoder(**queries).to_dense(reduce="sum")
        else:
            return self.q_encoder(**queries)

    def encoder_docs(self, to_dense=True, **docs):
        if to_dense:
            if self.opt.shared:
                return self.q_encoder(**docs).to_dense(reduce="sum")
            else:
                return self.d_encoder(**docs)
        else:
            if self.opt.shared:
                return self.q_encoder(**docs).to_dense(reduce="sum")
            else:
                return self.d_encoder(**docs)

    def forward(self, q_tokens, q_masks, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        max_num_steps = kwargs.pop('max_num_steps', n_segments)
        batch_size = q_tokens[0].size(0)

        # encode query request and query feedback
        q_reps = []
        for i in range(max_num_steps):
            if i == 0:
                q_rep = self.q_encoder.encode_queries(q_tokens[0], q_masks[0])
            else:
                f_rep = self.f_encoder.encode_queries(q_tokens[i], q_masks[i])  # B H
                q_rep = self.modifier(q_rep[-1], f_rep) # can be either first q or modified q
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
                d_rep = self.d_encode_docs(d_tokens[i], d_masks[i])  # B H
                d_reps.append(d_rep)
            d_reps = torch.stack(d_reps, dim=1) # B N_cand H

            scores = (q_reps[:, 0, :]/self.tau) @ (d_reps[:, 0, :]).T
            labels = torch.arange(0, batch_size, dtype=torch.long, device=qembs.device)
            loss_r = CELoss(scores, labels) # first query and document

        return BiencoderOutput(
            qembs=qembs,
            dembs=dembs,
            loss=loss_r,
            scores=scores, 
            logs={'InfoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.qr_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.qf_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.d_encoder.gradient_checkpointing_enable(**kwargs)

    def get_encoder(self):
        return self.qf_encoder, self.qf_encoder

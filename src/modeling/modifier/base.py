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
# from .dist_utils import gather

@dataclass
class BiencoderOutput(BaseModelOutput):
    qembs: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    scores: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class ModifierHead(nn.Module):
    """
    Type1: fuse two independant embeddings (query-request and query-feedback)
        - query = Porj (query-request (B 1 H), query-feedback (B 1 H))
    """
    def __init__(self, input_size, output_size=None, **kwargs):
        super().__init__()
        output_size = (output_size or input_size)
        self.fc_1 = nn.Linear(input_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        dropout_prob = kwargs.pop("dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob != 0.0 else nn.Identity()

    def forward(self, qremb, qfemb):
        emb = torch.cat( (qremb, qfemb), -1)
        output = self.dropout(emb)
        if output.dtype != self.fc_1.weight.dtype:
            output = output.to(self.fc_1.weight.dtype)
        output = self.fc_2(self.fc_1(output))
        return output

class FeedbackQueryModifier(nn.Module):

    def __init__(
        self, 
        opt, 
        qr_encoder, 
        qf_encoder=None,
        d_encoder=None,
    ):
        super().__init__()
        self.opt = opt
        self.qr_encoder = qr_encoder
        self.qf_encoder = qf_encoder
        self.d_encoder = (d_encoder or qr_encoder)

        self.is_ddp = dist.is_initialized()
        self.tau = opt.tau
        self.modifier = ModifierHead(
            qr_encoder.config.hidden_size + qf_encoder.config.hidden_size,
            qr_encoder.config.hidden_size
        )

        for n, p in self.named_parameters():
            if 'd_encoder' in n:
                p.requires_grad = False
            if 'qr_encoder' in n:
                p.requires_grad = False

    def forward(self, q_tokens, q_masks, d_tokens=None, d_masks=None, **kwargs):
        n_segments = len(q_tokens)
        include_n_feedbacks = kwargs.pop('include_n_feedbacks', n_segments)
        batch_size = q_tokens[0].size(0)

        # encode query request and query feedback
        qembs = []
        for i in range(include_n_feedbacks):
            if i == 0:
                qemb = self.qr_encoder(q_tokens[0], q_masks[0]).emb
            else:
                qfemb = self.qf_encoder(q_tokens[i], q_masks[i]).emb  # B H
                qemb = self.modifier(qembs[0], qfemb)
            qembs.append(qemb)
        qembs = torch.stack(qembs, dim=1) # B N_seg H

        # encode document if applicable
        scores, loss_r = None, 0.0
        if (d_tokens is not None):
            dembs = self.d_encoder(d_tokens[0], d_masks[0]).emb  # B H

            # if self.is_ddp:
            #     gather_fn = gather
            #     demb_ibn = gather_fn(demb_ibn)

            labels = torch.arange(0, batch_size, dtype=torch.long, device=qembs.device)
            scores = torch.einsum("id, jd->ij", qembs[:, 0, :]/self.tau, dembs)

            CELoss = nn.CrossEntropyLoss()
            loss_r = CELoss(scores, labels)

        # conetxt list-wise ranking for b-th batch
        # logits = scores = torch.max(all_scores, 1).values # B N_cand
        # ranking = (-scores).argsort(-1) # B N_cand

        return BiencoderOutput(
            qembs=qembs,
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

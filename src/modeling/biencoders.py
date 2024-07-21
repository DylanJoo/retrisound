import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
# from .dist_utils import gather
from dist_utils import gather

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    probs: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class RankingValueHead(nn.Module):
    """ estimate the value of entire ranking """

    def __init__(self, input_size, **kwargs):
        super().__init__()

        self.summary = nn.Linear(input_size, 1)
        summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.0)
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob != 0.0 else nn.Identity()

    def forward(self, logits):
        output = self.dropout(logits)

        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output

class AdaptiveReranker(nn.Module):

    def __init__(
        self, 
        opt, 
        q_encoder, 
        d_encoder=None,
        n_candidates=10,
        do_contrastive=False,
    ):
        super().__init__()
        self.opt = opt
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder
        self.is_ddp = dist.is_initialized()
        self.tau = opt.tau
        self.vhead = RankingValueHead(input_size=10)

        self.do_contrastive = do_contrastive

    def forward(
        self, 
        q_tokens, q_mask, 
        d_tokens, d_mask, 
        **kwargs
    ):
        loss, loss_r = 0.0, 0.0
        n_segments = len(q_tokens)
        n_candidates = len(d_tokens)
        batch_size = q_tokens[0].size(0)

        qembs = self.q_encoder(q_tokens, q_mask).last_hidden_state  # B N_seg H

        dembs = []
        for i in range(n_candidates):
            demb = self.d_encoder(d_tokens[i], d_mask[i]).emb  # B H
            dembs.append(demb)
        dembs = torch.stack(dembs, dim=1) # B N_cand H

        ## 1) conetxt list-wise ranking for b-th batch
        ### ranking (candidates) <- N_seg N_cand
        # alpha = 0
        # r_ranking = 1/(alpha + 1 + (-score).argsort(-1)) # reciprocal

        ### mode1: max pooling over segmentes
        sim_logits = qembs @ dembs.mT # B N_seg N_cand
        probs = sim_logits
        # probs = F.softmax(sim_logits, dim=1)
        logits = torch.max(probs, 1).values

        ## mode1: reciprocal
        # ranking_scores = 1/(alpha + 1 + (-scores).argsort(-1)) 
        # reranking = ranking_scores.sum(-2).argsort(-1) # B N_cand

        ## 2) constrastive learning
        if self.do_contrastive:
            qemb_ibn = qembs[:, 0, :] # the first segment (B H)
            demb_ibn = dembs[:, 0, :] # 

            if self.is_ddp:
                gather_fn = gather
                demb_ibn = gather_fn(demb_ibn)

            labels = torch.arange(0, batch_size, dtype=torch.long, device=qemb_ibn.device)
            rel_scores = torch.einsum("id, jd->ij", qemb_ibn/self.tau, demb_ibn)
            CELoss = nn.CrossEntropyLoss()
            loss_r = CELoss(rel_scores, labels)

        return InBatchOutput(
            loss=loss_r,
            probs=probs,
            logits=logits,
            logs={'infoNCE': loss_r}
        )

    def get_encoder(self):
        return self.q_encoder, None


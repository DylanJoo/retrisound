import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from .dist_utils import gather

@dataclass
class InBatchOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    reranking: Optional[torch.FloatTensor] = None
    indices: Optional[List] = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class InBatchInteraction(nn.Module):

    def __init__(
        self, 
        opt, 
        q_encoder, 
        d_encoder=None,
        miner=None, 
        fixed_d_encoder=False
    ):
        super().__init__()
        self.opt = opt
        self.q_encoder = q_encoder
        self.d_encoder = d_encoder if d_encoder is not None else copy.deepcopy(q_encoder)

        # distributed 
        # self.is_ddp = dist.is_initialized()
        self.is_ddp = False

        # learning hyperparameter
        self.tau = opt.temperature

        ## negative miner
        self.miner = miner
        self.n_negative_samples = opt.n_negative_samples

    def forward(
        self, 
        q_tokens, q_mask, 
        d_tokens, d_mask, 
        data_index=None,
        **kwargs
    ):
        loss = 0.0
        loss_ret = 0.0
        batch_size = len(q_tokens)
        max_length = d_tokens[0]

        qembs = self.q_encoder(
            input_ids=q_tokens, 
            attention_mask=q_mask
        ).last_hidden_state  # B N_seg H

        dembs = []
        for i in range(len(d_tokens)):
            demb = self.d_encoder(
                input_ids=d_tokens[i], 
                attention_mask=d_mask[i]
            ).emb  # B H
            dembs.append(demb)
        dembs = torch.stack(dembs, dim=1) # B N_cand H

        ## 1) conetxt list-wise ranking for b-th batch
        ### q <- qembs[b, :, :] N_seg H 
        ### d <- dembs[b, :, :] N_cand H 
        ### ranking (candidates) <- N_seg N_cand
        reranking = []
        alpha = 0
        # for b in range(batch_size):
        #     score = qembs[b] @ demb[b].T # (N_seg H) x (N_cand H)
        #     r_ranking = 1/(alpha + 1 + (-score).argsort(-1)) # reciprocal
        #     print(r_ranking)
        #     reranking.append(r_ranking.sum(-2)) # N_seg x N_cand
        # print(reranking)
        # reranking = torch.stack(reranking, dim=0)

        scores = qembs @ dembs.mT # (B N_seg H) x (B N_cand H) = B N_seg N_cand
        r_ranking = 1/(alpha + 1 + (-scores).argsort(-1)) # reciprocal
        reranking = r_ranking.sum(-2).argsort(-1).detach() # B N_cand

        ## 2) constrastive learning
        ### q <- qembs[:, 0, :] B (1) H. the first segment
        ### d <- dembs[:, 0, :] B (1) H. the first context (would change)
        ### scores (constrastive) <- B B 
        qemb_ibn = qembs[:, 0, :] # B H
        demb_ibn = dembs[:, 0, :] # B H
        ## [NOTE] here can also add negative by selecting bottom doc in cand

        if self.is_ddp:
            gather_fn = gather
            demb_ibn = gather_fn(demb_ibn)

        labels = torch.arange(
            0, qemb_ibn.size(0), 
            dtype=torch.long, 
            device=qemb_ibn.device
        )

        rel_scores = torch.einsum("id, jd->ij", qemb_ibn/self.tau, demb_ibn)

        ## computing losses
        CELoss = nn.CrossEntropyLoss()
        loss_ret = CELoss(rel_scores, labels)
        logs = {'infoNCE': loss_ret}

        return InBatchOutput(
            loss=loss_ret,
            reranking=reranking,
            indices=data_index,
            logs=logs, 
        )

    def get_encoder(self):
        return self.q_encoder, None


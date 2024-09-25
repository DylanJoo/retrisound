import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class ValueOutput(BaseModelOutput):
    logits: torch.FloatTensor = None
    last_hidden_states: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    logs: Optional[Dict[str, torch.FloatTensor]] = None

class RankingValueHead(nn.Module):
    """ estimate the value of entire ranking """

    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, 768)
        self.fc_2 = nn.Linear(768, 1)
        dropout_prob = kwargs.pop("dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob != 0.0 else nn.Identity()

    def forward(self, logits):
        output = self.dropout(logits)
        if output.dtype != self.fc_1.weight.dtype:
            output = output.to(self.fc_1.weight.dtype)
        output = self.fc_2(self.fc_1(output))
        return output

class ValueCrossEncoder(nn.Module):

    def __init__(
        self, 
        opt, 
        cross_encoder,
        d_encoder,
        n_max_candidates=None,
    ):
        super().__init__()
        self.opt = opt
        # BERTForSequenceClassification
        self.cross_encoder = cross_encoder
        self.d_encoder = d_encoder 
        self.n_max_candidates = n_max_candidates

    @staticmethod
    def _maybe_reshape(x):
        if x.dim() != 3:
            x = x[:, None, :]
        return x

    def _prepare_inputs(self, qemb, dembs):
        """ listwise re-ranker using embeddings to represent piece of texts """
        # prepare special tokens
        cls, sep, device = 101, 102, self.cross_encoder.device
        # size: [1, 2]
        embeds = self.cross_encoder.bert.embeddings(
            torch.tensor([[cls, sep]]).repeat( (qemb.size(0), 1) )
        ).to(device) 
        cls_emb = embeds[:, 0:1]
        sep_emb = embeds[:, 1:2]

        # prepare text embeddings
        qemb = self._maybe_reshape(qemb) # B N H
        dembs = self._maybe_reshape(dembs) # B M H

        # concat everything
        print(cls_emb.shape)
        print(sep_emb.shape)
        print(qemb.shape)
        print(dembs.shape)
        embeds = torch.cat([cls_emb, qemb, sep_emb, dembs, sep_emb], axis=1)
        token_type_ids = torch.ones( (qemb.size(0), (3+dembs.size(-2)+1)) , dtype=torch.long)
        token_type_ids[:, :3] = 0

        return {'input_ids': None,
                'inputs_embeds': embeds.to(device), 
                'attention_mask': None,
                'token_type_ids': token_type_ids.to(device),
                'output_hidden_states': True}

    ## [NOTE] Rewards ~ r = CrossEncoder(E_qr, E_qf)
    def forward(self, qembs, dembs, **kwargs):
        ## [CLS] <e_q> [SEP] <e_d1> <e_d2> ... [SEP]
        loss = 0.0
        batch_size = qembs.size(0)

        # encode candidates
        # B = 3 | N = 4 | M = 2
        logits = []
        print('debug')
        for i in range(qembs.size(1)):
            inputs = self._prepare_inputs(qembs[:, i], dembs) # (B (1) H) x (B M H)
            model_output = self.cross_encoder(**inputs, **kwargs)
            logits.append(model_output.logits)
        logits = torch.stack(logits, dim=1)     # B M 2
        logits = torch.max(logits, 1).values    # B 2

        return ValueOutput(
            logits=logits,
            last_hidden_states=model_output.hidden_states[-1], # this means nothing unless you check all
            loss=loss,
            logs={}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.cross_encoder.model.gradient_checkpointing_enable(**kwargs)


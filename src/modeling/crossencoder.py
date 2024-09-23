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
    qemb: torch.FloatTensor = None
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
        n_max_candidates=10,
    ):
        super().__init__()
        self.opt = opt
        self.cross_encoder = cross_encoder
        self.d_encoder = d_encoder
        self.n_max_candidates = n_max_candidates

    @staticmethod
    def _maybe_reshape(x):
        if x.dim() != 3:
            x = x[:, None, :]
        return x

    def _prepare_inputs(
        self,
        embed_1, 
        embed_2, 
        embeds_3,
    ):
        """ listwise re-ranker using embeddings to represent piece of texts """
        # prepare special tokens
        cls, sep, device = 101, 102, self.cross_encoder.device
        # size: [1, 2]
        embeds = self.cross_encoder.embeddings(
            torch.tensor([[cls, sep]]).repeat( (embed_1.size(0), 1) )
        ).to(device) 
        cls_emb = embeds[:, 0:1]
        sep_emb = embeds[:, 1:2]

        # prepare text embeddings
        embed_1 = self._maybe_reshape(embed_1)
        embed_2 = self._maybe_reshape(embed_2)

        # concat everything
        embeds = torch.cat(
            [cls_emb, embed_1, sep_emb, embed_2, sep_emb, embeds_3, sep_emb], axis=1
        )
        token_type_ids = torch.ones( (embed_1.size(0), (5+embeds_3.size(-2)+1)) , dtype=torch.long)
        token_type_ids[:, :5] = 0

        return {'input_ids': None,
                'inputs_embeds': embeds.to(device), 
                'attention_mask': None,
                'token_type_ids': token_type_ids.to(device)}

    ## [NOTE] Rewards ~ r = CrossEncoder(E_qr, E_qf)
    def forward(
        self, 
        qr_embed, 
        qf_embed,
        c_tokens,
        c_masks,
        **kwargs
    ):
        ## [CLS] <e_qr> [SEP] <e_qf> [SEP] <e_d1> <e_d2> ... [SEP]
        loss, loss_r = 0.0, 0.0
        n_candidates = len(c_tokens)
        batch_size = c_tokens[0].size(0)

        # encode candidates
        c_embeds = []
        for i in range(n_candidates):
            c_embed = self.d_encoder(c_tokens[i], c_masks[i]).emb  # B H
            # get single embedding
            c_embeds.append(c_embed)
        c_embeds = torch.stack(c_embeds, dim=1) # B N_cand H

        inputs = self._prepare_inputs(qr_embed, qf_embed, c_embeds)
        model_output = self.cross_encoder(**inputs, **kwargs)

        return ValueOutput(
            qemb=model_output.emb,
            last_hidden_states=model_output.last_hidden_states,
            loss=loss_r,
            logs={'infoNCE': loss_r}
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.cross_encoder.model.gradient_checkpointing_enable(**kwargs)


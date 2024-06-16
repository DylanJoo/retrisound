import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from modeling import dist_utils

@dataclass
class InBatchOutput(BaseModelOutput):
    ret_nll: torch.FloatTensor = None
    gen_nll: torch.FloatTensor = None
    rg_kl: torch.FloatTensor = None
    qemb: Optional[torch.FloatTensor] = None
    cemb: Optional[torch.FloatTensor] = None
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
        if fixed_d_encoder:
            for p in self.d_encoder.parameters():
                p.requires_grad = False

        # distributed 
        self.is_ddp = dist.is_initialized()

        # learning hyperparameter
        self.tau = opt.temperature

        ## negative miner
        self.miner = miner
        self.n_negative_samples = opt.n_negative_samples

    def forward(
        self, 
        q_tokens, q_mask, 
        c_tokens, c_mask, 
        data_index=None,
        **kwargs
    ):
        loss = 0.0
        loss_ret = 0.0
        loss_div = 0.0
        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask)[0]
        cemb = self.d_encoder(input_ids=c_tokens, attention_mask=c_mask)[0]

        if self.is_ddp:
            gather_fn = dist_utils.gather
            qemb = gather_fn(qemb)
            cemb = gather_fn(cemb)
            data_index = gather_fn(data_index).detach().cpu().numpy().tolist()

        bsz = qemb.size(0) # query-centric batch 
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        ## [st-st]
        if self.miner is not None:
            if self.miner.negative_jsonl is not None:
                # use prebuilt negatives
                neg_inputs = self.miner.batch_get_negative_inputs(
                        data_index,
                        n=self.n_negative_samples
                )
            else:
                # mine online
                neg_inputs = self.miner.crop_depedent_from_docs(
                        embeds_1=qemb.clone().detach().cpu(), 
                        embeds_2=cemb.clone().detach().cpu(),
                        indices=data_index,
                        n=self.n_negative_samples, k0=0, k=100, 
                        exclude_overlap=False,
                        to_return='spans_tokens',
                )
            neg_vectors = self.encoder(
                    input_ids=neg_inputs[0].to(self.encoder.device),
                    attention_mask=neg_inputs[1].to(self.encoder.device)
            )[0]
        else:
            neg_vectors = None

        if neg_vectors is not None:
            scores_q = torch.einsum("id, jd->ij", 
                    qemb / self.tau, torch.cat([cemb, neg_vectors], dim=0))
            scores_c = torch.einsum("id, jd->ij", 
                    cemb / self.tau, torch.cat([qemb, neg_vectors], dim=0))
        else:
            scores_q = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
            scores_c = torch.einsum("id, jd->ij", cemb / self.tau, qemb)

        ## computing losses
        CELoss = nn.CrossEntropyLoss()
        KLLoss = nn.KLDivLoss(reduction='batchmean')

        logs = {}
        loss_ret = (CELoss(scores_q, labels) + CELoss(scores_c, labels)) / 2

        logs.update({'ret_nll': loss_ret, 'gen_nll': loss_gen, 'div': loss_div})

        return InBatchOutput(
            loss=loss_ret + loss_gen + loss_div,
            acc=accuracy, 
            logs=logs, 
            qemb=qemb, 
        )

    def get_encoder(self):
        return self.q_encoder, None


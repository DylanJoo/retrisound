import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    ret_nll: torch.FloatTensor = None
    gen_nll: torch.FloatTensor = None
    rg_kl: torch.FloatTensor = None

class RerankAugmentedGeneration(nn.Module):

    def __init__(self, llm, biencoders=None):
        super().__init__()
        self.llm = llm
        self.biencoders = biencoders # could be inbatch

        # freeze G and R's d-encoder
        for p in self.llm.parameters():
            p.requires_grad = False

        if biencoders is not None:
            for p in self.biencoders.d_encoder.parameters():
                p.requires_grad = False

    def forward(
        self, 
        questions, 
        labels=None,
        passages=None,
        pids=None, 
        inputs_for_retrieval=None,
        **kwargs
    ):
        """
        params
        ------
        question: the initial questions.
        labels: this is for generation.
        passages: the retrieved passages (could be furtehr updated during train)
        pids: the context budgets
        """
        loss, loss_ret, loss_gen = 0.0, 0.0, 0.0

        ## step1. Retrieve documents and prepare the context
        if inputs_for_retrieval is not None:
            ### top-k retrieval 
            #### reranking by bi-encoders
            scores = self.biencoders(**inputs_for_retrieval)
            retrieved_pids = dosomething(scores)
            updated_pids = dosomething(retrieved_pids, pids)
            passages = [corpus[pid] for pid in updated_pids]
            # [TODO] can revise the pids with this one.

        ## step2. prepare context
        contexts = []
        for i in range(len(questions)):
            D = apply_docs_prompt(passages[i], field='text')
            prompt = apply_inst_prompt(
                Q=questions[i], D=D,
                instruction=instruction_prompt,
                add_prefix=True
            )
            contexts.append(prompt)
        ### tokenization
        inputs = self.tokenizer(prompts, return_tensors='pt').to(self.model.device)

        ## step3. Generate with prompt (with context)
        output = self.llm(**inputs, labels=labels)

        ## computing losses
        CELoss = nn.CrossEntropyLoss()
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        logs = {}

        logs.update({'ret_nll': loss_ret, 'gen_nll': loss_gen, 'div': loss_div})

        return RAGOutput(
            loss=loss_ret+loss_gen,
            ret_nll=loss_ret,
            gen_nll=loss_gen,
            rg_kl=None
        )

class RetrievalAugmentedGeneration():

    def __init__(self):
        pass

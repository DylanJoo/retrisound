import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from typing import Optional, Union, List, Dict, Tuple, Any

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers.modeling_outputs import BaseModelOutput
from prompt.qampari import *

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    ret_nll: torch.FloatTensor = None
    gen_nll: torch.FloatTensor = None
    rg_kl: torch.FloatTensor = None

class RerankAugmentedGeneration(nn.Module):

    def __init__(self, llm, tokenizer, biencoders=None):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.biencoders = biencoders # could be inbatch

        # freeze G and R's d-encoder
        for p in self.llm.parameters():
            p.requires_grad = False

        if biencoders is not None:
            for p in self.biencoders.d_encoder.parameters():
                p.requires_grad = False

    def forward(
        self, 
        question: List[str],
        answers: List[List[str]] = None,
        contexts: Optional[List[List[Dict]]] = None,
        pids: Optional[List[List[str]]] = None,
        inputs_for_retrieval: Optional[Dict] = None,
        **kwargs
    ):
        """
        params
        ------
        question: the initial questions.
        labels: this is for generation.
        contexts: i.e., the retrieved passages 
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
        ## [TODO] cleaner to move this section to another function
        prompts = []
        for i in range(len(question)):
            D = apply_docs_prompt(contexts[i], field='text')
            prompt = apply_inst_prompt(
                Q=question[i], 
                D=D,
                instruction=instruction_prompt,
                add_prefix=True
            ).replace('{DEMO}', '')
            prompts.append(prompt)

        ### Prepare source target
        inputs = self.tokenizer([f"{prompt} {answer}" \
            for (prompt, answer) in zip(prompts, answers)],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.llm.device)
        labels = inputs['input_ids'].clone()

        #### get source length
        source_len = [len(token) for token in self.tokenizer(
            prompts, truncation=True
        )]
        source_mask = torch.zeros( labels.shape)
        for i, s in enumerate(source_len):
            source_mask[i, :(s-1)] = 1
        #### replace label as
        labels.masked_fill_(source_mask.bool(), -100)

        ## step3. Generate with prompt (with context)
        output = self.llm(**inputs, labels=labels)

        ## computing losses
        CELoss = nn.CrossEntropyLoss()
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        logs = {'ret_nll': loss_ret, 'gen_nll': loss_gen}

        return RAGOutput(
            loss=loss_ret+loss_gen,
            ret_nll=loss_ret,
            gen_nll=loss_gen,
            rg_kl=None
        )

class RetrievalAugmentedGeneration():

    def __init__(self):
        pass

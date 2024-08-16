import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
from typing import Optional, Union, List, Dict, Tuple, Any
from torch.nn import CrossEntropyLoss

import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Mapping
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from prompts.qampari import *

from utils import get_expected_inputs

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    loss_r: torch.FloatTensor = None
    loss_g: torch.FloatTensor = None
    loss_kl: torch.FloatTensor = None
    answers: Optional[str] = None
    prompts_fbk: Optional[str] = None
    feedbacks: Optional[str] = None

class RerankAugmentedGenerationWrapper(LlamaForCausalLM):

    def __init__(
        self, 
        config, 
        stop_token_ids,
        num_budget=1,
        is_reference=False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.stop_token_ids = stop_token_ids
        self.num_budget = num_budget
        self.is_peft_model = False
        self.is_reference = is_reference
        self.tokenizer = None
        self.biencoders = None

        for n, p in self.model.named_parameters():
            if "embed_tokens" in n:
                p.requires_grad = False
            else:
                p.requires_grad = False

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_biencoders(self, biencoders):
        self.biencoders = biencoders
        for n, p in self.biencoders.d_encoder.named_parameters():
            p.requires_grad = False
        for n, p in self.biencoders.q_encoder.named_parameters():
            p.requires_grad = True

    def _forward_retrieval(
        self, 
        q_tokens, 
        q_mask, 
        d_tokens, 
        d_mask, 
        questions,
        candidates,
    ):
        if self.is_reference:
            contexts = [cand[:self.num_budget] for cand in candidates]
            output_r = None
        else:
            # retrieval
            output_r = self.biencoders(q_tokens, q_mask, d_tokens, d_mask)
            # prepare context
            assert len(candidates) == len(output_r.reranking)
            contexts = []
            for i, reranking in enumerate(output_r.reranking):
                reranked_context = [candidates[i][j] for j in reranking]
                contexts.append(reranked_context[:self.num_budget])

        # prepare prompts
        prompts = []
        prompts_fbk = []  
        prompts_last = [] 

        for i in range(len(questions)):
            ## for answering
            D = apply_docs_prompt(contexts[i], field='text')
            prompt = apply_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=instruction_prompt,
                prefix="Answer:"
            ).replace('{DEMO}', '')
            prompts.append(prompt)

            ## for getting feedback
            prompt_fbk = apply_fbk_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=fbk_instruction_prompt,
                prefix=fbk_prefix
            )
            prompts_fbk.append(prompt_fbk)

            D = apply_docs_prompt(candidates[i][:self.num_budget], field='text')
            prompt_last = apply_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=instruction_prompt,
                prefix="Answer:"
            ).replace('{DEMO}', '')
            prompts_last.append(prompt_last)

        return (prompts, prompts_fbk, prompts_last, output_r)

    def forward(
        self, 
        questions: List[str] = None,
        retriever_inputs: Optional[Dict] = None, 
        candidates: Optional[List[List[Dict]]] = None,
        responses: torch.tensor = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        **kwargs
    ):
        ## step0. 
        loss, loss_r, loss_g = 0.0, 0.0, 0.0

        if input_ids is None:
            prompts, prompts_fbk, prompts_last, output_r = \
                self._forward_retrieval(
                    **retriever_inputs, 
                    questions=questions, 
                    candidates=candidates
                )
            queries = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).input_ids.to(self.model.device)

            if responses is not None:
                input_ids = torch.cat([queries, responses], dim=1)
                attention_mask = input_ids != self.tokenizer.pad_token_id

        output_g = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        )

        return output_g


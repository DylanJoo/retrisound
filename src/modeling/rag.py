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
from transformers.modeling_outputs import BaseModelOutput
from prompt.qampari import *

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    loss_r: torch.FloatTensor = None
    loss_g: torch.FloatTensor = None
    loss_kl: torch.FloatTensor = None

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
        targets: List[str] = None,
        candidates: Optional[List[List[Dict]]] = None,
        k: int = 5,
        inputs_for_retriever: Optional[Dict] = None,
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
        loss, loss_r, loss_g = 0.0, 0.0, 0.0

        ## step1. Retrieve documents and prepare the context
        ## [NOTE] this could be move the `train.py`
        if inputs_for_retriever is not None:
            #### reranking by bi-encoders
            output_r = self.biencoders(**inputs_for_retriever)
            ## reorder/select documents
            contexts = []
            for candidate, ranking in zip(candidates, output_r.reranking):
                context = [p for p, _ in sorted(zip(candidate, ranking))]
                contexts.append(context[:k])

            # [TODO] this is the post-G actions
            # retrieved_pids = dosomething(output_r.scores)
            # updated_pids = dosomething(retrieved_pids, pids)
            # passages = [corpus[pid] for pid in updated_pids]
        else:
            contexts = [ctx[:k] for ctx in candidates]

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
        inputs = self.tokenizer([f"{prompt} {target}" \
            for (prompt, target) in zip(prompts, targets)],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.llm.device)

        #### get source length and revise labels
        labels = inputs['input_ids'].clone()
        tokenized_prompt = self.tokenizer(prompts, truncation=True)['input_ids']
        source_len = [len(tokens) for tokens in tokenized_prompt]
        for i, s in enumerate(source_len):
            labels[i, :(s-1)] = -100

        ## step3. Generate with prompt (with context)
        output_g = self.llm(**inputs, labels=None)

        ## verbose. it's slow
        # output = self.llm.generate(**inputs)
        # print(self.tokenizer.batch_decode(output))

        loss_r = output_r.loss
        # loss_g = output_g.loss # we will use loss from every example
        loss_g = self.compute_nll(output_g.logits, labels)
        logs = {'InfoNCE': loss_r, 'mle': loss_g.mean()}

        ## computing additional alignment loss
        # KLLoss = nn.KLDivLoss(reduction='batchmean')
        return RAGOutput(
            loss=loss_r+loss_g.mean(),
            loss_r=loss_r,
            loss_g=loss_g,
            loss_kl=None
        )

    def compute_nll(self, logits, labels):
        ## extract the batch-wise mean
        batch_size, _, vocab_size = logits.shape
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(batch_size, -1).mean(-1)
        return loss


class RetrievalAugmentedGeneration():

    def __init__(self):
        pass

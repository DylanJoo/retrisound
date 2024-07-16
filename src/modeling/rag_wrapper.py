"""
# computing additional alignment loss
KLLoss = nn.KLDivLoss(reduction='batchmean')
"""
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
from trl import AutoModelForCausalLMWithValueHead
from prompt.qampari import *

@dataclass
class RAGOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    loss_r: torch.FloatTensor = None
    loss_g: torch.FloatTensor = None
    loss_kl: torch.FloatTensor = None
    answers: Optional[str] = None
    prompts_fbk: Optional[str] = None
    feedbacks: Optional[str] = None

class RerankAugmentedGenerationWrapper(AutoModelForCausalLMWithValueHead):

    def __init__(
        self, 
        pretrained_model, 
        biencoders,
        stop_token_ids,
        num_budget,
        **kwargs
    ):
        super().__init__(pretrained_model, **kwargs)
        self.biencoders = biencoders
        self.stop_token_ids = stop_token_ids
        self.num_budget = num_budget
        self.is_peft_model = False

        # freeze params
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        for p in self.biencoders.d_encoder.parameters():
            p.requires_grad = False

    def _forward_retrieval(
        self, 
        q_tokens, 
        q_mask, 
        d_tokens, 
        d_mask, 
        questions,
        candidates,
    ):
        """
        params
        ------
        *_tokens: List of 3d tensor
        *_masks: List of 2d tensor
        candidate: List[Dict['text': str, 'title': str]]: 
            the N_cand passages for each question.  After ranking, 
            it would be the context for generator

        returns
        -------
        """
        # retrieval
        output_r = self.biencoders(q_tokens, q_mask, d_tokens, d_mask)

        # prepare context
        contexts = []
        for batch_i, (candidate, ranking) in enumerate(zip(candidates, output_r.reranking)):
            print(ranking)
            context = [p for p, r in sorted(zip(candidate, ranking))]
            contexts.append(context[:self.num_budget])

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
        responses: List[torch.tensor] = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        **kwargs
    ):
        """
        params for generator
        --------------------
        retriever_inputs: [R] the recurrent query and document inputs

        returns
        -------
        """
        ## step0. 
        loss, loss_r, loss_g = 0.0, 0.0, 0.0

        ## step1. prepare the context via retrieve/rerank documents
        ## step2. prepare prompt of context for generation
        ## [NOTE] this could be move the `train.py`
        if retriever_inputs is not None:
            prompts, prompts_fbk, prompts_last, output_r = \
                self._forward_retrieval(**retriever_inputs, questions=questions, candidates=candidates)

            ### Prepare query and response 
            model_inputs = self.tokenizer([f"{query} {response}" \
                for (query, response) in zip(prompts, responses)],
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(self.llm.device)
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']

            loss_r = output_r.loss 

        ## step3. forward pass with inputs 
        output_g = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        )

        ## step4. calculate likelihood  (optional)
        #### Revise labels
        # labels = input_ids.clone()
        # labels[labels==self.tokenizer.convert_tokens_to_ids(",")] = -100 
        # labels[labels==self.tokenizer.pad_token_id] = -100 # remove the padded tokens
        # tokenized_prompt = self.tokenizer(prompts, truncation=True)['input_ids']
        # source_len = [len(tokens) for tokens in tokenized_prompt]
        # for i, s in enumerate(source_len):
        #     labels[i, :(s-1)] = -100
        # loss_g = self.compute_nll(output_g.logits, labels)
        # logs = {'InfoNCE': loss_r, 'mle': loss_g.mean()}

        ## step4. generate feedbacks 
        # inputs_fbk = self.tokenizer(
        #     prompts_fbk,
        #     padding=True,
        #     truncation=True,
        #     return_tensors='pt'
        # ).to(self.llm.device)
        # output_fbk = self.llm.generate(
        #     **inputs_fbk, 
        #     do_sample=True,
        #     temperature=1.0,
        #     top_p=0.95,
        #     max_new_tokens=32,
        #     eos_token_id=self.stop_token_ids,
        #     pad_token_id=self.tokenizer.pad_token_id
        # )
        # feedbacks = []
        # for i, output in enumerate(output_fbk):
        #     feedback = self.tokenizer.decode(
        #         output[inputs_fbk['input_ids'][i].size(-1):], 
        #         skip_special_tokens=True
        #     )
        #     feedback = feedback.split('\n\n')[0]
        #     feedbacks.append(feedback)
        return output_g

        # return RAGOutput(
        #     loss=loss_r+loss_g.mean(),
        #     loss_r=loss_r,
        #     loss_g=loss_g,
        #     loss_kl=None,
        #     answers=None,
        #     prompts_fbk=prompts_fbk,
        #     feedbacks=feedbacks
        # )

    def gradient_checkpointing_enable(self, **kwargs):
        self.biencoders.q_encoder.model.gradient_checkpointing_enable(**kwargs)
        self.biencoders.d_encoder.gradient_checkpointing_enable(**kwargs)
        # self.llm.gradient_checkpointing_enable(**kwargs)

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

    def postprocess(self, x):
        x = x.split('\n\n')[0] 
        x = x.split('Question:')[0] 
        return x

    @torch.no_grad()
    def get_likelihood(self, prompts, targets):
        pad_second = self.tokenizer.padding_side == "right"
        model_inputs = self.tokenizer(
            [f"{prompt} {target}" for (prompt, target) in zip(prompts, targets)],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(ppo_trainer.device)

        #### Get the ignored input (without computing nll)   
        labels = model_inputs['input_ids'].clone()
        labels[labels==self.tokenizer.convert_tokens_to_ids(",")] = -100
        labels[labels==tokenizer.pad_token_id] = -100
        for i, s in enumerate(query_input_ids):
            labels[i, :(s-1)] = -100

        #### get likelihood 
        outputs = self.forward(**model_inputs, labels=None)
        loss_g_nll = self.compute_nll(outputs.logits, labels)
        return loss_g_nll


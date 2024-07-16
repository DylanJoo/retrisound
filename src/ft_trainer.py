import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# from trl import RewardTrainer
from transformers import PreTrainedModel, get_scheduler, Trainer
from transformers.utils import logging, is_peft_available

if is_peft_available():
    from peft import PeftModel
if is_safetensors_available():
    import safetensors.torch

from transformers.modeling_utils import unwrap_model

from prompt.qampari import *
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class IRTrainer(Trainer):

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ 
        Revise the pipeline to RAG's. Details are in `dataset` and `datacollator`
        """
        indices = inputs.pop('index', None)
        candidates = inputs.pop('candidates', [])
        questions = inputs.pop('questions', [])
        targets = inputs.pop('targets', [])

        inputs_for_retriever = inputs.pop('inputs_for_retriever')
        loss_r_co, loss_g_nll, loss_g_rl = 0, 0, 0

        ## [RAG]
        ### Retrieval and re-ranking
        outputs_retriever = model.biencoders(**inputs_for_retriever)
        contexts = []
        for candidate, ranking in zip(candidates, outputs_retriever.reranking):
            context = [p for _, p in sorted(zip(ranking, candidate))]
            contexts.append(context[:model.k])

        loss_r_co = outputs_retriever.loss # CL loss

        ### Prepare prompts 
        prompts, prompts_fbk = self._prepare_prompts(questions, contexts)

        #### (1) [answer] Zero-shot w/ answer 
        # [NOTE] I think we dont need to do inference of ZS
        inputs_zeroshot = self.tokenizer(
            prompts, 
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(model.llm.device)
        source_len = [len(toks) for toks in inputs_zeroshot['input_ids']]    

        #### (2) [answer] RAG w/ answer
        inputs_targets = self.tokenizer(
            [f"{p} {a}" for (p, a) in zip(prompts, targets)],
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(model.llm.device)

        labels = inputs_targets['input_ids'].clone()
        labels[labels==self.tokenizer.pad_token_id] = -100
        for i, s in enumerate(source_len):
            labels[i, :(s-1)] = -100

        ### forward
        #### (1) [answer] Zero-shot w/ answer 
        #### (2) [answer] RAG w/ answer (teacher-forcing)
        outputs_g = model.llm(**inputs_targets, labels=None)
        loss_g_nll = self.compute_nll(outputs_g.logits, labels)

        ### generation
        #### (1) [answer] RAG w/ answer (autoregressive)
        self.tokenizer.padding_side = "left"  
        inputs_zeroshot = self.tokenizer(
            prompts, 
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(model.llm.device)

        # outputs_answers = model.llm.generate(
        #     **inputs_zeroshot,
        #     do_sample=True,
        #     temperature=1.0,
        #     top_p=0.95,
        #     max_new_tokens=32,
        #     eos_token_id=model.stop_token_ids,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     use_cache=True
        # )

        #### (2) retrieval-agumentated feedback wo/ answer
        inputs_feedbacks = self.tokenizer(
            prompts_fbk,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(model.llm.device)
        self.tokenizer.padding_side = "right" # for finetuning

        outputs_fbk = model.llm.generate(
            **inputs_feedbacks,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=32,
            eos_token_id=model.stop_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True
        )

        feedbacks = []
        for i, output in enumerate(outputs_fbk):
            feedback = self.tokenizer.decode(
                output[inputs_feedbacks['input_ids'][i].size(-1):],
                skip_special_tokens=True
            )
            feedback = feedback.split('\n\n')[0]
            feedbacks.append(feedback)

        # calculate loss, optionally modulate with margin
        # loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
        loss = loss_r_co.mean() + loss_g_nll.mean() + loss_g_rl

        if self.accelerator.process_index == 0:
            log = {"loss_r_co": loss_r_co.clone().detach().item(),
                   "loss_g_nll": loss_g_nll.mean().clone().detach().item(),
                   "loss_g_rl": 0}
            self.log(log)
            self.accelerator.log(log)

        return loss

    def _prepare_prompts(self, questions, contexts):
        prompts = []
        prompts_fbk = []
        for i in range(len(questions)):
            D = apply_docs_prompt(contexts[i], field='text')

            prompt = apply_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=instruction_prompt,
                prefix="Answer: "
            ).replace('{DEMO}', '') # no demo
            prompts.append(prompt)

            # prompt for getting feedback
            prompt_fbk = apply_fbk_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=fbk_instruction_prompt,
                prefix=fbk_prefix
            )
            prompts_fbk.append(prompt_fbk)

        return prompts, prompts_fbk

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

    # [optimizer] Split weights in two groups, one with weight decay and the other not.
    def create_optimizer_and_scheduler(self, num_training_steps):
        # optimizer
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay) ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay) ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        # scheduler
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=int(num_training_steps * self.args.warmup_ratio),
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        ## [rag > [biencoders], llm > d_encoder, [q_encoder] > [model] (bert)
        if not isinstance(self.model.biencoders.q_encoder.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.biencoders.q_encoder.model.state_dict()

            if isinstance(unwrap_model(self.model.biencoders.q_encoder.model), supported_classes):
                unwrap_model(self.model.biencoders.q_encoder.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.biencoders.q_encoder.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

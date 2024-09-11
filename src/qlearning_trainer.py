import gc
import math
import os
import time
import json
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
import dataclasses

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl.trainer.reward_trainer import RewardTrainer

from transformers.modeling_utils import (
    PreTrainedModel, 
    load_sharded_checkpoint, 
    unwrap_model
)
from utils import (
    convert_texts_to_tensors, 
    multiple_sample_and_log_probability, 
    augmentation_response,
    augmentation_feedback,
    get_mini_batch_dict,
    load_searcher
)

class Trainer(RewardTrainer):

    def __init__(self, reward_model, index_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.searcher = load_searcher(index_dir, dense=True)
        self._move_model_to_device(self.reward_model, self.args.device)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        ## collect inputs 
        ### [RL states]: question, candidates, feedback
        candidates = inputs["candidates"]
        questions = inputs["questions"]
        targets = inputs["targets"]
        data_indices = inputs["index"] # for the next iteration

        ### sampling
        for t in range(0, self.args.num_steps + 1):
            if t == 0:
                retriever_inputs = inputs["inputs_for_retriever"]
            else: 
                baseline = reward # last reward as baseline
                del retriever_inputs, outputs

                gc.collect()
                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )

            outputs = model(**retriever_inputs, include_n_feedbacks=t+1)
            queries = outputs.qembs[:, -1, :] # the last reformulated qemb

            ### re-ranking without prepare contexts
            # logit = outputs.logits
            # ranking, logprob = multiple_sample_and_log_probability(logit, 1, batch=True)
            # ranking = ranking.squeeze(1) 

            ### retrieval and prepare contexts
            hits = self.searcher.batch_search(
                queries.float().detach().cpu().numpy(), q_ids=list(range(queries.size()[0])), k=len(candidates[0])
            )
            candidates = [] 
            for i, key in enumerate(hits):
                candidate = [self.train_dataset.corpus[h.docid] for h in hits[key]]
                candidates.append( candidate )

            ### generation
            gen_batch = (1 or self.args.generation_batch)
            ### (1) response
            prompt = augmentation_response(
                questions=questions, 
                candidates=candidates, 
                n_context=self.args.n_contexts,
                rankings=None if self.searcher is not None,
                dataset_prefix=self.args.dataset_prefix
            )
            response = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_response = self.reward_model._inference(prompt[i:i+gen_batch])
                b_response = [r.rsplit('Question', 1)[0] for r in b_response]
                response += b_response

            ### (2) feedback 
            prompt = augmentation_feedback(
                questions=questions, 
                answers=response,
                candidates=candidates, 
                n_context=self.args.n_contexts,
                rankings=None if self.searcher is not None,
                dataset_prefix=self.args.dataset_prefix
            )
            feedback = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_feedback = self.reward_model._inference(prompt[i:i+gen_batch])
                feedback += b_feedback

            for i in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[i], feedback[i])

            ### calculate rewards (on policy)
            reward = self.reward_model.get_rewards(response, targets).view(-1, 1)
            reward = reward.to(model.device)

        ## baseline can be the improved one-shot retrieval
        logprob = -1
        reinforce_loss = ((reward - baseline) * (-logprob)).mean()
        contrastive_loss = outputs.loss

        loss = reinforce_loss + contrastive_loss

        self.log({"train/reward": reward.mean().item()})
        self.log({"loss/RL": reinforce_loss.mean().item()})
        self.log({"loss/CT": contrastive_loss.mean().item()})

        print('\nprompt: ', prompt[0])
        print('\nquestion: ', questions[0])
        print('\nresponse: ', response[0])
        print('\nfeedback: ', feedback[0])

        ## logging
        if self.accelerator.is_main_process:
            df = pd.DataFrame({"question": questions, "response": response, "feedback": feedback})
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        if return_outputs:
            return loss, {
                "rewards_chosen": response,
                "rewards_rejected": feedback
            }
        return loss


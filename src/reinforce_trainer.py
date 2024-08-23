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
    get_mini_batch_dict
)

class Trainer(RewardTrainer):

    def __init__(self, reward_model, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
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
                del retriever_inputs, outputs, logit, ranking, logprob
                gc.collect()
                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )

            ### retrieval (ranking) and prepare contexts
            outputs = model(**retriever_inputs, include_n_feedbacks=t+1)
            logit = outputs.logits
            ranking, logprob = multiple_sample_and_log_probability(logit, 1, batch=True)
            ranking = ranking.squeeze(1) 

            gen_batch = (1 or self.args.generation_batch)

            ### generation
            ### (1) response
            prompt = augmentation_response(
                questions=questions, 
                candidates=candidates, 
                rankings=ranking, 
                n_context=self.args.n_contexts,
                dataset_prefix=self.args.dataset_prefix
            )
            response = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_response = self.reward_model._inference(prompt[i:i+gen_batch])
                response += b_response

            ### (2) feedback 
            prompt = augmentation_feedback(
                questions=questions, 
                answers=response,
                candidates=candidates, 
                rankings=ranking, 
                n_context=self.args.n_contexts,
                dataset_prefix=self.args.dataset_prefix
            )
            feedback = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_feedback = self.reward_model._inference(prompt[i:i+gen_batch])
                feedback += b_feedback

            for i in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[i], feedback[i])

        ### calculate rewards (on policy)
        reward = self.reward_model.get_rewards(response, targets).view(-1, 1).to(self.args.device)
        reward = reward.to(model.device)

        ## baseline can be the improved one-shot retrieval
        baseline = 0
        reinforce_loss = ((reward - baseline) * (-logprob)).mean()
        loss = reinforce_loss

        if return_outputs:
            return loss, {
                "rewards_chosen": reward,
                "rewards_rejected": None,
            }
        return loss


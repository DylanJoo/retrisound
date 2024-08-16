import gc
import math
import os
import time
import json
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import dataclasses

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from trl.trainer.reward_trainer import RewardTrainer

from utils import (
    convert_texts_to_tensors, 
    multiple_sample_and_log_probability, 
    augmentation_response,
    augmentation_feedback,
    get_mini_batch_dict
)

class Trainer(RewardTrainer):

    def __init__(self, reward_model, **kwargs):
        self.reward_model = reward_model
        super().__init__(**kwargs)

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
        data_indices = inuts["index"] # for the next iteration

        ### sampling
        for t in range(0, self.args.num_steps + 1):
            if t == 0:
                retriever_inputs = data["inputs_for_retriever"]
            else: 
                del retriever_inputs, outputs, logits, ranking, logprob
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

            gen_batch = (self.args.generation_batch or len(prompt))

            ### generation
            ### (1) response
            prompt = augmentation_response(questions, candidates, ranking, args.n_contexts)
            response = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_response = self.reward_model._inference(prompt[i:i+gen_batch])
                response += b_response

            ### (2) feedback 
            prompt = augmentation_feedback(questions, candidates, ranking, args.n_contexts)
            feedback = []
            for i in range(0, len(prompt_fbk), gen_batch):
                _, _, b_feedback = reward_model._inference(prompt_fbk[i:i+gen_batch])
                feedback += b_feedback

            for i in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[i], feedback[i])

        ### calculate rewards (on policy)
        reward = self.reward_model.get_rewards(response, targets).view(-1, 1).to(self.reward_model.device)
        reward = reward.to(model.device)

        ## baseline can be the improved one-shot retrieval
        baseline = 0
        reinforce_loss = ((rewards - baseline) * (-logprob)).mean()
        loss = reinforce_loss

        if return_outputs:
            return loss, {
                "rewards_chosen": reward,
                "rewards_rejected": None,
            }
        return loss


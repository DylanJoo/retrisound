# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import time
import json
#
from collections import defaultdict
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
    augmentation_response,
    augmentation_feedback,
    load_searcher
)

class Trainer(RewardTrainer):

    def __init__(self, reward_model, index_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.searcher = load_searcher(index_dir, lexical=True)
        self._move_model_to_device(self.reward_model, self.args.device)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        ## collect inputs 
        ### [RL states]: question, candidates, feedback
        candidates = bm25_candidates = inputs["candidates"]
        candidate_pids = inputs["candidate_pids"]
        questions = inputs["questions"]
        targets = inputs["targets"]
        data_indices = inputs["index"] # for the next iteration

        ### sampling
        logprobs = []
        rewards = []
        for t in range(0, self.args.num_steps + 1):
            if t == 0:
                retriever_inputs = inputs["inputs_for_retriever"]
            else: 
                del retriever_inputs, outputs
                gc.collect()

                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )
                candidates = [self.train_dataset[idx]['candidates'] for idx in data_indices]
                candidate_pids = [self.train_dataset[idx]['candidate_pids'] for idx in data_indices]

            outputs = model(**retriever_inputs, include_n_feedbacks=t+1)
            query = outputs.q_reps[:, -1] # the last reformulated qemb
            logprob = outputs.q_logprobs[:, -1]

            ### re-ranking without prepare contexts (deprecated)
            ### retrieval and prepare contexts
            hits = self.searcher.batch_search(
                query.float().detach().cpu().numpy(), 
                q_ids=[str(i) for i in range(query.size()[0])],
                k=len(candidates[0])
            )
            candidates_ = []
            candidate_pids_ = []
            for i, key in enumerate(hits):
                new_docids = [h.docid for h in hits[key] if h.docid not in candidate_pids[i]]
                candidate_pids_.append(new_docids)

                new_docs = [self.train_dataset.corpus[docid] for docid in new_docids]
                candidates_.append( new_docs )

            ### generation
            gen_batch = (1 or self.args.generation_batch)
            ### (1) response
            prompt = augmentation_response(
                questions=questions, 
                candidates=candidates_, 
                n_context=self.args.n_contexts,
                rankings=None,
                dataset_prefix=self.args.dataset_prefix,
                answers=targets
            )
            response = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_response = self.reward_model._inference(prompt[i:i+gen_batch])
                b_response = [r.rsplit('</r>', 1)[0] for r in b_response]
                response += b_response

            ### (2) feedback 
            prompt = augmentation_feedback(
                questions=questions, 
                candidates=candidates_, 
                n_context=self.args.n_contexts,
                rankings=None,
                dataset_prefix=self.args.dataset_prefix
            )
            feedback = []
            for i in range(0, len(prompt), gen_batch):
                _, _, b_feedback = self.reward_model._inference(prompt[i:i+gen_batch])
                b_feedback = [f.rsplit('</f>', 1)[0] for f in b_feedback]
                feedback += b_feedback

            ### calculate rewards (on policy)
            reward = self.reward_model.get_rewards(response, targets).view(-1)
            reward = reward.to(model.device)
            rewards.append(reward)
            logprobs.append(logprob)

            ### add feedback if rewards is good enough
            for i in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[i], feedback[i])
                if reward[i] >= 1:
                    self.train_dataset.update_candidates(data_indices[i], candidate_pids_[i])

        rewards = torch.stack(rewards, 1)
        logprobs = torch.stack(logprobs, 1)

        ## baseline can be the improved one-shot retrieval
        # reinforce_loss = (rewards * (-logprobs)).mean()
        reinforce_loss = (rewards[:, 1:] * (-logprobs[:, 1:])).mean()
        contrastive_loss = outputs.loss

        loss = (reinforce_loss * self.args.rl_coef) + \
                (contrastive_loss * self.args.cont_coef)

        self.log({"train/reward": reward.mean().item()})
        self.log({"loss/RL": reinforce_loss.mean().item()})
        self.log({"loss/CT": contrastive_loss.mean().item()})

        print('---')
        print('\nquestion: ', questions[0])
        print('\nAction top-k terms', \
                self.tokenizer.batch_decode(torch.argsort(outputs.q_values[0], -1, descending=True)[:, :8]))
        print('\nQuery top-k terms', \
                self.tokenizer.batch_decode(torch.argsort(outputs.q_reps[0], -1, descending=True)[:, :8]))
        print('\nOld candidate titles: ', [c['title'] for c in candidates[0]])
        print('New candidate titles: ', [c['title'] for c in candidates_[0]])
        print('\nanswer: ', targets[0])
        print('\nresponse: ', response[0])
        print('\nfeedback: ', feedback[0])
        print('\nrewards: ', rewards[0])
        print('---')

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


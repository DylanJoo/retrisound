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
import re
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

from transformers.modeling_utils import PreTrainedModel
from transformers import Trainer as Trainer_hf
from utils import (
    augmentation_response,
    augmentation_feedback,
    load_searcher
)
from sentence_transformers.evaluation import NanoBEIREvaluator

class Trainer(Trainer_hf):

    def __init__(self, generator, index_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.searcher = load_searcher(index_dir, lexical=True)
        self._move_model_to_device(self.generator, self.args.device)

    def measure_ranking(self):
        pass

    def compute_loss_reward(
        self, 
        query, 
        questions,
        cached_judges=None, # this mean each passage will be judged indepedently
        truth=None,
        reward_type='normal'
    ):
        gen_batch = (1 or self.args.generation_batch)

        hits = self.searcher.batch_search(
            logits=query.float().detach().cpu().numpy(), 
            q_ids=[str(i) for i in range(query.size()[0])],
            k=self.args.n_contexts,
            threads=32
        )
        hits = {int(k): v for k, v in hits.items()}
        hits = dict(sorted(hits.items()))

        rewards = []
        candidates = []
        for i, key in enumerate(hits):

            pids = [h.docid for h in hits[key]]
            candidate = [self.train_dataset.corpus[h.docid] for h in hits[key]]
            candidates.append(candidate)

            reward = self.measure_ranking(pids, truth)

            # update
            for pid, judge in zip(new_pids, new_reward):
                cached_judges[i][pid] = judge.item()

            rewards.append(reward)

        rewards = torch.stack(rewards, 0) # B 1 
        return rewards, candidates, cached_judges

    def compute_loss_feedback(self, questions, contexts):

        gen_batch = (1 or self.args.generation_batch)

        def remove_citations(sent):
            return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

        prompt = augmentation_feedback(
            questions=questions, 
            candidates=contexts, 
            n_context=self.args.n_contexts,
        )
        feedback = []
        for i in range(0, len(prompt), gen_batch):
            _, _, b_feedback = self.generator.generate(prompt[i:i+gen_batch])
            b_feedback = [remove_citations(f) for f in b_feedback]
            feedback += b_feedback

        return feedback

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        ## collect inputs 
        ### [RL states]: question, candidates, feedback
        questions = inputs["query"]
        data_indices = inputs["index"] # for the next iteration
        ids = [self.train_dataset.ids[idx] for idx in data_indices]
        truth = [self.train_dataset.qrels[id] for id in ids]
        judges = [self.train_dataset.judgements[id] for id in ids]

        batch_size, step_size = len(questions), self.args.num_steps

        ### sampling
        reps = []
        ct_losses = []
        logprobs = []
        rewards = []
        for t in range(0, self.args.num_steps+1):

            if t == 0:
                retriever_inputs = inputs["inputs_for_retriever"]
                output = model(
                    q_tokens=retriever_inputs['q_tokens'][0], 
                    q_masks=retriever_inputs['q_masks'][0], 
                    step=0
                )
                reward, candidates, judges = self.compute_loss_reward(
                    output.reps, questions, cached_judges=judges, truth=truth
                )
                feedback = self.compute_loss_feedback(questions, candidates)
                candidates_0 = candidates
                q_out = output.q_out
            else: 
                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )
                output = model(
                    output=q_out,
                    q_tokens=retriever_inputs['q_tokens'][0], 
                    q_masks=retriever_inputs['q_masks'][0], 
                    f_tokens=retriever_inputs['q_tokens'][t], 
                    f_masks=retriever_inputs['q_masks'][t], 
                    d_tokens=retriever_inputs['d_tokens'],
                    d_masks=retriever_inputs['d_masks'],
                    step=t,
                )
                reward, candidates, judges = self.compute_loss_reward(
                    output.reps, questions, cached_judges=judges, truth=truth
                )
                feedback = self.compute_loss_feedback(questions, candidates) 
                ct_losses.append(output.loss_ct)

            reps.append(output.reps)
            rewards.append(reward)

            # [NOTE] here we use the last sample as the stored feedback
            for j in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[j], feedback[j])
                self.train_dataset.add_judgements(data_indices[j], judges[j], info=questions[j])


        rewards = torch.stack(rewards, 1)
        contrastive_loss = torch.stack(ct_losses, 0)

        contrastive_loss = contrastive_loss.mean()

        loss = (contrastive_loss * self.args.ct_coef) 

        self.log({"train/reward": rewards.mean().item()})
        self.log({"loss/RL": 0})
        self.log({"loss/CT": contrastive_loss.mean().item()})
        self.log({"loss/MR": 0})

        print('---')
        print('\nquestion: ', questions[0])
        print('\nDocument +/- ', self.train_dataset[data_indices[0]]['contexts'])
        print('\nRetrieved doc (q0):', [c['text'][:30] for c in candidates_0[0]])
        print('\nRetrieved doc (q0 & f1):', [c['text'][:30] for c in candidates[0]])
        print('\nFeedback: ', self.train_dataset.feedbacks[data_indices[0]])

        print('\n\nTop-k terms vs rewards')

        for i in range(len(reps)):
            t = self.tokenizer.batch_decode(
                torch.argsort(reps[i], -1, descending=True)[0, :15]
            )
            r = rewards[0][i].tolist()
            print(r, t)
        print('---')

        ## logging
        if self.accelerator.is_main_process:
            df = pd.DataFrame({"question": questions, "feedback": feedback})
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        if return_outputs:
            return loss, {
                "rewards_chosen": 'none',
                "rewards_rejected": feedback
            }
        return loss


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
import ir_measures
from ir_measures import nDCG, R
from utils import (
    augmentation_response,
    augmentation_feedback,
    load_searcher
)

class Trainer(Trainer_hf):

    def __init__(self, generator, index_dir=None, dense=False, lexical=False, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.searcher = load_searcher(
            index_dir, 
            lexical=lexical,
            dense=dense
        )

        if dense:
            self.rep_type = 'dense'
        elif lexical:
            self.rep_type = 'sparse'

    def measure_ranking(self, pids_pred, pids_truth):
        qrel = {"dummy": pids_truth}
        run = {"dummy": {k: 1/(1+i) for i, k in enumerate(pids_pred)}}
        result = ir_measures.calc_aggregate([nDCG@10, R@10], qrel, run)[R@10]
        return result

    def compute_loss_reward(
        self, 
        query, 
        questions,
        truth=None,
    ):
        gen_batch = (self.args.generation_batch or 1)

        if self.rep_type == 'dense':
            hits = self.searcher.batch_search(
                queries=query.clone().float().detach().cpu().numpy(), 
                q_ids=[str(i) for i in range(query.size()[0])],
                k=self.args.n_max_candidates,
                threads=32
            )
        elif self.rep_type == 'sparse':
            hits = self.searcher.batch_search(
                logits=query.clone().float().detach().cpu().numpy(), 
                q_ids=[str(i) for i in range(query.size()[0])],
                k=self.args.n_max_candidates,
                threads=32
            )
        hits = {int(k): v for k, v in hits.items()}
        hits = dict(sorted(hits.items()))

        rewards = []
        candidates = []
        for i, key in enumerate([int(k) for k in range(query.size()[0])]):
            try: 
                pids = [h.docid for h in hits[key]]
                candidate = [self.train_dataset.corpus[h.docid] for h in hits[key]]
                reward = self.measure_ranking(pids, truth[i])
            except: # no retrieved results
                candidate = []
                reward = 0.0

            candidates.append(candidate)
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        return rewards, candidates

    def compute_loss_feedback(self, questions, contexts):

        gen_batch = (self.args.generation_batch or 1)

        def remove_citations(sent):
            return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

        prompt = augmentation_feedback(
            questions=questions, 
            candidates=contexts, 
            n_context=self.args.n_contexts,
        )
        feedback = []
        for i in range(0, len(prompt), gen_batch):
            b_feedback = self.generator.generate(prompt[i:i+gen_batch])
            b_feedback = [remove_citations(f) for f in b_feedback]
            feedback += b_feedback

        return feedback

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        ## collect inputs 
        ### [RL states]: question, candidates, feedback
        questions = inputs["query"]
        data_indices = inputs["index"] # for the next iteration
        ids = [self.train_dataset.ids[idx] for idx in data_indices]
        qrels = [self.train_dataset.qrels[id] for id in ids]

        batch_size, step_size = len(questions), self.args.num_steps

        ### sampling
        reps = []
        ct_losses = 0
        reg_losses = 0
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
                reward_0, candidates = self.compute_loss_reward(
                    output.reps, questions, truth=qrels
                )

                feedback = []
                for i, qrel in enumerate(qrels):
                    feedback.append(self.train_dataset.corpus[list(qrel.keys())[0]]['text'])
                # feedback = self.compute_loss_feedback(questions, candidates)
                candidates_0 = candidates
                q_out = output
            else: 
                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )
                output = model(
                    q_tokens=retriever_inputs['q_tokens'][0], 
                    q_masks=retriever_inputs['q_masks'][0], 
                    f_tokens=retriever_inputs['q_tokens'][t], 
                    f_masks=retriever_inputs['q_masks'][t], 
                    d_tokens=retriever_inputs['d_tokens'],
                    d_masks=retriever_inputs['d_masks'],
                    prev_output=q_out,
                    step=t,
                )
                reward, candidates = self.compute_loss_reward(
                    output.reps, questions, truth=qrels
                )
                feedback = []
                for i, qrel in enumerate(qrels):
                    feedback.append(self.train_dataset.corpus[list(qrel.keys())[0]]['text'])
                # feedback = self.compute_loss_feedback(questions, candidates)
                ct_losses += output.loss_ct 
                reg_losses += output.loss_flop 
                rewards.append(reward)

            reps.append(output.reps)

            # [NOTE] here we use the last sample as the stored feedback
            for j in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[j], feedback[j])

        rewards = torch.stack(rewards, 0)
        contrastive_loss = ct_losses
        regularization_loss = reg_losses
        loss = (contrastive_loss * self.args.ct_coef) + (regularization_loss * self.args.reg_coef)

        self.log({"train/reward_0": reward_0.mean().item()})
        self.log({"train/reward": rewards.mean().item()})
        self.log({"loss/RL": 0})
        self.log({"loss/CT": contrastive_loss.mean().item()})
        self.log({"loss/REG": regularization_loss.mean().item()})
        self.log({"loss/MR": 0})

        print('---')
        print('\nDocument +/- ', self.train_dataset[data_indices[0]]['contexts'])
        # print('\nRetrieved doc (q0):', [c['text'][:30] for c in candidates_0[0]])
        # print('\nRetrieved doc (q0 & f1):', [c['text'][:30] for c in candidates[0]])
        print('\nRetrieved doc (q0):', [c['title'] for c in candidates_0[0]])
        print('\nRetrieved doc (q0 & f1):', [c['title'] for c in candidates[0]])
        print('\nFeedback: ', self.train_dataset.feedbacks[data_indices[0]])

        print('\nquestion: ', questions[0])
        print('\n\nTop-k terms vs rewards')
        for i in range(len(reps)):
            t = self.tokenizer.batch_decode(
                torch.argsort(reps[i], -1, descending=True)[0, :15]
            )
            if i == 0:
                r = reward_0[0].tolist()
            else:
                r = rewards[i-1, 0].tolist()
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


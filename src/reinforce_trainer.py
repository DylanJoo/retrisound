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

    def compute_loss_reward(
        self, 
        query, 
        questions,
        cached_judges=None, # this mean each passage will be judged indepedently
        reward_type='normal'
    ):
        hits = self.searcher.batch_search(
            logits=query.float().detach().cpu().numpy(), 
            q_ids=[str(i) for i in range(query.size()[0])],
            k=self.args.n_contexts,
            threads=32
        )
        gen_batch = (1 or self.args.generation_batch)

        ## resort (sometimes the batch is larger than 10 would be ordered by first digit 1 11 12 13...)
        hits = {int(k): v for k, v in hits.items()}
        hits = dict(sorted(hits.items()))

        rewards = []
        candidates = []
        for i, key in enumerate(hits):

            pids = [h.docid for h in hits[key]]
            candidate = [self.train_dataset.corpus[h.docid] for h in hits[key]]
            candidates.append(candidate)

            # run new rewards
            new_pids = [h.docid for h in hits[key] if (h.docid not in cached_judges[i])]
            new_candidates = [c for (c, h) in zip(candidate, hits[key]) if h.docid in new_pids]
            prompt = augmentation_response(
                questions=questions[i], 
                candidates=new_candidates,
                n_context=self.args.n_contexts,
                independent=(cached_judges is not None)
            )
            response = []
            for j in range(0, len(prompt), gen_batch):
                _, _, b_response = self.reward_model._inference(prompt[j:j+gen_batch])
                response += b_response
            new_reward = self.reward_model.get_rewards(response, answers[i])

            # update
            for pid, judge in zip(new_pids, new_reward):
                cached_judges[i][pid] = judge.item()

            if reward_type == 'cumulative':
                new_pids = new_pids if len(new_pids) > 0 else [-1]
                reward = torch.tensor([float(cached_judges[i][pid]) for pid in new_pids]).mean().to(query.device)
            elif reward_type == 'irrelevant_pushing':
                score = [float(cached_judges[i][h.docid]) for h in hits[key]] + [0]
                rank_negative = 1 - 1 / ( (score.index(0)+1) )
                reward = torch.tensor(float(rank_negative)).to(query.device)
            else:
                reward = torch.tensor([float(cached_judges[i][h.docid]) for h in hits[key]]).mean().to(query.device)
            rewards.append(reward)

        rewards = torch.stack(rewards, 0) # B 1 

        if cached_judges is None:
            return rewards, candidates
        else:
            return rewards, candidates, cached_judges


    def compute_loss_feedback(
        self, 
        questions,
        contexts,
    ):
        gen_batch = (1 or self.args.generation_batch)
        ### (2) feedback 
        def remove_citations(sent):
            return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

        prompt = augmentation_feedback(
            questions=questions, 
            candidates=contexts, 
            n_context=self.args.n_contexts,
            rankings=None,
            dataset_prefix=self.args.dataset_prefix
        )
        feedback = []
        for i in range(0, len(prompt), gen_batch):
            _, _, b_feedback = self.reward_model._inference(prompt[i:i+gen_batch])
            b_feedback = [remove_citations(f) for f in b_feedback]
            feedback += b_feedback

        return feedback

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        model.q_encoder.eval()
        model.d_encoder.eval()
        model.modifier.q_encoder.eval()

        ## collect inputs 
        ### [RL states]: question, candidates, feedback
        questions = inputs["questions"]
        targets = inputs["targets"]
        data_indices = inputs["index"] # for the next iteration
        qids = [self.train_dataset.qids[idx] for idx in data_indices]
        judges = [self.train_dataset.judgements[qid] for qid in qids]

        batch_size, step_size = len(questions), self.args.num_steps

        ### sampling
        logprobs = []
        rewards = []
        for t in range(0, self.args.num_steps+1):

            if t == 0:
                retriever_inputs = inputs["inputs_for_retriever"]
                bm25_candidates = inputs["candidates"]
                prev_output = model.q_encoder(
                    retriever_inputs['q_tokens'][0], 
                    retriever_inputs['q_masks'][0]
                )
                rewards_0, candidates_0, judges = self.compute_loss_reward(
                    prev_output.reps, questions, targets, cached_judges=judges
                )
                feedback = self.compute_loss_feedback(questions, candidates_0)
            else: 
                del retriever_inputs
                retriever_inputs = self.data_collator.get_inputs_for_retriever(
                    [self.train_dataset[idx] for idx in data_indices],
                    device=model.device
                )
                output = model(**retriever_inputs, prev_output=prev_output, include_n_feedbacks=t)
                prev_output = output.q_out

                # get rewards over samples
                # [todo] sampled search to speed up
                for i in range(output.q_logprobs.size(1)):
                    query = output.q_reps[:, i, :] # B N V --> B 1 V
                    logprob = output.q_logprobs[:, i]
                    reward, candidates, judges = self.compute_loss_reward(
                        query, questions, targets, 
                        cached_judges=judges,
                        reward_type=self.args.reward_type
                    )
                    reward = reward.view(-1).to(model.device)
                    rewards.append(reward)
                    logprobs.append(logprob)

                # the expected retrieved candidates 
                feedback = self.compute_loss_feedback(questions, candidates) 
                losses.append(output.loss)


            # [NOTE] here we use the last sample as the stored feedback
            for j in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[j], feedback[j])
                self.train_dataset.add_judgements(data_indices[j], judges[j], info=questions[j])

        rewards = torch.stack(rewards, 1)
        logprobs = torch.stack(logprobs, 1)
        contrastive_loss = torch.stack(losses, 1)

        ## baseline can be the improved one-shot retrieval
        reinforce_loss = (rewards * (-logprobs)).mean()
        contrastive_loss = contrastive_loss.mean()

        loss = (reinforce_loss * self.args.rl_coef) + \
                (contrastive_loss * self.args.cont_coef)

        self.log({"train/reward": rewards.mean().item()})
        self.log({"loss/RL": reinforce_loss.mean().item()})
        self.log({"loss/CT": contrastive_loss.mean().item()})

        print('---')
        print('\nquestion: ', questions[0])
        print('\nRetrieved doc (q0):', [c['title'] for c in candidates_0[0]])
        print('\nRetrieved doc (q0 & f1):', [c['title'] for c in candidates[0]])
        print('\nFeedback: ', feedback[0])
        print('\nAnswer: ', targets[0])
        print('\n\nTop-k terms vs rewards')
        sample_terms = self.tokenizer.batch_decode(torch.argsort(output.q_reps[0], -1, descending=True)[:, :8])
        sample_rewards = rewards[0].tolist()
        for tt, rr in zip(sample_terms, sample_rewards):
            print(rr, tt)
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


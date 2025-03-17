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
from transformers.utils import logging, is_peft_available
logger = logging.get_logger(__name__)
import safetensors.torch

from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers import Trainer
import ir_measures
from ir_measures import nDCG, R, RR
from utils import load_searcher
from tools.annealing import Annealer
from modeling.llm.utils import remove_citations, replace_tags
from prompts.generic import apply_docs_prompt, apply_fbk_inst_prompt, apply_report_inst_prompt

def augmentation_feedback(questions, candidates, n_context, R=None):
    if R is None:
        R = [None] * len(questions)
    prompts = []
    for i in range(len(questions)):
        D = apply_docs_prompt(candidates[i][:n_context], field='text')
        prompt = apply_report_inst_prompt(Q=questions[i], D=D, R=R[i])
        prompts.append(prompt)
    return prompts

class PolicyTrainer(Trainer):

    def __init__(
        self, 
        generator, 
        searcher=None, 
        eval_searcher=None,
        index_dir=None, 
        num_generation=False, 
        dataset_name=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.searcher = searcher
        self.annealer = Annealer(self.args.max_steps, shape='cosine', cyclical=True)
        self.num_generation = num_generation
        self.eval_searcher = eval_searcher
        self.is_eval = False
        self.dataset_name = dataset_name

    @staticmethod
    def measure_ranking(pids_pred, pids_truth):
        qrel = {"dummy": pids_truth}
        run = {"dummy": {k: 1/(1+i) for i, k in enumerate(pids_pred)}}
        result = ir_measures.calc_aggregate([nDCG, R@10, RR@10], qrel, run)[nDCG] 
        return result

    def get_candidates(self, hits):
        corpus = self.train_dataset.corpus
        if self.is_eval:
            corpus = (self.eval_dataset.corpus or corpus)
        candidate = [corpus[h.docid] for h in hits]
        return candidate

    def compute_loss_reward(self, query, questions, truth=None):
        searcher = self.searcher
        if self.is_eval:
            searcher = (self.eval_searcher or searcher)

        hits = searcher.batch_search(
            logits=query.clone().float().detach().cpu().numpy(), 
            q_ids=[str(i) for i in range(query.size()[0])],
            k=self.args.n_max_candidates,
            threads=64
        )
        hits = {int(k): v for k, v in hits.items()}
        hits = dict(sorted(hits.items()))

        rewards = []
        candidates = []
        for i, key in enumerate([int(k) for k in range(query.size()[0])]):
            try: 
                pids = [h.docid for h in hits[key]]
                candidate = self.get_candidates(hits[key])
                reward = self.measure_ranking(pids, truth[i])
            except: # no retrieved results
                candidate = []
                reward = 0.0

            candidates.append(candidate)
            rewards.append(reward)

        rewards = torch.tensor(rewards)

        return rewards, candidates

    def compute_loss_feedback(self, questions, contexts, feedbacks=None):
        gen_batch = (self.args.generation_batch or 1)

        prompt = augmentation_feedback(
            questions=questions, 
            candidates=contexts, 
            n_context=self.args.n_contexts,
            R=feedbacks
        )
        feedback = []
        for i in range(0, len(prompt), gen_batch):
            b_feedback = self.generator.generate(prompt[i:i+gen_batch], max_tokens=256)
            # b_feedback = [remove_citations(f) for f in b_feedback]
            b_feedback = [replace_tags(f, 'p') for f in b_feedback]
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
        logprobs = []
        ct_losses = 0
        tc_losses = 0
        rewards = []
        pos_ratio_truth = []
        pos_ratio = []
        for t in range(0, self.args.num_steps+1):

            if t == 0:
                retriever_inputs = inputs["inputs_for_retriever"]
                output = model(
                    q_tokens=retriever_inputs['q_tokens'][0],
                    q_masks=retriever_inputs['q_masks'][0],
                    step=0,
                    tokenizer=self.tokenizer
                )
                reward_0, candidates = self.compute_loss_reward(
                    output.reps, questions, truth=qrels
                )

                # msmarco can use pre-computed feedback. other used the online generated
                if 'msmarco' in self.dataset_name:
                    feedback = [self.train_dataset.feedbacks[idx][0] for idx in data_indices]
                else:
                    feedback = self.compute_loss_feedback(questions, candidates)
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
                    sub_token_type_ids=retriever_inputs['q_types'][t],
                    step=t,
                    tokenizer=self.tokenizer
                )
                for rep, logprob in zip(output.samples['reps'], output.samples['logprobs']):
                    reward, candidates = self.compute_loss_reward(rep, questions, truth=qrels)
                    rewards.append(reward.detach().cpu())
                    logprobs.append(logprob)

                # ignore the followup feedback  # [TODO] add control for the looping feedback
                if self.num_generation > t:
                    feedback = self.compute_loss_feedback(questions, candidates, feedbacks=feedback)

                ct_losses += output.loss_ct 
                tc_losses += output.loss_tc

                pos_ratio_truth.append(output.logs['PosRatioTruth'])
                pos_ratio.append(output.logs['PosRatio'])

            reps.append(output.reps)

            # [NOTE] here we use the last sample as the stored feedback
            for j in range(len(data_indices)):
                self.train_dataset.add_feedback(data_indices[j], feedback[j])

        pos_ratio_truth = torch.stack(pos_ratio_truth, 0)
        pos_ratio = torch.stack(pos_ratio, 0)
        logprobs = torch.stack(logprobs, 0)

        # normal REINFORCE
        rewards = torch.stack(rewards, 0).to(logprobs.device)

        # baseline-enhanced REINFORCE
        # rewards = [r - reward_0 for r in rewards]
        # rewards = torch.stack(rewards, 0).to(logprobs.device)

        # ignore after the reaching the optimal reward 
        if self.args.rl_coef == -1:
            rl_coef = self.annealer(1.0)
            tc_coef = 1 - rl_coef
            self.annealer.step()
        else:
            tc_coef = self.args.tc_coef
            rl_coef = self.args.rl_coef

        rl_losses = (rewards * (-logprobs)).mean()

        loss = (tc_losses * tc_coef) + \
               (rl_losses * rl_coef) + \
               (ct_losses * self.args.ct_coef) 

        self.log({"train/reward_0": reward_0.mean().item()})
        self.log({"train/reward": rewards.mean().item()})
        self.log({"train/pos_ratio_truth": pos_ratio_truth.mean().item()})
        self.log({"train/pos_ratio": pos_ratio.mean().item()})
        self.log({"loss/RL": rl_losses.mean().item()})
        self.log({"loss/CT": ct_losses.mean().item()})
        self.log({"loss/TC": tc_losses.mean().item()})
        self.log({"loss/MR": 0})

        print('---')
        print('\nDocument +/- ', self.train_dataset[data_indices[0]]['contexts'])
        print('\nRetrieved doc (q0):', [c['text'][:30] for c in candidates_0[0]])
        print('\nRetrieved doc (q0 & f1):', [c['text'][:30] for c in candidates[0]])
        # print('\nRetrieved doc (q0):', [c['title'] for c in candidates_0[0]])
        # print('\nRetrieved doc (q0 & f1):', [c['title'] for c in candidates[0]])
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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        ## only save the query encoder
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)

        if state_dict is None:
            state_dict = self.model.q_encoder.state_dict()

        if isinstance(self.accelerator.unwrap_model(self.model.q_encoder), supported_classes):
            self.accelerator.unwrap_model(self.model.q_encoder).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if self.args.save_safetensors:
                # safetensors.torch.save_file(
                #     state_dict, os.path.join(output_dir, 'model.safetensors'), metadata={"format": "pt"}
                # )
                safetensors.torch.save_model(self.model.q_encoder, os.path.join(output_dir, 'model.safetensors'))
            else:
                torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Set up metric storage
        metrics = {}
        self.is_eval = True
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Run the iterative evaluation for testing
        with torch.no_grad():
            eval_metrics = self.iteration_loop(
                eval_dataloader,
                description=f"Evaluation ({metric_key_prefix})",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        
        metrics.update(eval_metrics)
        self.is_eval = False
        self.log(metrics)
        return metrics

    def iteration_loop(
        self,
        dataloader,
        description,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Initialize metrics
        metrics = {}
        total_rewards = []

        # Initialize metrics for this batch
        rewards = {0: [], 1: [], 2: []}
        print(f"In total, {len(dataloader)} examples")
        for batch in dataloader:

            questions = batch["query"]
            data_indices = batch["index"]
            ids = [self.eval_dataset.ids[idx] for idx in data_indices]
            qrels = [self.eval_dataset.qrels[id] for id in ids]

            for t in range(0, 2):
                if t == 0:
                    retriever_inputs = batch["inputs_for_retriever"]
                    output = self.model(
                        q_tokens=retriever_inputs['q_tokens'][0],
                        q_masks=retriever_inputs['q_masks'][0],
                        step=0,
                        tokenizer=self.tokenizer
                    )
                    reward, candidates = self.compute_loss_reward(output.reps, questions, truth=qrels)
                    feedback = self.compute_loss_feedback(questions, candidates)
                    q_out = output
                    rewards[0].append(reward.detach().cpu())
                else:
                    retriever_inputs = self.data_collator.get_inputs_for_retriever(
                        [self.eval_dataset[idx] for idx in data_indices],
                        device=self.args.device
                    )
                    output = self.model(
                        q_tokens=retriever_inputs['q_tokens'][0],
                        q_masks=retriever_inputs['q_masks'][0],
                        f_tokens=retriever_inputs['q_tokens'][t],
                        f_masks=retriever_inputs['q_masks'][t],
                        d_tokens=None,
                        d_masks=None,
                        prev_output=output,
                        sub_token_type_ids=retriever_inputs['q_types'][t],
                        step=t,
                        tokenizer=self.tokenizer
                    )
                    reward, candidates = self.compute_loss_reward(output.reps, questions, truth=qrels)
                    feedback = self.compute_loss_feedback(questions, candidates, feedbacks=feedback)
                    rewards[t].append(reward.detach().cpu())

                # Store feedback 
                for j in range(len(data_indices)):
                    self.eval_dataset.add_feedback(data_indices[j], feedback[j])

        # finish one batch
        rewards_0 = torch.cat(rewards[0]) # B N
        rewards_1 = torch.cat(rewards[1]) # B N
        metrics['value-0'] = rewards_0.mean().cpu().detach().numpy().item()
        metrics['value-1'] = rewards_1.mean().cpu().detach().numpy().item()
        metrics['failed'] = (rewards_1 == 0).sum().cpu().detach().numpy().item()
        metrics['win'] = (rewards_1 > rewards_0).sum().cpu().detach().numpy().item()
        metrics['lose'] = (rewards_0 > rewards_1).sum().cpu().detach().numpy().item()

        # rewards_2 = torch.cat(rewards[2]) # B N
        # metrics['value-2'] = rewards_1.mean()
        return metrics

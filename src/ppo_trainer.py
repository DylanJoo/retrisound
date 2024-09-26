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
from functools import wraps
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import dataclasses

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from trl.trainer.ppov2_config import PPOv2Config
from trl.trainer.utils import trl_sanitze_kwargs_for_tagging
from utils import (
    multiple_sample_and_log_probability,
    augmentation_response,
    augmentation_feedback
)


INVALID_LOGPROB = 1.0


def vanilla_mean(
    values: torch.Tensor, 
    mask: torch.Tensor = None, 
    axis: Optional[bool] = None
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(values)
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
    #     self.critic_backbone = getattr(value_model, value_model.base_model_prefix)
    #
    # sepearate the policy and value model during training
    # def forward(self, **kwargs):
    #     output = self.critic_backbone(
    #         **kwargs,
    #     )
    #     logits = self.value_model.score(output.hidden_states[-1])
    #     return self.policy(**kwargs), logits

from trl.trainer.ppov2_trainer import PPOv2Trainer
class Trainer(PPOv2Trainer):

    def __init__(
        self,
        config: PPOv2Config,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        reward_model: nn.Module,
        value_model: nn.Module,
        train_dataset: Dataset,
        data_collator = None,
        eval_dataset = None,
        optimizers = (None, None),
        callbacks = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer

        self.policy = policy
        self.reward_model = reward_model
        self.value_model = value_model
        self.model = PolicyAndValueWrapper(policy, value_model)
        # data
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        # self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        # args.world_size = 1
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        print('local_batch_size', args.local_batch_size) # 8 * 2 * 4 
        print('micro_batch_size', args.micro_batch_size) # 8
        print('batch_size', args.batch_size)             # 64
        print('mini_batch_size', args.mini_batch_size)   # 16
        print('local_mini_batch_size', args.local_mini_batch_size)

        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, value_model, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        # self.model = PolicyAndValueWrapper(policy, value_model)
        # self.model.config = policy.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        # self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        torch.manual_seed(self.local_seed)  # reset the local seed again

        # [NOTE] remove so far
        # self.eval_dataloader = DataLoader(
        #     self.eval_dataset,
        #     batch_size=args.per_device_eval_batch_size,
        #     collate_fn=DataCollatorWithPadding(self.tokenizer),
        #     drop_last=True,
        # )  # no need to shuffle eval dataset
        # self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            # self.ref_policy = prepare_deepspeed(
            #     self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            # )
        else:
            # self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def train(self):
        args = self.args
        with open('logs/arguments.txt', 'w') as f:
            f.write(json.dumps(dataclasses.asdict(args))+'\n')

        self.save_model(args.output_dir, _internal_call=False)
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        policy = self.policy
        # ref_policy = self.ref_policy
        reward_model = self.reward_model
        value_model = self.value_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        # entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # [Iterations]
        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            # tau = (self.args.tau or policy.tau)
            tau = policy.tau

            #### sampled trajetories
            logprobs = []
            rankings = []
            responses = []
            feedbacks = []
            reward_scores = []
            values = []

            #### data for retrieval
            data = next(iter_dataloader)
            data_indices = data["index"]
            # State includes question, candidates (and feedback)
            retriever_inputs = data["inputs_for_retriever"]
            questions = data["questions"]
            candidates = data["candidates"]
            targets = data["targets"]
            max_num_steps = max(data['n_feedbacks']) + 1

            with torch.no_grad():

                for n in range(max_num_steps):
                    # policy forward (encoding)
                    # print('n is', n)
                    outputs = policy(**retriever_inputs, include_n_feedbacks=n+1)
                    qemb = outputs.qembs[:, :(n+1), :] # B N H
                    dembs = outputs.dembs              # B M H
                    modified_scores = torch.einsum("bnd, bkd->bnk", qemb/tau, dembs)
                    modified_scores = torch.max(modified_scores, 1).values

                    # argsort ranking
                    ranking, logprob = multiple_sample_and_log_probability(
                        modified_scores, 1, batch=True, sort=True
                    )
                    ranking = ranking.squeeze(1) 
                    # logprob = logprob.squeeze(1)
                    # print(ranking.shape)
                    # print(logprob.shape)

                    # reference logprob could be 
                    # (1) original bm25 ranking (with reciprocal ranking as scores)
                    # (2) original bm25 ranking (with re-sorted relevance scores)

                    # value
                    # value = value_model(qemb, dembs).logits
                    value = value_model(qemb, dembs).logprobs
                    gen_batch = len(prompt) if args.generation_batch is None else args.generation_batch

                    # Response generation and Reward
                    prompt = augmentation_response(
                        questions=questions, 
                        candidates=candidates, 
                        n_context=self.args.n_contexts,
                        rankings=ranking,
                        dataset_prefix=self.args.dataset_prefix,
                        answers=targets
                    )
                    response = []
                    for i in range(0, len(prompt), gen_batch):
                        _, _, b_response = self.reward_model._inference(
                            prompt[i:i+gen_batch],
                            max_new_tokens=8 # for judge
                        )
                        b_response = [r.rsplit('Question', 1)[0] for r in b_response]
                        response += b_response
                    reward_score = reward_model.get_rewards(response, targets).view(-1, 1).to(device)
                    reward_score = reward_score.to(model.device)

                    # Feedback generation and write
                    prompt = augmentation_feedback(
                        questions=questions, 
                        answers=response,
                        candidates=candidates, 
                        n_context=self.args.n_contexts,
                        rankings=ranking,
                        dataset_prefix=self.args.dataset_prefix
                    )
                    feedback = []
                    for i in range(0, len(prompt), gen_batch):
                        _, _, b_feedback = reward_model._inference(
                            prompt[i:i+gen_batch], 
                            max_new_tokens=64
                        )
                        feedback += b_feedback

                    for i in range(len(data_indices)):
                        self.train_dataset.add_feedback(data_indices[i], feedback[i])

                    # append the trajetory at t step and n segment
                    logprobs.append(logprob) # B 1
                    values.append(value)     # B 2
                    reward_scores.append(reward_score)   # B 1

                logprobs = torch.stack(logprobs, 1)[:, :, 0]  # B N 1
                values = torch.stack(values, 1)[:, :, 1]      # B N 1 
                reward_scores = torch.stack(reward_scores, 1)[:, :, 0]    # B N 1
                # del (logprob, ref_logprob, value, outputs)
                torch.cuda.empty_cache()
                gc.collect()

                # 4. compute rewards
                kl = logprobs - (-0.301) 
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward + reward_scores

                # print(logprobs.shape)
                # print(values.shape)
                # print(rewards.shape)

                # 5. whiten rewards
                # 6. compute advantages and returns
                """
                next value: the "(next-)feedback-aware query representation"
                """
                lastgaelam = 0
                advantages_reversed = []
                # gen_length = responses.shape[1]
                for t in range(max_num_steps):
                    nextvalues = values[:, t + 1] if t < max_num_steps - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                # advantages = masked_whiten(advantages, ~padding_mask)
                # advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            ##### check trajectory here #####
            with open('trajectory.json', 'a') as f:
                data = [self.train_dataset[idx] for idx in data_indices]
                for i, d in enumerate(data):
                    _ = [d.pop(k) for k in ['index', 'n_feedbacks', 'candidate_pids', 'candidates']]
                    d['response'] = response[i]
                    json.dump(d, f)
                    f.write('\n')

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size) # this should equal loader batch size
                minibatch_idx = 0

                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]

                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Get micro batch data
                            mb_retriever_inputs = self.data_collator.get_inputs_for_retriever(
                                [self.train_dataset[data_indices[idx]] for idx in micro_batch_inds],
                                device=model.device
                            )
                            mb_candidates = [candidates[idx] for idx in micro_batch_inds]
                            mb_targets = [targets[idx] for idx in micro_batch_inds]

                            # Trajectory (b n 1)
                            mb_advantage = advantages[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            # FP
                            outputs = model.policy(**mb_retriever_inputs)

                            new_logprobs = []
                            new_values = []
                            # for n in range(len(mb_retriever_inputs['q_tokens'])):
                            for n in range(max_num_steps):
                                qemb = outputs.qembs[:, n, :]  # b H
                                dembs = outputs.dembs          # b M H
                                modified_scores = torch.einsum("bd, bkd->bk", qemb/tau, dembs)
                                new_ranking, new_logprob = multiple_sample_and_log_probability(
                                    modified_scores, 1, batch=True, sort=True
                                )
                                new_ranking = new_ranking.squeeze(1) 
                                # new_value = value_model(qemb, dembs).logits
                                new_value = value_model(qemb, dembs).logprobs

                                # append the trajetory at t step and n segment
                                new_logprobs.append(new_logprob) # b 1
                                new_values.append(new_value)    # b 2

                            new_logprobs = torch.stack(new_logprobs, 1)[:, :, 0]
                            vpred = new_values = torch.stack(new_values, 1)[:, :, 1]
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * vanilla_mean(vf_loss_max)
                            vf_clipfrac = vanilla_mean((vf_losses2 > vf_losses1).float())
                            # print(new_logprobs.shape)
                            # print(mb_logprobs.shape)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = vanilla_mean(pg_loss_max)
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = vanilla_mean((pg_losses2 > pg_losses).float())
                                # prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                # entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                # entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        outputs, new_logprobs, 
                        vpred, vpredclipped, vf_losses1, vf_losses2, vf_loss, vf_clipfrac, 
                        logprobs_diff, ratio, 
                        pg_losses, pg_losses2, pg_loss_max, pg_loss, loss, pg_clipfrac, 
                        approxkl, mb_return, mb_advantage, mb_values, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + reward_scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(reward_scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                # metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                # metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del (
                kl, mean_kl, mean_entropy, 
                mean_non_score_reward, reward_scores, metrics, non_score_reward
            )
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                logprobs,
                values,
                rewards,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        if output_dir is None:
            output_dir = self.args.output_dir

        if not _internal_call:  # `push_to_hub` already swaps out the self.model with policy
            self.backup_model = self.model
            self.model = self.accelerator.unwrap_model(self.model) # save only the policy
            agent_base = self.model.policy.qf_encoder # save only the qencoder's bert
        else:
            agent_base = self.backup_model.policy.qf_encoder

        state_dict = self.accelerator.get_state_dict(agent_base)

        if self.accelerator.is_main_process:

            if self.args.should_save:
                agent_base.save_pretrained(
                    self.args.output_dir, 
                    is_main_process=self.accelerator.is_main_process, 
                    save_function=self.accelerator.save, 
                    state_dict=state_dict
                )

        if not _internal_call:
            self.model = self.backup_model



import gc
import math
import os
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

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
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback

from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
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
from trl.trainer.ppov2_trainer import PPOv2Trainer

from utils import (
    convert_texts_to_tensors, 
    multiple_sample_and_log_probability, 
    augmentation,
    get_mini_batch_dict
)

INVALID_LOGPROB = 1.0

# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`

# class PolicyAndValueWrapper(nn.Module):
#     def __init__(self, policy, value_model) -> None:
#         super().__init__()
#         self.policy = policy
#         self.value_model = value_model
#
#     def forward(self, **kwargs):
#         output = self.policy(**kwargs)
#         logits = self.value_model.score(output.hidden_states[-1])
#         return self.policy(**kwargs), logits

class RAGRLTrainer(PPOv2Trainer):

    def __init__(
        self,
        config: PPOv2Config,
        tokenizer: PreTrainedTokenizer,
        model: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator=None,
        eval_dataset=None,
        optimizers=(None, None),
        callbacks=None,
    ) -> None:
        """
        """
        self.args = config
        args = config
        self.tokenizer = tokenizer

        self.model = model
        self.reward_model = reward_model

        # disable `pad_token_id` and `eos_token_id` because we just want to
        # self.reward_model.generator.generation_config.eos_token_id = None  
        # generate tokens without truncation / padding
        # self.reward_model.generator.generation_config.pad_token_id = None  

        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
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
        for module in [model, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        # self.model = PolicyAndValueWrapper(policy, value_model)
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
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
        self.backup_model = None

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

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            self.reward_model = self.accelerator.prepare(self.reward_model)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device

        optimizer = self.optimizer
        policy = self.model
        value_model = self.model.vhead
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        accelerator.print("===training policy===")
        start_time = time.time()

        policy.train()

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

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)

            #### data for retrieval
            data_indices = data["index"] 

            #### sampled trajetories
            # observations = []
            logits = []
            logprobs = []
            rankings = []
            responses = []
            rewards = []
            values = []

            for step in range(0, self.args.num_steps+1):
                # State: question, candidates, feedback
                if step == 0:
                    retriever_inputs = data["inputs_for_retriever"]
                    candidates = data["candidates"]
                    questions = data["questions"]
                    targets = data["targets"]
                    # target_tensors = convert_texts_to_tensors(targets)
                    # max_sequence_length = target_tensors.shape[1]
                else: 
                    retriever_inputs = self.data_collator.get_inputs_for_retriever(
                        [self.train_dataset[idx] for idx in data_indices],
                        device=device
                    )

                with torch.no_grad():
                    # Action: scoring and sampling ranking
                    sim_logits = policy(**retriever_inputs).scores
                    ranking, logprob = multiple_sample_and_log_probability(
                        sim_logits, 1, batch=True
                    )
                    ranking = ranking.squeeze(1) # use only one sampeld ranking
                    value = value_model(sim_logits)

                    # Reward
                    ## using "metric" as reward
                    ## [TODO] if using "likelihood" as reward
                    prompt, prompt_fbk = augmentation(
                        questions, candidates, ranking, args.n_contexts
                    )
                    _, _, response = reward_model._inference(prompt, responses=None)
                    reward = reward_model.get_rewards(response, targets).view(-1, 1)
                    reward = reward.to(device)

                    # Feedback
                    _, _, feedback = reward_model._inference(prompt_fbk, responses=None)
                    for i in range(len(data_indices)):
                        self.train_dataset.add_feedback(data_indices[i], feedback[i])

                # Feedback if applicable
                # observations = retriever_inputs
                logits.append(sim_logits)
                rankings.append(ranking)
                logprobs.append(logprob)
                responses.append(response)
                values.append(value)
                rewards.append(reward) 

                # (1) from value_model > generate > metric evaluation (2) from value_model > target likelihood

                del (sim_logits, ranking, logprob, value, prompt, reward)
                torch.cuda.empty_cache()
                gc.collect()

            logits = torch.cat(logits, 1)       # B N_step N_cand
            rankings = torch.cat(rankings, 1)   # B N_step N_cand

            logprobs = torch.cat(logprobs, 1)   # B N_step 
            rewards = torch.cat(rewards, 1)     # B N_step
            values = torch.cat(values, 1)       # B N_step

            # Advantage: next value and previous value comparison
            # with torch.no_grad():
            lastgaelam = 0
            advantages_reversed = []
            for t in reversed(range(args.num_steps + 1)):
                next_values = values[:, t + 1] if t < args.num_steps - 1 else 0.0
                delta = rewards[:, t] + args.gamma * next_values - values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)

            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            torch.cuda.empty_cache()

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)

                # only forward once
                # mb_retriever_inputs = get_mini_batch_dict(observations, mb_inds)
                all_sim_logits = policy(**retriever_inputs).all_scores # B N_seg N_cand

                for start in range(0, args.batch_size, args.mini_batch_size):
                    end = start + args.mini_batch_size
                    mb_inds = b_inds[start:end]
                    # print(len(candidates))
                    # print(mb_inds)
                    # mb_candidates = [candidates[i] for i in mb_inds]
                    # mb_questions = [questions[i] for i in mb_inds]

                    mb_logprobs = logprobs[mb_inds]
                    mb_rewards = rewards[mb_inds]
                    mb_values = values[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    mb_returns = returns[mb_inds]

                    new_logprobs = torch.zeros_like(mb_logprobs)
                    new_values = torch.zeros_like(mb_values)

                    for t in range(args.num_steps):
                        mb_logit = all_sim_logits[mb_inds, t]

                        _, new_logprob = multiple_sample_and_log_probability(
                            mb_logit, 1, batch=True)
                        new_value = value_model(mb_logit)

                        new_logprobs[:, t] = new_logprob.flatten()
                        new_values[:, t] = new_value.flatten()

                    logratio = new_logprobs - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]


                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if args.clip_vloss:
                        v_loss_unclipped = (new_value - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            new_value - mb_values,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                    # entropy_loss = entropy.mean()
                    loss = pg_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                del (
                    mb_logprobs, mb_rewards, mb_logit,
                    mb_values, mb_advantages, mb_returns, new_value, new_logprob
                )

            torch.cuda.empty_cache()
            with torch.no_grad():
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics["loss/value_loss"] = self.accelerator.gather(v_loss).mean().item()
                metrics["loss/policy_loss"] = self.accelerator.gather(pg_loss).mean().item()
                # metrics["loss/entropy"] = self.accelerator.gather(entropy_loss).mean().item()
                metrics["loss/old_approx_kl"] = self.accelerator.gather(old_approx_kl).mean().item()
                metrics["loss/approx_kl"] = self.accelerator.gather(approx_kl).mean().item()
                metrics["loss/clipfrac"] = np.mean(clipfracs)
                metrics["val/ratio"] = self.accelerator.gather(ratio).mean().item()

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del old_approx_kl, approx_kl, clipfracs, metrics
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()

            del (
                rankings, logprobs, rewards,
                values, advantages, returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


import gc
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
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback

from trl.core import masked_mean, masked_whiten
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    generate,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from trl.trainer.ppov2_config import PPOv2Config
from trl.trainer.ppov2_trainer import PPOv2Trainer
from modeling.rewards import MetricRewards
from utils import get_expected_inputs

INVALID_LOGPROB = 1.0


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, reward_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = reward_model

    def forward(self, **kwargs):
        return self.policy(**kwargs)


class RAGPPOv2Trainer(PPOv2Trainer):

    def __init__(
        self,
        config: PPOv2Config,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks

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
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)

        #########
        # setup model, optimizer, and others
        #########
        # for module in [policy, ref_policy, value_model, reward_model]:
        #     disable_dropout_in_model(module)
        for module in [policy, ref_policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
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
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        # args.bf16 = False
        # args.fp16 = True
        if self.is_deepspeed_enabled:
            if isinstance(reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.bf16, args.fp16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.bf16, args.fp16
            )
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()
        for update in range(1, args.num_updates + 1):
            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            with torch.no_grad():

                ####  reformualte the data inputs
                data_indices = data.pop("index", None)
                retriever_inputs = data.pop("inputs_for_retriever")
                target_texts = data.pop("targets", None)

                ##### forward pass the biencoder
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    prompts, prompts_fbk, prompts_last, _ = unwrapped_model.policy._forward_retrieval(
                        **retriever_inputs,
                        questions=data["questions"],
                        candidates=data["candidates"]
                    )
                    queries = tokenizer(
                        prompts,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).input_ids.to(device)
                    queries_last = tokenizer(
                        prompts_last,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).input_ids.to(device)

                    queries_target_new, target_length = get_expected_inputs(queries, target_texts, tokenizer)
                    queries_target_old, _ = get_expected_inputs(queries_last, target_texts, tokenizer)

                ####
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                values = []
                scores = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                        query = queries[i : i + args.local_rollout_forward_batch_size]
                        query_response, logits = generate(
                            unwrapped_model.policy,
                            query,
                            tokenizer.pad_token_id,
                            generation_config,
                        )
                        query_target_new = queries_target_new[i : i + args.local_rollout_forward_batch_size]
                        query_target_old = queries_target_old[i : i + args.local_rollout_forward_batch_size]
                        target_text = target_texts[i : i + args.local_rollout_forward_batch_size]

                        response = query_response[:, context_length:]
                        target = query_target_new[:, -target_length:]

                        # use the logits during generation directly, instead of using the following
                        ## [NOTE] we use the forward again, since we use target instead.
                        act_output = forward(ref_policy, query_target_new, tokenizer.pad_token_id)
                        logits = act_output.logits[:, -target_length:]
                        logits /= args.temperature + 1e-7
                        all_logprob = F.log_softmax(logits, dim=-1)
                        logprob = torch.gather(all_logprob, 2, target.unsqueeze(-1)).squeeze(-1)
                        del act_output, logits, all_logprob
                        torch.cuda.empty_cache()

                        ref_output = forward(ref_policy, query_target_old, tokenizer.pad_token_id)
                        ref_logits = ref_output.logits[:, -target_length:]
                        ref_logits /= args.temperature + 1e-7
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, target.unsqueeze(-1)).squeeze(-1)
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()

                        # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                        # [NOTE] targets have not stop token
                        postprocessed_response = response
                        if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                            postprocessed_response = truncate_response(
                                args.stop_token_id, tokenizer.pad_token_id, response
                            )

                        # Response Processing 2. run reward model on the truncated responses
                        ## [NOTE] In RAG, it's a bit weird as we only change the query-side
                        # postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                        sequence_length = first_true_indices(target == tokenizer.pad_token_id) - 1 
                        # [NOTE] change the value model to query-targe pair 
                        # as we are calculating the logit values of target
                        # unwrapped_value_model = accelerator.unwrap_model(model).value_model
                        value_output = forward(model.policy, query_target_new, tokenizer.pad_token_id) 
                        value_logits = value_output.logits[:, -target_length:]
                        value = torch.gather(value_logits, 2, target.unsqueeze(-1)).squeeze(-1)
                        del value_output, value_logits
                        torch.cuda.empty_cache()

                        # [NOTE] here the example used a hf model to do rewarding
                        # response_text  = [tokenizer.decode(r) for r in postprocessed_response]
                        score = reward_model.calculate_rewards(prompts, target_text)

                        # [NOTE] Now we can replace the response with target
                        postprocessed_response = response = target
                        query_response = query_target_new

                        query_responses.append(query_response)
                        responses.append(response)
                        postprocessed_responses.append(postprocessed_response)
                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        values.append(value)
                        sequence_lengths.append(sequence_length)
                        scores.append(score)
                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                values = torch.cat(values, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0).to(self.accelerator.device)
                del (logprob, ref_logprob, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                sequence_lengths_p1 = sequence_lengths + 1
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            ### add action to dataset (fbk)
            queries_fbk = tokenizer(
                prompts_fbk,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(device)

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                query_feedbacks, _ = generate(
                    unwrapped_model.policy,
                    query,
                    tokenizer.pad_token_id,
                    generation_config,
                )
                feedbacks = query_feedbacks[:, queries_fbk.shape[1]:]
                feedbacks = [tokenizer.decode(fbk.squeeze()) for fbk in feedbacks]
                for i, feedback in enumerate(feedbacks):
                    self.train_dataset.add_action(data_indices[i], feedback)

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0

                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0

                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_return = returns[micro_batch_inds]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            ### retreive the set of context
                            mb_retriever_inputs = {}
                            for key, list_of_item in retriever_inputs.items():
                                mb_retriever_inputs[key] = []
                                for item in list_of_item:
                                    mb_retriever_inputs[key].append( item[micro_batch_inds] )

                            mb_prompts, _, _, output_r = model.policy._forward_retrieval(
                                **mb_retriever_inputs,
                                questions=[data["questions"][i] for i in micro_batch_inds],
                                candidates=[data["candidates"][i] for i in micro_batch_inds]
                            )
                            mb_queries = tokenizer(
                                prompts,
                                padding=True,
                                truncation=True,
                                return_tensors='pt'
                            ).input_ids.to(device)
                            mb_query_responses, _ = get_expected_inputs(mb_queries, mb_responses, tokenizer)
                            mb_logprobs = logprobs[micro_batch_inds]
                            output = forward(model.policy, mb_query_responses, tokenizer.pad_token_id)

                            # here we set the value as output.logit
                            logits = output.logits[:, -target_length:]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )

                            vpred_temp = output.logits
                            vpred = vpred_temp[:, -target_length:]
                            vpred = torch.gather(vpred, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            vpred = torch.masked_fill(
                                vpred, padding_mask_p1[micro_batch_inds], 0
                            )
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            print(model.policy.model.embed_tokens.weight)
                            print(model.policy.model.hello.weight)
                            print(model.policy.biencoders.q_encoder.model.embeddings.word_embeddings.weight)
                            accelerator.backward(loss)
                            # print(loss.grad)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = vf_clipfrac
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = global_step
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

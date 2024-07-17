import math
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from trl import PPOTrainer
from trl.models import PreTrainedModelWrapper
from trl.core import PPODecorators, stats_to_np, logprobs_from_logits, stack_dicts, WANDB_PADDING, convert_to_scalar
from transformers import PreTrainedModel, get_scheduler

from prompt.qampari import *

class RAGRLTrainer(PPOTrainer):

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        questions: List[str], 
        retriever_inputs: Optional[dict], 
        candidates: List[str],
        targets: List[str], 
        response_masks: Optional[List[torch.LongTensor]] = None,
    ): 
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args: 
            queries (List[`torch.LongTensor`]): 
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        # [NOTE] remove it since this is not aligned to our collator
        preserved_length = max([len(r) for r in responses])
        model_inputs = self.prepare_model_inputs(queries, responses, preserved_length)

        # no idea why this should be done
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.eos_token_id, # pad token is not used
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        # [NOTE] since the only tunable component in the active model is retriever, we need only generation once.
        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries, # this will be changed
                responses,
                model_inputs=None,
                questions=questions,
                retriever_inputs=retriever_inputs,
                candidates=candidates,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
                preserved_length=preserved_length
            )
            # [NOTE] maybe a better reference model is the normal bm25 top k context
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries, # this will be changed
                    responses,
                    model_inputs=None,
                    questions=questions,
                    retriever_inputs=retriever_inputs,
                    candidates=candidates,
                    response_masks=response_masks,
                    return_logits=full_kl_penalty,
                    preserved_length=preserved_length,
                    reference_contexts=True
                )
                # ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                #     self.model if self.is_peft_model else self.ref_model,
                #     queries,
                #     responses,
                #     model_inputs=model_inputs,
                #     return_logits=full_kl_penalty,
                #     preserved_length=preserved_length
                # )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, all_logprobs, ref_logprobs, masks
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,      # List[str]
            "responses": responses,  # List[str]
            "questions": questions,  # List[List[str]]
            "candidates": candidates,
            "retriever_inputs": retriever_inputs,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "questions": [batch_dict["questions"][i] for i in mini_batch_inds],
                        "candidates": [batch_dict["candidates"][i] for i in mini_batch_inds],
                        "retriever_inputs": self.get_minibatched_dictionary(retriever_inputs, mini_batch_inds),
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            queries=mini_batch_dict["queries"], # this queries is in fact useless
                            responses=mini_batch_dict["responses"],
                            model_inputs=None,          # since we have to update queries first 
                            questions=mini_batch_dict["questions"],
                            retriever_inputs=mini_batch_dict["retriever_inputs"],
                            candidates=mini_batch_dict["candidates"],
                            return_logits=True,
                            preserved_length=max([len(r) for r in responses]),
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict = None,
        questions: dict = None,
        retriever_inputs: dict = None,
        candidates: dict = None,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
        preserved_length: Union[bool, int] = False,
        reference_contexts: bool = False
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            X queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
            preserved_length (`bool`, `int`, defaults to `False`):
                whether and the length of truncation should offset the input query + response.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(responses)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        # get the queries 
        if model_inputs is None:
            prompts, _, _, _ = model._forward_retrieval(
                **retriever_inputs, 
                questions=questions, 
                candidates=candidates,
                reference_contexts=reference_contexts
            )
            queries = [self.tokenizer(q, return_tensors='pt').input_ids[0] for q in prompts]
            queries = [tensor.to(self.current_device) for tensor in queries]

            model_inputs = self.prepare_model_inputs(queries, responses, preserved_length)

        # print('r', max([len(r) for r in responses]))
        # print('m', max([len(m) for m in model_inputs['input_ids']])) # [L] * B

        for i in range(math.ceil(bs / fbs)):
            # model inputs
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)
            # print(logits.shape) # B L V
            # print(values.shape) # B L 

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        if preserved_length:
            outputs = (
                torch.cat(all_logprobs)[:, -preserved_length:],
                torch.cat(all_logits)[:, -(1+preserved_length):-1] if return_logits else None,
                torch.cat(all_values)[:, -(1+preserved_length):-1],
                torch.cat(all_masks)[:, -(1+preserved_length):-1],
            )
        else:
            outputs = (
                torch.cat(all_logprobs),
                torch.cat(all_logits)[:, :-1] if return_logits else None,
                torch.cat(all_values)[:, :-1],
                torch.cat(all_masks)[:, :-1],
            )

        return outputs

    @staticmethod
    def get_minibatched_dictionary(retriever_inputs, indices): # q_tokens, q_masks
        reshaped_retriever_inputs = {}
        for key, list_of_item in retriever_inputs.items():
            reshaped_retriever_inputs[key] = []
            for item in list_of_item:
                reshaped_retriever_inputs[key].append( item[indices, :] )
        # print('q', len(retriever_inputs['q_tokens']))
        # print('d', len(retriever_inputs['d_tokens']))
        return reshaped_retriever_inputs

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor, max_response_length: int = None):
        ## retokenize the response, we would like it to be consistent.
        if max_response_length is not None:
            self.tokenizer.padding_side = 'right'
            response_inputs = self.tokenizer.pad(
                {"input_ids": responses},
                padding='max_length',
                max_length=max_response_length
            )
            responses = response_inputs.input_ids.to(queries[0].device)
            # response_masks = response_inputs.to(queries[0].device)
            self.tokenizer.padding_side = 'left'

        ## retokenize the response, we would like it to be consistent.
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator(
            [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
        ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

    # def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
    #     input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
    #     input_data = self.data_collator(
    #         [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
    #     ).to(self.current_device)
    #     input_data.pop("labels", None)  # we don't want to compute LM losses
    #     # attention_mask: [padding] * n + [query] + [response]
    #     query_mask = None
    #     if return_query_mask:
    #         query_mask = torch.zeros_like(input_data["attention_mask"].clone())
    #         ## query_mask: [...] + [response]
    #         for i, response in enumerate(responses):
    #             query_mask[i, :-(1+len(response)) ] = 0
    #
    #     query_mask = None
    #     if return_query_mask:
    #         query_mask = input_data["attention_mask"].clone()
    #         ## query_mask: select only the response part
    #         for i, query, response in enumerate(zip(queries, responses)):
    #             query_mask[i, -len(input_ids):-(1+len(response)) ] = 0
    #     return input_data, query_mask

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: Non score rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `response_length`)
        """
        # print(scores.shape)
        # print(logprobs.shape)
        # print(ref_logprobs.shape)
        # print(masks.shape)
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]
            # print(reward)

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v.contiguous().to(self.accelerator.device), dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def _kl_penalty(
        self, 
        logprob: torch.FloatTensor, 
        ref_logprob: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError


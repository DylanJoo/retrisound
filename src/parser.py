#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import sys
import random
import datasets
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from dataclasses import asdict
from copy import deepcopy

from trl import AutoModelForCausalLMWithValueHead
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed
)

from transformers.utils import logging 
from utils import update_tokenizer

logger = logging.get_logger("transformers")


def main():

    from options import ModelOptions, DataOptions, TrainOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    os.environ["WANDB_PROJECT"] = train_opt.wandb_project
    set_seed(train_opt.seed)

    # [Retriever Bi-encoder]
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    from modeling.rmt import RMTEncoder
    from modeling.rife import Contriever
    ada_encoder = RMTEncoder(
        base_model=Contriever.from_pretrained(model_opt.retriever_name_or_path),
        tokenizer=tokenizer_r,
        num_mem_tokens=model_opt.num_mem_tokens,
        n_max_segments=train_opt.n_max_segments,
        input_size=128,
        sum_loss=False,
    )
    from modeling.inbatch import InBatchInteraction
    bi_encoders = InBatchInteraction(
        model_opt,
        q_encoder=ada_encoder,
        d_encoder=Contriever.from_pretrained(model_opt.retriever_name_or_path),
        fixed_d_encoder=True
    )

    # [Generatir Config & tokenizer & Model]
    ## [TODO] Check further the accurate setup of tokenizer for llama
    from utils import update_tokenizer
    tokenizer_g = AutoTokenizer.from_pretrained(
            model_opt.generator_name_or_path, 
            padding_side='left',
            use_fast=True
    )
    tokenizer_g = update_tokenizer(tokenizer_g, '<|reserved_special_token_0|>')
    config = AutoConfig.from_pretrained(model_opt.generator_name_or_path)
    generator = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_opt.generator_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation,
        torch_dtype=torch.bfloat16
    ).eval()

    stop = ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>", "<|end_of_text|>"]
    stop_token_ids = [tokenizer_g.eos_token_id] + [tokenizer_g.convert_tokens_to_ids(token) for token in stop]
    stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))  

    # [RAG]
    from modeling import RerankAugmentedGeneration
    model = RerankAugmentedGeneration(
        llm=generator,
        tokenizer=tokenizer_g,
        biencoders=bi_encoders,
        stop_token_ids=stop_token_ids,
        k=model_opt.num_contexts
    )

    # [Reward model] 
    reward_model = answer_recall()

    # [Data]
    ## [NOTE] execute multithread once and wait
    from data.qampari import ContextQADataset, ContextQACollator
    train_dataset = ContextQADataset(
        data_file=data_opt.train_file, 
        n_max_segments=train_opt.n_max_segments,
        n_max_candidates=train_opt.n_max_candidates,
        budget=model_opt.budget,
        depth=data_opt.depth,
        corpus_file=data_opt.corpus_file,
        retrieval_file=data_opt.retrieval_file,
        quick_test=train_opt.quick_test
    )

    # [Data] data ollator
    ## [TODO] add sampler by action size (e.g., less to more)
    data_collator = ContextQACollator(
        tokenizer_r=tokenizer_r,
        tokenizer_g=tokenizer_g,
    )

    # [trainer]
    from trl import PPOConfig, PPOTrainer
    config = PPOConfig(
	model_name=model_opt.retriever_name_or_path + model.opt.generator_name_or_path,
	log_with=train_opt.report_to,
	query_dataset=data_opt.train_file,
	reward_model=model.opt.generator_name_or_path,
	learning_rate=train_opt.learning_rate,
	steps=train_opt.max_steps,
	batch_size=train_opt.per_device_train_batch_size,
	mini_batch_size=train_opt.per_device_train_batch_size,
	gradient_accumulation_steps=train_opt.gradient_accumulation_steps,
	optimize_device_cache=True
    )
    ppo_trainer = PPOTrainer(
	model=model,
	config=config,
	dataset=train_dataset,
	tokenizer=tokenizer_g,
    )

    # train!
    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    
        #### Get the adaptive context from retriever
        retriever_inputs = batch.pop("inputs_for_retriever")
        prompts, prompts_fbk, prompts_last, loss_r = ppo_trainer.model._forward_retrieval(
            **retriever_inputs,
            questions=batch["questions"],
            candidates=batch["candidates"]
        )
        # instruction + batch["questions"] + context
        batch["queries"] = prompts
        #### tokenized the adaptive context, as the query
        tokenizer.padding_side = "left"
        query_inputs = tokenizer_g(
            batch["queries"],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(ppo_trainer.device)

        response_inputs = tokenizer_g(
            batch['targets'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(ppo_trainer.device)

        if train_opt.policy_on == 'metrics': 
            #### Get response 
            predicted_targets = ppo_trainer.generate(
                **query_inputs,
                min_length=-1,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer_g.pad_token_id,
                max_new_tokens=64
            )
            #### Compute reward score
            texts = [q + r for q, r in zip(batch["queries"], predicted_targets)]
            rewards = reward_model(texts, batch['answers'])

        # if train_opt.policy_on == 'likelihood':
        #     #### [NOTE] may need to revise `step()` as it also calculates nll
        #     #### get likelihood
        #     active_rewards = ppo_trainer.model.get_likelihood(prompts, targets)
        #     ref_rewards = ppo_trainer.model.get_likelihood(prompts_last, targets)
        #     rewards = active_rewards / ref_rewards

	#### Run PPO step
        tokenizer.padding_side = "left"
        stats = ppo_trainer.step(
            queries=query_inputs['input_ids'], 
            responses=response_inputs['input_ids'], 
            scores=rewards,
            retriever_inputs=retriever_inputs,
            targets=batch["target"]
        )
        ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained("my_ppo_model")

if __name__ == '__main__':
    main()

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

from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
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
    if data_opt.config_file is not None:
        model_opt, data_opt, train_opt = parser.parse_yaml_file(data_opt.config_file)
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
    ).train()
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
    tokenizer_g = update_tokenizer(tokenizer_g)
    config = AutoConfig.from_pretrained(model_opt.generator_name_or_path)

    stop = ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>", "<|end_of_text|>"]
    stop_token_ids = [tokenizer_g.eos_token_id] + [tokenizer_g.convert_tokens_to_ids(token) for token in stop]
    stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))  

    # [RAG]
    from modeling import RerankAugmentedGenerationWrapper
    llm = AutoModelForCausalLM.from_pretrained(
        model_opt.generator_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation,
        torch_dtype=torch.bfloat16
    ).eval()
    model = RerankAugmentedGenerationWrapper(
        pretrained_model=llm,
        biencoders=bi_encoders,
        stop_token_ids=stop_token_ids,
        num_budget=model_opt.num_budget,
    ).train()

    ref_model = RerankAugmentedGenerationWrapper(
        pretrained_model=model.pretrained_model,
        biencoders=None, # the reference model is actually a one-shot RAG with bm25
        stop_token_ids=stop_token_ids,
        num_budget=model_opt.num_budget,
        is_reference=True
    ).eval()

    # [Reward model] 
    if train_opt.policy_on == 'metrics': 
        import evaluate
        rouge = evaluate.load('rouge')

        def reward_model(xs, ys):
            results = rouge.compute(predictions=xs, references=ys, use_aggregator=False)
            return [torch.tensor(s) for s in results['rouge1']]

    # [Data]
    ## [NOTE] execute multithread once and wait
    from data.qampari import ContextQADataset, ContextQACollator
    train_dataset = ContextQADataset(
        data_file=data_opt.train_file, 
        n_max_segments=train_opt.n_max_segments,
        n_max_candidates=train_opt.n_max_candidates,
        budget=model_opt.num_budget,
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
    from ppo_trainer import RAGRLTrainer
    config = PPOConfig(
	model_name=model_opt.retriever_name_or_path + model_opt.generator_name_or_path,
	log_with=train_opt.report_to,
	query_dataset=data_opt.train_file,
	reward_model=model_opt.generator_name_or_path,
	learning_rate=train_opt.learning_rate,
	steps=train_opt.max_steps,
	batch_size=train_opt.per_device_train_batch_size,
	mini_batch_size=train_opt.per_device_train_batch_size // 2,
	gradient_accumulation_steps=train_opt.gradient_accumulation_steps,
	optimize_device_cache=True,
	# init_kl_coef=0.0,
	# adap_kl_ctrl=False,
	# tracker_project_name=train_opt.wandb_project
    )
    ppo_trainer = RAGRLTrainer(
	config=config,
	model=model,
	ref_model=ref_model,
	tokenizer=tokenizer_g,
	dataset=train_dataset,
	data_collator=data_collator
    )

    # train!
    for epoch in tqdm(range(train_opt.num_train_epochs), "epochs: "):
        for batch in tqdm(ppo_trainer.dataloader):
    
            #### Get the adaptive context from retriever
            data_indices = batch.pop("index", None)
            retriever_inputs = batch.pop("inputs_for_retriever")
            prompts, prompts_fbk, prompts_last, loss_r = ppo_trainer.model._forward_retrieval(
                **retriever_inputs,
                questions=batch["questions"],
                candidates=batch["candidates"]
            )
            # instruction + batch["questions"] + context
            batch["query"] = prompts
            #### tokenized the adaptive context, as the query
            tokenizer_g.padding_side = "left"
            query_tensors = [tokenizer_g(q, return_tensors='pt').input_ids[0] for q in batch["query"]]

            if train_opt.policy_on == 'metrics': 
                #### Get response 
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    top_k=0.0,
                    top_p=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer_g.eos_token_id,
                    min_length=-1,
                    max_new_tokens=64
                )
                batch['response'] = [tokenizer_g.decode(r.squeeze()) for r in response_tensors]
                #### Compute reward score
                rewards = reward_model(batch['response'], batch['targets'])

            #### Run PPO step
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                scores=rewards,
                questions=batch['questions'],
                retriever_inputs=retriever_inputs,
                candidates=batch['candidates'],
                targets=batch["targets"]
            )
            print(ppo_trainer.model.biencoders.q_encoder.model.embeddings.word_embeddings.weight)

            ### add action to dataset
            if data_indices is not None:
                query_tensors = [tokenizer_g(q, return_tensors='pt').input_ids[0] for q in prompts_fbk]
                feedback_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer_g.eos_token_id,
                    max_new_tokens=64
                )
                feedbacks = [tokenizer_g.decode(fbk.squeeze()) for fbk in feedback_tensors]
                for i, feedback in enumerate(feedbacks):
                    train_dataset.add_action(data_indices[i], feedback)
                    # print(i, feedback.replace('\n', ''))

            ppo_trainer.log_stats(stats, batch, rewards)

        #### Save model
        ppo_trainer.save_pretrained("my_ppo_model")

if __name__ == '__main__':
    main()

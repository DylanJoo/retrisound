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

from utils import update_tokenizer, init_generation_config

logger = logging.get_logger("transformers")

def main():

    from options import ModelOptions, DataOptions, RLTrainOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, RLTrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()
    if data_opt.config_file is not None:
        model_opt, data_opt, train_opt = parser.parse_yaml_file(data_opt.config_file)

    set_seed(train_opt.seed)

    # [Retriever]
    ## Config & tokenizer
    from modeling import RMTEncoder, AdaptiveReranker, Contriever
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    ada_encoder = RMTEncoder(
        base_model=Contriever.from_pretrained(model_opt.retriever_name_or_path),
        tokenizer=tokenizer_r,
        num_mem_tokens=model_opt.num_mem_tokens,
        n_max_segments=train_opt.n_max_segments,
        input_size=256,
        sum_loss=False,
    )
    ada_reranker = AdaptiveReranker(
        model_opt,
        q_encoder=ada_encoder,
        d_encoder=Contriever.from_pretrained(model_opt.retriever_name_or_path),
        n_max_candidates=train_opt.n_max_candidates,
        do_contrastive=True
    ).train()

    # [Generator]
    ## Config & tokenizer
    from utils import update_tokenizer
    from modeling import GenerativeRewardWrapper, Metric
    tokenizer_g = AutoTokenizer.from_pretrained(
        model_opt.generator_name_or_path, 
        padding_side='left',
        use_fast=True
    )
    tokenizer_g = update_tokenizer(tokenizer_g)
    config = AutoConfig.from_pretrained(model_opt.generator_name_or_path)

    llm = AutoModelForCausalLM.from_pretrained(
        model_opt.generator_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation,
        torch_dtype=torch.bfloat16,
    ).eval()
        # load_in_4bit=True

    # [RAG]
    generation_config = init_generation_config(model_opt, tokenizer_g)

    reward_model = GenerativeRewardWrapper(
        generator=llm, 
        tokenizer=tokenizer_g, 
        utility=Metric('rouge'),
        generation_config=generation_config
    ).eval()

    # [Data]
    if 'qampari' in data_opt.train_file:
        from data.qampari import ContextQADataset, ContextQACollator
    if 'asqa' in data_opt.train_file:
        from data.asqa import ContextQADataset, ContextQACollator
    train_dataset = ContextQADataset(
        data_file=data_opt.train_file, 
        n_max_segments=train_opt.n_max_segments,
        n_max_candidates=train_opt.n_max_candidates,
        budget=model_opt.num_budget,
        depth=data_opt.depth,
        corpus_file=data_opt.corpus_file,
        retrieval_file=data_opt.retrieval_file,
    )

    ## data ollator
    data_collator = ContextQACollator(
        tokenizer_r=tokenizer_r,
        tokenizer_g=tokenizer_g,
    )

    # [trainer]
    from rlrag_trainer import RAGRLTrainer
    trainer = RAGRLTrainer(
        config=train_opt,
        tokenizer=tokenizer_g,
        model=ada_reranker,
        reward_model=reward_model,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(train_opt.output_dir)

if __name__ == '__main__':
    main()

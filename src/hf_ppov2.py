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

    from options import ModelOptions, DataOptions, RLTrainOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, RLTrainOptions))
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
    tokenizer_g = update_tokenizer(tokenizer_g, "[PAD]")

    stop = ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>", "<|end_of_text|>"]
    stop_token_ids = [tokenizer_g.eos_token_id] + [tokenizer_g.convert_tokens_to_ids(token) for token in stop]
    stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))  

    # [RAG]
    config = AutoConfig.from_pretrained(model_opt.generator_name_or_path)
    model_kwargs = {
        'stop_token_ids': stop_token_ids,
        'num_budget': model_opt.num_budget,
    }
    from modeling.rag_wrapper2 import RerankAugmentedGenerationWrapper
    model = RerankAugmentedGenerationWrapper.from_pretrained(
        model_opt.generator_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation,
        torch_dtype=torch.bfloat16,
        **model_kwargs
    )
    model.set_biencoders(bi_encoders)
    model.set_tokenizer(tokenizer_g)
    ref_model = RerankAugmentedGenerationWrapper.from_pretrained(
        model_opt.generator_name_or_path,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation,
        torch_dtype=torch.bfloat16,
        stop_token_ids=stop_token_ids,
        num_budget=model_opt.num_budget,
        is_reference=True
    )
    ref_model.set_tokenizer(tokenizer_g)
    # [Model for RL]
    from modeling.rewards import MetricRewards
    reward_model = MetricRewards('rouge')

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
    from ppov2_trainer import RAGPPOv2Trainer
    ppo_trainer = RAGPPOv2Trainer(
	config=train_opt,
	tokenizer=tokenizer_g,
	policy=model,
	ref_policy=ref_model,
	reward_model=reward_model,
	value_model=reward_model,
	train_dataset=train_dataset,
	data_collator=data_collator
    )
    ppo_trainer.train()


if __name__ == '__main__':
    main()

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

from utils import update_tokenizer, init_generation_config

logger = logging.get_logger("transformers")

def main():

    from options import ModelOptions, DataOptions, ReinforceOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, ReinforceOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()
    set_seed(train_opt.seed)

    # [Retriever]
    ## Config & tokenizer
    from modeling import Contriever, FeedbackQueryModifier
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    ada_retriever = FeedbackQueryModifier(
        model_opt,
        qr_encoder=Contriever.from_pretrained(
            model_opt.retriever_name_or_path
        ),
        qf_encoder=Contriever.from_pretrained(
            model_opt.retriever_name_or_path, pooling='cls'
        ).train(),
    )

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
    train_opt.dataset_prefix = data_opt.train_file.lower()
    if 'qampari' in train_opt.dataset_prefix:
        from data.qampari import ContextQADataset, ContextQACollator
    elif 'asqa' in train_opt.dataset_prefix:
        from data.asqa import ContextQADataset, ContextQACollator
    else:
        print(train_opt.dataset_prefix)
        raise ValueError('no available dataset')

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

    ## data ollator
    data_collator = ContextQACollator(
        tokenizer_r=tokenizer_r,
        tokenizer_g=tokenizer_g,
    )

    # [trainer]
    train_opt.gradient_checkpointing_kwargs={"use_reentrant": False}
    from reinforce_trainer import Trainer
    trainer = Trainer(
        reward_model=reward_model,
        index_dir=model_opt.faiss_index_dir,
        model=ada_retriever,
        tokenizer=tokenizer_g,
        args=train_opt,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # trainer.save_model(train_opt.output_dir)

if __name__ == '__main__':
    main()

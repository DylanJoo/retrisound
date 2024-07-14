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

import transformers
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoModelForCausalLM,
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
    generator = AutoModelForCausalLM.from_pretrained(
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

    # Train!
    from trainer import IRTrainer
    trainer = IRTrainer(
        model=model,
        tokenizer=tokenizer_g,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        args=train_opt,
    )
    trainer.train()

if __name__ == '__main__':
    main()

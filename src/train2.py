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
    SchedulerType,
    get_scheduler,
    set_seed
)
from transformers.utils import logging 

from utils import update_tokenizer, init_generation_config

torch.autograd.set_detect_anomaly(True)

logger = logging.get_logger("transformers")

def main():

    from options import ModelOptions, DataOptions, ReinforceOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, ReinforceOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()
    set_seed(train_opt.seed)

    # [Retriever]
    from modeling.biencoders.sparse_doc_crossattn_testing import SparseAdaptiveEncoders
    from modeling.base_encoder_new import SparseEncoder
    from modeling.base_encoder_testing import SparseEncoder as SparseEncoder_test
    encoder = SparseEncoder(model_name_or_path=model_opt.retriever_name_or_path, cross_attention=False).eval()
    cattn_encoder = SparseEncoder_test.from_pretrained(
        model_opt.retriever_name_or_path,
        add_cross_attention=False, is_decoder=False, num_hidden_layers=12
    )
    ada_retriever = SparseAdaptiveEncoders(
        q_encoder=cattn_encoder, 
        encoder=encoder, 
        n_candidates=train_opt.n_max_candidates
    )

    from options import LLMOptions
    from modeling.llm import vLLM, dummyLLM
    llm_opt = LLMOptions()
    if model_opt.generator_name_or_path is None:
        generator = dummyLLM()
    else:
        generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)

    # [Data]
    train_opt.dataset_prefix = data_opt.train_file.lower()
    from data.beir_cellar import PRFDataset, PRFCollator

    train_dataset = PRFDataset(
        dataset_dir=data_opt.train_file, 
        split=data_opt.split,
        n_max_segments=train_opt.n_max_segments,
        n_negative_samples=model_opt.n_negative_samples,
        retrieval_file=data_opt.retrieval_file,
        judgement_file=data_opt.judgement_file,
        quick_test=train_opt.quick_test,
    )

    ## data ollator
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    data_collator = PRFCollator(tokenizer=tokenizer_r)

    # [trainer]
    os.environ["WANDB_PROJECT"] = train_opt.wandb_project
    train_opt.gradient_checkpointing_kwargs={"use_reentrant": False}
    from trainer import Trainer
    trainer = Trainer(
        args=train_opt,
        generator=generator,
        index_dir=model_opt.lucene_index_dir,
        model=ada_retriever,
        tokenizer=tokenizer_r,
        train_dataset=train_dataset,
        data_collator=data_collator,
        lexical=True if 'splade' in model_opt.retriever_name_or_path else False,
        dense=True if 'splade' not in model_opt.retriever_name_or_path else False
    )
    trainer.train()
    trainer.save_model(train_opt.output_dir)

if __name__ == '__main__':
    main()

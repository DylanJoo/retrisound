#!/usr/bin/env python
# coding=utf-8
import os
import json
from dataclasses import asdict
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    set_seed
)

def main():

    from options import ModelOptions, DataOptions, ReinforceOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, ReinforceOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()
    set_seed(train_opt.seed)

    # [Retriever]
    from modeling.biencoders.query_adapter import SparseAdaptiveEncoders
    from modeling.encoder import SparseEncoder, SparseEncoderForTokenClf
    encoder = SparseEncoder.from_pretrained(model_opt.retriever_name_or_path)
    q_encoder = SparseEncoderForTokenClf.from_pretrained(model_opt.retriever_name_or_path,
        add_cross_attention=False, is_decoder=False, num_hidden_layers=2
    )
    retriever = SparseAdaptiveEncoders(q_encoder=q_encoder, encoder=encoder)

    # [Environment: Generator]
    from options import LLMOptions
    from modeling.llm import vLLM, dummyLLM
    llm_opt = LLMOptions()
    if model_opt.generator_name_or_path is None:
        generator = dummyLLM()
    else:
        generator = vLLM(model=model_opt.generator_name_or_path, temperature=0.7)

    # [Environment: Searcher]
    from utils import load_searcher
    searcher = load_searcher(model_opt.index_dir, lexical=True)

    # [data]
    # train_opt.dataset_prefix = data_opt.train_file.lower()
    from data.beir_cellar import PRFDataset, PRFCollator
    dataset = PRFDataset(
        dataset_dir=data_opt.train_file, 
        split=data_opt.split,
        n_max_segments=train_opt.n_max_segments,
        n_negative_samples=model_opt.n_negative_samples,
        quick_test=train_opt.quick_test,
    )
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    data_collator = PRFCollator(tokenizer=tokenizer_r)

    # [trainer]
    os.environ["WANDB_PROJECT"] = train_opt.wandb_project
    train_opt.gradient_checkpointing_kwargs={"use_reentrant": False}
    from trainer import PolicyTrainer
    trainer = PolicyTrainer(
        args=train_opt,
        model=retriever,
        generator=generator,
        searcher=searcher,
        tokenizer=tokenizer_r,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(train_opt.output_dir)

if __name__ == '__main__':
    main()

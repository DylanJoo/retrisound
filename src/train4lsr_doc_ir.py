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
    from modeling.biencoders.query_adapter import SparseAdaptiveRetriever
    from modeling.encoder import SparseEncoder, SparseEncoderForTokenClf
    encoder = SparseEncoder.from_pretrained(model_opt.retriever_name_or_path).eval()
    q_encoder = SparseEncoderForTokenClf.from_pretrained(
        (model_opt.query_encoder_name_or_path or model_opt.retriever_name_or_path),
        add_cross_attention=False, is_decoder=False, num_hidden_layers=model_opt.num_layers,
        num_labels=2
    )
    retriever = SparseAdaptiveRetriever(
        q_encoder=q_encoder, encoder=encoder, sample_type=train_opt.sample_type,
        num_samples=train_opt.num_samples
    )

    # [Environment: Generator]
    from options import LLMOptions
    from modeling.llm.vllm_api import LLM
    from modeling.llm.hf_back import dummyLLM
    llm_opt = LLMOptions()
    if model_opt.generator_name_or_path is None:
        generator = dummyLLM()
    else:
        generator = LLM(
            model=model_opt.generator_name_or_path, temperature=0.7,
            max_num_batched_tokens=20480, max_model_len=20480,
            gpu_memory_utilization=0.5
        )

    # [Environment: Searcher]
    from utils import load_searcher
    searcher = load_searcher(model_opt.index_dir, lexical=True)

    # [data]
    from data import PRFDataset, PRFCollator
    train_dataset = PRFDataset(
        dataset_dir=data_opt.train_file, 
        split=data_opt.split,
        n_max_segments=train_opt.n_max_segments,
        n_negative_samples=model_opt.n_negative_samples,
    )
    if train_opt.do_eval:
        eval_dataset = PRFDataset(
            dataset_dir=(data_opt.eval_file or data_opt.train_file),
            split='test',
            n_max_segments=train_opt.n_max_segments,
            n_negative_samples=model_opt.n_negative_samples,
        )
    else:
        eval_dataset = None
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    data_collator = PRFCollator(tokenizer=tokenizer_r, max_src_length=model_opt.max_src_length)

    # [trainer]
    os.environ["WANDB_PROJECT"] = train_opt.wandb_project
    train_opt.gradient_checkpointing_kwargs={"use_reentrant": False}
    from trainer_ir import PolicyTrainer
    trainer = PolicyTrainer(
        args=train_opt,
        model=retriever,
        generator=generator,
        searcher=searcher,
        tokenizer=tokenizer_r,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        num_generation=model_opt.num_generation
    )
    if train_opt.do_eval:
        trainer.evaluate()

    trainer.train()
    trainer.save_model(train_opt.output_dir)

if __name__ == '__main__':
    main()

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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
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
)

from utils import init_tokenizer

logger = get_logger(__name__)


def main():


    from options import ModelOptions, DataOptions, TrainOptions
    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    os.environ["WANDB_PROJECT"] = train_opt.wandb_project

    # [Accelerator] 
    accelerator_log_kwargs = {}

    if train_opt.with_tracking:
        accelerator_log_kwargs["log_with"] = train_opt.report_to
        accelerator_log_kwargs["project_dir"] = train_opt.output_dir

    accelerator = Accelerator(
            gradient_accumulation_steps=train_opt.gradient_accumulation_steps, 
            **accelerator_log_kwargs
    )
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    set_seed(train_opt.seed)

    if accelerator.is_main_process:
        if train_opt.output_dir is not None:
            os.makedirs(train_opt.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    # from transformers import AutoModel
    # test = AutoModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16)
    # print(test.embeddings.word_embeddings)
    # print(test.embeddings.word_embeddings.weight)

    # [Retriever Bi-encoder]
    tokenizer_r = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
    from modeling.rmt import RMTEncoder
    from modeling.rife import Contriever
    ada_encoder = RMTEncoder(
        base_model=Contriever.from_pretrained(model_opt.retriever_name_or_path),
        tokenizer=tokenizer_r,
        num_mem_tokens=model_opt.num_mem_tokens,
        n_max_segments=train_opt.n_max_segments,
        input_size=(256-model_opt.num_mem_tokens-3),
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
    tokenizer_g = AutoTokenizer.from_pretrained(
            model_opt.generator_name_or_path, 
            use_fast=True
    )
    tokenizer_g, _= init_tokenizer(tokenizer_g, model_opt.use_special_tokens)
    config = AutoConfig.from_pretrained(model_opt.generator_name_or_path)
    generator = AutoModelForCausalLM.from_pretrained(
        model_opt.generator_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        attn_implementation=model_opt.attn_implementation, 
    )
    # We resize the embeddings only when necessary to avoid index errors. 
    # If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = generator.get_input_embeddings().weight.shape[0]
    if len(tokenizer_g) > embedding_size:
        generator.resize_token_embeddings(len(tokenizer_g))

    # [RAG]
    from modeling import RerankAugmentedGeneration
    model = RerankAugmentedGeneration(
        llm=generator,
        tokenizer=tokenizer_g,
        biencoders=bi_encoders
    )
    if train_opt.gradient_checkpointing:
        model.biencoders.q_encoder.model.gradient_checkpointing_enable()
        model.biencoders.d_encoder.gradient_checkpointing_enable()
        model.llm.gradient_checkpointing_enable()


    # [Data]
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
    logger.info(f"Sample 0: {train_dataset[0]}.")
    logger.info(f"Sample 1: {train_dataset[1]}.")
    # eval_dataset = ContextQADataset(data_opt.eval_file)

    # [Data] dataloader with collator
    ## [TODO] add sampler by action size (e.g., less to more)
    data_collator = ContextQACollator(
        tokenizer_r=tokenizer_r,
        tokenizer_g=tokenizer_g,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_opt.per_device_train_batch_size,
    )


    # [optimizer] Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) ],
    #         "weight_decay": train_opt.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay) ],
            "weight_decay": train_opt.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay) ],
            "weight_decay": 0.0,
        },
    ]

    freezed = ['llm', 'd_encoder']

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_opt.learning_rate)

    ## [Optimizer] calculate maximum steps 
    overrode_max_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_opt.gradient_accumulation_steps)
    if train_opt.max_steps is None:
        train_opt.max_steps = train_opt.num_train_epochs * num_update_steps_per_epoch
        overrode_max_steps = True

    num_training_steps_for_scheduler = train_opt.max_steps if overrode_max_steps else train_opt.max_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=train_opt.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * train_opt.warmup_ratio),
    )

    ## [NOTE] here is a bit tricky, use 'auto' would be ideal
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info(n)

    ## Recalculate maximum steps (bc of accelerator may change loader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_opt.gradient_accumulation_steps)
    if overrode_max_steps:
        train_opt.max_steps = train_opt.num_train_epochs * num_update_steps_per_epoch
    train_opt.num_train_epochs = math.ceil(train_opt.max_steps / num_update_steps_per_epoch)
    logger.info(f"{train_opt.num_train_epochs} | {train_opt.max_steps}" )

    if train_opt.with_tracking:
        experiment_config = asdict(train_opt)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("retrisound-debug", experiment_config)

    # Train!
    total_batch_size = train_opt.per_device_train_batch_size * accelerator.num_processes * train_opt.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_opt.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_opt.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {train_opt.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_opt.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(train_opt.max_steps)), disable=not accelerator.is_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if train_opt.resume_from_checkpoint:
        if train_opt.resume_from_checkpoint is not None or train_opt.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {train_opt.resume_from_checkpoint}")
            accelerator.load_state(train_opt.resume_from_checkpoint)
            path = os.path.basename(train_opt.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * train_opt.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, train_opt.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if train_opt.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and completed_steps < resume_step:
                    if step % train_opt.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if train_opt.logging_steps and completed_steps % train_opt.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / train_opt.gradient_accumulation_steps / train_opt.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if train_opt.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if train_opt.save_strategy == 'steps':
                    if completed_steps % train_opt.save_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if train_opt.output_dir is not None:
                            output_dir = os.path.join(train_opt.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= train_opt.max_steps:
                    break

        if train_opt.save_strategy == "epoch":
            output_dir = f"epoch_{epoch}"
            if train_opt.output_dir is not None:
                output_dir = os.path.join(train_opt.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if train_opt.with_tracking:
        accelerator.end_training()

    if train_opt.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer_r.save_pretrained(train_opt.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model.biencoders.q_encoder)
        unwrapped_model.save_pretrained(
            train_opt.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )

if __name__ == '__main__':
    main()

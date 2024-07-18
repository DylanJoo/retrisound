import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
    # retriever_tokenizer_name: Optional[str] = field(default=None)
    generator_name_or_path: Optional[str] = field(default=None)
    # generator_tokenizer_name: Optional[str] = field(default=None)
    add_pooling_layer: Optional[bool] = field(default=False)
    n_negative_samples: Optional[int] = field(default=0)
    fixed_d_encoder: Optional[bool] = field(default=False)
    attn_implementation: Literal[None, 'sdpa', 'flash_attention_2'] = field(default=None)
    # use_special_tokens: Optional[bool] = field(default=False) # check
    num_mem_tokens: Optional[int] = field(default=16)
    num_budget: Optional[int] = field(default=5)
    tau: Optional[float] = field(default=1.0)

@dataclass
class DataOptions:
    config_file: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)
    retrieval_file: Optional[str] = field(default=None)
    depth: Optional[int] = field(default=30)

from transformers import TrainingArguments
@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
    low_cpu_mem_usage: Optional[bool] = field(default=False) 
    n_max_segments: Optional[int] = field(default=8)
    n_max_candidates: Optional[int] = field(default=10)
    quick_test: Optional[int] = field(default=None)
    wandb_project: Optional[str] = field(default='testing')
    with_tracking: Optional[bool] = field(default=False)
    max_steps: int = field(default=-1) # different from HF's  
    num_processes: Optional[int] = field(default=1)
    remove_unused_columns: Optional[bool] = field(default=False)
    policy_on: str = field(default='metrics')
    learning_rate: float = field(default=5e-5)

from trl.trainer.ppov2_config import PPOv2Config
@dataclass
class RLTrainOptions(PPOv2Config):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
    low_cpu_mem_usage: Optional[bool] = field(default=False) 
    n_max_segments: Optional[int] = field(default=8)
    n_max_candidates: Optional[int] = field(default=10)
    wandb_project: Optional[str] = field(default='testing')
    max_steps: int = field(default=-1) # different from HF's  
    num_processes: Optional[int] = field(default=1)
    remove_unused_columns: Optional[bool] = field(default=False)
    learning_rate: float = field(default=5e-5)
    num_mini_batches: int = 1
    num_sample_generations: int = 0
    quick_test: Optional[int] = field(default=None)

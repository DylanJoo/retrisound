import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
    # retriever_tokenizer_name: Optional[str] = field(default=None)
    generator_name_or_path: Optional[str] = field(default=None)
    # generator_tokenizer_name: Optional[str] = field(default=None)
    add_pooling_layer: Optional[bool] = field(default=False)
    use_special_tokens: Optional[bool] = field(default=False) # check
    temperature: Optional[float] = field(default=1.0)
    n_negative_samples: Optional[int] = field(default=0)
    fixed_d_encoder: Optional[bool] = field(default=False)
    num_mem_tokens: Optional[int] = field(default=16)
    budget: Optional[int] = field(default=5)
    attn_implementation: Literal[None, 'sdpa', 'flash_attention_2'] = field(default=None)

@dataclass
class DataOptions:
    train_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)
    retrieval_file: Optional[str] = field(default=None)
    depth: Optional[int] = field(default=10)

@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
    low_cpu_mem_usage: Optional[bool] = field(default=False) 
    n_max_segments: Optional[int] = field(default=2)
    n_max_candidates: Optional[int] = field(default=10)
    quick_test: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default='testing')
    with_tracking: Optional[bool] = field(default=False)
    max_steps: int = field(default=None) # different from HF's  

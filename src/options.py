import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default=None)
    retriever_tokenizer_name: Optional[str] = field(default=None)
    generator_name_or_path: Optional[str] = field(default=None)
    generator_tokenizer_name: Optional[str] = field(default=None)
    add_pooling_layer: Optional[bool] = field(default=False)
    use_special_tokens: Optional[bool] = field(default=False) # check
    temperature: Optional[float] = field(default=1.0)
    n_negative_samples: Optional[int] = field(default=0)
    fixed_d_encoder: Optional[bool] = field(default=False)

@dataclass
class DataOptions:
    corpus_jsonl: Optional[str] = field(default=None)
    corpus_spans_jsonl: Optional[str] = field(default=None)
    prebuilt_faiss_dir: Optional[str] = field(default=None)
    prebuilt_negative_jsonl: Optional[str] = field(default=None)
    chunk_length: Optional[int] = field(default=256)
    preprocessing: Optional[str] = field(default='replicate')

@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
    low_cpu_mem_usage: Optional[bool] = field(default=False) 
    attn_implementation: str = field(default=None)


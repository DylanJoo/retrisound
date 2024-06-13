import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    model_name: Optional[str] = field(default=None)
    model_path: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    add_pooling_layer: Optional[bool] = field(default=False)
    # SSL DR
    pooling: Optional[str] = field(default="mean")
    span_pooling: Optional[str] = field(default="mean")
    norm_doc: Optional[bool] = field(default=False)
    norm_query: Optional[bool] = field(default=False)
    norm_spans: Optional[bool] = field(default=False)
    # span
    # objective, mining source
    temperature: Optional[float] = field(default=1.0)
    temperature_span: Optional[float] = field(default=1.0)
    alpha: float = field(default=1.0)
    beta: float = field(default=1.0) 
    gamma: float = field(default=1.0)
    mine_neg_using: Optional[str] = field(default=None)
    n_negative_samples: Optional[int] = field(default=0)
    # Multivec (previous)
    # late_interaction: Optional[bool] = field(default=False)
    fixed_d_encoder: Optional[bool] = field(default=False)

@dataclass
class DataOptions:
    # positive_sampling: Optional[str] = field(default='ind_cropping')
    corpus_jsonl: Optional[str] = field(default=None)
    corpus_spans_jsonl: Optional[str] = field(default=None)
    # negative sampling
    prebuilt_faiss_dir: Optional[str] = field(default=None)
    prebuilt_negative_jsonl: Optional[str] = field(default=None)
    # independent cropping
    chunk_length: Optional[int] = field(default=256)
    ratio_min: Optional[float] = field(default=0.1)
    ratio_max: Optional[float] = field(default=0.5)
    augmentation: Optional[str] = field(default=None)
    prob_augmentation: Optional[float] = field(default=0.0)
    # preprocessing
    preprocessing: Optional[str] = field(default='replicate')
    min_chunk_length: Optional[int] = field(default=32)
    # span contrastive
    select_span_mode: Optional[str] = field(default=None)
    span_mask: Optional[bool] = field(default=None)
    span_online_update: bool = field(default=False)

@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')

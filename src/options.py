import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="")
    generator_name_or_path: Optional[str] = field(default=None)
    index_dir: Optional[str] = field(default="")
    add_pooling_layer: Optional[bool] = field(default=False)
    n_negative_samples: Optional[int] = field(default=1)
    fixed_d_encoder: Optional[bool] = field(default=False)
    attn_implementation: Literal[None, 'sdpa', 'flash_attention_2'] = field(default=None)
    tau: Optional[float] = field(default=1.0)
    max_new_tokens: Optional[int] = field(default=32)
    fusion_type: Optional[str] = field(default='ff')
    zero_init: bool = field(default=False)
    # num_budget: Optional[int] = field(default=5)
    # sft: bool = field(default=False)
    # samples: int = field(default=1)
    # num_mem_tokens: Optional[int] = field(default=16)

@dataclass
class DataOptions:
    train_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)
    retrieval_file: Optional[str] = field(default=None)
    judgement_file: Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)
    depth: Optional[int] = field(default=30)

# from trl.trainer.reward_config import RewardConfig
from transformers import TrainingArguments
@dataclass
class ReinforceOptions(TrainingArguments):
    output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
    n_contexts: Optional[int] = field(default=5)
    n_max_segments: Optional[int] = field(default=2)
    n_max_candidates: Optional[int] = field(default=10)
    num_steps: Optional[int] = field(default=1)
    run_name: Optional[str] = field(default='testing')
    wandb_project: Optional[str] = field(default='adarag')
    max_steps: int = field(default=-1) # different from HF's  
    num_processes: Optional[int] = field(default=1)
    remove_unused_columns: Optional[bool] = field(default=False)
    learning_rate: float = field(default=5e-5)
    update_epochs: Optional[int] = field(default=4)
    generation_batch: Optional[int] = field(default=2)
    report_to: Optional[str] = field(default="wandb")
    ct_coef: Optional[float] = field(default=0.0)
    tc_coef: Optional[float] = field(default=0.0)
    mr_coef: Optional[float] = field(default=0.0)
    rl_coef: Optional[float] = field(default=1.0)
    reg_coef: Optional[float] = field(default=1.0)
    quick_test: Optional[int] = field(default=None)

    # num_mini_batches: int = 2
    # ampere_gpu: Optional[bool] = field(default=False)
    # reward_function: Optional[str] = field(default='metric')

@dataclass
class LLMOptions:
    model: Optional[str] = field(default="")
    temperature: Optional[float] = field(default=0.0)
    top_p: Optional[float] = field(default=1.0)

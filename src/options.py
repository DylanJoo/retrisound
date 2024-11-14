import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="")
    generator_name_or_path: Optional[str] = field(default="")
    faiss_index_dir: Optional[str] = field(default="")
    lucene_index_dir: Optional[str] = field(default="")
    add_pooling_layer: Optional[bool] = field(default=False)
    n_negative_samples: Optional[int] = field(default=0)
    fixed_d_encoder: Optional[bool] = field(default=False)
    attn_implementation: Literal[None, 'sdpa', 'flash_attention_2'] = field(default=None)
    num_mem_tokens: Optional[int] = field(default=16)
    tau: Optional[float] = field(default=1.0)
    num_budget: Optional[int] = field(default=5)
    max_new_tokens: Optional[int] = field(default=32)
    fusion_type: Optional[str] = field(default='ff')
    zero_init: bool = field(default=False)
    sft: bool = field(default=False)
    samples: int = field(default=1)

@dataclass
class DataOptions:
    train_file: Optional[str] = field(default=None)
    corpus_file: Optional[str] = field(default=None)
    retrieval_file: Optional[str] = field(default=None)
    judgement_file: Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)
    depth: Optional[int] = field(default=30)

from trl.trainer.reward_config import RewardConfig
@dataclass
class ReinforceOptions(RewardConfig):
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
    num_mini_batches: int = 2
    quick_test: Optional[int] = field(default=None)
    update_epochs: Optional[int] = field(default=4)
    reward_function: Optional[str] = field(default='metric')
    ampere_gpu: Optional[bool] = field(default=False)
    generation_batch: Optional[int] = field(default=None)
    report_to: Optional[str] = field(default="wandb")
    ct_coef: Optional[float] = field(default=0.0)
    mr_coef: Optional[float] = field(default=0.0)
    rl_coef: Optional[float] = field(default=1.0)
    half_with_bottom: bool = field(default=False)
    reward_type: str = field(default='normal')

# from trl.trainer.ppov2_config import PPOv2Config
# @dataclass
# class PolicyTrainOptions(PPOv2Config):
#     output_dir: str = field(default='/ivi/ilps/personal/dju/checkpoints/')
#     n_contexts: Optional[int] = field(default=5)
#     n_max_segments: Optional[int] = field(default=2)
#     n_max_candidates: Optional[int] = field(default=10)
#     wandb_project: Optional[str] = field(default='adarag')
#     max_steps: int = field(default=-1) # different from HF's  
#     num_processes: Optional[int] = field(default=1)
#     num_steps: Optional[int] = field(default=1)
#     remove_unused_columns: Optional[bool] = field(default=False)
#     learning_rate: float = field(default=5e-5)
#     num_mini_batches: int = 2
#     num_sample_generations: int = 0
#     quick_test: Optional[int] = field(default=None)
#     whiten_rewards: Optional[bool] = field(default=False)
#     update_epochs: Optional[int] = field(default=4)
#     norm_adv: Optional[bool] = field(default=True)
#     clip_coef: Optional[float] = field(default=0.2)
#     vf_coef: Optional[float] = field(default=0.1)
#     clip_vloss: Optional[bool] = field(default=True)
#     max_grad_norm: float = field(default=0.5)
#     target_kl: Optional[float] = field(default=None)
#     cont_coef: Optional[float] = field(default=0.0)
#     rl_coef: Optional[float] = field(default=1.0)
#     kl_coef: Optional[float] = field(default=0.05)
#     reward_function: Optional[str] = field(default='metric')
#     ampere_gpu: Optional[bool] = field(default=False)
#     generation_batch: Optional[int] = field(default=None)
#     world_size: Optional[int] = field(default=1)
#     report_to: Optional[str] = field(default="wandb")
#     run_name: Optional[str] = field(default="test")
#     num_ppo_epochs: int = 4
#     half_with_bottom: bool = field(default=False)
#     gamma: float = 0.5
#     debugging: bool = field(default=False)

#!/bin/sh
#SBATCH --job-name=adarag-base
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
export CUDA_HOME=/usr/local/cuda

# Start the experiment.

# Setups
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=facebook/contriever-msmarco
MODEL_SIZE=8B
BASE_LLM=meta-llama/Meta-Llama-3.1-8B-Instruct

deepspeed --num_gpus $NUM_GPUS train.py \
    --num_processes $NUM_GPUS \
    --bf16 \
    --deepspeed configs/zero2_config_accelerate.json \
    --retriever_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --attn_implementation flash_attention_2 \
	--train_file /home/dju/datasets/asqa/ASQA.json \
	--corpus_file /home/dju/datasets/wikipedia_split/ \
	--retrieval_file /home/dju/datasets/asqa/train_data_bm25-top100.run \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --world_size 2 \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --report_to wandb \
    --num_train_epochs 1 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
	--num_mini_batches 2 \
	--total_episodes 10000 \
	--n_contexts 5 \
	--n_max_candidates 30 \
	--depth 30 \
	--n_max_segments 5 \
	--num_steps 5 \
	--reward_function metric \
	--generation_batch 1 \
	--cont_coef 1.0 \
    --wandb_project retrisound \
    --quick_test 5000 \
    --max_steps 5000 \
    --save_steps 1000 \
    --logging_steps 1

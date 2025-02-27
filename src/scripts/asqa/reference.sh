#!/bin/sh
#SBATCH --job-name=5hr-adarag
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
export CUDA_HOME=/usr/local/cuda
# export MASTER_PORT=29500

# Start the experiment.

# Setups
NUM_GPUS=1
BATCH_SIZE_PER_GPU=32
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=OpenMatch/cocodr-base-msmarco
MODEL_SIZE=1B
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct

RUN='(half-random-zero-init)10/10/100-modifier-plus'

deepspeed --num_gpus $NUM_GPUS --master_port 29600 ppo.py \
    --num_processes $NUM_GPUS \
    --wandb_project debug \
    --run_name $RUN \
    --fusion_type plus \
    --half_with_bottom \
    --zero_init \
    --gamma 0.01 \
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
    --num_ppo_epochs 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --report_to wandb \
    --num_train_epochs 1 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
	--num_mini_batches 2 \
	--total_episodes 10000 \
	--n_contexts 10 \
	--n_max_candidates 10 \
	--depth 100 \
	--n_max_segments 5 \
	--num_steps 5 \
	--reward_function metric \
	--generation_batch 1 \
	--cont_coef 0.0 \
	--cliprange 0.2 \
    --max_steps 5000 \
    --save_steps 1000 \
    --logging_steps 1
    # --debugging \

#!/bin/sh
#SBATCH --job-name=adarag-ppo
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
# Setups
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --num_gpus $NUM_GPUS hf_ppo.py \
    --num_processes $NUM_GPUS \
    --bf16 \
    --deepspeed stage3_no_offloading_accelerate.conf \
    --config_file configs/testing.yaml \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS
    --n_max_segments 3 \
    --n_max_candidates 10 \
    --num_budget 5 \
    --depth 30 

# accelerate launch --config_file configs/deepspeed_zero3.yaml \
#     hf_ppo.py \
#     --config_file configs/testing.yaml \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --n_max_segments 3 \
#     --n_max_candidates 10 \
#     --num_budget 5 \
#     --depth 30 \
#     --quick_test 100

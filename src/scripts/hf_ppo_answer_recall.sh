#!/bin/sh
#SBATCH --job-name=adarag-ppo
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
# Setups
NUM_GPUS=2
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --num_gpus $NUM_GPUS hf_ppo.py \
    --num_processes $NUM_GPUS \
    --bf16 \
    --deepspeed stage3_no_offloading_accelerate.conf \
    --config_file configs/testing.yaml \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS


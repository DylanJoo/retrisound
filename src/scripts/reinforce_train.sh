#!/bin/sh
#SBATCH --job-name=adarag
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

# Start the experiment.
index_dir=${HOME}/indexes/asqa
data_dir=${HOME}/datasets/asqa

# Setups
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=facebook/contriever-msmarco

MODEL_SIZE=1.1B
BASE_LLM=TinyLlama/TinyLlama-1.1B-Chat-v0.6

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/zero2_config_accelerate.json \
    reinforce_train.py \
    --retriever_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --attn_implementation flash_attention_2 \
	--train_file /home/dju/datasets/asqa/ASQA.json \
	--corpus_file /home/dju/datasets/wikipedia_split/ \
	--retrieval_file /home/dju/datasets/asqa/train_data_bm25-top100.run \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --report_to wandb \
    --generation_batch 2 \
    --bf16 true \
    --logging_steps 1

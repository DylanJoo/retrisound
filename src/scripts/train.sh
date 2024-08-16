#!/bin/sh
#SBATCH --job-name=adarag
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
index_dir=${HOME}/indexes/qampari
data_dir=${HOME}/datasets/qampari

# Setups
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=facebook/contriever

MODEL_SIZE=1.1B
BASE_LLM=TinyLlama/TinyLlama-1.1B-Chat-v0.6
# MODEL_SIZE=7B
# BASE_LLM=meta-llama/Meta-Llama-3-8B-Instruct

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --retriever_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --attn_implementation flash_attention_2 \
    --train_file /home/dju/datasets/qampari/train_data.jsonl \
    --corpus_file /home/dju/datasets/qampari/wikipedia_chunks/chunks_v5 \
    --retrieval_file /home/dju/datasets/qampari/train_data_bm25-top100.run \
    --dataloader_num_workers 32 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to wandb \
    --quick_test 1000 \
    --logging_steps 1


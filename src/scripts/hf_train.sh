#!/bin/sh
#SBATCH --job-name=adarag-base
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
index_dir=${HOME}/indexes/qampari
data_dir=${HOME}/datasets/qampari

# Setups
NUM_GPUS=2
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
# BASE_RET=facebook/contriever
BASE_RET=facebook/contriever-msmarco

MODEL_SIZE=7B
BASE_LLM=meta-llama/Meta-Llama-3-8B-Instruct

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --num_gpus $NUM_GPUS hf_base.py \
    --num_processes $NUM_GPUS \
    --bf16 \
    --retriever_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --attn_implementation flash_attention_2 \
    --deepspeed stage3_no_offloading_accelerate.conf \
    --train_file /home/dju/datasets/qampari/train_data.jsonl \
    --corpus_file /home/dju/datasets/qampari/wikipedia_chunks/chunks_v5 \
    --retrieval_file /home/dju/datasets/qampari/train_data_bm25-top100.run \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to wandb \
    --wandb_project retrisound \
    --quick_test 5000 \
    --max_steps 5000 \
    --save_steps 1000 \
    --logging_steps 1


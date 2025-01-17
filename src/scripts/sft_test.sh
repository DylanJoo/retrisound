#!/bin/sh
#SBATCH --job-name=5hr.res
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
. /home/dju/miniconda3/etc/profile.d/conda.sh
source ${HOME}/.bashrc
conda activate retrisound
export CUDA_HOME=/usr/local/cuda
cd /home/dju/retrisound/src/

# Setups
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=naver/splade-v3-doc
MODEL_SIZE=1B
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
# dataset=litsearch
dataset=beir-cellar/scidocs
# dataset=beir-cellar/fiqa
dataset=msmarco-passage

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs" 
echo "$BATCH_SIZE_PER_GPU batch size per GPU" 
echo "$GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --config_file configs/default_config.yaml \
    --main_process_port 29500 \
    train2.py \
    --retriever_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --train_file $DATASET_DIR/${dataset} \
    --split train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --max_steps 500 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --report_to wandb \
    --generation_batch 4 \
    --n_contexts 10 --n_max_candidates 10 --n_negative_samples 2 \
    --num_steps 3 --n_max_segments 15 \
    --ct_coef 0.0 \
    --tc_coef 1.0 \
    --reg_coef 0.0 \
    --mr_coef 0.0 \
    --rl_coef 0.0 \
    --do_train \
    --fp16 \
    --index_dir /home/dju/indexes/${dataset}.lucene_doc \
    --logging_steps 1 --run_name 'MLP(q, f)-(TC)-q_enc'

#!/bin/sh
#SBATCH --job-name=5hr.neuclir
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
# BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
BASE_LLM=meta-llama/Llama-3.2-3B-Instruct
dataset=neuclir1-mt/fas

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs" 
echo "$BATCH_SIZE_PER_GPU batch size per GPU" 
echo "$GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --config_file configs/default_config_${NUM_GPUS}.yaml \
    train.py \
    --retriever_name_or_path $BASE_RET \
    --query_encoder_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --train_file $DATA_DIR/${dataset} \
    --num_layers 6 \
    --num_samples 100 \
    --split test \
    --sample_type random \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --max_steps 500 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --report_to wandb \
    --generation_batch 4 \
    --n_contexts 5 --n_max_candidates 5 --n_negative_samples 2 \
    --num_steps 3 --n_max_segments 15 \
    --ct_coef 0.0 \
    --tc_coef 0.0 \
    --rl_coef 1.0 \
    --do_train \
    --fp16 \
    --index_dir ${INDEX_DIR}/${dataset}/splade-v3-doc.lucene \
    --logging_steps 1 --run_name ${dataset##*/}:random:RL1
    # --do_eval \
    # --eval_strategy steps \
    # --eval_steps 50 \

#!/bin/sh
#SBATCH --job-name=10hr.msmarco
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
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
MODEL_SIZE=gpt

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs" 
echo "$BATCH_SIZE_PER_GPU batch size per GPU" 
echo "$GRADIENT_ACC_STEPS gradient accumulation steps"

BASE_LLM=gpt3.5
num_labels=1
num_layers=2
tc_coef=1
rl_coef=1
num_gen=1

for topk in 30; do
for num_samples in 10;do
for rl_coef in -1; do

dataset=msmarco-passage/train
exp=$dataset-random-L${num_layers}-TC${tc_coef}-RL${rl_coef}-num_labels${num_labels}-gen${num_gen}
accelerate launch \
    --config_file configs/default_config_${NUM_GPUS}.yaml \
    --main_process_port 29600 \
    train_pl.py \
    --retriever_name_or_path $BASE_RET \
    --query_encoder_name_or_path $BASE_RET \
    --train_file ${dataset} \
    --eval_file ${dataset/train/trec-dl-2019} \
    --num_layers $num_layers \
    --num_samples $num_samples \
    --topk $topk \
    --num_generation $num_gen \
    --split train \
    --sample_type deterministic \
    --max_src_length 512 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --max_grad_norm 0.5 \
    --max_steps 10000 \
    --save_steps 2500 \
    --output_dir ${MODEL_DIR}/ada_lsr_${MODEL_SIZE}/ \
    --report_to wandb \
    --generation_batch 16 \
    --n_contexts 10 --n_max_candidates 10 --n_negative_samples 2 \
    --num_steps 1 --n_max_segments 15 \
    --ct_coef 0.0 \
    --tc_coef $tc_coef \
    --rl_coef $rl_coef \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --fp16 \
    --index_dir ${INDEX_DIR}/${dataset}/splade-v3-doc.lucene \
    --logging_steps 1 --run_name $exp
done
done
done

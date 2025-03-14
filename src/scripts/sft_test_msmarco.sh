#!/bin/sh
#SBATCH --job-name=10hr.msmarco
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
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
BATCH_SIZE_PER_GPU=64
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MODEL_DIR=/ivi/ilps/personal/dju/checkpoints
BASE_RET=naver/splade-v3-doc
MODEL_SIZE=1B
BASE_LLM=meta-llama/Llama-3.2-1B-Instruct
dataset=msmarco-passage/train

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs" 
echo "$BATCH_SIZE_PER_GPU batch size per GPU" 
echo "$GRADIENT_ACC_STEPS gradient accumulation steps"

num_layers=2
num_labels=1 # train.py
# num_labels=2 # train4lsr_doc_ir.py
tc_coef=1
rl_coef=-1
num_gen=1
exp=$dataset-random-L${num_layers}-TC${tc_coef}-RL${rl_coef}-num_labels${num_labels}-gen${num_gen}
    # train4lsr_doc_ir.py \
    # train.py \
    # train4lsr_doc_ir.py \

accelerate launch \
    --config_file configs/default_config_${NUM_GPUS}.yaml \
    --main_process_port 29601 \
    train.py \
    --retriever_name_or_path $BASE_RET \
    --query_encoder_name_or_path $BASE_RET \
    --train_file ${dataset} \
    --eval_file msmarco-passage/trec-dl-2019 \
    --num_layers 2 \
    --num_samples 100 \
    --topk 30 \
    --split train \
    --sample_type random \
    --max_src_length 384 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0. \
    --max_steps 10000 \
    --output_dir ${MODEL_DIR}/adarag_${MODEL_SIZE}/ \
    --report_to wandb \
    --generation_batch 16 \
    --n_contexts 10 --n_max_candidates 10 --n_negative_samples 2 \
    --num_steps 3 --n_max_segments 15 \
    --ct_coef 0.0 \
    --tc_coef 1.0 \
    --rl_coef -1.0 \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --fp16 \
    --index_dir ${INDEX_DIR}/${dataset}/splade-v3-doc.lucene \
    --logging_steps 1 --run_name $exp

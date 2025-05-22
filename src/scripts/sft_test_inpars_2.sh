#!/bin/sh
#SBATCH --job-name=72hr-inpars-mb
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
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

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs" 
echo "$BATCH_SIZE_PER_GPU batch size per GPU" 
echo "$GRADIENT_ACC_STEPS gradient accumulation steps"

# MODEL_SIZE=3B
# BASE_LLM=meta-llama/Llama-3.2-3B-Instruct
MODEL_SIZE=8B
# BASE_LLM=allenai/Llama-3.1-Tulu-3.1-8B
BASE_LLM=meta-llama/Llama-3.1-8B-Instruct

num_labels=2
num_gen=1
init=splade-v3-doc

for dataset_name in trec-covid dbpedia-entity climate-fever webis-touche2020 nfcorpus scidocs;do
for topk in -1; do
for num_samples in 100;do
for tc_coef in 1; do
for rl_coef in 0 1 -1; do
for num_layers in 1; do

dataset=inpars-v2/$dataset_name
exp=${dataset_name}-${init}-L${num_layers}-${MODEL_SIZE}-TC${tc_coef}-RL${rl_coef}-num_labels${num_labels}-gen${num_gen}
accelerate launch \
    --config_file configs/default_config_${NUM_GPUS}.yaml \
    --main_process_port 29607 \
    train_mb.py \
    --retriever_name_or_path $BASE_RET \
    --query_encoder_name_or_path $BASE_RET \
    --generator_name_or_path $BASE_LLM \
    --train_file $DATA_DIR/${dataset} \
    --eval_file $DATA_DIR/${dataset/inpars-v2/beir-cellar} \
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
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 25 \
    --weight_decay 0. \
    --max_grad_norm 5 \
    --max_steps 500 \
    --save_steps 500 \
    --output_dir ${MODEL_DIR}/ada_lsr_${MODEL_SIZE}/${dataset##*/}/${num_labels} \
    --report_to wandb \
    --generation_batch 16 \
    --n_contexts 10 --n_max_candidates 10 --n_negative_samples 1 \
    --num_steps 1 --n_max_segments 15 \
    --ct_coef 0.0 \
    --tc_coef $tc_coef \
    --rl_coef $rl_coef \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --fp16 \
    --index_dir ${INDEX_DIR}/${dataset/inpars-v2/beir-cellar}.lucene_doc \
    --logging_steps 1 --run_name $exp
done
done
done
done
done
done

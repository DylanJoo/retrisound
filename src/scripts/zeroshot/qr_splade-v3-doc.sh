#!/bin/sh
#SBATCH --job-name=zsqr
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-9%1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

cd /home/dju/retrisound/src/

# Setups
RETRIEVER=naver/splade-v3-doc
MULTIJOBS=/home/dju/temp/beir_multijobs.txt

# Generate embeddings
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each

python3 BEIR_eval.py \
    --dataset_dir /home/dju/datasets/${each} \
    --index_dir /home/dju/indexes/${each}.lucene_doc \
    --d_encoder_name $RETRIEVER \
    --generator_name meta-llama/Llama-3.2-1B-Instruct \
    --split test \
    --batch_size 4 \
    --prompt_type qr \
    --iteration 1  --repeat_query 1 \
    --device cuda \
    --exp query-rewrite

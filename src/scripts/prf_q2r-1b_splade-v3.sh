#!/bin/sh
#SBATCH --job-name=prfq2r
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-10%1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

cd /home/dju/retrisound/src/

# Setups
VLLM_ATTENTION_BACKEND=XFORMERS
RETRIEVER=naver/splade-v3
MULTIJOBS=/home/dju/temp/beir_multijobs.txt

# Generate embeddings
# echo Runing $file 
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each

python3 BEIR_eval.py \
    --dataset_dir /home/dju/datasets/beir-cellar/${each} \
    --index_dir /home/dju/indexes/beir-cellar/${each}.lucene \
    --d_encoder_name $RETRIEVER \
    --q_encoder_name_or_path $RETRIEVER \
    --generator_name meta-llama/Llama-3.2-1B-Instruct \
    --split test \
    --batch_size 4 \
    --iteration 1 --expansion 1 --top_k 5 --prompt_type prf_q2r \
    --context_masking \
    --device cuda \
    --exp debug

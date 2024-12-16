#!/bin/sh
#SBATCH --job-name=debug
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# export CUDA_HOME=/usr/local/cuda
cd /home/dju/retrisound/src/

# Setups
NUM_GPUS=1

python3 BEIR_eval.py \
    --dataset_dir /home/dju/datasets/beir-cellar/dbpedia-entity \
    --index_dir /home/dju/indexes/beir-cellar/dbpedia-entity.lucene \
    --d_encoder_name naver/splade-v3 \
    --q_encoder_name_or_path naver/splade-v3 \
    --split test \
    --top_k 5 \
    --iteration 0 \
    --batch_size 2 \
    --device cuda

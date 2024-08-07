#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate pyserini

# Start the experiment.
# Setups
python -m pyserini.encode input \
    --corpus ${DATASET_DIR}/wikipedia_split/raw/psgs_w100.jsonl \
    --fields text \
    --delimiter "\n" \
    --shard-id 0 \
    --shard-num 1 output \
    --embeddings ${INDEX_DIR}/wikipedia_split/contriever.psgs_w100.faiss \
    --to-faiss encoder \
    --encoder facebook/contriever \
    --encoder-class contriever \
    --fields text \
    --batch 64 \
    --fp16  

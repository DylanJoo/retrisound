#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
export CUDA_HOME=/usr/local/cuda


# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3

# Generate embeddings
# echo Runing $file 
python retrieval/mlm_encode.py \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection ${DATASET_DIR}/lit-search/corpus_data.jsonl \
    --collection_output ${INDEX_DIR}/lit-search/splade-v3.lit-search.encoded/title_abstract.jsonl${num} \
    --batch_size 64 \
    --max_length 512 \
    --quantization_factor 100


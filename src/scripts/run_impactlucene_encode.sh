#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate pyserini
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3

# Generate embeddings
# echo Runing $file 
for num in $(seq 13280 21015);do
    file=${DATASET_DIR}/wikipedia_split/psgs_w100.jsonl${num}
    python retrieval/mlm_encode.py \
        --model_name_or_path ${RETRIEVER} \
        --tokenizer_name ${RETRIEVER} \
        --collection ${file} \
        --collection_output ${DATASET_DIR}/wikipedia_split/splade/${file##*/} \
        --batch_size 256 \
        --max_length 256 \
        --quantization_factor 100
done

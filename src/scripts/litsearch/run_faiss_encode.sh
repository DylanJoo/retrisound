#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
# Setups
RETRIEVER=facebook/dragon-plus-context-encoder

# Generate embeddings
python -m pyserini.encode \
  input   --corpus lit-search.corpus_clean_data.json \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ${INDEX_DIR}/litsearch/dragon-plus.lit-search.flat.faiss  \
          --to-faiss \
  encoder --encoder ${RETRIEVER} \
          --fields text \
          --batch 256 \
          --fp16


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
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3

# Generate embeddings
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${INDEX_DIR}/wikipedia_split/splade-v3.psgs_w100.encoded \
  --index ${INDEX_DIR}/wikipedia_split/splade-v3.psgs_w100.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized

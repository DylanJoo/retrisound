#!/bin/sh
#SBATCH --job-name=bm25
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
index_dir=${HOME}/indexes/qampari
data_dir=${HOME}/datasets/qampari

python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${data_dir}/wikipedia_chunks/chunks_v5 \
    --index ${index_dir}/wikipedia_080121.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 64

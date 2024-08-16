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

### QAMPARI ###
# index_dir=${HOME}/indexes/qampari
# data_dir=${HOME}/datasets/qampari

# Indexing
# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input ${data_dir}/wikipedia_chunks/chunks_v5 \
#     --index ${index_dir}/wikipedia_080121.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 64

# Search
# for split in train dev test;do
#     python retrieval/bm25_search.py \
#         --k 100 --k1 0.9 --b 0.4 \
#         --index ${index_dir}/wikipedia_080121.lucene \
#         --topic ${data_dir}/${split}_data.jsonl \
#         --batch_size 32 \
#         --output ${data_dir}/${split}_data_bm25-top100.run
# done

### ASQA ###
# index_dir=${HOME}/indexes/asqa
# data_dir=${HOME}/datasets/asqa

# Indexing
# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input ${DATASET_DIR}/wikipedia_split/ \
#     --index ${index_dir}/wikipedia_122018.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 64

# Search
# for split in train dev;do
#     python retrieval/bm25_search.py \
#         --k 100 --k1 0.9 --b 0.4 \
#         --index ${index_dir}/wikipedia_122018.lucene \
#         --topic ${data_dir}/${split}.jsonl \
#         --batch_size 32 \
#         --output ${data_dir}/${split}_data_bm25-top100.run
# done

#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1-1%1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
cd ~/retrisound/src/

# Start the experiment.
# Setups
MULTIJOBS=/home/dju/temp/beir_multijobs.txt

each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)

echo $each
# Indexing
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input ${INDEX_DIR}/${each}.encoded \
    --index ${INDEX_DIR}/${each}.bm25_lucene_doc \
    --generator DefaultLuceneDocumentGenerator \
    --threads 64 --storeDocvectors 

### QAMPARI ###
# index_dir=${HOME}/indexes/qampari
# data_dir=${HOME}/datasets/qampari

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

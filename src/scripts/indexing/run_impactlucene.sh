#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1-3%1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate retrisound
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3-doc
MULTIJOBS=/home/dju/temp/beir_multijobs.txt
MULTIJOBS=/home/dju/temp/wikipedia_split_dpr_multijobs.txt
MULTIJOBS=/home/dju/temp/neuclir_multijobs.txt

# IR benchmarks
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
# echo $each
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input ${INDEX_DIR}/${each}.encoded_doc \
#   --index ${INDEX_DIR}/${each}.lucene_doc \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --storeDocvectors --impact --pretokenized

# QA benchmarks: wiki
# python -m pyserini.index.lucene \
#   --collection JsonVectorCollection \
#   --input ${INDEX_DIR}/wikipedia_split_dpr/encoded \
#   --index ${INDEX_DIR}/wikipedia_split_dpr/splade-v3-doc.lucene \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 36 \
#   --storeDocvectors --impact --pretokenized

# NeuCLIR
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each
python -m pyserini.index.lucene \
    --collection JsonVectorCollection \
    --input ${INDEX_DIR}/${each}/encoded \
    --index ${INDEX_DIR}/${each}/splade-v3-doc.lucene \
    --generator DefaultLuceneDocumentGenerator \
    --threads 36 \
    --storeDocvectors --impact --pretokenized

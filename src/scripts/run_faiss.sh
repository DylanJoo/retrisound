#!/bin/sh
#SBATCH --job-name=faiss
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
# Setups
RETRIEVER=OpenMatch/cocodr-base-msmarco
HPARAMS_FILE=${HOME}/temp/hparams_encode_psgs_w100.txt

# Construct FAISS index
# for num in {0..19};do
#     python -m pyserini.index.faiss \
#         --input ${INDEX_DIR}/wikipedia_split/cocodr.psgs_w100.encoded/cocodr.psgs_w100.faiss$num \
#         --output ${INDEX_DIR}/wikipedia_split/cocodr.psgs_w100.faiss-pq-$num \
#         --pq
# done

python -m pyserini.index.merge_faiss_indexes \
    --prefix ${INDEX_DIR}/wikipedia_split/cocodr.psgs_w100.faiss-pq- \
    --shard-num 20


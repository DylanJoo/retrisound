#!/bin/sh
#SBATCH --job-name=merge
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=320G
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

source ${HOME}/.bashrc
conda activate retrisound

python3 -m pyserini.index.merge_faiss_indexes \
  --prefix ~/indexes/wikipedia_split/contriever.psgs_w100.faiss \
  --shard-num 20

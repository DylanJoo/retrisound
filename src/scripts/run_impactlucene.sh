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
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${DATASET_DIR}/wikipedia_split/temp \
  --index ${INDEX_DIR}/wikipedia_split/splade-v3.psgs_w100.full.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized

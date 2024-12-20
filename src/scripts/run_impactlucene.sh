#!/bin/sh
#SBATCH --job-name=lucene
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1-13%4
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

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${INDEX_DIR}/beir-cellar/${each}.encoded \
  --index ${INDEX_DIR}/beir-cellar/${each}.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --storeDocvectors --impact --pretokenized

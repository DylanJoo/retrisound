#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-4%4
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate pyserini

# Start the experiment.
# Setups
HPARAMS_FILE=${HOME}/temp/hparams_encode_psgs_w100.txt

python -m pyserini.encode input \
    --corpus ${DATASET_DIR}/wikipedia_split/raw/psgs_w100.jsonl \
    --fields text \
    $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1) \
    --delimiter "\n" output \
    --embeddings ${INDEX_DIR}/wikipedia_split/contriever.psgs_w100.faiss \
    --to-faiss encoder \
    --encoder facebook/contriever-msmarco \
    --encoder-class contriever \
    --fields text \
    --batch 64 \
    --fp16  

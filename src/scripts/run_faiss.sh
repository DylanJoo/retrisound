#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-20%2
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate pyserini

# Start the experiment.
# Setups
RETRIEVER=OpenMatch/cocodr-base-msmarco
HPARAMS_FILE=${HOME}/temp/hparams_encode_psgs_w100.txt

# Generate embeddings
python -m pyserini.encode input \
    --corpus ${DATASET_DIR}/wikipedia_split/raw/psgs_w100.jsonl \
    --fields text \
    $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1) encoder \
    --encoder ${RETRIEVER} \
    --encoder-class auto \
    --pooling cls \
    --fields text \
    --batch 256 \
    --fp16  

# Construct FAISS index

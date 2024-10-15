#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=1-22%4
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3
HPARAMS_FILE=/home/dju/temp/hparams_encode_psgs_w100.txt

# Generate embeddings
# echo Runing $file 
num=$(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
echo $num
python retrieval/mlm_encode.py \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection ${DATASET_DIR}/wikipedia_split/raw/splits/psgs_w100.jsonl${num} \
    --collection_output ${INDEX_DIR}/wikipedia_split/splade-v3.psgs_w100.encoded/psgs_w100.jsonl${num} \
    --batch_size 384 \
    --max_length 256 \
    --quantization_factor 100


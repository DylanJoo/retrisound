#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=9-13%2
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound
cd ~/retrisound/src/

# Start the experiment.
# Setups
RETRIEVER=naver/splade-v3
MULTIJOBS=/home/dju/temp/hparams_encode_psgs_w100.txt
MULTIJOBS=/home/dju/temp/beir_multijobs.txt

# Generate embeddings
# echo Runing $file 
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each
python -m retrieval.mlm_encode \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection ${DATASET_DIR}/beir-cellar/${each}/corpus.jsonl \
    --collection_output ${INDEX_DIR}/beir-cellar/${each}.encoded/vectors.jsonl \
    --batch_size 384 \
    --max_length 256 \
    --quantization_factor 100


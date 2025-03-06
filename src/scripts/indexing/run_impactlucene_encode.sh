#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --array=1-1%1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

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
# python -m retrieval.mlm_encode \
#     --model_name_or_path ${RETRIEVER} \
#     --tokenizer_name ${RETRIEVER} \
#     --collection ${DATASET_DIR}/${each}/corpus.jsonl \
#     --collection_output ${INDEX_DIR}/${each}.encoded/vectors_doc.jsonl \
#     --batch_size 384 \
#     --max_length 256 \
#     --quantization_factor 100

# QA benchmarks: wiki
# each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
# echo $each
# python3 -m retrieval.mlm_encode \
#     --model_name_or_path ${RETRIEVER} \
#     --tokenizer_name ${RETRIEVER} \
#     --collection /home/dju/datasets/${each} \
#     --collection_output /home/dju/indexes/wikipedia_split_dpr/encoded/vectors_doc.${each##*\jsonl}.jsonl \
#     --batch_size 384 \
#     --max_length 256 \
#     --quantization_factor 100

# NeuCLIR
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each
python3 -m retrieval.mlm_encode \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection /home/dju/datasets/${each}/corpus.jsonl \
    --collection_output /home/dju/indexes/${each}/encoded/vectors_doc.${each##*/}.jsonl \
    --batch_size 384 \
    --max_length 256 \
    --quantization_factor 100

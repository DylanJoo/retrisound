#!/bin/sh
#SBATCH --job-name=encode
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --array=3-22%2
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

# Generate embeddings
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each
python3 -m retrieval.mlm_encode \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection /home/dju/datasets/${each} \
    --collection_output /home/dju/indexes/wikipedia_split_dpr/encoded/splade-v3-doc.wikipedia_split_dpr.${each##*\jsonl}.jsonl \
    --batch_size 384 \
    --max_length 256 \
    --quantization_factor 100

#!/bin/sh
#SBATCH --job-name=search
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --array=1-2%1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate retrisound
export CUDA_HOME=/usr/local/cuda
cd /home/dju/retrisound/src/

# Setups
RETRIEVER=naver/splade-v3-doc
MULTIJOBS=/home/dju/temp/beir_multijobs.txt
MULTIJOBS=/home/dju/temp/msmarco_multijobs.txt

# Generate embeddings
each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo $each

python3 IR_eval.py \
    --dataset_dir /home/dju/datasets/${each} \
    --index_dir /home/dju/indexes/msmarco-passage/train/splade-v3-doc.lucene \
    --d_encoder_name $RETRIEVER \
    --q_encoder_name_or_path $RETRIEVER \
    --split test \
    --batch_size 32 \
    --iteration 0 \
    --device cpu \
    --exp ${each}-baseline-doc

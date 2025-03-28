#!/bin/sh
#SBATCH --job-name=zsqr
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate retrisound
export CUDA_HOME=/usr/local/cuda
cd /home/dju/retrisound/src/

# Setups
RETRIEVER=naver/splade-v3-doc
MULTIJOBS=/home/dju/temp/beir_multijobs.txt

for dataset_name in trec-covid dbpedia-entity climate-fever webis-touche2020 scifact nfcorpus fiqa scidocs;do
each=beir-cellar/${dataset_name}
echo $each

python3 IR_eval.py \
    --dataset_dir /home/dju/datasets/${each} \
    --index_dir /home/dju/indexes/${each}.lucene_doc \
    --d_encoder_name $RETRIEVER \
    --generator_name meta-llama/Llama-3.2-3B-Instruct \
    --split test \
    --batch_size 128 \
    --prompt_type qr \
    --iteration 1 --repeat_query 1 \
    --device cuda \
    --exp query-rewrite-iter1
done

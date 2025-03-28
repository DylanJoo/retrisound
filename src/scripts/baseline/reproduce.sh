#!/bin/sh
#SBATCH --job-name=search
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
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

for dataset_name in trec-covid dbpedia-entity climate-fever webis-touche2020 scifact nfcorpus scidocs;do
each=beir-cellar/${dataset_name}
echo $each

python3 IR_eval.py \
    --dataset_dir /home/dju/datasets/${each} \
    --index_dir /home/dju/indexes/${each}.lucene_doc \
    --d_encoder_name $RETRIEVER \
    --q_encoder_name_or_path $RETRIEVER \
    --split test \
    --batch_size 128 \
    --iteration 0 \
    --count \
    --device cpu \
    --exp ${each}-baseline-doc
done

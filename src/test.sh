#!/bin/sh
#SBATCH --job-name=test
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
index_dir=${HOME}/indexes/qampari
data_dir=${HOME}/datasets/qampari

# Setups

python3 test.py


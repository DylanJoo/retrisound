#!/bin/sh
#SBATCH --job-name=adarag
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
export WANDB_PROJECT=retrisound
source ${HOME}/.bashrc
conda activate retrisound

# Start the experiment.
# Setups

accelerate launch --config_file configs/deepspeed_zero2.yaml \
    train.py --config_file configs/tinyllama-metric+cont.yaml

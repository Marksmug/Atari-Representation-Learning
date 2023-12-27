#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-960 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-40:00:00

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch

python test.py

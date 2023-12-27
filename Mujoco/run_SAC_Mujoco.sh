#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-960 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-00:05:00

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch


#python SAC_mujoco.py  --wandb-track=True --env-name=HalfCheetah-v4  --seed-value=1
python SAC_cleanRL.py --track=True --seed=1





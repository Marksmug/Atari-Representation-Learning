#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-960 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1  # We're launching 2 nodes with 4 Nvidia T4 GPUs each
#SBATCH -t 0-80:00:00 
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch


#python SAC_clean_pixel.py --track=True --task-name=pushing  
#python SAC_pixel_DMC.py --wandb-track=True --task-name=pushing  --seed-value=1
python SAC_VAE_DMC.py --wandb-track=True --task-name=pushing  --seed-value=1
#python SAC_structured.py --wandb-track=True --task-name=pushing --auto-alpha=True --seed-value=1




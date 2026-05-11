#!/bin/bash
#SBATCH --job-name=tft_v3
#SBATCH --account=project_2001220
#SBATCH --partition=gpusmall
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/project_2001220/kaukolampo/tft_v3_%j.log

module purge
module load pytorch/2.4
source /scratch/project_2001220/kaukolampo/venv/bin/activate
cd /scratch/project_2001220/kaukolampo

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
python -u train_tft_csc_v3.py
echo "Job finished: $(date)"

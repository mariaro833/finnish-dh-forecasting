#!/bin/bash

#SBATCH --account=project_2001220
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00

#SBATCH --mem=16G

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --job-name=evaluate_tft

#SBATCH --output=/scratch/project_2001220/kaukolampo/eval_%j.log



cd /scratch/project_2001220/kaukolampo

module load pytorch/2.4

source venv/bin/activate

python evaluate_real.py


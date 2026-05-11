#!/bin/bash

#SBATCH --job-name=xgb_v3

#SBATCH --account=project_2001220

#SBATCH --partition=small

#SBATCH --time=02:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=20G

#SBATCH --output=/scratch/project_2001220/kaukolampo/xgb_v3_%j.log



module purge

module load pytorch/2.4

source /scratch/project_2001220/kaukolampo/venv/bin/activate

cd /scratch/project_2001220/kaukolampo



echo "Job started: $(date)"

echo "Node: $(hostname)"

python -u xgb_v3.py

echo "Job finished: $(date)"


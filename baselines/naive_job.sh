#!/bin/bash

#SBATCH --job-name=naive_bl

#SBATCH --account=project_2001220

#SBATCH --partition=small

#SBATCH --time=00:10:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=4G

#SBATCH --output=/scratch/project_2001220/kaukolampo/naive_%j.log



module purge

module load pytorch/2.4

source /scratch/project_2001220/kaukolampo/venv/bin/activate

cd /scratch/project_2001220/kaukolampo



echo "Job started: $(date)"

echo "Node: $(hostname)"

python -u naive_baselines.py

echo "Job finished: $(date)"


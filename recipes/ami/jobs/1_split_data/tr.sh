#!/bin/bash
#SBATCH --output=jobs.out/1_split_data/tr.out
#SBATCH --error=jobs.out/1_split_data/tr.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# SBATCH --partition=V100
#SBATCH --time=3:00:00
srun python ./scripts/1_split_data.py job --mode tr 

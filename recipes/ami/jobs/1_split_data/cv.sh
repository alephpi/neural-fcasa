#!/bin/bash
#SBATCH --output=jobs.out/1_split_data/cv.out
#SBATCH --error=jobs.out/1_split_data/cv.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=CPU
#SBATCH --time=3:00:00
srun python ./scripts/1_split_data.py job --mode cv 

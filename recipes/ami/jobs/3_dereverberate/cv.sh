#!/bin/bash
#SBATCH --output=jobs.out/3_dereverberate/cv.out
#SBATCH --error=jobs.out/3_dereverberate/cv.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=V100
#SBATCH --time=12:00:00
srun python ./scripts/3_dereverberate.py job --mode cv 

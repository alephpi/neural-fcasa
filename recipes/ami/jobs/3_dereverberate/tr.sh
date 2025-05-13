#!/bin/bash
#SBATCH --output=jobs.out/3_dereverberate/tr.out
#SBATCH --error=jobs.out/3_dereverberate/tr.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=1
#SBATCH --partition=A100
#SBATCH --time=12:00:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate cd

python  ./scripts/3_dereverberate.py job --mode tr 

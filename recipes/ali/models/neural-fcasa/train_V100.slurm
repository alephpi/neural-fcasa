#!/bin/bash
#SBATCH --job-name=fcasa
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH --partition=V100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --nodes=4
#SBATCH --nodelist=node[08,10,12,15]
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=2

export MAIN_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MAIN_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=${MAIN_ADDR}
export MASTER_PORT=${MAIN_PORT}

# working_dir=$(dirname "${BASH_SOURCE[0]}")
# Activate environment

echo "Job started at: $(date)"

srun python train.py config_V100.yaml

echo "Job finished at: $(date)"
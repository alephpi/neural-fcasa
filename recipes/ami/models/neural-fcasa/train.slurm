#!/bin/bash
#SBATCH --job-name=fcasa
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=2

# export CUDA_VISIBLE_DEVICES=0,1
export MAIN_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MAIN_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=${MAIN_ADDR}
export MASTER_PORT=${MAIN_PORT}

export OMPI_COMM_WORLD_RANK="${SLURM_PROCID:-0}"
export OMPI_COMM_WORLD_LOCAL_RANK="${SLURM_LOCALID:-0}"
export OMPI_COMM_WORLD_SIZE="${SLURM_NTASKS:-1}"
export OMPI_COMM_WORLD_LOCAL_SIZE="${SLURM_NTASKS_PER_NODE:-1}"
export OMPI_MCA_orte_num_nodes="${SLURM_JOB_NUM_NODES:-1}"

# working_dir=$(dirname "${BASH_SOURCE[0]}")
# Activate environment
source ~/.anaconda3/etc/profile.d/conda.sh
conda activate fcasa

# srun python -m aiaccel.apps.train config.yaml --working_directory .
srun python -m aiaccel.apps.train config.yaml

echo "Job finished at: $(date)"
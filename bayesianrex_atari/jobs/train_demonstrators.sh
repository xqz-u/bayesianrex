#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_demonstrators
#SBATCH --cpus-per-task=3
#SBATCH --time=08:00:00
#SBATCH --mem=12G
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

code_dir=$HOME/bayesianrex

source $code_dir/scripts/activate_env.sh

cd $code_dir/bayesianrex
echo "Start @ $(date)"
srun python demonstrators.py --all
echo "End @ $(date)"

#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_demonstrators
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --output=/home/%u/job_logs/%x_%u_%A.out

# NOTE output naming convention:
# %x job name
# %A job id
# %u user

module purge

env=$1
run_name=$2

# use default breakout if env name not provided
env_flag=$([ -z $env ] && echo "" || echo "--env $env")
run_flag=$([ -z $run_name ] && echo "" || echo "--run-name $run_name")

echo "env -> $env"
echo "run_flag -> $run_flag"

cd $HOME/bayesianrex
source scripts/activate_env.sh

# to fix an error occurring when importing wandb, comment out if you don't
# encounter it (fixes wrong import of libstdc++.so.6.0.30)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/micromamba/envs/bayesianrex-dl2/lib"
echo "Start @ $(date)"

# NOTE they impose a limit of 4 CPU cores per user
srun python -m bayesianrex.dataset.demonstrators \
  --seed 0 \
  --n-envs 4 \
  --assets-dir /project/gpuuva022/shared/b-rex \
  --log-level 1 \
  $env_flag \
  $run_flag

echo "End @ $(date)"

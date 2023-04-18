#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:2
#SBATCH --job-name=pretrain_bayesianrex
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --output=/home/%u/job_logs/%x_%A_%u.out

code_dir=$HOME/bayesianrex

source $code_dir/scripts/activate_env

cd $code_dir/code
echo "Start @ $(date)"
srun python LearnAtariRewardLinear.py --env_name breakout --reward_model_path ../pretrained_networks/breakout_pretrained.params --models_dir ..
echo "End @ $(date)"

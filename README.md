# TIIIITTTTLLLLEEEEE

# Running the project

## Installing the environment

You can install the environment for the project using any Python package manager, e.g. with Conda:
```sh
conda env create -f env.yml
```
We also provide an `env.gpu.yml` file that installs PyTorch with CUDA support on the 
[Lisa compute cluster](https://www.surf.nl/en/lisa-compute-cluster-extra-processing-power-for-research) as of 05-27-2023; 
check out the correct version of PyTorch-CUDA for your machine on 
[PyTorch's website](https://pytorch.org/get-started/locally/). 

Alternatively, on a Linux system, running `bash lisa_scripts/install_env.sh` will install the 
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) package manager in your home directory and 
create an environment from `env.yml`.

## Demonstrators
We reproduce the setup described by Brown et al. ([2019](https://arxiv.org/pdf/1904.06387.pdf)) to train a PPO agent on 
Atari games using default hyperparameters found in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml) (`atari` section). We take ~25 checkpoints during training, and use them to generate 
trajectories for Inverse Reinforcement Learning (IRL) from preferences.

Pretrained checkpoints for some Atari games are available at **<ins>TODO pretrained ckpts download link</ins>**. These 
demonstrators were trained for **<ins>number of hours</ins>** on a **<ins>GPU model</ins>** GPU. The corresponding learning 
curves can be seen at **<ins>TODO wandb link to learning curves??</ins>** for comparison with standard ones from 
[Stable Baselines' Zoo](https://wandb.ai/openrlbenchmark/sb3). 

To train a PPO demonstrator on the default game Breakout (pass the `-h` flag to print all possible options):
```sh
python -m bayesianrex.dataset.demonstrators
```
This will by default save Weighs&Biases and TensorBoard logs, checkpoints and final trained model in corresponding folders 
under `assets/`. 

## Training the reward embedding model

```sh
python -m bayesianrex.learn_reward_fn --env breakout \
	--checkpoints-dir assets/demonstrators/BreakoutNoFrameskip-v4
```
The above command will load the demonstrators checkpoints from `checkpoints-dir`, generate demonstrations with each 
checkpoint, and them to learn an embedding model of the reward function. The parameters of the learned model are 
saved by default at `assets/reward_model_<env_name>.pth`; this path can be changed by specifying a 
`--reward-model-save-path` argument.

Playing the game with a checkpointed agent does 
not take long; however, it is also possible to pre-generate the demonstrations and trajectories pairs used to learn the 
embedding model by first running
```sh
python -m bayesianrex.dataset.generate_demonstrations \
	--checkpoints-dir assets/demonstrators/BreakoutNoFrameskip-v4
    --train-data-save-dir assets/train_data/BreakoutNoFrameskip-v4
```
Demonstrations and training data are then saved under the given `train-data-save-dir` (defaults to 
`assets/train_data/<environment_id>`) as `trajectories` and `train_pairs.npz`, respectively. It is then possible to load 
them and learn the embedding model of the reward function as such:
```sh
python -m bayesianrex.learn_reward_fn --env breakout \
	--trajectories-path assets/train_data/BreakoutNoFrameskip-v4/trajectories \
    --train-data-path assets/train_data/BreakoutNoFrameskip-v4/train_pairs.npz
```

## Learning reward function posterior via MCMC <ins>TODO </ins>

## Training with the learned reward function <ins>TODO </ins>

## Evaluation with the ground truth reward function <ins>TODO </ins>
# A Study on Risk Evaluation in Imitation Learning
## Re: *Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences*
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
python -m bayesianrex.train_rl_agent
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

## Learning reward function posterior via B-REX MCMC
```sh
python -m bayesianrex.mcmc_reward_fn \
	--pretrained-model-path assets/reward_model_breakout.pth 
    --checkpoint-dir assets/demonstrators/BreakoutNoFrameskip-v4
```
The above command will run MCMC sampling over the T-Rex layer of the embedding model learned in the previous section 
for the game Breakout, stored into `assets/reward_model_breakout.pth`. It will finish by saving the MCMC chain and 
likelihoods into `assets/mcmc_chain_breakout.npz` and the MAP reward function into 
`assets/reward_model_breakout_MAP.pth`; both filenames can be specified with the arguments `mcmc-chain-save-path` and 
`map-model-save-path`, respectively.

In the command above, the preferences used to express the MCMC likelihood ratios during search are generated anew from 
`checkpoint-dir`. Instead of passing this argument, you can pass `--trajectories-path <path-to-trajectories>` to give 
pregenerated trajectories - make sure they are different than the ones used in the section **Training the reward 
embedding model**. 

## Training with the learned reward function
To train  a PPO agent using the MAP reward function learned via MCMC:
```sh
python -m bayesianrex.train_rl_agent --custom-reward \
	--reward-model-path assets/reward_model_breakout_MAP.pth
    --env breakout
```
To train  a PPO agent using the mean reward function from the MCMC chain:
```sh
python -m bayesianrex.train_rl_agent --custom-reward \
	--reward-model-path assets/reward_model_breakout_MAP.pth \
    --mean --mcmc-chain-path assets/mcmc_chain_breakout.npz \
    --env breakout
```
This will run the same training loop as the one used to generate the original demonstrators, but will replace the ground 
truth reward function with the mean or MAP one for policy learning. This round of training will take longer than the one 
with ground truth rewards, since the mean/MAP forward inference time is factored in. Checkpoints are saved by default 
under `<assets_dir>/demonstrators/<env>_custom` folder, where `<assets_dir>` and `<env>` are respectively the name of the 
passed `assets-dir` and `env` arguments. 

## Evaluation with the ground truth reward function
Finally, to evaluate all checkpointned PPO agents trained on a learned reward function, run:
```sh
python -m bayesianrex.evaluate_policy \
	--checkpointpath assets/demonstrators/breakout_custom \
    --eval_all
```
If `eval_all` is omitted and `checkpointpath` is a path to a valid PPO checkpoints e.g. 
`assets/demonstrators/breakout_custom/PPO_400000_steps.zip`, only the given checkpoint is evaluated. 










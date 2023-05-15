# Environment installation
```sh
micromamba create -f env.yml
```

## Installing [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
```sh
cd bayesianrex_cc/rl-baselines3-zoo
pip install -e .
cd ..
ln -s rl-baselines3-zoo rl_baselines3_zoo
```

**NOTE** The symlink to `rl_baselines3_zoo` is necessary to be able to import
the rl baselines zoo library in user code (also renaming the directory is fine),
since Python packages cannot contain dashes. To avoid having to write `import
sys; sys.path.append('./rl-baselines3-zoo/')` everywhere we want to use the zoo
as a library, run a Python file as a module adding the zoo to the PYTHONPATH,
e.g. with `training_data.py`:

```sh
cd bayesianrex_cc
PYTHONPATH="./rl-baselines3-zoo:${PYTHONPATH}" python -m training_data
```

Or `export PYTHONPATH="./rl-baselines3-zoo:${PYTHONPATH}"` once in your current
shell.

<!-- ```sh -->
<!-- cd bayesianrex_cc -->
<!-- git clone git@github.com:DLR-RM/rl-baselines3-zoo.git -->
<!-- cd rl-baselines3-zoo -->
<!-- git checkout 382dcabbd9815cb9557503a7caa0c54a562fa7fb -->
<!-- pip install -e . -->
<!-- ``` -->

# Train demonstrators
```sh
micromamba activate bayesianrex-cc
cd bayesianrex_cc
python -m demonstrators
```

wandb, tensorboard and csv logs for the environments listed in
`bayesianrex_cc.constants.envs_id_mapper` will be logged under the root of
the git repo; they are also available
[here](https://drive.google.com/drive/folders/1usrIO5k9-KNfgFJQf74Qr0NfDwy9xKSM?usp=share_link).
Hyperparameters for the `stable_baselines3` PPO agent for
each environment are read from
`bayesianrex_cc/rl-baselines3-zoo/hyperparams/ppo.yml`.

# Generate training data (demonstrations)
Assuming PPO checkpoints trained on CartPole-v1 are stored at
`./logs/ppo/CartPole-v1_1` and training data will be saved at `./data`:
```sh
cd bayesianrex_cc
PYTHONPATH="./rl-baselines3-zoo:$PYTHONPATH" python -m training_data --env cartpole --ckpts-dir ../logs/ppo/CartPole-v1_1 --snippet-min-len 50 --snippet-max-len 100 --save-dir ../data
```

Training data created by Marco on 14/05/2023 using the default seed (42) are
available
[here](https://drive.google.com/drive/folders/11K4-8kksIFYM5IAe_9JcyLsU_Kiomv0-?usp=share_link).

# NOTES
## Training data
For Atari the authors do 2 things:

1. Generate `num_trajs` (downsampled) full trajectories by picking 2 random
   trajectories and giving preference labels; these are used only on Enduro
   - partial preferences
   - not all trajectories used, some used more than once

2. Generate `num_snippets` (default 60000) short trajectories:
   1. pick 2 random trajectories
   2. pick a random snippet length < min trajectory length picked
   3. ensure that the snippet sampled from the worse trajectory occurs
	  temporally later than the one sampled from the better trajectory -
	  probably to signal that a better trajectory is so since the early states

## Prediction/T-REX loss
The NN model directly predicts the reward based
purely on _states_, not even actions (these are used in the self-supervised
losses, cf. `train_net.forward()` and `train_net.cum_return()`). The T-REX
loss is applied to couples of trajectories as a cross entropy loss between a
trajectory's (estimated) return (i.e. continuous) and whether it was better
than the other trajectory (i.e. discrete), corresponding to formula 4 in Brown
et al. (2020):
  ```math
  P(D, \mathcal{P} \mid R_{\theta}) = \prod_{(i,j)\in\mathcal{P}}\frac{e^{\beta R_{\theta}(\tau_j)}}{e^{\beta R_{\theta}(\tau_i)} + e^{\beta R_{\theta}(\tau_j)}}
  ```


# TODO

## Bureaucracy
- [X] transfer repo under `xqz-u/bayesianrex` new branch `classic_control`
- [ ] generate more diverse demonstrators (the majority of them are too good already)
- [ ] git submodule for `rl_zoo3`
- [ ] `isort` + `black` github actions; provisioned
	[here](https://towardsdatascience.com/black-with-git-hub-actions-4ffc5c61b5fe)
	but it adds a new commit, which I don't like. Maybe install a local
	pre-commit hook
- [ ] Lisa gpu-compatible env (?)
- [ ] Add possibility to specify assets folders where needed, e.g. in Lisa it's
	  important to dump stuff under `/scratch/` or `/project/`

## Code
A lot, I'll write a better description tmrw

# Environment installation
```sh
micromamba create -f env.yml
```

## Installing [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
```sh
cd bayesianrex_cc/rl-baselines3-zoo
pip install -e .
```
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
`bayesianrex_cc.constants.envs_id_mapper` are logged under at the root of the
git repo. Hyperparameters for the `stable_baselines3` PPO agent for
each environment are read from
`bayesianrex_cc/rl-baselines3-zoo/hyperparams/ppo.yml`.

# TODO

## Bureaucracy
- [X] transfer repo under `xqz-u/bayesianrex` new branch `classic_control`
- [ ] git submodule for `rl_zoo3`
- [ ] `isort` + `black` github actions; provisioned
	[here](https://towardsdatascience.com/black-with-git-hub-actions-4ffc5c61b5fe)
	but it adds a new commit, which I don't like. Maybe install a local
	pre-commit hook

## Code
A lot, I'll write a better description tmrw

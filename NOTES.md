# How to run stuff
`bayesianrex/dataset/demonstrators.py`,`bayesianrex/dataset/generate_demonstrations.py`
and `bayesianrex/learn_reward_fn.py` all accept command-line arguments, visible
by calling each script with a `-h` flag e.g.
```sh
python -m bayesianrex.learn_reward_fn -h
```
You don't need to run `generate_demonstrations.py`, its functions are called by
`learn_reward_fn.py` but I also made it a runnable as a standalone.

**NOTE** that `breakout` demonstrators checkpoints are available on Lisa at
`/project/gpuuva22/shared/b-rex` thx to Marga

## Important
For those files that accept the `assets-dir` flag, **if you are on Lisa**,
specify it like `--assets-dir /project/gpuuva22/shared/b-rex`

# TODO

## Steps missing to complete the pipeline
- Learn a reward fn with `bayesianrex.learn_reward_fn` (running it)
- Strip the weights of the `RewardNetwork` and keep only the last layer
- MCMC over the aforementioned weights
- Train with the MAP/mean reward fn from MCMC sampling with a custom
  environment (there is already a copy-pasted skeleton in
  `bayesianrex/environments.py`, and David also worked on the overall logic),
  and saving the final trained agent
- Normal evaluation of the aforementioned trained agent
- Plots (?)
- ... extensions ?

## Overall
- [ ] Go through the original codebase and the current rewrite, and double-check
	  that we are nailing the correct logic
- [ ] Get a better idea on how PPO works - also for the report & individual
	  essay
- [ ] Functions docstrings
- [ ] README section on how to run the different files
- [ ] Speed up generation of 6e4 trajectory snippets (default number in original
	  codebase). **Actually** they are generated fast; the problem might be with
	  some 'faulty' checkpoints, which run forever in the RL game loop
	  (investigate)

# Possible ways to improve results
- `bayesianrex/dataset/demonstrators.py`: demonstrators trained on normal
  environments **without lives/score masking** (T-Rex paper said to train
  demonstrators with OpenAI Baselines' default parameters), but all the later
  B-Rex pipelines involving RL environments use games with masked lives/scores.
- **Bad checkpoints**: maybe too few checkpoints over default 10e6 demonstrators
  training steps, and they all come from an already well-trained region ->
  preferences are not really informative. Standard training curves are
  [here](https://wandb.ai/openrlbenchmark/sb3), but we should check ours too.
- **Wrong preference labels**: see the first *NOTE* comment on
  `bayesianrex.dataset.generate_demonstrations.training_snippets()`.

import argparse
from pathlib import Path
import multiprocessing as mp
import os

import numpy as np
import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan

from bayesianrex import config, constants, utils
from bayesianrex.environments import create_atari_env


def evaluate_learned_policy(env_name, checkpointpath, num_episodes, conf, run):
    """
    Evaluate a learned policy stored in a checkpoint file on the specified environment.

    :param checkpointfile: path to checkpoint file
    :param env_name: name of the environment to evaluate the policy on
    :param num_episodes: number of episodes to run during evaluation
    :param conf: configuration parameters required for evaluation
    :param run: The wandb run object
    :return: A list of returns obtained from each evaluated episode.
    """
    conf["env_args"] = {"n_envs": args.n_envs or mp.cpu_count(), "seed": args.seed}
    env_id = constants.envs_id_mapper.get(env_name)
    env = create_atari_env(env_id, **conf["env_args"], vec_env_cls=SubprocVecEnv)
    env = VecCheckNan(env, raise_exception=True)

    agent = PPO.load(checkpointpath, custom_objects={"tensorboard_log": None})

    learning_returns = []
    print(checkpointpath)

    for episode in range(num_episodes):
        if args.seed is not None:
            env.seed(args.seed + episode)
        r, steps, acc_reward = 0, 0, 0
        ob, done = env.reset(), False
        while True:
            action, *_ = agent.predict(ob)
            ob, r, done, *_ = env.step(action)
            steps += 1
            acc_reward += r[0]
            if done:
                print(f"episode: {episode} steps: {steps}, return: {acc_reward}")
                run.log({"steps": steps, "acc_reward": acc_reward})
                break
        learning_returns.append(acc_reward)
    run.log({"avg_reward": np.mean(learning_returns), "std": np.std(learning_returns)})
    # env.close()
    return learning_returns


def eval_checkpoint(checkpointfile, env_name, num_episodes, conf):
    """
    Evaluate a checkpoint file containing learned policy weights on the specified environment.
    (This is mainly a wrapper over evaluate_learned_policy)

    :param checkpointfile: path to checkpoint file
    :param env_name: name of the environment to evaluate the policy on
    :param num_episodes: number of episodes to run during evaluation
    :param conf: configuration parameters required for evaluation
    """
    n_steps = checkpointfile.stem.split("_")[1]
    run_name = f"{env_name}_{n_steps}"
    run = wandb.init(
        project="evaluate_checkpoints",
        entity="bayesianrex-dl2",
        name=run_name,
        monitor_gym=True,
    )

    print("*" * 10)
    print(checkpointfile)
    print("*" * 10)
    _ = evaluate_learned_policy(env_name, checkpointfile, num_episodes, conf, run)
    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for experiments"
    )
    parser.add_argument(
        "--env_name", default="breakout", help="Select the environment name to run, i.e. pong"
    )
    parser.add_argument(
        "--checkpointpath", type=Path, help="path to checkpoint to run eval on"
    )
    parser.add_argument(
        "--num_episodes", default=30, type=int, help="number of rollouts"
    )
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="enable to evaluate all the checkpoints in the checkpointpath directory",
    )
    args = parser.parse_args()

    # set seeds
    set_random_seed(seed=args.seed, using_cuda=utils.torch_device().type != "cpu")

    with open(config.ROOT_DIR / "atari_conf.yml") as fd:
        conf = yaml.safe_load(fd)

    if args.eval_all:
        for cpt_file in os.listdir(args.checkpointpath):
            if ".zip" in cpt_file:
                path = f"{args.checkpointpath}/{cpt_file[:-4]}"
                eval_checkpoint(path, args.env_name, args.num_episodes, conf)
    else:
        assert args.checkpointpath is not None
        eval_checkpoint(args.checkpointpath, args.env_name, args.num_episodes, conf)

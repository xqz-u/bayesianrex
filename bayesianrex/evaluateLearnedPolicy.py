import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan

from bayesianrex import config, constants
from bayesianrex.environments import create_atari_env


def evaluate_learned_policy(env_name, checkpointpath, num_episodes, conf):
    """
    Evaluate a learned policy stored in a checkpoint file on the specified environment.

    :param checkpointfile: path to checkpoint file
    :param env_name: name of the environment to evaluate the policy on
    :param num_episodes: number of episodes to run during evaluation
    :param conf: configuration parameters required for evaluation
    :return: A list of returns obtained from each evaluated episode.

    """
    env_id = constants.envs_id_mapper.get(env_name)

    conf["env_args"] = {"n_envs": args.n_envs or mp.cpu_count(), "seed": args.seed}

    env = create_atari_env(env_id, **conf["env_args"], vec_env_cls=SubprocVecEnv)
    env = VecCheckNan(env, raise_exception=True)

    agent = PPO.load(checkpointpath)

    learning_returns = []
    print(checkpointpath)

    episode_count = num_episodes
    for i in range(episode_count):
        done = False
        r = 0

        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:
            action, _states = agent.predict(ob)
            ob, r, done, _ = env.step(action)

            steps += 1
            acc_reward += r[0]
            if done:
                print("steps: {}, return: {}".format(steps, acc_reward))
                wandb.log({"steps": steps, "acc_reward": acc_reward})
                break
        learning_returns.append(acc_reward)

    wandb.log(
        {"avg_reward": np.mean(learning_returns), "std": np.std(learning_returns)}
    )
    env.close()

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
    n_steps = checkpointfile.split("/")[-1]
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
    _ = evaluate_learned_policy(env_name, checkpointfile, num_episodes, conf)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for experiments"
    )
    parser.add_argument(
        "--env_name", default="", help="Select the environment name to run, i.e. pong"
    )
    parser.add_argument(
        "--checkpointpath", default="", help="path to checkpoint to run eval on"
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
    env_name = args.env_name
    # set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config.ROOT_DIR / "atari_conf.yml") as fd:
        conf = yaml.safe_load(fd)

    if args.eval_all:
        for cpt_file in os.listdir(args.checkpointpath):
            if ".zip" in cpt_file:
                path = f"{args.checkpointpath}/{cpt_file[:-4]}"
                eval_checkpoint(path, args.env_name, args.num_episodes, conf)
    else:
        eval_checkpoint(args.checkpointpath, args.env_name, args.num_episodes, conf)

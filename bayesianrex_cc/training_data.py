"""Generate IRL training data: trajectories of states, actions and rewards."""
import logging
from pathlib import Path
from typing import List, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_trajectory_from_ckpt(
    ckpt_path: Union[Path, str],
    env: GymEnv,
    n_traj: int = 1,
    seed: int = 42,
    seed_globally: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate demonstrations for an arbitrary environment from a PPO checkpoint.

    :param ckpt_path: (Absolute) path to the zipped checkpoint
    :param env: The gym environment used to generate the checkpoint
    :param n_traj: Number of episodes to run (a trajectory is a full episode)
    :param seed: The RNG seed passed to the environment, increased by `n_traj`
    :param seed_globally: Whether to use StableBaseline's RNG seeding utility
    :return: The states, actions and rewards for each trajectory
    """
    logger.debug("Loading ckpt '%s'", ckpt_path)
    agent = PPO.load(ckpt_path, env, verbose=1)
    env = agent.get_env()
    if seed_globally:
        logger.info("Set global RNG seeds *WARN* should be called once")
        set_random_seed(seed, using_cuda=device.type != "cpu")
    states, actions, rewards = [], [], []
    for i in range(n_traj):
        env.seed(seed + i)
        done = False
        obs = env.reset()
        ep_states, ep_actions, ep_rewards = [], [], []
        while not done:
            ep_states.append(obs)
            action, *_ = agent.predict(obs)
            obs, reward, done, *_ = env.step(action)
            ep_actions.append(action)
            ep_rewards.append(reward)
        states.append(np.array(ep_states).squeeze())
        actions.append(np.array(ep_actions).squeeze())
        rewards.append(np.array(ep_rewards).squeeze())
    return states, actions, rewards


def generate_trajectories():
    ...


if __name__ == "__main__":
    ...

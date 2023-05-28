import logging
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common import env_util
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from torch import nn

from bayesianrex import constants
from bayesianrex.models.reward_model import RewardNetwork

logger = logging.getLogger(__name__)


# NOTE grayscale image normalization (/ 255) occurs in ActorCriticCnnPolicy.extract_features()
def mask_score(obs: np.ndarray, env_name: str) -> np.ndarray:
    """
    Obfuscates the score and life count pixel portions of some Atari 2600 games.

    :param obs: The original observation returned by the Atari Gymnasium
        environment with shape (H,W,C).
    :param env_name: A string representing the Atari game id as found in
        `bayesianrex.code.rl_utils.environments_mapper`.
    :return: The original observation without game score/life information.
    """
    obs_copy = obs.copy()
    if env_name in ["spaceinvaders", "breakout", "pong", "montezumarevenge"]:
        obs_copy[:10, :, :] = 0
    elif env_name == "beamrider":
        obs_copy[:16, :, :] = 0
        obs_copy[-11:, :, :] = 0
    elif env_name == "enduro":
        obs_copy[-14:, :, :] = 0
    elif env_name == "hero":
        obs_copy[-30:, :, :] = 0
    # elif env_name == "qbert":
    #     obs_copy[:12, :, :] = 0
    elif env_name == "seaquest":
        # cuts out divers and oxygen
        obs_copy[:12, :, :] = 0
        obs_copy[-16:, :, :] = 0
    # elif env_name == "mspacman":
    #     obs_copy[-15:, :, :] = 0  # mask score and number of lives left
    # elif env_name == "videopinball":
    #     obs_copy[:15, :, :] = 0
    else:
        logger.warning("Don't know how to mask '%s', pass", env_name)
    return obs_copy


def create_atari_env(
    env_id: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    wrapper_kwargs: Optional[dict] = None,
    **kwargs
) -> VecFrameStack:
    logger.debug("Env: %s", env_id)
    return VecFrameStack(
        env_util.make_atari_env(
            env_id, n_envs=n_envs, seed=seed, wrapper_kwargs=wrapper_kwargs, **kwargs
        ),
        n_stack=4,
    )


def create_hidden_lives_atari_env(
    env_name: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    atari_kwargs: Optional[dict] = None,
    **kwargs
) -> VecFrameStack:
    env_id = constants.envs_id_mapper.get(env_name)
    logger.info("Env name %s Env id %s", env_name, env_id)
    return VecFrameStack(
        env_util.make_vec_env(
            env_id,
            n_envs=n_envs,
            seed=seed,
            **kwargs,
            wrapper_class=lambda env, **_: TransformObservation(
                AtariWrapper(env, **(atari_kwargs or {})),
                lambda obs: mask_score(obs, env_name),
            ),
        ),
        n_stack=4,
    )


class MAPRewardWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, reward_model: RewardNetwork):
        super().__init__(venv=venv)
        self.reward_model = reward_model
        # NOTE necessary to avoid reparameterization trick in
        # RewardNetwork.cum_return, as in original code
        self.reward_model.eval()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        reward = self.reward_model.cum_return(torch.Tensor(obs))[0].detach().numpy()
        return (obs, reward[..., None], done, info)


class MeanRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_net, chain_path, embedding_dim, device):
        super().__init__(env)
        self.device = device
        self.env = env
        self.reward_net = reward_net

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.0

        # load the mean of the MCMC chain
        burn = 5000
        skip = 20
        reader = open(chain_path)
        data = []
        for line in reader:
            parsed = line.strip().split(",")
            np_line = []
            for s in parsed[:-1]:
                np_line.append(float(s))
            data.append(np_line)
        data = np.array(data)

        # get average across chain and use it as the last layer in the network
        mean_weight = np.mean(data[burn::skip, :], axis=0)
        self.reward_net.trex = nn.Linear(
            embedding_dim, 1, bias=False
        )  # last layer just outputs the scalar reward = w^T \phi(s)

        new_linear = torch.from_numpy(mean_weight)
        with torch.no_grad():
            # unsqueeze since nn.Linear wants a 2-d tensor for weights
            new_linear = new_linear.unsqueeze(0)
            with torch.no_grad():
                self.reward_net.trex.weight.data = new_linear.float().to(self.device)
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.0

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        obs = torch.cat((torch.Tensor(self.env.state), torch.Tensor([action])))
        custom_reward, _, _, _, _ = self.reward_model.cum_reward(obs)

        return next_state, custom_reward.item(), done, truncated, info

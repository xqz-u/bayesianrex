import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common import env_util
from stable_baselines3.common.atari_wrappers import AtariWrapper
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
        # NOTE necessary to avoid reparameterization trick in
        # RewardNetwork.cum_return(), as in original code (EmbeddingNet of
        # custom_reward_wrapper.py in baselines)
        self.reward_model = reward_model.eval()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        # reward embeddings were trained on float representation of pixels
        model_obs = torch.Tensor(obs / 255.0).to(self.reward_model.device)
        reward = self.reward_model.cum_return(model_obs)[0].detach().numpy()
        return (obs, reward[..., None], done, info)


class MeanRewardWrapper(MAPRewardWrapper):
    def __init__(self, venv: VecEnv, reward_model: RewardNetwork, chain_path: Path):
        super().__init__(venv, reward_model)
        # load MCMC data
        mcmc_data = np.load(chain_path)
        mcmc_chain = mcmc_data["chain"]
        # last layer just outputs the scalar reward = w^T \phi(s)
        self.reward_model.trex = nn.Linear(reward_model.trex.in_features, 1, bias=False)
        # get average across (downsampled) chain and use it as the last layer in
        # the network
        burn, skip = int(5e3), 20
        if len(mcmc_chain) >= burn + 100:
            logger.info(
                "No proposal burning/downsampling, chain size: %d", len(mcmc_chain)
            )
            burn, skip = 0, 1
        mean_reward_fn = mcmc_chain[burn::skip].mean(0)
        mean_reward_fn = (
            torch.from_numpy(mean_reward_fn[None, ...])
            .to(torch.float32)  # default torch dtype
            .to(reward_model.device)
        )
        self.reward_model.trex.weight = nn.Parameter(mean_reward_fn)
        logger.info("Succesfully set learned reward fn as mean of MCMC chain")

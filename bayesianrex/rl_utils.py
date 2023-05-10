import logging

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import config
import constants

logger = logging.getLogger(__name__)


def framestack_env(env: VecEnv, n_stack: int = 4) -> VecFrameStack:
    return VecFrameStack(env, n_stack=n_stack)


def make_base_atari_env(env_id: str, **kwargs) -> VecEnv:
    logger.debug("Env: %s args: %s", env_id, kwargs)
    return framestack_env(make_atari_env(env_id, **kwargs))


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


# TODO add possibility to load from checkpoints
# TODO (opt) VecVideoRecorder
# TODO make it so that it works for both the T-REX agents (train to generate
# demonstrations), so maybe no score masking and different params, and with the
# same params as B-REX
# NOTE grayscale image normalization (/ 255) occurs in ActorCriticCnnPolicy.extract_features()
def make_ppo_agent(
    atari_env_name: str,
    n_envs: int = 1,
    hide_scores: bool = True,
    seed: int = 42,
    verbose: int = 0,
) -> PPO:
    """
    Utility to create a PPO agent from stable-baselines3 with the same
    parameters as in B-REX (https://arxiv.org/pdf/2002.09089.pdf).
    Creates a vectorized Atari environment with observations hiding game score
    and lives left.

    :param atari_env_name: The id of the desired Atari environment e.g.
        'ALE/Breakout-v5'; see https://gymnasium.farama.org/environments/atari/
    :param n_envs: Number of environments to run in parallel, defaults to 1
    :param seed: Random seed for reproducibility, passed both to the agent and
        the environment.
    :param verbose: Agent logging level, value integer between 0 and 2.
    :return: A PPO agent that works with pixel observations.
    """
    env_id = constants.envs_id_mapper.get(atari_env_name)
    logger.debug(
        "Env name: %s Env id: %s Life obfuscation: %s",
        atari_env_name,
        env_id,
        hide_scores,
    )
    if not env_id:
        raise NotImplementedError(f"Unsupported environment: '{atari_env_name}'")
    vec_env_kwargs = {"n_envs": n_envs, "seed": seed}
    atari_wrapper_kwargs = {"clip_reward": False, "terminal_on_life_loss": False}
    if hide_scores:
        env = framestack_env(
            make_vec_env(
                env_id,
                **vec_env_kwargs,
                wrapper_class=lambda env, **_: TransformObservation(
                    AtariWrapper(env, **atari_wrapper_kwargs),
                    lambda obs: mask_score(obs, atari_env_name),
                ),
            )
        )
    else:
        env = make_base_atari_env(
            env_id, **vec_env_kwargs, wrapper_kwargs=atari_wrapper_kwargs
        )
    return PPO(
        "CnnPolicy",
        env,
        ent_coef=0.0,
        vf_coef=0.0,
        max_grad_norm=0.0,
        seed=seed,
        verbose=verbose,
    )


# TODO check that obfuscation is the same as in old code
def test_score_obfuscation(n_envs: int = 1):
    """
    Checks that scores and lives for Atari games known in
    `constants.envs_id_mapper` are masked correctly, and outputs corresponding
    frames in `plot_dir`.
    """

    def plot_and_save(ag, env_name, hide):
        vec_env = ag.get_env()
        obs = vec_env.reset()
        print(f"observation shape: {obs.shape}")
        for i, env_obs in enumerate(obs):
            # after a env.reset() only the last of 4 frames is populated
            plt.imshow(env_obs[3], cmap="gray")
            fname = (
                config.TEST_ASSETS_DIR
                / f"{constants.envs_id_mapper[env_name]}_{i}{'_dark' if hide else ''}.png"
            )
            print(f"Save to {fname}")
            plt.savefig(fname)

    for env_name in constants.envs_id_mapper:
        print(env_name)
        ag = make_ppo_agent(env_name, hide_scores=False, n_envs=n_envs)
        plot_and_save(ag, env_name, False)
        ag = make_ppo_agent(env_name, n_envs=n_envs)
        plot_and_save(ag, env_name, True)

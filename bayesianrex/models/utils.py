import logging
from pathlib import Path
from typing import Optional

import torch
from bayesianrex import constants, utils
from bayesianrex.environments import create_atari_env
from bayesianrex.models.reward_model import RewardNetwork

logger = logging.getLogger(__name__)


def make_reward_network(
    env: str,
    encoding_dims: int = constants.reward_net_latent_space,
    device: Optional[torch.device] = None,
) -> RewardNetwork:
    n_actions = create_atari_env(constants.envs_id_mapper.get(env)).action_space.n
    device = device or utils.torch_device()
    return RewardNetwork(encoding_dims, n_actions, device)


def load_reward_network(model_path: Path, env: str, **kwargs) -> RewardNetwork:
    reward_net = make_reward_network(env, **kwargs)
    logger.info("Loading reward model params from %s", model_path)
    reward_net.load_state_dict(
        torch.load(model_path, map_location=kwargs.get("device"))
    )
    return reward_net

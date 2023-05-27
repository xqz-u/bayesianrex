from typing import Dict, Tuple

import torch
from torch import Tensor as T
from torch.nn import functional as F

from bayesianrex.networks import RewardNetwork

# NOTE everywhere, `mu` is a trajectory encoding


def trex_loss(returns: T, preferences: T) -> T:
    return F.cross_entropy(returns, preferences)


# NOTE `fwd_dynamics_distance` corresponds to framestack
def forward_dynamics_loss(
    reward_net: RewardNetwork, mu: T, actions: T, fwd_dynamics_distance: int = 5
) -> T:
    num_states = len(actions)
    # one-hot encoding the actions like this makes sure the right dimensionality
    # is preserved even if some action was never performed in the episode
    actions_onehot = torch.zeros((num_states, reward_net.action_dims))
    actions_onehot[torch.arange(actions_onehot.size(0)), actions] = 1
    actions_onehot = actions_onehot.to(reward_net.device)

    # dropping last state/action
    next_state_encoding = reward_net.forward_dynamics(
        mu[:-fwd_dynamics_distance], actions_onehot[:-fwd_dynamics_distance]
    )
    # simulate estimating next frame, for each frame (?)
    for fd_i in range(1, fwd_dynamics_distance):
        next_state_encoding = reward_net.forward_dynamics(
            next_state_encoding,
            actions_onehot[fd_i : num_states - fwd_dynamics_distance + fd_i],
        )
    return 100 * F.mse_loss(next_state_encoding, mu[fwd_dynamics_distance:])


def inverse_dynamics_loss(reward_net: RewardNetwork, mu: T, target_actions: T) -> T:
    # estimate actions taken from the concatenation of state encodings, dropping
    # first and last states
    actions = reward_net.estimate_inverse_dynamics(mu[:-1], mu[1:])
    return F.cross_entropy(actions, target_actions[1:]) / 1.9


def temporal_distance_loss(reward_net: RewardNetwork, mu: T, times: T) -> T:
    # pick 2 random states from same trajectory
    t1, t2 = torch.randint(0, len(times), size=(2,))
    estimated_dt = reward_net.estimate_temporal_difference(mu[t1], mu[t2])
    real_dt = (times[t2] - times[t1]) / 100.0
    return 4 * F.mse_loss(estimated_dt, real_dt[None, ...])


def reconstruction_loss(
    reward_net: RewardNetwork, encoding: T, target: T, mu: T, logvar: T
) -> T:
    recon_loss = F.binary_cross_entropy(reward_net.decode(encoding), target)
    kld_regularizer = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_regularizer /= target.numel()
    return 10 * (recon_loss + kld_regularizer)


def self_supervised_loss(
    reward_net: RewardNetwork, states: T, actions: T, times: T, info: Dict[str, T]
) -> Tuple[T, T, T, T]:
    mu, var, z = info["mu"], info["var"], info["z"]
    return sum(
        (
            reconstruction_loss(reward_net, z, states, mu, var),
            temporal_distance_loss(reward_net, mu, times),
            inverse_dynamics_loss(reward_net, mu, actions),
            forward_dynamics_loss(reward_net, mu, actions),
        )
    )


def self_supervised_losses(
    reward_net: RewardNetwork,
    states: T,
    actions: T,
    times: T,
    info: Tuple[Dict[str, T]],
) -> T:
    return self_supervised_loss(
        reward_net, states[0], actions[0], times[0], info[0]
    ) + self_supervised_loss(reward_net, states[1], actions[1], times[1], info[1])

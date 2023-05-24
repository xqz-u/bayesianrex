from typing import Dict, Optional, Tuple

import torch
import torch.distributions as tdist
from torch import Tensor as T
from torch import nn


class RewardNetwork(nn.Module):
    def __init__(self, encoding_dims: int, action_dims: int, device: torch.device):
        super().__init__()
        self.encoding_dims = encoding_dims
        self.action_dims = action_dims
        self.device = device
        # This is the width of the layer between the convolved framestack
        # and the actual latent space. Scales with self.encoding_dims
        intermediate_dimension = min(784, max(64, self.encoding_dims * 2))
        # Define CNN embedding
        self.cnn_embedding = nn.Sequential(
            # Conv layers
            nn.Conv2d(4, 16, 7, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, stride=1),
            nn.LeakyReLU(),
            # Reshapes into [traj_size, 784]
            nn.Flatten(start_dim=1),
            # Brings the convolved frame down to intermediate dimension just
            # before being sent to latent space
            nn.Linear(784, intermediate_dimension),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dims, intermediate_dimension),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dimension, 1568),
            nn.LeakyReLU(),
            nn.Unflatten(1, (2, 28, 28)),
            nn.ConvTranspose2d(2, 4, 3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 16, 6, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 7, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 4, 10, stride=1),
            nn.Sigmoid(),
        )
        # - given 2 state encodings, predict temporal difference
        self.temporal_difference_M = nn.Linear(encoding_dims * 2, 1, bias=False)
        # - given state encodings B and A, predict action sequence A -> B
        self.inverse_dynamics_M = nn.Linear(encoding_dims * 2, action_dims, bias=False)
        # - given state encoding and action, predict next state encoding
        self.forward_dynamics_M = nn.Linear(
            encoding_dims + action_dims, encoding_dims, bias=False
        )
        # VAE modules (kept divided since fc_mu is often used alone)
        self.fc_mu = nn.Linear(intermediate_dimension, self.encoding_dims)
        self.fc_var = nn.Linear(intermediate_dimension, self.encoding_dims)
        self.normal = tdist.Normal(0.0, 1.0)
        # T-Rex layer, linear combination of self.encoding_dims
        self.trex = nn.Linear(self.encoding_dims, 1)

    def reparameterize(self, mu: T, var: T) -> T:
        std = torch.exp(var / 2)
        eps = self.normal.sample(mu.shape).to(self.device)
        return eps * std + mu

    def cum_return(self, traj: T) -> Tuple[T, T, T, Optional[T], T]:
        """Estimates the return of a trajectory."""
        # get embedding of trajectory in BNCHW layout from BNHWC one
        traj_embedding = self.get_embedding(traj.permute(0, 3, 1, 2))
        mu = self.fc_mu(traj_embedding)
        var = self.fc_var(traj_embedding)  # var is actually the log variance
        # If training the embeddings, sample latent space to compute reward
        # else take the mean
        z = self.reparameterize(mu, var) if self.training else mu
        reward = self.trex(z)
        return_ = reward.sum()
        abs_return_ = torch.abs(reward).sum()
        return return_, abs_return_, mu, var, z

    def forward(self, traj_i: T, traj_j: T) -> Tuple[T, T, Dict[str, T]]:
        """Compute cumulative reward for each trajectory and return logits."""
        cum_r_i, abs_r_i, *rest_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j, *rest_j = self.cum_return(traj_j)
        keys = ["mu", "var", "z"]
        return (
            torch.hstack((cum_r_i, cum_r_j))[None, ...],
            abs_r_i + abs_r_j,
            dict(zip(keys, rest_i)),
            dict(zip(keys, rest_j)),
        )

    def get_embedding(self, traj: T) -> T:
        return self.cnn_embedding(traj)

    def estimate_temporal_difference(self, z1: T, z2: T) -> T:
        return self.temporal_difference_M(torch.hstack((z1, z2)))

    def estimate_inverse_dynamics(self, z1: T, z2: T) -> T:
        return self.inverse_dynamics_M(torch.hstack((z1, z2)))

    def forward_dynamics(self, z, actions):
        return self.forward_dynamics_M(torch.hstack((z, actions)))

    def decode(self, encoding: T) -> T:
        return self.decoder(encoding).permute(0, 2, 3, 1)

    def state_features(self, traj: T) -> T:
        with torch.no_grad():
            x = self.cnn_embedding(traj.permute(0, 3, 1, 2))
            mu = self.fc_mu(x)
            mu_sum = torch.sum(mu, dim=0)
        return mu_sum

    def state_feature(self, obs: T) -> T:
        with torch.no_grad():
            x = self.cnn_embedding(obs.permute(0, 3, 1, 2))
            mu = self.fc_mu(x)
        return mu

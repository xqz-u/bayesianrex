import torch
from torch import functional as F
from torch import nn
import torch.distributions as tdist

class RewardNet(nn.Module):
    def __init__(self, ENCODING_DIMS, ACTION_DIMS, training, device):
        super().__init__()
        self.training = training  ## Denotes if network is being trained (True during embedding network training, False otherwise)
        self.ENCODING_DIMS = ENCODING_DIMS
        self.device = device

        # This is the width of the layer between the convolved framestack
        # and the actual latent space. Scales with self.ENCODING_DIMS
        intermediate_dimension = min(784, max(64, self.ENCODING_DIMS * 2))

        ## Define CNN embedding
        self.cnn_embedding = nn.Sequential(
            # Conv layers
            nn.Conv2d(4, 16, 7, stride=3), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, stride=1), nn.LeakyReLU(),
            # Reshapes into [traj_size, 784]
            nn.Flatten(start_dim=1),
            # Brings the convolved frame down to intermediate dimension just before being sent to latent space
            nn.Linear(784, intermediate_dimension), nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.ENCODING_DIMS, intermediate_dimension), nn.LeakyReLU(),
            nn.Linear(intermediate_dimension, 1568), nn.LeakyReLU(),
            nn.Unflatten(1, (2, 28, 28)),
            nn.ConvTranspose2d(2, 4, 3, stride=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 16, 6, stride=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 16, 7, stride=2), nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 4, 10, stride=1), nn.Sigmoid(),
        )
        
        # Nets for the losses
        self.temporal_difference_M = nn.Linear(ENCODING_DIMS*2, 1, bias=False)
        self.inverse_dynamics_M = nn.Linear(ENCODING_DIMS*2, ACTION_DIMS, bias=False)
        self.forward_dynamics_M = nn.Linear(ENCODING_DIMS+ACTION_DIMS, ENCODING_DIMS, bias=False)

        # These allow sampling for the autoencoder
        self.fc_mu = nn.Linear(intermediate_dimension, self.ENCODING_DIMS)
        self.fc_var = nn.Linear(intermediate_dimension, self.ENCODING_DIMS)
        self.normal = tdist.Normal(0, 1)

        # This is the actual T-REX layer; linear comb. from self.ENCODING_DIMS
        self.trex = nn.Linear(self.ENCODING_DIMS, 1)

    def get_embedding(self, traj):
        return self.cnn_embedding(traj)

    ## Formerly: EmbeddingNet.forward (used in environments.py)
    def cum_return(self, traj):
        """Compute the return of a trajectory"""
        traj = traj.permute(0, 3, 1, 2)
        traj_embedding = self.get_embedding(traj)

        mu = self.fc_mu(traj_embedding)

        # If training the embeddings, sample latent space to compute reward
        # else take the mean
        if self.training:
            var = self.fc_var(traj_embedding)   # Var is actually the log variance
            std = var.mul(0.5).exp()
            eps = self.normal.sample(mu.shape).to(self.device)
            z = eps.mul(std).add(mu)
            reward = self.trex(z)
            sum_rewards = torch.sum(reward)
            sum_abs_rewards = torch.sum(torch.abs(reward))
            return sum_rewards, sum_abs_rewards, mu, var, z

        reward = self.trex(mu)
        sum_rewards = torch.sum(reward)
        sum_abs_rewards = torch.sum(torch.abs(reward))
        return sum_rewards, sum_abs_rewards, mu
    
    def estimate_temporal_difference(self, z1, z2):
        return self.temporal_difference_M(torch.cat((z1, z2), 1))
    
    def forward_dynamics(self, z, actions):
        x = torch.cat((z, actions), dim=1)
        return self.forward_dynamics_M(x)

    def estimate_inverse_dynamics(self, z1, z2):
        x = torch.cat((z1, z2), 1)
        return self.inverse_dynamics_M(x)
    
    def decode(self, encoding):
        x = self.decoder(encoding)
        return x.permute(0, 2, 3, 1)

    def compare_trajs(self, traj_i, traj_j):
        """compute cumulative return for each trajectory and return logits"""
        cum_r_i, abs_r_i, mu1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2 = self.cum_return(traj_j)
        return (
            torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0),
            abs_r_i + abs_r_j,
            mu1,
            mu2,
        )
        
    def state_features(self, traj):
        with torch.no_grad():
            traj = traj.permute(0, 3, 1, 2)
            x = self.cnn_embedding(traj)
            mu = self.fc_mu(x)
            print(mu.shape)
        return torch.sum(mu, dim=0)

    def state_feature(self, obs):
        with torch.no_grad():
            x = obs.permute(0, 3, 1, 2)
            x = self.cnn_embedding(x)
            mu = self.fc_mu(x)
        return mu
    
if __name__ == '__main__':
    device = torch.device('cuda:0')
    traj = torch.randn(20, 84, 84, 4).to(device)
    net = RewardNet(64, 4, True, device).to(device)

import torch
from torch import Tensor, nn


# TODO documentation
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def cum_return(self, traj: Tensor) -> Tensor:
        x = self.get_embedding(traj)
        rewards = self.fc2(x)
        # sum rewards at each state to get total return
        return torch.sum(rewards)
    
    def step_reward(self, state_action: Tensor) -> Tensor:
        x = self.get_embedding(state_action)
        rewards = self.fc2(x)
        return rewards.squeeze(0)

    def forward(self, traj_i: Tensor, traj_j: Tensor) -> Tensor:
        return torch.hstack((self.cum_return(traj_i), self.cum_return(traj_j)))

    def get_embedding(self, traj: Tensor) -> Tensor:
        embedding = self.act_fn(self.fc1(traj))
        return embedding

    def sum_embeddings(self, traj: Tensor) -> Tensor:
        with torch.no_grad():
            embeddings = self.get_embedding(traj)
            return torch.sum(embeddings, dim=0)

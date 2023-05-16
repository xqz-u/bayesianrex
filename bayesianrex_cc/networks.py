import torch
from torch import Tensor, nn


# TODO documentation
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def cum_return(self, traj: Tensor) -> Tensor:
        x = self.act_fn(self.fc1(traj))
        rewards = self.fc2(x)
        # sum rewards at each state to get total return
        return torch.sum(rewards)

    def forward(self, traj_i: Tensor, traj_j: Tensor) -> Tensor:
        return torch.hstack((self.cum_return(traj_i), self.cum_return(traj_j)))

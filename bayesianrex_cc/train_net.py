import joblib
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import wandb
import time

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, hidden_dim)
        self.act_fn = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, traj_i, traj_j):
        cum_return_i = self.cum_return(traj_i)
        cum_return_j = self.cum_return(traj_j)
        return torch.cat((cum_return_i.unsqueeze(0), cum_return_j.unsqueeze(0)), 0)
    
    def cum_return(self, traj):
        x = self.act_fn(self.fc1(traj))
        rewards = self.fc2(x)
        return torch.sum(rewards)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    data = joblib.load('./train_data/train_data_cartpole.gz')

    ## 60100 pairs of trajectories --> [60100 x (traj_i, traj_j)]
    ## each traj has [500, 4] states [500] actions
    ## each traj_i, traj_j pair has a label 1 or 0 indicating which is the better trajectory
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    reward_net = MLP(64, 1).to(device)
    trex_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_net.parameters())

    ## Wandb
    wandb.init(project='cartpole-mlp', entity='bayesianrex-dl2')

    start = time.time()
    train_data = []    
    for i in range(len(data['states'])):
        states_i, states_j = data['states'][i]
        actions_i, actions_j = data['actions'][i]
        actions_i, actions_j = np.expand_dims(actions_i, -1), np.expand_dims(actions_j, -1)

        sa_i, sa_j = np.hstack([states_i, actions_i]), np.hstack([states_j, actions_j]) # [N x 5]
        sa_i, sa_j = torch.from_numpy(sa_i).to(device), torch.from_numpy(sa_j).to(device)
        label = torch.from_numpy(data['labels'][i]).to(device)

        train_data.append((sa_i, sa_j, label))

    epochs = 1  # this is what the do in the original repo as well and it seems to work fine
    # loss is nearly constant already halfway through the epoch.
    for _ in range(epochs):
        # epoch_loss = 0
        for sa_i, sa_j, label in tqdm(train_data):
            reward_net.train()
            optimizer.zero_grad()

            outputs = reward_net(sa_i, sa_j)
            loss = trex_criterion(outputs, label.long())

            loss.backward()
            optimizer.step()

            # epoch_loss += loss.item()
            wandb.log({'loss': loss.item()})
    
    ## Save model
    torch.save(reward_net.state_dict(), '../reward_net.params')


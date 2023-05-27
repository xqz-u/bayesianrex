import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
import torch
import torch.nn as nn

class CustomMAPRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.env = env
        self.reward_model = reward_model
        
    def step(self, action):

        next_state, reward, done, truncated, info = self.env.step(action)
        obs = torch.cat((torch.Tensor(self.env.state), torch.Tensor([action])))
        custom_reward, _, _, _, _ = self.reward_model.cum_reward(obs)

        return next_state, custom_reward.item(), done, truncated, info
    
class CustomMeanRewardWrapper(gym.Wrapper):
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
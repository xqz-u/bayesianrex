import gymnasium as gym
import numpy as np
import torch

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.env = env
        self.reward_model = reward_model
        
    def step(self, action):

        next_state, reward, done, truncated, info = self.env.step(action)
        obs = torch.cat((torch.Tensor(self.env.state), torch.Tensor([action])))
        custom_reward, _, _, _, _ = self.reward_model.cum_reward(obs)

        return next_state, custom_reward.item(), done, truncated, info
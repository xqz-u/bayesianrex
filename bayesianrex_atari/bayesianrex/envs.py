import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

import constants
import networks

logger = logging.getLogger(__name__)


# NOTE grayscale image normalization (/ 255) occurs in ActorCriticCnnPolicy.extract_features()
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


def create_atari_env(
    env_id: str, n_envs: int = 1, seed: int = None, wrapper_kwargs: dict = None
) -> VecFrameStack:
    logger.debug("Env: %s", env_id)
    # default atari wrapper kwargs found in B-REX
    wrapper_kwargs = wrapper_kwargs or {
        "clip_reward": False,
        "terminal_on_life_loss": False,
    }
    return VecFrameStack(
        make_atari_env(env_id, n_envs=n_envs, seed=seed, wrapper_kwargs=wrapper_kwargs),
        n_stack=4,
    )


def create_hidden_lives_atari_env(
    env_name: str,
    n_envs: int = 1,
    seed: int = None,
    clip_reward: bool = False,
    terminal_on_life_loss: bool = False,
) -> VecFrameStack:
    env_id = constants.envs_id_mapper.get(env_name)
    logger.info("Env name %s Env id %s", env_name, env_id)
    return VecFrameStack(
        make_vec_env(
            env_id,
            n_envs=n_envs,
            seed=seed,
            wrapper_class=lambda env, **_: TransformObservation(
                AtariWrapper(
                    env,
                    clip_reward=clip_reward,
                    terminal_on_life_loss=terminal_on_life_loss,
                ),
                lambda obs: mask_score(obs, env_name),
            ),
        ),
        n_stack=4,
    )


class VecMCMCMAPAtariReward(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        reward_net_path: Union[Path, str],
        embedding_dim: int,
        env_name: str,
    ):
        super().__init(venv)
        self.reward_net = networks.EmbeddingNet(embedding_dim)
        self.reward_net.load_state_dict(torch.load(reward_net_path, map_location="cpu"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.0
        # self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # obs shape: [num_env,84,84,4] in case of atari games
        # plt.subplot(1,2,1)
        # plt.imshow(obs[0][:,:,0])
        # crop off top of image
        # n = 10
        # no_score_obs = copy.deepcopy(obs)
        # obs[:,:n,:,:] = 0

        # Need to normalize for my reward function
        # normed_obs = obs / 255.0
        # mask and normalize for input to network
        # NOTE my comment
        # normed_obs = preprocess(obs, self.env_name)
        # plt.subplot(1,2,2)
        # plt.imshow(normed_obs[0][:,:,0])
        # plt.show()
        # print(traj[0][0][40:60,:,:])

        with torch.no_grad():
            learned_rewards = self.reward_net().float().to(self.device).numpy()
            # learned_rewards = (
            #     self.reward_net(torch.from_numpy(np.array(obs)).float().to(self.device))
            #     # .cpu()
            #     .numpy().squeeze()
            # )
        print(learned_rewards.shape)

        return obs, learned_rewards, news, infos

    def reset(self, **kwargs) -> VecEnvObs:
        return self.venv.reset()


# # TODO: need to test with RL
# class VecMCMCMeanAtariReward(VecEnvWrapper):
#     def __init__(
#         self, venv, pretrained_reward_net_path, chain_path, embedding_dim, env_name
#     ):
#         VecEnvWrapper.__init__(self, venv)
#         self.reward_net = EmbeddingNet(embedding_dim)
#         # load the pretrained weights
#         self.reward_net.load_state_dict(
#             torch.load(pretrained_reward_net_path, map_location=torch.device("cpu"))
#         )
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         # load the mean of the MCMC chain
#         burn = 5000
#         skip = 20
#         reader = open(chain_path)
#         data = []
#         for line in reader:
#             parsed = line.strip().split(",")
#             np_line = []
#             for s in parsed[:-1]:
#                 np_line.append(float(s))
#             data.append(np_line)
#         data = np.array(data)
#         # print(data[burn::skip,:].shape)

#         # get average across chain and use it as the last layer in the network
#         mean_weight = np.mean(data[burn::skip, :], axis=0)
#         # print("mean weights", mean_weight[:-1])
#         # print("mean bias", mean_weight[-1])
#         # print(mean_weight.shape)
#         self.reward_net.fc2 = nn.Linear(
#             embedding_dim, 1, bias=False
#         )  # last layer just outputs the scalar reward = w^T \phi(s)

#         new_linear = torch.from_numpy(mean_weight)
#         print("new linear", new_linear)
#         print(new_linear.size())
#         with torch.no_grad():
#             # unsqueeze since nn.Linear wants a 2-d tensor for weights
#             new_linear = new_linear.unsqueeze(0)
#             # print("new linear", new_linear)
#             # print("new bias", new_bias)
#             with torch.no_grad():
#                 # print(last_layer.weight)
#                 # print(last_layer.bias)
#                 # print(last_layer.weight.data)
#                 # print(last_layer.bias.data)
#                 self.reward_net.fc2.weight.data = new_linear.float().to(self.device)

#             # TODO: print out last layer to make sure it stuck...
#             print("USING MEAN WEIGHTS FROM MCMC")
#             # with torch.no_grad():
#             #    for param in self.reward_net.fc2.parameters():
#             #        print(param)

#         self.reward_net.to(self.device)

#         self.rew_rms = RunningMeanStd(shape=())
#         self.epsilon = 1e-8
#         self.cliprew = 10.0
#         self.env_name = env_name

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         # obs shape: [num_env,84,84,4] in case of atari games
#         # plt.subplot(1,2,1)
#         # plt.imshow(obs[0][:,:,0])
#         # crop off top of image
#         # n = 10
#         # no_score_obs = copy.deepcopy(obs)
#         # obs[:,:n,:,:] = 0

#         # Need to normalize for my reward function
#         # normed_obs = obs / 255.0
#         # mask and normalize for input to network
#         normed_obs = preprocess(obs, self.env_name)
#         # plt.subplot(1,2,2)
#         # plt.imshow(normed_obs[0][:,:,0])
#         # plt.show()
#         # print(traj[0][0][40:60,:,:])

#         with torch.no_grad():
#             rews_network = (
#                 self.reward_net.forward(
#                     torch.from_numpy(np.array(normed_obs)).float().to(self.device)
#                 )
#                 .cpu()
#                 .numpy()
#                 .squeeze()
#             )

#         return obs, rews_network, news, infos

#     def reset(self, **kwargs):
#         obs = self.venv.reset()

#         ##############
#         # If the reward is based on LSTM or something, then please reset internal state here.
#         ##############

#         return obs

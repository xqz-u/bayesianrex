import stable_baselines3 as sb3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import envs
import time
import torch
import os
import torch.nn as nn
import numpy as np

def generate_demos():
    vec_env = make_vec_env("CartPole-v1", n_envs=1)
    agent = PPO("MlpPolicy", vec_env, verbose=1)
    checkpoints = list(os.walk('../cartpole_models'))[0][2]
    episode_cnt = 1

    demonstrations, learning_returns, learning_rewards = [], [], []
    for checkpoint in checkpoints:
        agent.load(f'../cartpole_models/{checkpoint[:-4]}')

        for i in range(episode_cnt):
            done = False
            traj, actions, gt_rewards = [], [], []            
            r, steps, acc_reward, frameno = 0, 0, 0, 0

            obs = vec_env.reset()
            while True:  
                action, _ = agent.predict(obs)
                ob, r, done, _ = vec_env.step(action)
                # print(r)
                # quit(1)

                traj.append(ob)
                actions.append(action)
                frameno += 1
                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]

                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break

            print(f"Traj length {len(traj)}, demo length: {len(demonstrations)}")
            demonstrations.append([traj, actions])
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    sorted_returns = sorted(learning_returns)

    ## TODO change the demonstrations to instead of N state,action pairs become M trajectories of same length
    ## TODO train a 2layer MLP using trex-loss
    ## TODO freeze first layer, retrain second layer using MCMC 
    ## TODO train RL agent with custom reward function from prev step
    ## succes

    return demonstrations, sorted_returns, learning_rewards   

def train_cartpole():
    # Parallel environments
    vec_env = make_vec_env("CartPole-v1", n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1)

    ckpt_callback = CheckpointCallback(
        save_freq=1000,
        save_path='../cartpole_models',
        name_prefix="PPO",
    )

    model.learn(
        total_timesteps=25000,
        callback=ckpt_callback,
        progress_bar=True,
        )
    
def mcmc_map(demos, sorted_returns, learning_rewards, device):
    pairwise_prefs = []
    for i in range(len(demonstrations)):
        for j in range(i+1, len(demonstrations)):
            if sorted_returns[i] < sorted_returns[j]:
                pairwise_prefs.append((i,j))
            else: # they are equal
                print("not using equal prefs", i, j, sorted_returns[i], sorted_returns[j])

    # demos = [state (4), actions (1)]
    # print(len(demos))       ## total number of demonstrations 
    # print(len(demos[0]))    ## [state, actions]
    # print(demos[0][0], demos[0][1])  ## trajectories and actions for 1 episode

    reshaped = []
    for traj, actions in demos:
        for state, action in zip(traj, actions):
            reshaped.append(np.hstack([state[0], action]))
    reshaped = np.array(reshaped)

    weights = nn.Linear(in_features=5, out_features=1).to(device)

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence=1):
    '''use (i,j) indices and precomputed feature counts to do p pairwise ranking loss'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)
    #don't need any gradients
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        #print(list(reward_net.fc2.parameters()))
        linear = last_layer.weight.data  #not using bias
        #print(linear)
        #print(bias)
        weights = linear.squeeze() #append bias and weights from last fc layer together
        #print('weights',weights)
        #print('demo_cnts', demo_cnts)
        demo_returns = confidence * torch.mv(demo_cnts, weights)

        #positivity prior
        if demo_returns[0] < 0.0:
            return torch.Tensor([-float("Inf")])


        loss_criterion = nn.CrossEntropyLoss(reduction='sum') #sum up losses
        cum_log_likelihood = 0.0
        outputs = torch.zeros(len(pairwise_prefs),2) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()

        #outputs = outputs.unsqueeze(0)
        #print(outputs)
        #print(labels)
        cum_log_likelihood = -loss_criterion(outputs, labels)
            #if labels == 0:
            #    log_likelihood = torch.log(return_i/(return_i + return_j))
            #else:
            #    log_likelihood = torch.log(return_j/(return_i + return_j))
            #print("ll",log_likelihood)
            #cum_log_likelihood += log_likelihood
    return cum_log_likelihood



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    demonstrations, sorted_returns, learning_rewards = generate_demos()
    mcmc_map(demonstrations, sorted_returns, learning_rewards, device)


# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")



# env = envs.create_hidden_lives_atari_env("breakout", seed=0)
# eval_env = envs.create_atari_env('BreakoutNoFrameskip-v4', seed=0)
# map_env = envs.VecMCMCMAPAtariReward(env, "../mcmc_data/breakout_map.params", 64, 'breakout')

# log_path = '../rl_logs/'
# logger = configure(log_path, ['stdout', 'tensorboard'])
# model = PPO(policy="MlpPolicy", env=map_env, verbose=1, device=device)
# eval_callback = EvalCallback(eval_env, best_model_save_path='../best_rl_models/', log_path='../rl_logs', eval_freq=10000, n_eval_episodes=1)

# model.set_logger(logger)
# model = model.learn(total_timesteps=10000, progress_bar=True, callback=eval_callback)
# print(evaluate_policy(model, eval_env, n_eval_episodes=1))


# # <stable_baselines3.common.vec_env.vec_transpose.VecTransposeImage object at 0x000002299CF62490>
# # <envs.VecMCMCMAPAtariReward object at 0x000002299D6F3A10>

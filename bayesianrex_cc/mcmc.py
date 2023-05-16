from networks import MLP
import torch.nn as nn
import torch
import joblib
import numpy as np
import copy
from tqdm import tqdm

def generate_feature_embeddings(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.fc2.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.sum_embeddings(traj).squeeze().float().to(device)
    return feature_cnts.to(device)

def compute_l2(linear):
    with torch.no_grad():
        weights = linear.cpu().numpy()
    return np.linalg.norm(weights) 

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_embeds, criterion, confidence=1):
    with torch.no_grad():
        weights = last_layer.weight.data.squeeze()
        demo_returns = confidence * torch.mv(demo_embeds, weights)

        ## NOTE reinstate this once we know the demonstrations are correct
        # # positivity prior
        # if demo_returns[0] < 0.0:
        #     return torch.Tensor([-float('Inf')])

        outputs = demo_returns[pairwise_prefs.long()]
        labels = torch.ones(len(pairwise_prefs)).long().to(device)

        return -criterion(outputs, labels)

def mcmc_map_search(reward_net, pairwise_prefs, demo_embeds, num_steps, step_size, device):
    last_layer = reward_net.fc2

    with torch.no_grad():
        linear = last_layer.weight.data
        linear.add_(torch.randn(linear.size()).to(device) * step_size)
        linear = last_layer.weight.data
        l2_norm = np.array([compute_l2(linear)])
        linear.div_(torch.from_numpy(l2_norm).float().to(device))

    loglik_loss = nn.CrossEntropyLoss(reduction='sum')
    starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_embeds, loglik_loss)

    map_loglik, cur_loglik = starting_loglik, starting_loglik
    map_reward, cur_reward = copy.deepcopy(reward_net.fc2), copy.deepcopy(reward_net.fc2)

    reject_cnt, accept_cnt = 0, 0

    for i in tqdm(range(num_steps)):
        # take proposal step
        proposal_reward = copy.deepcopy(cur_reward)

        # add random noise
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.add_(torch.randn(param.size()).to(device) * step_size)
        l2_norm = np.array([compute_l2(proposal_reward.weight.data)])
        #normalize the weight vector...
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.div_(torch.from_numpy(l2_norm).float().to(device))

        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, pairwise_prefs, demo_embeds, loglik_loss)
        if prop_loglik > cur_loglik:
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)
                print("proposal loglik", prop_loglik.item())
                print("updating map to ", prop_loglik)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < torch.exp(prop_loglik - cur_loglik).item():
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward


if __name__ == '__main__':
    # Load pretrained model
    hidden_dim = 64
    pretrained = torch.load('../reward_net.params')
    reward_net = MLP(5, hidden_dim)
    reward_net.load_state_dict(pretrained)

    # Re-initialize last layer
    reward_net.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    reward_net.to(device)

    # Freeze all parameters
    for param in reward_net.parameters():
        param.requires_grad = False

    trajectories = joblib.load('./data/trajectories_cartpole.gz')

    demonstrations, learning_returns = [], []
    for i in range(len(trajectories['states'])):
        states, actions = trajectories['states'][i], trajectories['actions'][i]
        rewards = sum(trajectories['rewards'][i])
        state_action = torch.hstack((torch.from_numpy(states), torch.from_numpy(actions[..., None])))

        demonstrations.append(state_action)
        learning_returns.append(rewards)

    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    sorted_returns = sorted(learning_returns)

    # Get the summed feature embeddings
    demo_embed = generate_feature_embeddings(demonstrations, reward_net)

    ## Create pairwise preferences (as in old codebase)
    ## the returns are all the same, namely 500. so we cant really sort and everything is the same. sooo i think we need more varied demonstrations.

    # for now, i just add everything so i know the code runs wrt shapes and stuff :) 
    pairwise_prefs = []
    for i in range(len(demonstrations)):
        for j in range(i+1, len(demonstrations)):
            ## temporary solution:
            pairwise_prefs.append((i,j))
            ## original:
            # if sorted_returns[i] < sorted_returns[j]:
            #     pairwise_prefs.append((i,j))
            # else: # they are equal
            #     print("not using equal prefs", i, j, sorted_returns[i], sorted_returns[j])
    pairwise_prefs = torch.Tensor(pairwise_prefs)

    step_size, num_mcmc_steps = 5e-3, 10#2e5
    mcmc_map = mcmc_map_search(reward_net, pairwise_prefs, demo_embed, num_mcmc_steps, step_size, device)

    reward_net = MLP(5, hidden_dim)
    reward_net.load_state_dict(pretrained)
    reward_net.fc2 = mcmc_map

    torch.save(reward_net.state_dict(), '../mcmc_net.params')
    print('Succesfully saved MAP estimate')
    # ## we need some unseen data for this, so either generate more 
    # ## or split the data into like 80% train and 20% mcmc?


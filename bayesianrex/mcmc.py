import random

import torch.nn as nn
import torch
import joblib
import numpy as np
import copy
from tqdm import tqdm
import argparse

from bayesianrex import config, constants, environments, utils
from bayesianrex.networks import RewardNetwork
from bayesianrex.dataset import generate_demonstrations as gen_demos

def generate_feature_embeddings(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.trex.in_features) #no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        #print(len(trajectory))
        feature_cnts[i,:] = reward_net.state_features(traj).squeeze().float().to(device)
    return feature_cnts.to(device)

def print_traj_returns(reward_net, demonstrations):
    #print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

def predict_traj_return(net, traj):
    traj = torch.from_numpy(traj).float().to(net.device)
    return round(net.cum_return(traj)[0].item(), 2)

def compute_l2(linear):
    with torch.no_grad():
        weights = linear.cpu().numpy()
    return np.linalg.norm(weights) 

def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_embeds, criterion, confidence=1):
    with torch.no_grad():
        weights = last_layer.weight.data.squeeze()
        demo_returns = confidence * torch.mv(demo_embeds, weights)

        # # positivity prior
        # if demo_returns[0] < 0.0:
        #     return torch.Tensor([-float('Inf')])

        outputs = demo_returns[pairwise_prefs.long()]
        labels = torch.ones(len(pairwise_prefs)).long().to(device)

        return -criterion(outputs, labels)

def dcg_at_k(sorted_labels, k):
    if k > 0:
        k = min(sorted_labels.shape[0], k)
    else:
        k = sorted_labels.shape[0]
    denom = 1.0 / np.log2(np.arange(k) + 2.0)
    nom = 2 ** sorted_labels - 1.0
    dcg = np.sum(nom[:k] * denom)
    return dcg

def listwise_loss(predictions: list[nn.Tensor], labels: list[int]):
    N = len(labels)
    # obtain positional indexes based on predicted rewards
    # e.g. [13, 0, 14, 5] -> [1, 3, 0, 2].
    predicted_order = np.argsort(predictions)
    ideal_DCG = dcg_at_k(labels, N)
    NDCG_OG = dcg_at_k(predicted_order, N) / ideal_DCG

    # Compute lambda_ij
    si_sj = (predictions - predictions.T)
    Sij = torch.sign(labels - labels.T)
    lambda_ij = (1 - Sij) / 2 - 1 / (1 + torch.exp((si_sj)))

    arange_vector = torch.arange(N)
    index = torch.vstack((torch.repeat_interleave(arange_vector, N), arange_vector.repeat(N))).T

    swap = torch.max(index, dim=1)[0]
    all_indices = torch.arange(N).repeat(N * N).reshape((N * N, N))
    all_indices[range(N * N), torch.max(index, dim=1)[0]] = torch.min(index, dim=1)[0]
    all_indices[range(N * N), torch.min(index, dim=1)[0]] = swap

    swapped_scores = scores[all_indices].squeeze()  # Each row is a swapped score
    NDCG_swapped = vectorized_ndcg(labels, swapped_scores)
    NDCG_changed = np.abs(NDCG_swapped - NDCG_OG).reshape((N, N))

    delta = np.abs(NDCG_changed - 1)
    loss = lambda_ij * delta
    return loss.sum(dim=1).view(-1, 1)

def calc_linearized_listwise_ranking_loss(last_layer, listwise_prefs, demo_embeds, criterion, confidence=1):
    with torch.no_grad():
        weights = last_layer.weight.data.squeeze()
        demo_returns = confidence * torch.mv(demo_embeds, weights)

        # # positivity prior
        # if demo_returns[0] < 0.0:
        #     return torch.Tensor([-float('Inf')])

        outputs = demo_returns[pairwise_prefs.long()]
        labels = torch.ones(len(pairwise_prefs)).long().to(device)

        return -criterion(outputs, labels)

def mcmc_map_search(reward_net, preferences, demo_embeds, num_steps, step_size, weight_output_filename, device, pairwise=True):
    last_layer = reward_net.trex

    writer = open(weight_output_filename,'w')

    with torch.no_grad():
        linear = last_layer.weight.data
        linear.add_(torch.randn(linear.size()).to(device) * step_size)
        linear = last_layer.weight.data
        l2_norm = np.array([compute_l2(linear)])
        linear.div_(torch.from_numpy(l2_norm).float().to(device))

    loglik_loss = nn.CrossEntropyLoss(reduction='sum')
    starting_loglik = 0
    if pairwise:
        starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, preferences, demo_embeds, loglik_loss)
    else:
        starting_loglik = calc_linearized_listwise_ranking_loss(last_layer, preferences, demo_embeds, loglik_loss)

    map_loglik, cur_loglik = starting_loglik, starting_loglik
    map_reward, cur_reward = copy.deepcopy(reward_net.trex), copy.deepcopy(reward_net.trex)

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

        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, preferences, demo_embeds, loglik_loss)
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
        
        write_weights_likelihood(cur_reward, cur_loglik, writer)

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    writer.close()
    return map_reward

def write_weights_likelihood(last_layer, loglik, file_writer):
    #convert last layer to numpy array
    np_weights = get_weight_vector(last_layer)
    for w in np_weights:
        file_writer.write(str(w)+",")
    file_writer.write(str(loglik.item()) + "\n")

def get_weight_vector(last_layer):
    '''take fc2 layer and return numpy array of weights and bias'''
    linear = last_layer.weight.data
    with torch.no_grad():
        weights = linear.squeeze().cpu().numpy()
    return weights

if __name__ == '__main__':
    parser_conf = {
        "pretrained_model_path": {
            "type": str,
            "help": 'where to find the pretrained embedding network'
        },
        "encoding_dim": {
            "type": int,
            "default": 64,
            "help": "dimension of latent space"
        },
        "weight_output_path" : {
            "type": str,
            "help": "where to save the mcmc chain"
        },
        "mcmc_net_output_path" : {
            "type": str,
            "help": "where to save the network with mcmc estimate of final layer"
        },
        "listwise" : {
            "type": int,
            "default": 0,
            "help": "select integer N for learning on N-wise preferences. anything below 2 results in pairwise."
        },
        **gen_demos.parser_conf
    }
    p = utils.define_cl_parser(parser_conf)
    args = p.parse_args()

    pairwise = True if args.listwise <= 2 else False

    device = utils.torch_device()

    # Load pretrained model
    n_actions = environments.create_atari_env(
        constants.envs_id_mapper.get(args.env)
    ).action_space.n

    pretrained = torch.load(args.pretrained_model_path)
    reward_net = RewardNetwork(args.encoding_dim, n_actions, device)
    reward_net.load_state_dict(pretrained)

    # Re-initialize last layer
    reward_net.trex = nn.Linear(args.encoding_dim, 1, bias=False)

    reward_net.to(device)

    # Freeze all parameters
    for param in reward_net.parameters():
        param.requires_grad = False

    ## TODO rewrite this to the new version
    states, actions, rewards = list(joblib.load('./train-data/trajectories_breakout').values())
    returns = [sum(r) for r in rewards]

    demonstrations = [s for s, _ in sorted(zip(states, returns), key= lambda pair: pair[1])]
    sorted_returns = sorted(returns)

    # Get the summed feature embeddings
    demo_embed = generate_feature_embeddings(demonstrations, reward_net)

    # prepare data for pairwise
    pairwise_prefs = []
    listwise_prefs = []
    if pairwise:
        ## Create pairwise preferences (as in old codebase)
        for i in range(len(demonstrations)):
            for j in range(i+1, len(demonstrations)):
                if sorted_returns[i] < sorted_returns[j]:
                    pairwise_prefs.append((i,j))
                else: # they are equal
                    print("using equal prefs", i, j, sorted_returns[i], sorted_returns[j])
        pairwise_prefs = torch.Tensor(pairwise_prefs)
    # prepare listwise learning data
    else:
        for i in range(len(demonstrations)):
            # randomly select args.listwise number of preferences to comprise list
            selection = random.sample(range(i, len(demonstrations)), args.listwise)
            # sort selected preferences based on their returns
            sel_returns = [sorted_returns[j] for j in selection]
            sorted_sel = [idx for _, idx in sorted(zip(sel_returns, selection))]
            # add new list of preferences to dataset
            listwise_prefs.append(sorted_sel)
        listwise_prefs = torch.Tensor(listwise_prefs)


    step_size, num_mcmc_steps = 5e-3, int(2e5)
    if pairwise:
        mcmc_map = mcmc_map_search(reward_net, pairwise_prefs, demo_embed, num_mcmc_steps, step_size, args.weight_output_path, device)
    else:
        mcmc_map = mcmc_map_search(reward_net, listwise_prefs, demo_embed, num_mcmc_steps, step_size, args.weight_output_path, device)

    reward_net = RewardNetwork(args.encoding_dim, n_actions, device).to(device)
    reward_net.load_state_dict(pretrained)
    reward_net.trex = mcmc_map

    torch.save(reward_net.state_dict(), args.mcmc_net_output_path)
    print('Succesfully saved MAP estimate')

    print_traj_returns(reward_net, demonstrations)

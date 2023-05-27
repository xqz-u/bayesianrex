import copy
import logging
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from bayesianrex import constants, environments, utils
from bayesianrex.dataset import generate_demonstrations as gen_demos
from bayesianrex.models.reward_model import RewardNetwork
from bayesianrex.models.utils import load_reward_network

logger = logging.getLogger(__name__)
device = utils.torch_device()


def generate_feature_embeddings(demos, reward_net):
    feature_cnts = torch.zeros(len(demos), reward_net.trex.in_features)  # no bias
    for i in range(len(demos)):
        traj = np.array(demos[i])
        traj = torch.from_numpy(traj).float().to(device)
        # print(len(trajectory))
        feature_cnts[i, :] = (
            reward_net.state_features(traj).squeeze().float().to(device)
        )
    return feature_cnts.to(device)


def print_traj_returns(reward_net, demonstrations):
    # print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [
            predict_traj_return(reward_net, traj) for traj in demonstrations
        ]
    for i, p in enumerate(pred_returns):
        print(i, p, sorted_returns[i])


def predict_traj_return(net, traj):
    traj = torch.from_numpy(traj).float().to(net.device)
    return round(net.cum_return(traj)[0].item(), 2)


def compute_l2(linear):
    with torch.no_grad():
        weights = linear.cpu().numpy()
    return np.linalg.norm(weights)


def calc_linearized_pairwise_ranking_loss(
    last_layer, pairwise_prefs, demo_embeds, criterion, confidence=1
):
    with torch.no_grad():
        weights = last_layer.weight.data.squeeze()
        demo_returns = confidence * torch.mv(demo_embeds, weights)

        # # positivity prior
        # if demo_returns[0] < 0.0:
        #     return torch.Tensor([-float('Inf')])

        outputs = demo_returns[pairwise_prefs.long()]
        labels = torch.ones(len(pairwise_prefs)).long().to(device)

        return -criterion(outputs, labels)


def mcmc_map_search(
    reward_net, pairwise_prefs, demo_embeds, num_steps, step_size, device
):
    last_layer = reward_net.trex

    with torch.no_grad():
        linear = last_layer.weight.data
        linear.add_(torch.randn(linear.size()).to(device) * step_size)
        linear = last_layer.weight.data
        l2_norm = np.array([compute_l2(linear)])
        linear.div_(torch.from_numpy(l2_norm).float().to(device))

    loglik_loss = nn.CrossEntropyLoss(reduction="sum")
    starting_loglik = calc_linearized_pairwise_ranking_loss(
        last_layer, pairwise_prefs, demo_embeds, loglik_loss
    )

    map_loglik, cur_loglik = starting_loglik, starting_loglik
    map_reward, cur_reward = copy.deepcopy(reward_net.trex), copy.deepcopy(
        reward_net.trex
    )

    reject_cnt, accept_cnt = 0, 0

    for i in tqdm(range(num_steps)):
        # take proposal step
        proposal_reward = copy.deepcopy(cur_reward)

        # add random noise
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.add_(torch.randn(param.size()).to(device) * step_size)
        l2_norm = np.array([compute_l2(proposal_reward.weight.data)])
        # normalize the weight vector...
        with torch.no_grad():
            for param in proposal_reward.parameters():
                param.div_(torch.from_numpy(l2_norm).float().to(device))

        prop_loglik = calc_linearized_pairwise_ranking_loss(
            proposal_reward, pairwise_prefs, demo_embeds, loglik_loss
        )
        if prop_loglik > cur_loglik:
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            # check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)
                print("proposal loglik", prop_loglik.item())
                print("updating map to ", prop_loglik)
        else:
            # accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < torch.exp(prop_loglik - cur_loglik).item():
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                # reject and stick with cur_reward
                reject_cnt += 1

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward


def prepare_linear_comb_network(args: Namespace) -> RewardNetwork:
    encoding_dims = args.encoding_dims
    # load pretrained reward model
    reward_net = load_reward_network(
        args.pretrained_model_path,
        args.env,
        encoding_dims=encoding_dims,
        device=device,
    )
    logger.info("Initialize trex layer and unset all requires_grad")
    # re-initialize last layer
    reward_net.trex = nn.Linear(encoding_dims, 1, bias=False)
    # freeze all parameters
    for param in reward_net.parameters():
        param.requires_grad = False
    return reward_net.to(device)


if __name__ == "__main__":
    parser_conf = {
        "pretrained-model-path": {
            "type": Path,
            "help": "where to find the pretrained embedding network",
        },
        "encoding-dims": {
            "type": int,
            "default": constants.reward_net_latent_space,
            "help": "dimension of latent space",
        },
        **gen_demos.parser_conf,
    }
    p = utils.define_cl_parser(parser_conf)
    args = p.parse_args()

    reward_net = prepare_linear_comb_network(args)

    ## TODO rewrite this to the new version
    states, actions, rewards = list(
        joblib.load("./train-data/trajectories_breakout").values()
    )
    returns = [sum(r) for r in rewards]

    demonstrations = [
        s for s, _ in sorted(zip(states, returns), key=lambda pair: pair[1])
    ]
    sorted_returns = sorted(returns)

    # Get the summed feature embeddings
    demo_embed = generate_feature_embeddings(demonstrations, reward_net)

    ## Create pairwise preferences (as in old codebase)
    pairwise_prefs = []
    for i in range(len(demonstrations)):
        for j in range(i + 1, len(demonstrations)):
            if sorted_returns[i] < sorted_returns[j]:
                pairwise_prefs.append((i, j))
            else:  # they are equal
                print("using equal prefs", i, j, sorted_returns[i], sorted_returns[j])
    pairwise_prefs = torch.Tensor(pairwise_prefs)

    step_size, num_mcmc_steps = 5e-3, int(2e5)
    mcmc_map = mcmc_map_search(
        reward_net, pairwise_prefs, demo_embed, num_mcmc_steps, step_size, device
    )

    reward_net = RewardNetwork(args.encoding_dim, n_actions, device).to(device)
    reward_net.load_state_dict(pretrained)
    reward_net.trex = mcmc_map

    torch.save(reward_net.state_dict(), "../mcmc_net.params")
    print("Succesfully saved MAP estimate")

    print_traj_returns(reward_net, demonstrations)

import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.utils import set_random_seed
from torch import Tensor as T
from torch.nn import functional as F
from tqdm import tqdm

from bayesianrex import config, constants, utils
from bayesianrex.dataset import generate_demonstrations as gen_demos
from bayesianrex.models.reward_model import RewardNetwork
from bayesianrex.models.utils import load_reward_network

logger = logging.getLogger(__name__)
device = utils.torch_device()


def generate_feature_embeddings(traj_states: List[T], reward_net: RewardNetwork) -> T:
    """
    Compute embeddings of trajectory states using the reward network
    for each state (stack of frames) in a trajectory.

    :param traj_states: list of trajectories (list-type containing image-frame tensors)
    :param reward_net: a RewardNetwork network
    :return: tensor of embeddings for trajectory states computed by the network
    """

    # set up embeddings, no bias used
    feature_cnts = torch.zeros(
        len(traj_states), reward_net.trex.in_features, device=device
    )

    for i, traj in enumerate(traj_states):
        traj = traj.to(device)
        feature_cnts[i, :] = reward_net.trajectory_embedding(
            traj, sum_=True
        )  # sum over frames

    return feature_cnts


def predict_traj_return(reward_net: RewardNetwork, traj: T) -> float:
    """
    Compute the cumulative return over trajectory states with a reward_net.
    This function is mainly used with print_traj_returns to
    print out values, so the output is rounded to 2 significant digits.

    :param reward_net: a RewardNetwork network
    :param traj: single trajectory (list-type containing image-frame tensors)
    :return: rounded cumulative reward of trajectory
    """
    return round(reward_net.cum_return(traj.to(device))[0].item(), 2)


# print out predicted cumulative returns and actual returns
def print_traj_returns(reward_net: RewardNetwork, traj_states: List[T], returns_: T):
    """
    Compute the cumulative return over trajectory states with a reward_net and print it,
    alongside the returns computed by the real reward function.

    :param reward_net: a RewardNetwork network
    :param traj_states: trajectories (list-type containing image-frame tensors)
    :param returns_: tensor containing the cumulative returns of real reward function
    """
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in traj_states]
    for i, (estimated_ret, actual_ret) in enumerate(zip(pred_returns, returns_)):
        print(f"{i} predict {estimated_ret:.2f} actual {actual_ret.item():.2f}")


def linearized_pairwise_ranking_loss(
    layer: nn.Module, pairwise_prefs, states_embeddings, confidence: int = 1
) -> T:
    """
    Training loss of the MCMC-trained linearized neural net using pairwise ranking loss.
    The pairwise preferences are of the sort [(i,j),...] where trajectory j is preferred over i.

    :param layer: Linear neural network layer being trained
    :param pairwise_prefs: true trajectory pairwise preferences
    :param states_embeddings: stacked embeddings of trajectory states
    :param confidence: confidence in preference labels (value beta in eq 4,6)
    :return:

    """
    with torch.no_grad():
        weights = layer.weight.squeeze()
        # return as linear combination of state embeddings
        predicted_returns = confidence * torch.mv(states_embeddings, weights)
        # positivity prior from eq 10 from paper
        if predicted_returns[0] < 0.0:
            return torch.Tensor([-float("Inf")])
        # pick the returns of the trajectories for available preference information
        outputs = predicted_returns[pairwise_prefs]
        labels = torch.ones(len(pairwise_prefs), dtype=int, device=device)
        # eq. 6 in the B-Rex paper
        return -F.cross_entropy(outputs, labels, reduction="sum")


def inplace_MCMC_proposal(layer: nn.Module, mcmc_step_size: float):
    """
    Performs an in-place Markov Chain Monte Carlo (MCMC) proposal for a neural network layer.

    :param layer: `nn.Module` class instance (e.g. neural network layer)
    :param mcmc_step_size: The step size for the MCMC proposal_reward
    """
    with torch.no_grad():
        # layer.parameters() returns the pointers, so modifications are in-place
        for param in layer.parameters():
            param += torch.randn_like(param, device=device) * mcmc_step_size
            param /= torch.linalg.vector_norm(param)


def MCMC_MAP_search(
    trex_layer: nn.Linear,
    pairwise_prefs: List[Tuple[int, int]],
    states_embeddings: T,
    mcmc_steps: int,
    mcmc_step_size: float,
) -> Tuple[nn.Linear, np.ndarray, np.ndarray]:
    """
    Performs a MCMC search (Metropolis-Hastings) to find MAP reward function.
    The pairwise preferences are of the sort [(i,j),...] where trajectory j is preferred over i.

    :param trex_layer: initial reward function as an instance of `nn.Linear`
    :param pairwise_prefs: list of pairwise preferences
    :param states_embeddings: embeddings of the states of trajectories
    :param mcmc_steps: number of MCMC steps to perform
    :param mcmc_step_size: step size for the MCMC proposal

    :return: The MAP reward function, the MCMC chain and the proposal likelihoods.
    """

    # initialize the MAP reward function as the current proposal, together with
    # its likelihood
    map_reward, cur_reward = deepcopy(trex_layer), deepcopy(trex_layer)
    starting_loglik = linearized_pairwise_ranking_loss(
        cur_reward, pairwise_prefs, states_embeddings
    )
    logger.debug("Starting loglikelihood: %.2f", starting_loglik)
    map_loglik, cur_loglik = starting_loglik, starting_loglik

    mcmc_chain = np.zeros((mcmc_steps, trex_layer.in_features))
    mcmc_likelihoods = np.zeros(mcmc_steps)
    reject_cnt, accept_cnt = 0, 0

    for step in tqdm(range(mcmc_steps)):
        # update the MCMC chain and likelihoods
        with torch.no_grad():
            proposal_np = cur_reward.weight.cpu().numpy().squeeze()
        mcmc_chain[step] = proposal_np
        mcmc_likelihoods[step] = cur_loglik.item()
        # generate new proposal
        proposal_reward = deepcopy(cur_reward)
        inplace_MCMC_proposal(proposal_reward, mcmc_step_size)
        # compute proposal likelihood, given preferences and reward model
        prop_loglik = linearized_pairwise_ranking_loss(
            proposal_reward, pairwise_prefs, states_embeddings
        )

        if prop_loglik > cur_loglik:
            logger.debug(
                "Step %d: found better proposal, likelihood %.2f -> %.2f",
                step,
                prop_loglik.item(),
                cur_loglik.item(),
            )
            # update to proposed better reward model
            accept_cnt += 1
            cur_reward, cur_loglik = deepcopy(proposal_reward), prop_loglik
            if prop_loglik > map_loglik:
                # update MAP reward function
                logger.info(
                    "Step %d: update MAP reward fn, loglikelihood: %.2f -> %.2f",
                    step,
                    map_loglik,
                    prop_loglik.item(),
                )
                map_reward, map_loglik = deepcopy(proposal_reward), prop_loglik
        else:
            # accept proposal with prob exp(prop_loglik - cur_loglik)
            prob, threshold = torch.rand(()), torch.exp(prop_loglik - cur_loglik)
            logger.debug(
                "Accept prob %.2f threshold %.2f", prob.item(), threshold.item()
            )
            if prob < threshold:
                accept_cnt += 1
                cur_reward, cur_loglik = deepcopy(proposal_reward), prop_loglik
            else:
                # reject and stick with current reward function proposal
                reject_cnt += 1

    logger.info("# rejects: %d accepts: %d", reject_cnt, accept_cnt)
    return map_reward, mcmc_chain, mcmc_likelihoods


def prepare_linear_comb_network(args: Namespace) -> RewardNetwork:
    """
    Load pretrained reward network.

    :param args: namespace object including details about the network
    :return: pretrained reward network
    """
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
    logger.info(
        "Reward is linear combination of %d features", reward_net.trex.in_features
    )

    # freeze all parameters
    for param in reward_net.parameters():
        param.requires_grad = False

    return reward_net.to(device)


def triangular_preferences(returns_: T) -> T:
    """
    Create pairwise preferences as in old bayesianrex codebase

    :params returns_: true returns of trajectories
    :return: Pairwise preferences of the sort [(i,j),...] where
             trajectory j is preferred over i.
    """
    pairwise_prefs = []
    # NOTE assume that returns are sorted in increasing order
    for i in range(len(returns_)):
        for j in range(i + 1, len(returns_)):
            return_i, return_j = returns_[i], returns_[j]
            if return_i < return_j:
                pairwise_prefs.append((i, j))
            else:  # skip if equal preferences
                log_args = (i, return_i, j, return_j)
                logger.info("Skip equal preferences: %d %.2f, %d %.2f", *log_args)

    return torch.LongTensor(pairwise_prefs)


def main(args: Namespace):
    """
    Executes the main functionality of the program for training a reward model
    using MCMC and saving the MAP (Maximum A Posteriori) reward model.

    Pipeline:
        - Set the random seed if provided.
        - Prepare and initialize a linear combination reward network.
        - If `args.trajectories_path` is not provided, generate demonstrations using
          the checkpoints directory, environment, and other options.
        - Convert demonstrations to tensors and split into states, actions, and rewards.
        - Generate summed feature embeddings for the states using the reward network.
        - Generate preference labels based on the true cumulative returns.
        - Determine the MCMC chain path (where the parameters will be stored).
        - Obtain the MAP reward function using MCMC_MAP_search.
        - Save MAP reward model to file.
        - Print out the returns of trajectories using the MAP reward model.

    :params args: Namespace instance containing the command-line arguments and options.

    """
    # set seed in case it hasn't yet been
    if args.seed is not None:
        logger.info("`set_random_seed`, seed: %d", args.seed)
        set_random_seed(seed=args.seed)

    reward_net = prepare_linear_comb_network(args).to(device)

    if args.trajectories_path is None:
        demonstrations = gen_demos.generate_demonstrations_from_dir(
            args.checkpoints_dir,
            args.env,
            args.n_episodes,
            seed=args.seed,
            n_envs=args.n_envs,
        )
    else:
        # NOTE make sure that these demonstrations are sorted by increasing
        # order of return, or the whole logic does not apply
        demonstrations = gen_demos.load_trajectories(args.trajectories_path)
    demonstrations = utils.tensorify(demonstrations)
    # NOTE demonstrations use a lot of RAM, consider deleting them and re-loading
    # for print_traj_returns later
    states, _, rewards = demonstrations

    # get the summed feature embeddings
    states_embeddings = generate_feature_embeddings(states, reward_net)
    # generate preference labels
    returns_ = torch.Tensor([r.sum() for r in rewards])
    pairwise_prefs = triangular_preferences(returns_)

    map_reward_fn, mcmc_chain, mcmc_likelihoods = MCMC_MAP_search(
        reward_net.trex, pairwise_prefs, states_embeddings, int(2e5), 5e-3
    )
    mcmc_chain_path = (
        args.mcmc_chain_save_path or config.ASSETS_DIR / f"mcmc_chain_{args.env}.npz"
    )
    logger.info("Writing MCMC chain and likelihoods to %s", mcmc_chain_path)
    np.savez_compressed(mcmc_chain_path, chain=mcmc_chain, likelihoods=mcmc_likelihoods)

    # saved MAP reward model resulting from MCMC
    reward_net_mcmc = load_reward_network(
        args.pretrained_model_path,
        args.env,
        encoding_dims=args.encoding_dims,
        device=device,
    ).to(device)
    reward_net_mcmc.trex = map_reward_fn
    map_model_path = (
        args.map_model_save_path
        or config.ASSETS_DIR / f"reward_model_{args.env}_MAP.pth"
    )
    logger.info("Saving reward model MAP to %s", map_model_path)
    torch.save(reward_net_mcmc.state_dict(), map_model_path)

    print_traj_returns(reward_net_mcmc, states, returns_)


if __name__ == "__main__":
    invalid_parser_keys = (
        "n-traj",
        "n-snippets",
        "snippet-min-len",
        "snippet-max-len",
        "train-data-save-dir",
        # common key but different message here
        "trajectories-path",
    )
    common_parser = {
        k: v for k, v in gen_demos.parser_conf.items() if k not in invalid_parser_keys
    }
    parser_conf = {
        "pretrained-model-path": {
            "type": Path,
            "help": "where to find the pretrained reward embedding network",
        },
        "encoding-dims": {
            "type": int,
            "default": constants.reward_net_latent_space,
            "help": "dimension of latent space",
        },
        "mcmc-chain-save-path": {
            "type": Path,
            "help": (
                """where to save the MCMC chain, defaults to"""
                f""" {config.ASSETS_DIR / 'mcmc_chain_<env>.txt'}"""
            ),
        },
        "map-model-save-path": {
            "type": Path,
            "help": (
                """where to save the weights with the MAP reward fn,"""
                f""" defaults to {config.ASSETS_DIR / 'reward_model_<env>_MAP.pth'}"""
            ),
        },
        "trajectories-path": {
            "type": Path,
            "help": (
                """path to already generated trajectories (should be """
                """different than ones used to train reward embedding model)"""
            ),
        },
        **common_parser,
    }
    p = utils.define_cl_parser(parser_conf)
    args = p.parse_args()
    utils.setup_root_logging(args.log_level)

    # args.pretrained_model_path = config.ASSETS_DIR / "reward_model_breakout.pth"
    # args.trajectories_path = (
    #     config.TRAIN_DATA_DIR / "BreakoutNoFrameskip-v4" / "trajectories"
    # )
    # args.seed = 0

    main(args)

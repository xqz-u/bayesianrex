import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from networks import RewardNet

from stable_baselines3.common.type_aliases import GymEnv
from bayesianrex.dataset.generate_demonstrations import (
    create_training_data,
    generate_demonstrations,
)


def reconstruction_loss(decoded, target, mu, logvar):
    num_elements = decoded.numel()
    target_num_elements = target.numel()
    assert num_elements == target_num_elements

    bce = F.binary_cross_entropy(decoded, target)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= num_elements
    # print("bce: " + str(bce) + " kld: " + str(kld))
    return bce + kld


def compute_losses(reward_network, z, mu, logvar, traj, times, num_frames, actions_):
    inverse_dynamics_loss_fn = nn.CrossEntropyLoss()
    forward_dynamics_loss_fn = nn.MSELoss()
    temporal_difference_loss = nn.MSELoss()
    decoded = reward_network.decode(z)
    recon_loss = 10 * reconstruction_loss(decoded, traj, mu, logvar)
    t1, t2 = tuple(np.random.randint(low=0, high=len(times), size=2))
    est_dt = reward_network.estimate_temporal_difference(
        mu[t1].unsqueeze(0), mu[t2].unsqueeze(0)
    )
    real_dt = (times[t2] - times[t1]) / 100.0

    actions = reward_network.estimate_inverse_dynamics(mu[0:-1], mu[1:])
    target_actions = torch.LongTensor(actions_[1:]).to(device)

    inverse_dynamics_loss = inverse_dynamics_loss_fn(actions, target_actions) / 1.9
    forward_dynamics_distance = 5
    forward_dynamics_onehot_actions_1 = torch.zeros(
        (num_frames - 1, reward_network.ACTION_DIMS), dtype=torch.float32, device=device
    )
    forward_dynamics_onehot_actions_1.scatter_(1, target_actions.unsqueeze(1), 1.0)

    forward_dynamics = reward_network.forward_dynamics(
        mu[:-forward_dynamics_distance],
        forward_dynamics_onehot_actions_1[: (num_frames - forward_dynamics_distance)],
    )

    for fd_i in range(forward_dynamics_distance - 1):
        forward_dynamics = reward_network.forward_dynamics(
            forward_dynamics,
            forward_dynamics_onehot_actions_1[
                fd_i + 1 : (num_frames - forward_dynamics_distance + fd_i + 1)
            ],
        )

    forward_dynamics_loss = 100 * forward_dynamics_loss_fn(
        forward_dynamics, mu[forward_dynamics_distance:]
    )
    dt_loss_i = 4 * temporal_difference_loss(
        est_dt, torch.tensor(((real_dt,),), dtype=torch.float32, device=device)
    )

    return sum([dt_loss_i, forward_dynamics_loss, recon_loss, inverse_dynamics_loss])


# Train the network
def learn_reward(
    reward_network,
    optimizer,
    inputs,
    outputs,
    actions,
    times,
    num_iter,
    l1_reg,
    loss_fn,
    checkpoint_dir="../model_checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving outputs to {os.path.abspath(checkpoint_dir)}")
    # check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0
    training_data = list(zip(inputs, outputs, times, actions))
    for _ in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_times_sub, training_actions_sub = zip(
            *training_data
        )
        validation_split = 1.0
        for i in tqdm(range(len(training_labels)), mininterval=1):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            times_i, times_j = training_times_sub[i]
            actions_i, actions_j = training_actions_sub[i]

            traj_i, traj_j = np.array(traj_i), np.array(traj_j)
            traj_i, traj_j = torch.from_numpy(traj_i).float().to(
                device
            ), torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)
            num_frames_i, num_frames_j = len(traj_i), len(traj_j)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            (
                outputs,
                abs_rewards,
                z1,
                z2,
                mu1,
                mu2,
                logvar1,
                logvar2,
            ) = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)

            # compute losses
            losses_i = compute_losses(
                reward_network,
                z1,
                mu1,
                logvar1,
                traj_i,
                times_i,
                num_frames_i,
                actions_i,
            )
            losses_j = compute_losses(
                reward_network,
                z2,
                mu2,
                logvar2,
                traj_j,
                times_j,
                num_frames_j,
                actions_j,
            )
            trex_loss = loss_criterion(outputs, labels.long())

            if loss_fn == "trex":  # only use trex loss
                loss = trex_loss
            elif loss_fn == "ss":  # only use self-supervised loss
                loss = losses_i + losses_j
            elif loss_fn == "trex+ss":
                loss = losses_i + losses_j + trex_loss

            if i < len(training_data) * validation_split:
                loss.backward()
                optimizer.step()

            # print stats to see if learning
            item_loss = loss.item()
            # print("total", item_loss)
            cum_loss += item_loss
            if (i + 1) % 1000 == 0:
                # print(i)
                print("loss {}".format(cum_loss))
                print(f"abs rewards: {abs_rewards.item()}")
                cum_loss = 0.0
                # print("check pointing")
                torch.save(
                    reward_network.state_dict(), f"{checkpoint_dir}/epoch_{i}.pt"
                )
    print("finished training")


def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[
                0
            ].item()
            rewards_from_obs.append(r)
    return rewards_from_obs


def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_correct = 0.0
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i, traj_j = np.array(traj_i), np.array(traj_j)
            traj_i, traj_j = torch.from_numpy(traj_i).float().to(
                device
            ), torch.from_numpy(traj_j).float().to(device)

            # forward to get logits
            outputs, _, _, _, _, _, _, _ = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs, 0)
            if pred_label.item() == label:
                num_correct += 1.0
    return num_correct / len(training_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    # NOTE why this argument here? we learn those weights in this file
    parser.add_argument(
        "--reward_model_path",
        default="./reward_model_checkpoints/",
        help="name and location for learned model params, e.g. ./learned_models/breakout.params",
    )
    # parser.add_argument("--seed", default=0, help="random seed for experiments")
    parser.add_argument(
        "--encoding_dims",
        default=64,
        type=int,
        help="number of dimensions in the latent space",
    )
    parser.add_argument(
        "--loss_fn",
        default="trex+ss",
        help="ss: selfsupervised, trex: only trex, trex+ss: both trex and selfsupervised",
    )
    parser.add_argument(
        "--training_data_available",
        default=False,
        help="If training data not availabel we generate it locally",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help="""path to a directory containing PPO checkpoints, e.g."""
        """ 'data/demonstrations/BreakoutNoFrameskip-v4'""",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="breakout",
        help=(
            """name of Atari game to generate demonstrations (should be"""
            """ same as checkpoints training env)"""
        ),
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=1,
        help="number of episodes to play (1 episode = 1 full trajectory)",
    )
    parser.add_argument(
        "--n-traj", type=int, default=0, help="number of full training demonstrations"
    )
    parser.add_argument(
        "--n-snippets",
        type=int,
        default=int(6e4),
        help="number of short training demonstrations",
    )
    parser.add_argument(
        "--snippet-min-len",
        type=int,
        default=50,
        help="minimum length of short demonstrations",
    )
    parser.add_argument(
        "--snippet-max-len",
        type=int,
        default=100,
        help="max length of short demonstrations",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="number of envs to run in parallel to gather demonstrations",
    )
    parser.add_argument(
        "--seed",
        default=7,
        type=int,
        help="RNG seed",
    )

    args = parser.parse_args()

    lr, weight_decay = 1e-4, 1e-3
    encoding_dims = 64
    ACTION_DIMS = 4
    l1_reg = 0.0
    num_iter = 2 if args.env == "enduro" and args.loss_fn == "trex+ss" else 1

    if args.training_data_available:
        # load training data from checkpoints
        folder = "../training_data"
        obs = np.load(f"{folder}/training_obs.npy", allow_pickle=True)
        labels = np.load(f"{folder}/training_labels.npy", allow_pickle=True)
        actions = np.load(f"{folder}/training_actions.npy", allow_pickle=True)
        times = np.load(f"{folder}/training_times.npy", allow_pickle=True)
        demonstrations = np.load(f"{folder}/demonstrations.npy", allow_pickle=True)
        sorted_returns = np.load(f"{folder}/training_obs.npy", allow_pickle=True)
    else:
        # generate training data
        # checkpoints = args.checkpoints_dir
        from bayesianrex import config

        ckpt_dir = args.checkpoints_dir

        # TODO: Generalise this to accept any checkpoint
        checkpoints = [
            Path("./agent_checkpoints/PPO_9950_steps"),
        ]
        trajectories = generate_demonstrations(
            checkpoints,
            args.env,
            args.n_episodes,
            seed=args.seed,
            n_envs=args.n_envs,
        )
        training_data = create_training_data(
            trajectories,
            args.n_traj,
            args.n_snippets,
            np.random.default_rng(args.seed),
            (args.snippet_min_len, args.snippet_max_len),
        )
        obs, actions, times, labels = training_data
        demonstrations = trajectories
        sorted_returns = obs
        print(f"Generated data")

    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet(encoding_dims, ACTION_DIMS, training=True, device=device)
    reward_net.to(device)

    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
    learn_reward(
        reward_net,
        optimizer,
        obs,
        labels,
        actions,
        times,
        num_iter,
        l1_reg,
        args.loss_fn,
    )
    # save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)

    # print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [
            sum(predict_reward_sequence(reward_net, traj[0])) for traj in demonstrations
        ]
    for i, p in enumerate(pred_returns):
        print(i, p, sorted_returns[i])
    print("accuracy", calc_accuracy(reward_net, obs, labels))

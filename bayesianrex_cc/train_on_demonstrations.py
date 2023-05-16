"""Train an MLP to recognize better RL demonstrations."""
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config, constants, networks, utils
from dataset.data_torch import RLDemonstrationsDataset

logger = logging.getLogger(__name__)


def create_parser() -> ArgumentParser():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--train-data",
        type=Path,
        default=config.DATA_DIR / "train_data_cartpole.gz",
        help="Path to .gz compressed training data archive",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=constants.HIDDEN_DIM,
        help="Hidden dimensionality of 2-layers MLP",
    )
    p.add_argument(
        "--model-save-path",
        type=Path,
        default=config.DATA_DIR / "cartpole_reward_net.pt",
        help="Path where the trained model parameters are saved",
    )
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    return p


# NOTE should we do some validation too ?
def learn_preferences(
    model: nn.Module, train_data: DataLoader, epochs: int, device: torch.device
):
    model = model.to(device)
    trex_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_net.parameters())
    # loss is nearly constant already halfway through the epoch.
    for _ in range(epochs):
        # epoch_loss = 0
        for (traj_i, traj_j), label in tqdm(train_data):
            reward_net.train()
            optimizer.zero_grad()

            outputs = reward_net(traj_i.to(device), traj_j.to(device))
            loss = trex_criterion(outputs, label.squeeze().to(device))

            loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            wandb.log({"loss": loss.item()})


# TODO check again that trajectories generation is correct, it seems that
# each traj_i and traj_j are very similar if not the same
if __name__ == "__main__":
    p = create_parser()
    args = p.parse_args()

    utils.setup_root_logging()
    # NOTE was giving error: input tensor dtype and network dtype are different
    # torch.set_default_dtype(torch.float64)

    # 60100 pairs of trajectories --> [60100 x (traj_i, traj_j)]
    # 100 traj has [500, 4] states [500] actions, 60000 are shorter
    # each (traj_i,traj_j) pair has a label 1 or 0 indicating whether traj_i is
    # better than traj_j
    train_dataset = RLDemonstrationsDataset(demonstrations_path=args.train_data)
    logger.info("Training on demonstrations file '%s'", args.train_data)
    # TODO pass a sampler that picks random trajectories (they are already
    # randomized but it doesn't hurt to do it once more)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device '%s'", device)

    reward_net = networks.MLP(train_dataset.feature_shape, args.hidden_dim)

    # Wandb TODO dump configuration too + accept args for run name
    wandb.init(project="cartpole-mlp", entity="bayesianrex-dl2")

    learn_preferences(reward_net, train_dataloader, args.epochs, device)

    # Save model
    torch.save(reward_net.state_dict(), args.model_save_path)
    logger.info("Saved trained model to '%s'", args.train_data)

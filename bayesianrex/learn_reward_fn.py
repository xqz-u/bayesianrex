# TODO original codebase has a `predict_trajectory_return` evaluation step at
# the end
import logging
import time
from argparse import Namespace
from pathlib import Path
from pprint import pformat

import joblib
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from bayesianrex import config, constants, environments, losses, utils
from bayesianrex.dataset import generate_demonstrations as gen_demos
from bayesianrex.dataset.dataset_torch import make_demonstrations_loader
from bayesianrex.networks import RewardNetwork

logger = logging.getLogger(__name__)
device = utils.torch_device()


def learn_reward_fn(
    train_loader: DataLoader,
    reward_net: RewardNetwork,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    n_epochs: int,
    run: Run,
    log_freq: int = 300,
):
    for epoch in range(n_epochs):
        cum_loss, cum_ret, cum_abs_ret = 0, 0, 0
        for i, datapoint in enumerate(train_loader):
            optimizer.zero_grad()
            # squeeze out batch dimension due to DataLoader
            states, actions, times = [
                tuple(map(lambda t: t.squeeze().to(device), dp)) for dp in datapoint[:3]
            ]
            # NOTE `abs_returns_sum` used only for logging
            returns_, abs_returns_sum, *info = reward_net(*states)
            loss = 0.0
            if "trex" in loss_name:
                loss += losses.trex_loss(returns_, datapoint[3].long().to(device))
            if "ss" in loss_name:
                loss += losses.self_supervised_losses(
                    reward_net, states, actions, times, info
                )
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            cum_ret += returns_
            cum_abs_ret += abs_returns_sum
            logger.debug(
                "return %.2f abs_return %.2f loss %.2f",
                returns_,
                abs_returns_sum,
                loss.item(),
            )
            if i % log_freq == 0:
                return_i, return_j = cum_ret.squeeze()
                info = {
                    "cum_loss": cum_loss,
                    "cum_return_i": return_i,
                    "cum_return_j": return_j,
                    "cum_abs_return_both": cum_abs_ret,
                }
                logger.info("\n%s", pformat({**info, **{"epoch": epoch, "step": i}}))
                run.log(info)
                cum_loss, cum_ret, cum_abs_ret = 0, 0, 0


def main(args: Namespace):
    logger.info("Using device: %s", device)

    # load offline training data if they exist, otherwise generate new ones
    if args.train_data_dir is None:
        logger.info("Generate %d training datapoints", args.n_traj + args.n_snippets)
        train_data, _ = gen_demos.main(args)
    else:
        logger.info("Loading training data from %s", args.train_data_dir)
        train_data = list(joblib.load(args.train_data_dir).values())
    train_loader = make_demonstrations_loader(train_data)

    n_actions = environments.create_atari_env(
        constants.envs_id_mapper.get(args.env)
    ).action_space.n
    reward_net = RewardNetwork(args.encoding_dims, n_actions, device).to(device)
    optimizer = Adam(reward_net.parameters(), **constants.reward_net_hparams)

    # FIXME after training has started for a while, C-c is not captured
    # TODO capture SIGTERM/KILL too to die gracefully (relevant for wandb)
    try:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project or "reward-model",
            name=args.wandb_run_name,
            dir=args.assets_dir,
            config=vars(args),
        )
        start = time.time()
        logger.info("Start learning reward fn from demos")
        learn_reward_fn(
            train_loader,
            reward_net,
            optimizer,
            args.loss_fn,
            args.n_episodes,
            run,
            log_freq=args.log_frequency,
        )
    except KeyboardInterrupt:
        logger.info("Training aborted!")
        run.finish()
        exit(1)

    end = time.time()
    logger.info(
        "Learning reward fn for %d epochs on %d trajectories took %.2fs",
        args.n_episodes,
        len(train_loader),
        end - start,
    )

    savepath = args.reward_model_save_path
    logger.info("Saving learned reward function weights to %s", savepath)
    torch.save(reward_net.state_dict(), savepath)
    logger.info("Done")


if __name__ == "__main__":
    parser_conf = {
        "encoding-dims": {
            "type": int,
            "default": constants.reward_net_latent_space,
            "help": "latent space dimensionality",
        },
        "loss-fn": {
            "type": str,
            "choices": ("trex+ss", "trex", "ss"),
            "default": "trex+ss",
            "help": "loss type, ss: Self-Supervised, trex: T-Rex, trex+ss: T-Rex and Self-Supervised",
        },
        "train-data-dir": {
            "type": Path,
            "help": "(optional) path to folder with demonstrations",
        },
        "reward-model-save-path": {
            "type": Path,
            "default": config.ASSETS_DIR / "reward_model_breakout.pth",
            "help": "path where to save the trained reward model",
        },
        "log-frequency": {
            "type": int,
            "default": 300,
            "help": "learning stats logging frequency to stdout and W&B",
        },
        **gen_demos.parser_conf,
        **constants.wandb_cl_parser,
    }
    p = utils.define_cl_parser(parser_conf)
    args = p.parse_args()

    # args.seed = 0
    # # args.log_level = 1
    # # args.checkpoints_dir = config.DEMONSTRATIONS_DIR / "BreakoutNoFrameskip-v4"
    # args.checkpoints_dir = config.DEMONSTRATIONS_DIR.parent / "demonstrators_tiny"
    # args.n_traj = 1000
    # args.n_snippets = 2000
    args.wandb_entity = "bayesianrex-dl2"
    args.wandb_project = "reward-model"

    utils.setup_root_logging(args.log_level)
    main(args)
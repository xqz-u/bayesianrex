import logging
import multiprocessing as mp
from argparse import Namespace
from pathlib import Path
from pprint import pformat
from typing import Tuple

import wandb
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    VecEnv,
    VecVideoRecorder,
)
from wandb.integration.sb3 import WandbCallback

from bayesianrex import config, constants, utils
from bayesianrex.environments import (
    MAPRewardWrapper,
    MeanRewardWrapper,
    create_atari_env,
    create_hidden_lives_atari_env,
)
from bayesianrex.models.utils import load_reward_network

logger = logging.getLogger(__name__)


def make_env(args: Namespace, conf: dict) -> Tuple[VecEnv, str, Path]:
    # adjust env_args
    conf["env_args"] = {"n_envs": args.n_envs or mp.cpu_count(), "seed": args.seed}
    env_id = constants.envs_id_mapper.get(args.env)
    ckpt_path = args.assets_dir / "demonstrators" / env_id
    # run the environments in parallel, the overhead should be worth it with Atari
    if args.custom_reward:
        # our custom reward model was trained on atari environments with special
        # preprocessing to hide game info
        env = create_hidden_lives_atari_env(
            args.env, **conf["env_args"], vec_env_cls=SubprocVecEnv
        )
        ckpt_path = args.assets_dir / "demonstrators" / f"{env_id}_custom"
        reward_model_path = args.reward_model_path
        assert reward_model_path is not None, "You must give a path to reward fn model"
        device = device = utils.torch_device()
        logger.info("Loading learned reward fn weights from %s", reward_model_path)
        reward_net = load_reward_network(reward_model_path, args.env, device=device)
        if args.mean:
            chain_path = args.mcmc_chain_path
            assert chain_path is not None, "You must give a path to the MCMC chain"
            logger.info(
                "Using mean learned reward fn from MCMC chain at %s", chain_path
            )
            env = MeanRewardWrapper(
                env, reward_net, args.mcmc_chain_path, args.encoding_dims, device
            )
        else:
            logger.info("Using learned MAP reward fn")
            env = MAPRewardWrapper(env, reward_net)
    else:
        env = create_atari_env(
            env_id,
            **conf["env_args"],
            vec_env_cls=SubprocVecEnv,
        )
    # sometimes early into training there are numerical instability issues
    env = VecCheckNan(env, raise_exception=True)
    if args.video:
        # NOTE bugfix, stable_baselines3.common.vec_env.VecFrameStack does not set
        # 'render_mode' after wrapping, which is required by VecVideoRecorder
        # (make_atari_env sets 'render_mode' correctly under the hood)
        env.render_mode = "rgb_array"
        # frequency & duration from
        # https://huggingface.co/ThomasSimonini/ppo-BreakoutNoFrameskip-v4#training-code
        video_len, video_freq = int(2e3), int(1e5)
        video_dir = args.assets_dir / "videos"
        logger.info(
            "Saving videos of %d train steps every %d steps at %s",
            video_len,
            video_freq,
            video_dir,
        )
        env = VecVideoRecorder(
            env,
            video_dir,
            record_video_trigger=lambda x: x % video_freq == 0,
            video_length=video_len,
            name_prefix=f"PPO-{env_id}",
        )
    return env, env_id, ckpt_path


def learn_demonstrator(args: Namespace):
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    # parse default args from rl-baselines3-zoo for Atari envs
    with open(config.ROOT_DIR / "atari_conf.yml") as fd:
        conf = yaml.safe_load(fd)

    env, env_id, ckpt_path = make_env(args, conf)

    logger.info("Atari args:\n%s", pformat(conf))
    logger.info("Command line args:\n%s", pformat(vars(args)))

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project or "atari-demonstrators",
        dir=args.assets_dir,
        name=args.wandb_run_name,
        config={**conf, **{"cl_args": vars(args)}},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    # adjust for learning rate (or other hparams) scheduling
    utils.adjust_ppo_schedules(conf["ppo_args"])
    agent = PPO(
        env=env,
        **conf["ppo_args"],
        seed=args.seed,
        tensorboard_log=args.assets_dir / "tensorflow_runs" / env_id / run.id,
        verbose=2,
    )
    agent.learn(
        **conf["ppo_learn_args"],
        callback=CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(args.save_freq // conf["env_args"]["n_envs"], 1),
                    save_path=ckpt_path,
                    name_prefix="PPO",
                ),
                WandbCallback(
                    gradient_save_freq=int(1e4),
                    # path to final trained agent
                    model_save_path=ckpt_path / f"PPO_{env_id}_trained_demonstrator",
                    verbose=2,
                ),
            ]
        ),
        progress_bar=True,
    )


if __name__ == "__main__":
    import numpy as np
    import torch

    torch.autograd.set_detect_anomaly(True)
    np.seterr(all="raise")

    parser = {
        "env": {
            "type": str,
            "choices": tuple(constants.envs_id_mapper),
            "default": "breakout",
            "help": "environment name",
        },
        "seed": {"type": int, "help": "RNG seed"},
        "n-envs": {"type": int, "help": "number of environments to run in parallel"},
        "save-freq": {
            "type": int,
            "default": int(4e5),
            "help": "checkpointing frequency in steps (defaults to ~25 checkpoints)",
        },
        "video": {
            "default": False,
            "action": "store_true",
            "help": "record videos of the agent while training",
        },
        "custom-reward": {
            "action": "store_true",
            "default": False,
            "help": "whether to use a reward fn from the learned posterior",
        },
        "reward-model-path": {
            "type": Path,
            "help": "location of agent trained with a learned reward fn",
        },
        "encoding-dims": {
            "type": int,
            "default": constants.reward_net_latent_space,
            "help": "dimension of latent space",
        },
        "mcmc-chain-path": {
            "type": Path,
            "help": "where to find MCMC chain file for mean reward fn estimate",
        },
        "mean": {
            "action": "store_true",
            "default": False,
            "help": "whether to use the mean learned reward fn, or the MAP",
        },
        **constants.wandb_cl_parser,
    }

    p = utils.define_cl_parser(parser)
    args = p.parse_args()

    utils.setup_root_logging(args.log_level)

    learn_demonstrator(args)

import logging
import multiprocessing as mp
from argparse import Namespace
from pprint import pformat

import bayesianrex.environments as environments
import wandb
import yaml
from bayesianrex import config, constants, utils
from bayesianrex.dataset.wrapper import CustomRewardWrapper
from bayesianrex.environments import create_atari_env
from bayesianrex.networks import RewardNetwork
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    VecVideoRecorder,
)
from wandb.integration.sb3 import WandbCallback

logger = logging.getLogger(__name__)


def learn_demonstrator(args: Namespace):
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    # parse default args from rl-baselines3-zoo for Atari envs
    with open(config.ROOT_DIR / "atari_conf.yml") as fd:
        conf = yaml.safe_load(fd)

    # adjust env_args
    conf["env_args"] = {"n_envs": args.n_envs or mp.cpu_count(), "seed": args.seed}
    env_id = constants.envs_id_mapper.get(args.env)
    ckpt_path = args.assets_dir / "demonstrators" / env_id
    # run the environments in parallel, the overhead should be worth it with Atari
    env = create_atari_env(
        env_id,
        **conf["env_args"],
        vec_env_cls=SubprocVecEnv,
    )
    # TODO: test Custom Reward wrapper
    if args.Custom_Reward:
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        n_actions = environments.create_atari_env(
            constants.envs_id_mapper.get(args.env)
        ).action_space.n
        reward_net = RewardNetwork(args.encoding_dims, n_actions, device).to(device)
        checkpoint = torch.load(args.reward_net_location)
        reward_net.load_state_dict(checkpoint)
        env = CustomRewardWrapper(env, reward_net)
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
        "Custom_Reward": {
            "type": bool,
            "default": False,
            "help": "whether to use learned reward function",
        },
        "reward_net_location": {
            "type": str,
            "default": "",
            "help": "location of trained reward net checkpoint",
        },
        "encoding_dims": {
            "type": int,
            "default": constants.reward_net_latent_space,
            "help": "encoding_dim of the model",
        },
        **constants.wandb_cl_parser,
    }

    p = utils.define_cl_parser(parser)
    args = p.parse_args()

    utils.setup_root_logging(args.log_level)

    learn_demonstrator(args)

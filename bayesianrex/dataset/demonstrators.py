import logging
import multiprocessing as mp
from argparse import Namespace
from pathlib import Path
from pprint import pprint

import stable_baselines3 as sb3
import wandb
import yaml
from bayesianrex import config, constants, utils
from bayesianrex.environments import create_atari_env, create_hidden_lives_atari_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
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
    # env = create_hidden_lives_atari_env(
    #     args.env,
    #     **conf["env_args"],
    #     vec_env_cls=SubprocVecEnv,
    #     monitor_dir=ckpt_path / "monitor",
    # )
    env = create_atari_env(
        env_id,
        **conf["env_args"],
        vec_env_cls=SubprocVecEnv,
        monitor_dir=ckpt_path / "monitor",
    )

    # if args.video:
    #     # frequency & duration from
    #     # https://huggingface.co/ThomasSimonini/ppo-BreakoutNoFrameskip-v4#training-code
    #     video_len, video_freq = int(2e3), int(1e5)
    #     video_dir = args.assets_dir / "videos"
    #     logger.info(
    #         "Saving videos of %d train steps every %d steps at %s",
    #         video_len,
    #         video_freq,
    #         video_dir,
    #     )
    #     env = VecVideoRecorder(
    #         env,
    #         video_dir,
    #         lambda x: x % video_freq == 0,
    #         video_length=video_len,
    #         name_prefix=f"PPO-{env_id}",
    #     )

    print("Atari args:")
    pprint(conf)
    print("Command line args:")
    pprint(vars(args))

    run = wandb.init(
        project="atari-demonstrators",
        dir=args.assets_dir,
        name=args.run_name,
        config={**conf, **{"cl_args": vars(args)}},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    # adjust for learning rate (or other hparams) scheduling
    utils.adjust_ppo_schedules(conf["ppo_args"])
    agent = sb3.PPO(
        env=env,
        **conf["ppo_args"],
        seed=args.seed,
        tensorboard_log=args.assets_dir / "tensorflow_runs" / env_id / run.id,
        # verbose=1,
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
                    model_save_path=ckpt_path / f"PPO_{env_id}_trained_demonstrator",
                    verbose=2,
                ),
            ]
        ),
        progress_bar=True,
    )
    run.finish()


if __name__ == "__main__":
    parser = {
        "env": {"type": str, "default": "breakout", "help": "environment name"},
        "seed": {"type": int, "default": None, "help": "RNG seed"},
        "run-name": {"type": str, "default": None, "help": "wandb run name"},
        "n-envs": {
            "type": int,
            "default": None,
            "help": "number of environments to run in parallel",
        },
        "save-freq": {
            "type": int,
            "default": int(4e5),
            "help": "checkpointing frequency in steps (~25 total checkpoints)",
        },
        "assets-dir": {
            "type": Path,
            "default": Path("./assets").resolve(),
            "help": "Container folder to store artifacts (checkpoints, videos etc.)",
        },
        "video": {
            "default": False,
            "action": "store_true",
            "help": "Record videos of the agent while training",
        },
    }

    p = utils.define_cl_parser(parser)
    args = p.parse_args()

    utils.setup_root_logging(args.log_level)

    learn_demonstrator(args)



# from stable_baselines3.common.env_checker import check_env

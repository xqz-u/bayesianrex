import argparse
import logging
import multiprocessing as mp

import numpy as np
import stable_baselines3 as sb3
import wandb
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

import config
import constants
import rl_utils

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", type=str, default="breakout")
    parser.add_argument("--seed", help="RNG seed", type=int, default=None)
    return parser


# used for Enduro
class DelayedCheckpointCallback(CheckpointCallback):
    msg_done: bool = False

    def _on_step(self):
        if self.num_timesteps >= 3100:
            if not self.msg_done:
                print(
                    f"Start ckpts, num_timesteps: {self.num_timesteps} n_calls: {self.n_calls}"
                )
                self.msg_done = True
            super()._on_step()
        return True


# TODO video record? Monitor wrapper?
# TODO do the authors of T-REX use less than 1e6 total timesteps?!
# NOTE checkpointing is done every (save_freq // n_envs) * n_envs, so it can
# happen that it is not precisely save_freq (e.g. 50 // 4 * 4 = 48). So enforce
# the max compatible amount of environments.
if __name__ == "__main__":
    import utils

    utils.setup_root_logging()

    parser = create_parser()
    args = parser.parse_args()

    if args.env == "seaquest":
        save_freq = 5
        # easier to run a single environment since ckpts happen often
        n_envs = 1
        logger.info("Enforced n_envs=1 for environment 'seaquest'")
    else:
        save_freq = 50
        modulo_divisors = np.array([1, 2, 5, 10, 25, save_freq])
        diffs = mp.cpu_count() - modulo_divisors
        n_envs = diffs[diffs >= 0].min()
        logger.info(f"n_envs: {n_envs}")

    env_id = constants.envs_id_mapper.get(args.env)
    env = rl_utils.make_base_atari_env(env_id, n_envs=n_envs)

    run_config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": int(1e6),
        "env_name": env_id,
        "n_envs": n_envs,
    }
    if args.seed is not None:
        run_config["seed"] = args.seed
    run = wandb.init(
        project="sb3",
        dir=config.LOGS_DIR,
        config=run_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    agent = sb3.PPO(
        run_config["policy_type"],
        env,
        seed=args.seed,
        tensorboard_log=config.LOGS_DIR / "wandb" / "runs" / run.id,
        verbose=1,
    )

    ckpt_callback_class = (
        DelayedCheckpointCallback if args.env == "enduro" else CheckpointCallback
    )
    ckpt_callback = ckpt_callback_class(
        save_freq=max(save_freq // n_envs, 1),
        save_path=config.DEMONSTRATIONS_DIR / env_id,
        name_prefix="PPO",
    )

    agent.learn(
        run_config["total_timesteps"],  # 1e6 samples is the default in baselines too
        callback=CallbackList([ckpt_callback, WandbCallback(verbose=2)]),
        progress_bar=True,
    )

    run.finish()

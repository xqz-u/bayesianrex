import argparse
import logging
import multiprocessing as mp

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback

import constants
import rl_utils

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--env", help="environment ID", type=str, default="breakout")
    parser.add_argument("--seed", help="RNG seed", type=int, default=None)
    parser.add_argument(
        "--n_envs",
        help="Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari",
        default=mp.cpu_count(),
        type=int,
    )
    return parser


# TODO wandb
# TODO correct save_freq, seaquest and another game have a different one
# NOTE empty stdout?!
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    env = rl_utils.make_base_atari_env(
        constants.envs_id_mapper.get(args.env), n_envs=args.n_envs
    )
    agent = sb3.PPO("CnnPolicy", env, seed=args.seed)
    # 1e6 samples is the default in baselines too
    agent.learn(
        1e6,
        progress_bar=True,
        callback=CheckpointCallback(
            save_freq=50, save_path="./logs/", name_prefix="test"
        ),
    )

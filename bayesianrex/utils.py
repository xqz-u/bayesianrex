import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, List, Union

import numpy as np
import torch


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE unsafe in python because no tail recursion optimization, but in this
# project this is used with shallow structures only
def tensorify(array: List[np.ndarray]) -> List[torch.Tensor]:
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    return list(map(tensorify, array))


def define_cl_parser(args_dict: dict) -> ArgumentParser:
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    for k, v in args_dict.items():
        p.add_argument(f"--{k}", **v)
    if "log-level" not in args_dict:
        p.add_argument(
            "--log-level",
            **{
                "type": int,
                "default": 2,
                "help": (
                    """logging severity,"""
                    """ number in [1, 5] (increase to see less messages)"""
                ),
            },
        )
    return p


def setup_root_logging(level: int = 2):
    """
    Entry point to configure logging, called once from the program's main entry point.

    :param level: Minimum logging severity, integer between [1, 5]
    """
    assert level in range(
        1, 6
    ), "See https://docs.python.org/3/library/logging.html#logging-levels for valid logging levels"
    logging.basicConfig(
        level=level * 10,
        format="p%(process)s [%(asctime)s] [%(levelname)s]  %(message)s  (%(name)s:%(lineno)s)",
        datefmt="%y-%m-%d %H:%M",
    )


# from
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/382dcabbd9815cb9557503a7caa0c54a562fa7fb/rl_zoo3/utils.py#L298
def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def adjust_ppo_schedules(ppo_args: dict):
    for k in ["learning_rate", "clip_range"]:
        if ppo_args.get(k, "").startswith("lin_"):
            init_val = float(ppo_args[k][4:])
            ppo_args[k] = linear_schedule(init_val)
    return ppo_args

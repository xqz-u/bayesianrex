# TODO log trajectories returns on wandb
import itertools as it
import logging
import random
import time
from argparse import Namespace
from pathlib import Path
from pprint import pformat
from typing import List, Optional, Tuple

import joblib
import numpy as np
from bayesianrex import utils
from bayesianrex.environments import create_hidden_lives_atari_env
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import set_random_seed

RawTrajectories = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
PreferenceTraj = Tuple[np.ndarray, np.ndarray]
TrainTrajectories = Tuple[
    List[PreferenceTraj], List[PreferenceTraj], List[PreferenceTraj], List[np.ndarray]
]

logger = logging.getLogger(__name__)
device = utils.torch_device()


parser_conf = {
    "seed": {"type": int, "help": "RNG seed"},
    "checkpoints-dir": {
        "type": Path,
        "help": (
            """path to a directory containing PPO checkpoints, e.g."""
            """ 'data/demonstrations/BreakoutNoFrameskip-v4'"""
        ),
    },
    "env": {
        "type": str,
        "default": "breakout",
        "help": (
            """name of Atari game to generate demonstrations (should be"""
            """ same as checkpoints training env)"""
        ),
    },
    "n-episodes": {
        "type": int,
        "default": 1,
        "help": "number of episodes to play (1 episode = 1 full trajectory)",
    },
    "n-traj": {
        "type": int,
        "default": 0,
        "help": "number of full training demonstrations",
    },
    "n-snippets": {
        "type": int,
        "default": int(6e4),
        "help": "number of short training demonstrations",
    },
    "snippet-min-len": {
        "type": int,
        "default": 50,
        "help": "minimum length of short demonstrations",
    },
    "snippet-max-len": {
        "type": int,
        "default": 100,
        "help": "max length of short demonstrations",
    },
    "n-envs": {
        "type": int,
        "default": 1,
        "help": "number of envs to run in parallel to gather demonstrations",
    },
    "train-data-save-dir": {
        "type": Path,
        "help": "folder to save trajectories and training data indexes",
    },
}


def list_checkpoints(folder: Path) -> List[Path]:
    checkpoints = sorted(list(folder.glob("PPO_*_steps.zip")), reverse=True)
    if not checkpoints:
        raise ValueError(f"{folder} does not contain valid PPO checkpoints")
    logger.debug("Checkpoints in %s:\n%s", folder, pformat(checkpoints))
    return checkpoints


def generate_trajectory_from_ckpt(
    ckpt_path: Path, env: GymEnv, n_traj: int, seed: Optional[int] = None
) -> RawTrajectories:
    """
    Generate demonstrations for Atari environmenst with masked game info from a PPO checkpoint.

    :param ckpt_path: (Absolute) path to the zipped checkpoint
    :param env: The gym environment used to generate the checkpoint, with masked
        game score
    :param n_traj: Number of episodes to run (a trajectory is a full episode)
    :param seed: The (optional) RNG seed passed to the environment, increased
        by `n_traj`
    :return: The states, actions and rewards for each trajectory
    """
    agent = PPO.load(ckpt_path, verbose=1, custom_objects={"tensorboard_log": None})
    states, actions, rewards = [], [], []
    for i in range(n_traj):
        if seed is not None:
            env.seed(seed + i)
        obs, done = env.reset(), False
        ep_states, ep_actions, ep_rewards = [], [], []
        while not done:
            # NOTE pixel observations from Atari games are uint8, transform to
            # float for CNN processing
            ep_states.append((obs / 255.0).astype(np.float32))
            action, *_ = agent.predict(obs)
            obs, reward, done, *_ = env.step(action)
            ep_actions.append(action)
            ep_rewards.append(reward)
        logger.info("%s episode %d traj len %d", str(ckpt_path), i, len(ep_states))
        states.append(np.array(ep_states).squeeze())
        actions.append(np.array(ep_actions).squeeze())
        rewards.append(np.array(ep_rewards).squeeze())
    return states, actions, rewards


def generate_demonstrations(
    checkpoints: List[Path],
    env_name: str,
    n_traj_per_ckpt: int,
    seed: Optional[int] = None,
    **env_kwargs,
) -> RawTrajectories:
    """
    Generate demonstrations from checkpointed `sb3` PPO agents.

    :param checkpoints: List of PPO checkpoints
    :param env_name: The name of the environment used to checkpoint the agent
    :param n_traj_per_ckpt: Number of episodes a checkpointed agent is run for
    :param seed: (Optional) starting RNG seed passed to the Gym environment
        during play
    :param env_kwargs: Additional keyword argumenst passed to
        `bayesianrex.environments.create_hidden_lives_atari_env`
    :return: The trajectories sorted by ground truth return
    .. warning::
      For full reproducibility, `sb3.common.utils.set_random_seed` should be
      called once in a program before calling this function
    """
    if env_kwargs.get("seed") is None:
        env_kwargs["seed"] = seed
    # default Atari options found in B-Rex
    env = create_hidden_lives_atari_env(
        env_name,
        atari_kwargs={"clip_reward": False, "terminal_on_life_loss": False},
        **env_kwargs,
    )
    # structure: [(s,a,r,return), (s,a,r,return), ...], size: len(ckpts) * n_traj
    # use the return of a trajectory (sum of rewards) as sorting key
    trajectories = [
        (s, a, r, r.sum())
        for ckpt in checkpoints
        for s, a, r in zip(
            *generate_trajectory_from_ckpt(ckpt, env, n_traj_per_ckpt, seed=seed)
        )
    ]
    for s, a, r, _ in trajectories:
        assert (
            len(s) == len(a) == len(r)
        ), "State/action/reward chain length mismatch for same trajectory"
    # sort by worst return
    trajectories = sorted(trajectories, key=lambda el: el[3])
    # structure: [[states], [actions], [rewards]]
    # list before tuple since the latter splices the trajectory
    return tuple(list(zip(*trajectories)))[:3]


def sample_trajectory_pairs(
    states: List[np.ndarray], n: int, sort: bool = False
) -> Tuple[np.ndarray]:
    """
    Randomly select pairs of trajectories as training datapoints for IRL from preferences.

    :param states: A list of trajectories elements (could be states, actions, rewards)
    :param n: The number of pairs to generate
    :param sort: Whether to sort pairs in increasing order. This can be useful
        if `states` are already sorted; then, a pair [i, j] has the index of the
        better demonstration at location 0
    :return: An array of index pairs and an array of trajectory lengths for each pair
    """
    # same traj can be compared multiple times, but only with a different traj;
    # cannot use rng.choice easily here
    pairs = random.choices(list(it.permutations(range(len(states)), 2)), k=n)
    pairs = np.array(pairs)
    assert (pairs[:, 0] != pairs[:, 1]).all()
    if sort:
        pairs.sort(1)
    return pairs, np.array([(len(states[i]), len(states[j])) for i, j in pairs])


def pack_idxs(
    ids: np.ndarray,
    ends: np.ndarray,
    starts: Optional[np.ndarray] = None,
    step: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Represent training datapoints for IRL from preferences as an index array.

    :param ids: (N,2) array of trajectories indeces
    :param ends: (N,2) array of trajectories endpoints
    :param starts: (N,2) array of trajectories startpoints, defaults to 0
    :param step: (N,) array of downsampling steps, defaults to 1
    :return: A packed representation of the training datapoints, structured as
        `[worst_trj_idx,worst_trj_start,worst_trj_end,best_trj_idx,best_trj_start,best_trj_end,step]`
    """
    # structure: [w,start_w,end_w,b,start_b,end_b,step]
    packed = np.hstack((np.zeros((len(ids), 6)), np.ones((len(ids), 1)))).astype(int)
    # invariant assumption: worst trajectory at ids[:, 0]
    packed[:, 0], packed[:, 3] = ids[:, 0], ids[:, 1]
    if starts is not None:
        packed[:, 1], packed[:, 4] = starts[:, 0], starts[:, 1]
    packed[:, 2], packed[:, 5] = ends[:, 0], ends[:, 1]
    if step is not None:
        packed[:, 6] = step
    return packed


def full_trajectories_idxs(
    trajectories: RawTrajectories, n_traj: int, rng: np.random.Generator
) -> np.ndarray:
    # pick n_traj trajectory indexes
    pairs, lens = sample_trajectory_pairs(trajectories[0], n_traj, sort=True)
    # pick random starts with replacement
    starts = rng.choice(6, size=(n_traj, 2))
    steps = rng.integers(3, 7, size=n_traj)
    return pack_idxs(pairs, lens, starts, steps)


def snippet_trajectories_idxs(
    trajectories: RawTrajectories,
    n_snippets: int,
    snippet_min_len: int,
    snippet_max_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # only sample from trajectories long enough
    states = list(filter(lambda s: len(s) >= snippet_min_len, trajectories[0]))
    # pick n_snipptes trajectory indexes
    pairs, lens = sample_trajectory_pairs(states, n_snippets, sort=True)
    max_lens = np.repeat(snippet_max_len, len(lens))
    max_samplable_lens = np.hstack((lens, max_lens[..., None])).min(1)
    # pick some random snippet lengths achievable in all trajectories
    rand_lens = rng.integers(snippet_min_len, max_samplable_lens, endpoint=True)
    # desired: worse demo samples start earlier than better ones
    # NOTE when a short trajectory is paired with a long one, snippets can be
    # wildly discountuous with this sampling strategy
    worst_s = rng.integers(0, lens[np.arange(len(lens)), 1] - rand_lens, endpoint=True)
    best_s = rng.integers(
        worst_s, lens[np.arange(len(lens)), 1] - rand_lens, endpoint=True
    )
    ends = np.stack((worst_s + rand_lens, best_s + rand_lens), -1)
    return pack_idxs(pairs, ends, np.stack((worst_s, best_s), -1))


def create_training_idxs(
    trajectories: RawTrajectories,
    n_traj: int,
    n_snippets: int,
    rng: np.random.Generator,
    snippet_len_bounds: Tuple[int],
) -> np.ndarray:
    full_idxs = full_trajectories_idxs(trajectories, n_traj, rng)
    snippet_idxs = snippet_trajectories_idxs(
        trajectories, n_snippets, *snippet_len_bounds, rng
    )
    logger.info(
        "# full trajectories: %d # snippets: %d", len(full_idxs), len(snippet_idxs)
    )
    return np.vstack((full_idxs, snippet_idxs))


def construct_training_data(
    trajectories: RawTrajectories, idxs: np.ndarray
) -> TrainTrajectories:
    states, actions, _ = trajectories
    train_states, train_actions, train_times = [], [], []
    start = time.time()
    for slice_ in idxs:
        b, sb, eb, w, sw, ew, step = slice_
        train_states.append((states[w][sw:ew:step], states[b][sb:eb:step]))
        train_actions.append((actions[w][sw:ew:step], actions[b][sb:eb:step]))
        train_times.append((np.arange(sw, ew, step), np.arange(sb, eb, step)))
        logger.debug("[w,ws,we,b,bs,be,step]: %s", slice_)
    end = time.time()
    logger.info("Reconstructed %d train demos in %.2fs", len(idxs), end - start)
    return train_states, train_actions, train_times, np.ones(len(idxs), dtype=int)


# NOTE assumes trajectories are sorted in *increasing* order of return
def create_training_data(
    trajectories: RawTrajectories,
    n_traj: int,
    n_snippets: int,
    rng: np.random.Generator,
    snippet_len_bounds: Tuple[int],
) -> Tuple[TrainTrajectories, np.ndarray]:
    train_idxs = create_training_idxs(
        trajectories, n_traj, n_snippets, rng, snippet_len_bounds
    )
    return construct_training_data(trajectories, train_idxs), train_idxs


def main(args: Namespace) -> Tuple[TrainTrajectories, np.ndarray]:
    if args.env == "enduro":
        # NOTE we don't vary the loss function for now, always trex+ss
        assert (
            args.n_episodes >= 2
        ), f"Need at least 2 episodes per checkpoint on enduro, got {args.n_episodes}"
        logger.info(
            "Got n_snippets %d, but use only %d full trajs for enduro",
            args.n_snippets,
            int(1e4),
        )
        args.n_traj = 10000
        args.n_snippets = 0
    if args.seed is not None:
        logger.info("Calling sb3 `set_random_seed` with seed %d", args.seed)
        set_random_seed(args.seed, using_cuda=device.type != "cpu")
    checkpoints = list_checkpoints(args.checkpoints_dir)
    trajectories = generate_demonstrations(
        checkpoints,
        args.env,
        args.n_episodes,
        seed=args.seed,
        n_envs=args.n_envs,
    )
    train_data, train_idxs = create_training_data(
        trajectories,
        args.n_traj,
        args.n_snippets,
        np.random.default_rng(args.seed),
        (args.snippet_min_len, args.snippet_max_len),
    )

    savedir = args.train_data_save_dir
    if savedir is not None:
        savedir.mkdir(parents=True, exist_ok=True)

        traj_path = savedir / f"trajectories_{args.env}"
        trajectories_dict = dict(zip(["states", "actions", "rewards"], trajectories))
        logger.info("Saving %d trajectories to %s...", len(trajectories[0]), traj_path)
        joblib.dump(trajectories_dict, traj_path, compress=True)

        train_idxs_path = savedir / f"train_data_{args.env}"
        logger.info(
            "Saving %d training datapoints indexes to %s...",
            len(train_idxs),
            train_idxs_path,
        )
        np.savez_compressed(train_idxs_path, train_idxs)
        logger.info("DONE!")
    return train_data, train_idxs


if __name__ == "__main__":
    parser = utils.define_cl_parser(parser_conf)
    args = parser.parse_args()

    # from bayesianrex import config
    # args.seed = 0
    # # args.checkpoints_dir = config.DEMONSTRATIONS_DIR / "BreakoutNoFrameskip-v4"
    # args.checkpoints_dir = config.ASSETS_DIR / "demonstrators_tiny"
    # args.n_traj = 100
    # args.n_snippets = 200
    # args.n_episodes = 1
    # args.log_level = 1
    # args.train_data_save_dir = config.ASSETS_DIR

    utils.setup_root_logging(args.log_level)
    main(args)

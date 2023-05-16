"""Generate IRL training data: trajectories of states, actions and rewards."""
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import joblib
import numpy as np
import torch
from bayesianrex_cc import config, constants, utils
from bayesianrex_cc.rl_baselines3_zoo.rl_zoo3.utils import ALGOS
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import set_random_seed

RawTrajectory = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_trajectory_from_ckpt(
    ckpt_path: Union[Path, str],
    env: GymEnv,
    n_traj: int,
    seed: Optional[int] = None,
    seed_globally: Optional[bool] = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate demonstrations for an arbitrary environment from a PPO checkpoint.

    :param ckpt_path: (Absolute) path to the zipped checkpoint
    :param env: The gym environment used to generate the checkpoint
    :param n_traj: Number of episodes to run (a trajectory is a full episode)
    :param seed: The RNG seed passed to the environment, increased by `n_traj`
    :param seed_globally: Whether to use StableBaseline's RNG seeding utility
    :return: The states, actions and rewards for each trajectory
    """
    logger.debug("Loading ckpt '%s'", ckpt_path)
    agent = PPO.load(ckpt_path, env, verbose=1)
    env = agent.get_env()
    if seed_globally:
        logger.info("Set global RNG seeds *WARN* should be called once")
        set_random_seed(seed, using_cuda=device.type != "cpu")
    states, actions, rewards = [], [], []
    for i in range(n_traj):
        if seed is not None:
            env.seed(seed + i)
        done = False
        obs = env.reset()
        ep_states, ep_actions, ep_rewards = [], [], []
        while not done:
            ep_states.append(obs)
            action, *_ = agent.predict(obs)
            obs, reward, done, *_ = env.step(action)
            ep_actions.append(action)
            ep_rewards.append(reward)
        states.append(np.array(ep_states).squeeze())
        actions.append(np.array(ep_actions).squeeze())
        rewards.append(np.array(ep_rewards).squeeze())
    return states, actions, rewards


def list_sb3_ckpts(
    starting_dir: Path = None,
    agent_name: Optional[str] = None,
    env_id: Optional[str] = None,
) -> List[Path]:
    """
    Retrieve a list of `stable_baselines3` agent checkpoints.

    :param starting_dir: The folder where the checkpoints are stored, or the
        root of a logs directory created with `rl-baselines3-zoo`
    :param agent_name: The string name of the checkpointed agent
    :param env_id: The Gym id name of the checkpointed environment. `agent_name`
        and `env_id` are needed when `starting_dir` does not directly contain
        checkpoints
    :return: A list of `pathlib.Path` checkpoint filenames
    :raise Exception: If a folder scanned for checkpoints does not provide
        checkpoints (matched by the '*_steps' glob)
    """

    def ckpts_in_dir(folder: Path):
        ckpts = list(folder.glob("*_steps.zip"))
        if not ckpts:
            raise Exception(
                f"Folder {folder} does not contain any ckpt ending with '_steps.zip'"
            )
        return ckpts

    assert starting_dir is not None, "Missing `starting_dir`"
    if env_id in str(starting_dir):
        # assume direct agent/env_version folder was passed, glob directly in
        # this dir
        return ckpts_in_dir(starting_dir)
    else:
        assert agent_name in ALGOS, f"Unknown agent '{agent_name}'"
        assert (
            env_id in constants.envs_id_mapper.values()
        ), f"Unknown environment '{env_id}'"
        # assume the root of the checkpoints directories was passed, descend
        env_versions = (starting_dir / agent_name).glob(f"{env_id}*")
        return (ckpt for env in env_versions for ckpt in ckpts_in_dir(env))


# TODO tqdm somewhere
def gen_trajectories(
    env_name: str, starting_dir: Union[Path, str], seed: int, n_traj_per_ckpt: int = 1
) -> RawTrajectory:
    """
    Generate demonstrations from checkpointed `sb3` PPO agents stored under `starting_dir`.

    :param env_name: The name of the environment used to checkpoint the agent
    :param starting_dir: The root of a `rl-baselines3-zoo` logs folder, or the
        checkpoints folder itself
    :param seed: Starting RNG seed passed to the Gym environment during play
    :param n_traj_per_ckpt: Number of episodes a checkpointed agent is run for
    :return: The trajectories sorted by ground truth return
    .. warning::
      For full reproducibility, `sb3.common.utils.set_random_seed` should be
      called once in a program before calling this function
    """
    env_id = constants.envs_id_mapper.get(env_name)
    ckpts = list_sb3_ckpts(Path(starting_dir), "ppo", env_id)
    env = gym.make(env_id)
    # structure: [(s,a,r,return), (s,a,r,return), ...], size: len(ckpts) * n_traj
    # use the return of a trajectory (sum of rewards) as sorting key
    trajectories = [
        (s, a, r, r.sum())
        for ckpt in ckpts
        for s, a, r in zip(
            *gen_trajectory_from_ckpt(ckpt, env, n_traj=n_traj_per_ckpt, seed=seed)
        )
    ]
    for s, a, r, _ in trajectories:
        assert (
            len(s) == len(a) == len(r)
        ), "State/action/reward chain length mismatch for same trajectory"
    # sort by return
    trajectories = sorted(trajectories, key=lambda el: -el[3])
    # structure: [[states], [actions], [rewards]]
    return tuple(zip(*trajectories))[:3]


def gen_snippet_idx(
    arr: List[np.ndarray],
    target_len: int,
    rng: np.random.Generator,
    legal_len: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Pick two random trajectories at least `target_len` long.

    :param arr: Container of trajectories elements (either states, actions or
        rewards)
    :param target_len: Desired subtrajectory length
    :param rng: A Numpy Random Number Generator
    :param legal_len: Maximum available trajectory length (computed from `arr` if
        not given)
    :return: The indices of the sampled trajectories
    :raise ValueError: If `target_len > max_len`
    """
    legal_len = legal_len or max(len(s) for s in arr)
    if target_len > legal_len:
        raise ValueError(
            f"Desired snippet length: {target_len} max available length: {legal_len}"
        )
    while True:
        i, j = rng.choice(len(arr), size=2, replace=False)
        if len(arr[i]) >= target_len and len(arr[j]) >= target_len:
            break
    return i, j


# TODO a simple strategy to use each transition more is to compare it to other
# transitions in a fixed proportion, e.g. each transition is compared with at
# least 5% other transitions from the total (equal preference representation);
# this might still underrepresent important transitions, could fix by some sort
# of weighting
# TODO plot trj/snippets length distribution (there are many trajectories from
# well-trained policies)
def create_training_data(
    trajectories: RawTrajectory,
    n_traj: int,
    n_snippets: int,
    snippet_min_len: int,
    snippet_max_len: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Assign partial preference labels between trajectories and create training data.

    Give preference labels to `n_traj` random full trajectories. Additionally,
    generate `n_snippets` synthetic (short) trajectories of random length bound
    by [`snippet_min_len`, `snippet_max_len`).

    :param trajectories: A tuple of `np.array`s: (states,actions,rewards), all
        sorted by return
    :param n_traj: Number of full trajectories to compare (training points)
    :param n_snippets: Number of downsampled trajectories to compare (training
        points)
    :param snippet_min_len: Minimum length of downsampled trajectory
    :param snippet_max_len: Maximum length of downsampled trajectory
    :param rng: Numpy Random Number Generator used to sample trajectories and
        snippets lengths
    :return: Trajectories tuples suitable for preference learning, and
        corresponding labels
    """

    def helper(best_idx, worst_idx, snippet_len):
        best_size, worst_size = len(states[best_idx]), len(states[worst_idx])
        # subsample best trajectory choosing earlier states
        if best_size < worst_size:
            best_start = rng.integers(best_size - snippet_len + 1)
        else:
            # when better trajectory is longer than worse, don't sample it too
            # late or there might not be enough states left for the worse one
            best_start = rng.integers(max(worst_size - snippet_len - 20 + 1, 1))
        return best_start, best_start + snippet_len

    states, actions, _ = trajectories
    train_states, train_actions, train_labels = [], [], []
    picked = set()
    for _ in range(n_traj):
        # pick 2 different random trajectories
        i, j = rng.choice(np.arange(len(states)), size=2, replace=False)
        picked.add(i), picked.add(j)
        # NOTE for Atari they pick a random starting state among the first 6 +
        # they downsample
        train_states.append((states[i], states[j]))
        train_actions.append((actions[i], actions[j]))
        train_labels.append(np.array(i < j, dtype=int))

    logger.info(
        "Mean length full train trajectories: %.2f",
        np.mean([len(states[idx]) for idx in picked]),
    )

    snippet_sizes, max_tr_len = [], max(len(s) for s in states)
    for _ in range(n_snippets):
        # pick random subtrajectory length
        snippet_len = rng.integers(snippet_min_len, snippet_max_len)
        logger.debug("Snippet len: %d", snippet_len)
        # pick 2 different random trajectories both at least `snippet_len` long
        i, j = gen_snippet_idx(states, snippet_len, rng, legal_len=max_tr_len)
        logger.debug("i %d j %d i is best: %d", i, j, i < j)

        best, worst = (i, j) if i < j else (j, i)
        best_start, best_end = helper(best, worst, snippet_len)
        worst_end = best_end + snippet_len

        # the better trajectory is always `snippet_len` long, the worse one
        # might be shorter depending on starting point of better snippet
        best_states_snippet = states[best][best_start:best_end]
        worst_states_snippet = states[worst][best_end:worst_end]
        logger.debug(
            "best start %d end %d worst start %d worst end %d",
            best_start,
            best_end,
            best_end,
            worst_end,
        )
        logger.debug(
            "best len %d sampled len %d", len(states[best]), len(best_states_snippet)
        )
        logger.debug(
            "worst len %d sampled len %d", len(states[worst]), len(worst_states_snippet)
        )
        logger.debug("------------------------")

        train_states += [(best_states_snippet, worst_states_snippet)]
        train_actions += [
            (
                actions[best][best_start:best_end],
                actions[worst][best_end:worst_end],
            )
        ]
        snippet_sizes += [len(best_states_snippet), len(worst_states_snippet)]
        train_labels.append(np.array(best == i, dtype=int))

    logger.info("Mean length short train trajectories: %.2f", np.mean(snippet_sizes))
    logger.info("Generated %d training datapoints", len(train_states))
    return train_states, train_actions, train_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=constants.SEED, help="Random seed for experiments"
    )
    parser.add_argument(
        "--env", type=str, help="Name of environment used for checkpointing"
    )
    parser.add_argument(
        "--ckpts-dir",
        type=Path,
        default=config.LOGS_DIR,
        help="Path to folder containing the checkpointed agens",
    )
    parser.add_argument(
        "--n-traj",
        default=100,
        type=int,
        help="Number of full training trajectories",
    )
    parser.add_argument(
        "--n-snippets",
        default=60000,
        type=int,
        help="Number of short subtrajectories to sample",
    )
    parser.add_argument(
        "--snippet-min-len",
        default=20,
        type=int,
        help="Minimum length of subtrajectory",
    )
    parser.add_argument(
        "--snippet-max-len",
        default=60,
        type=int,
        help="Maximum length of subtrajectory",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=int,
        help="Logging level",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        type=Path,
        help="Optional folder where to save trajectories and training data",
    )
    args = parser.parse_args()

    utils.setup_root_logging(args.log_level)

    set_random_seed(args.seed, using_cuda=device.type != "cpu")

    traj = gen_trajectories(args.env, args.ckpts_dir, args.seed)
    train_data = create_training_data(
        traj,
        args.n_traj,
        args.n_snippets,
        args.snippet_min_len,
        args.snippet_max_len,
        np.random.default_rng(seed=args.seed),
    )
    if args.save_dir is not None:
        savedir = args.save_dir
        savedir.mkdir(parents=True, exist_ok=True)
        traj_path = savedir / f"trajectories_{args.env}.gz"
        train_data_path = savedir / f"train_data_{args.env}.gz"
        joblib.dump(dict(zip(["states", "actions", "rewards"], traj)), traj_path)
        joblib.dump(
            dict(zip(["states", "actions", "labels"], train_data)), train_data_path
        )
        logger.info(
            "Saved trajectories to %s and training data to %s",
            traj_path,
            train_data_path,
        )

# TODO log trajectories returns on wandb
import logging
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
        "help": (
            """folder to save trajectories and training data"""
            """\n*NOTE* saving training data might require a lot of time and disk space"""
        ),
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


def training_trajectories(
    trajectories: RawTrajectories, n_traj: int, rng: np.random.Generator
) -> TrainTrajectories:
    states, actions, _ = trajectories
    train_states, train_actions, train_times, labels = [], [], [], []
    # add full trajs (for use on Enduro)
    start = time.time()
    for n in range(n_traj):
        # pick 2 different random trajectories
        i, j = rng.choice(len(states), size=2, replace=False)
        # random downsampled partial trajs with random start and frame skip
        si, sj = rng.choice(6, size=2)
        step = rng.integers(3, 7)
        train_states.append((states[i][si::step], states[j][sj::step]))
        train_actions.append((actions[i][si::step], actions[j][sj::step]))
        train_times.append(
            (np.arange(si, len(states[i]), step), np.arange(sj, len(states[j]), step))
        )
        labels.append(np.array(i < j, dtype=int))
    end = time.time()
    logger.info("Generated %d full trajectories in %.2fs", n_traj, end - start)
    return train_states, train_actions, train_times, labels


# NOTE if training doesn't work, we should try swapping the condition in
# i_betterthan_j. rn trajectories are sorted in *increasing* order of return, so
# a trajectory with a greater index should have label 1 to my understanding. but
# this is reversed in LearnAtariRewardLinear.create_training_data (they also
# sort trajectories increasingly)
# NOTE assumes trajectories are sorted in *increasing* order of return
# FIXME generating 60K snippets takes long, could multiprocessing.map?
def training_snippets(
    trajectories: RawTrajectories,
    n_snippets: int,
    snippet_min_len: int,
    snippet_max_len: int,
    rng: np.random.Generator,
) -> TrainTrajectories:
    def helper():
        start_b = rng.integers(max_samplable_len - rand_len + 1)
        return start_b, rng.integers(start_b, len(states[worst]) - rand_len + 1)

    states, actions, _ = trajectories
    train_states, train_actions, train_times, labels = [], [], [], []
    # FIXED SIZE SNIPPETS WITH PROGRESS PRIOR
    # only sample from trajectories long enough
    states = list(filter(lambda s: len(s) >= snippet_min_len, states))
    start = time.time()
    for n in range(n_snippets):
        # pick 2 different random trajectories
        i, j = rng.choice(len(states), size=2, replace=False)
        # create random snippets without frame downsampling
        # find min length of both demos to ensure we pick a (sub)demo no earlier
        # than that chosen in worse demo, like in the B-Rex codebase
        max_samplable_len = min(len(states[i]), len(states[j]))
        max_rand_len = min(snippet_max_len, max_samplable_len) + 1
        rand_len = rng.integers(snippet_min_len, max_rand_len)

        i_betterthan_j = i > j
        best, worst = (i, j) if i_betterthan_j else (j, i)
        best_s, worst_s = helper()
        best_e, worst_e = best_s + rand_len, worst_s + rand_len

        state_snippets = (states[best][best_s:best_e], states[worst][worst_s:worst_e])
        action_snippets = (
            actions[best][best_s:best_e],
            actions[worst][worst_s:worst_e],
        )
        train_states.append(state_snippets)
        train_actions.append(action_snippets)
        labels.append(np.array(i_betterthan_j, dtype=int))
        train_times.append((np.arange(best_s, best_e), np.arange(worst_s, worst_e)))

        logger.debug(
            (
                """\nbest start %d end %d worst start %d worst end %d\nbest len"""
                """ %d sampled len %d\nworst len %d sampled len %d\n"""
                """------------------------"""
            ),
            best_s,
            best_e,
            worst_s,
            worst_e,
            len(states[best]),
            len(train_states[-1][0]),
            len(states[worst]),
            len(train_states[-1][1]),
        )
    end = time.time()
    logger.info("Generated %d snippet trajectories in %.2fs", n_snippets, end - start)
    return train_states, train_actions, train_times, labels


def create_training_data(
    trajectories: RawTrajectories,
    n_traj: int,
    n_snippets: int,
    rng: np.random.Generator,
    snippet_len_bounds: Tuple[int],
) -> TrainTrajectories:
    full_demos = training_trajectories(trajectories, n_traj, rng)
    snippets = training_snippets(trajectories, n_snippets, *snippet_len_bounds, rng)
    return [d + s for s, d in zip(snippets, full_demos)]


def main(args: Namespace) -> Tuple[TrainTrajectories, RawTrajectories]:
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
    training_data = create_training_data(
        trajectories,
        args.n_traj,
        args.n_snippets,
        np.random.default_rng(args.seed),
        (args.snippet_min_len, args.snippet_max_len),
    )
    return training_data, trajectories


if __name__ == "__main__":
    parser = utils.define_cl_parser(parser_conf)
    args = parser.parse_args()

    # from bayesianrex import config
    # args.seed = 0
    # args.checkpoints_dir = config.DEMONSTRATIONS_DIR / "BreakoutNoFrameskip-v4"
    # args.n_traj = 100
    # args.n_snippets = 2000
    # args.log_level = 1

    utils.setup_root_logging(args.log_level)
    training_data, trajectories = main(args)

    # NOTE saving training_data['states'] takes loads of time and 2+GB with the
    # default number of snippets and full trajectories
    savedir = args.train_data_save_dir
    if savedir is not None:
        savedir.mkdir(parents=True, exist_ok=True)

        traj_path = savedir / f"trajectories_{args.env}"
        trajectories_dict = dict(zip(["states", "actions", "rewards"], trajectories))
        logger.info("Saving %d trajectories to %s...", len(trajectories[0]), traj_path)
        joblib.dump(trajectories_dict, traj_path, compress=True)

        train_data_path = savedir / f"train_data_{args.env}"
        train_data_dict = dict(
            zip(["states", "actions", "times", "labels"], training_data)
        )
        logger.info(
            "Saving %d training datapoints to %s...",
            len(training_data[0]),
            train_data_path,
        )
        joblib.dump(train_data_dict, train_data_path, compress=True)
        logger.info("DONE!")

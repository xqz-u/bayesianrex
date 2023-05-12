"""Train PPO demonstrators, logging to wandb and saving intermediate checkpoints."""
import pprint as pp
import shlex
import signal
import subprocess as sp

import yaml

import config
import constants

# NOTE add --eval-freq -1 to avoid evaluation steps
if __name__ == "__main__":
    # NOTE idk if this works on Windows
    python_bin = sp.run(
        ["which", "python3"], capture_output=True, text=True
    ).stdout.strip()
    print(f"Using python3 executable '{python_bin}'")

    with open(config.CODE_DIR / "rl-baselines3-zoo/hyperparams/ppo.yml") as fd:
        ppo_conf = yaml.safe_load(fd)

    for i, env_id in enumerate(constants.envs_id_mapper.values()):
        # NOTE use ~25 checkpoints per game
        # (not extactly 25 if save_freq % n_envs != 0)
        ckpt_freq = int(ppo_conf[env_id]["n_timesteps"] / 25)
        args = (
            f"""{python_bin} -m rl-baselines3-zoo.train """
            f"""--seed {i} --algo ppo --env {env_id} --track """
            """--wandb-project-name sb3 --wandb-entity bayesianrex-dl2 """
            """--wandb-tags demonstrator-cc --progress --verbose 0 """
            f"""--save-freq {ckpt_freq} --log-folder {config.LOGS_DIR} """
            f"""--wandb-dir {config.WANDB_LOGS_DIR} -tb {config.TB_LOGS_DIR}"""
        )
        args = shlex.split(args)
        print("Command:")
        pp.pprint(args)

        try:
            p = sp.Popen(args)
            p.wait()
        except KeyboardInterrupt:
            p.send_signal(signal.SIGINT)
            print("Process interrupted")

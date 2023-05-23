# NOTE use *NoFrameskip-v4 versions of environments according to
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.AtariWrapper
envs_id_mapper = {
    "breakout": "BreakoutNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "spaceinvaders": "SpaceInvadersNoFrameskip-v4",
    "enduro": "EnduroNoFrameskip-v4",
    "montezumarevenge": "MontezumaRevengeNoFrameskip-v4",
    "hero": "HeroNoFrameskip-v4",
    "beamrider": "BeamRiderNoFrameskip-v4",
    "seaquest": "SeaquestNoFrameskip-v4",
}

reward_net_latent_space = 64
reward_net_hparams = {"lr": 1e-4, "weight_decay": 1e-3}

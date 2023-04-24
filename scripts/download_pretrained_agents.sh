#!/bin/bash

# TODO download to correct data partition, not user directory

# these are the PPO2 pretrained models used to generate the
# demonstrations for imitation learning
pretrained_models_url="https://github.com/dsbrown1331/learning-rewards-of-learners/releases/download/atari25"

tags="beamrider_25.tar.gz
breakout_25.tar.gz
enduro_25.tar.gz
hero_25.tar.gz
pong_25.tar.gz
qbert_25.tar.gz
seaquest_25.tar.gz
seaquest_5.tar.gz
spaceinvaders_25.tar.gz"

models_dir=$HOME/bayesianrex/models
mkdir $models_dir
cd $models_dir

while IFS= read -r env; do
    cmd="wget $pretrained_models_url/$env"
    echo $cmd
    $cmd
    cmd="tar -xf $env"
    echo $cmd
    $cmd
done <<< "$tags"

echo "Done" && exit 0

#!/bin/bash

# TODO download to correct data partition, not user directory
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

mkdir models
cd models

while IFS= read -r env; do
    cmd="wget $pretrained_models_url/$env"
    echo $cmd
    $cmd
    cmd="tar -xf $env"
    echo $cmd
    $cmd
done <<< "$tags"

echo "Done" && exit 0

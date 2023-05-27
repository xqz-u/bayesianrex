#!/bin/bash

# assumes to be run from the root of the repo
project_root=$PWD
env_file="$project_root/env.yml"

if ! [ -f $env_file ]; then
    echo "Cannot find env file $env_file; run this script from the repo's root."
    exit 1
fi

cd $HOME
echo "Installing micromamba"
curl micro.mamba.pm/install.sh | bash

micromamba_bin=$HOME/.local/bin/micromamba

echo "Installing from env file $code_dir/env.gpu.yml"
$micromamba_bin create -f $code_dir/env.gpu.yml

echo "Environment creation succesful"

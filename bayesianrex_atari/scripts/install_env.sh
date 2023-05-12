#!/bin/bash

code_dir=$HOME/bayesianrex

cd $HOME

echo "Installing micromamba..."
curl micro.mamba.pm/install.sh | bash

micromamba_bin=$HOME/.local/bin/micromamba

echo "Installing from env file $code_dir/env.yml..."
$micromamba_bin create -f $code_dir/env.yml

echo "DONE!"

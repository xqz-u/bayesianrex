#!/bin/bash

# we use Conda to source the environment since it's not as straightforward
# with MicroMamba, but use the latter for environment management (it's faster,
# same interface as Conda and allows easy environment modification)

module load 2022
module load Anaconda3/2022.05

# with Conda the full prefix is needed since the env lives outside of
# default $HOME/.conda
source activate $HOME/micromamba/envs/bayesianrex-dl2

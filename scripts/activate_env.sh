#!/bin/bash

# NOTE unnecessary if micromamba is already in $PATH
micromamba_bin=$HOME/.local/bin/micromamba

$micromamba_bin activate bayesianrex-dl2

cd $HOME/bayesianrex

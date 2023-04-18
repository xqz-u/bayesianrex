#!/bin/bash

module purge
module load 2022
module load Mamba/4.14.0-0
# needed or openAI baselines stuff does not work on compute nodes
module load OpenMPI/4.1.4-GCC-11.3.0

source activate bayesianrex

cd $HOME/bayesianrex
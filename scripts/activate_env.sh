#!/bin/bash

module purge
module load 2022
module load Mamba/4.14.0-0

source activate bayesianrex

cd $HOME/bayesianrex
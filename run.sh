#!/usr/bin/env bash
#
#SBATCH --job-name dmbal
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

# venv
source ./venv/test/bin/activate
export PYTHONPATH=$PYTHONPATH:./

python3 ./scripts/dmbal.py

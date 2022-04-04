#!/usr/bin/env bash

# VAE_CF
sbatch --nodes=1 --time=01:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_CF.sh

# DCE_VAE
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE.sh




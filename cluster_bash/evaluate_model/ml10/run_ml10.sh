#!/usr/bin/env bash

# VAE_CF
sbatch --nodes=1 --time=01:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAE_CF.sh

# DCE_VAE
sbatch --nodes=1 --time=06:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL DCE_VAE.sh


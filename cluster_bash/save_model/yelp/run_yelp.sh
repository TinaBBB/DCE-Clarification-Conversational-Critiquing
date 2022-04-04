#!/usr/bin/env bash

# DCE-VAE
sbatch --nodes=1 --time=03:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer_contrast.sh

# VAE-CF
sbatch --nodes=1 --time=00:05:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer.sh

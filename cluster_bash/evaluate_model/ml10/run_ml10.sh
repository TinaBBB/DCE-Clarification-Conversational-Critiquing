#!/usr/bin/env bash

sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast1.sh


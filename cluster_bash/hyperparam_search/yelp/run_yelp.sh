#!/usr/bin/env bash

# baseline VAE-CF model
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_CF1.sh  #invalid learning rate
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_CF2.sh
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_CF3.sh
sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_CF4.sh

# DCE-VAE model
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE5.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner DCE_VAE6.sh


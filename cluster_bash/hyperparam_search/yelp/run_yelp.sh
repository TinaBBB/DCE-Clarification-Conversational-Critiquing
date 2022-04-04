#!/usr/bin/env bash

# baseline VAE model
#sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEmultilayer1.sh  #invalid learning rate
#sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEmultilayer2.sh
#sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEmultilayer3.sh
#sbatch --nodes=1 --time=48:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEmultilayer4.sh

# DCE-VAE model
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning5.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning6.sh


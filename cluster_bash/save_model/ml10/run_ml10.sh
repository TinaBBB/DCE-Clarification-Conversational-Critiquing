#!/usr/bin/env bash

# VAE multilayer

#sbatch --nodes=1 --time=01:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer.sh

sbatch --nodes=1 --time=01:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast1.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEmultilayer_contrast1.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast3.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast4.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast5.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast6.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast7.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast8.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast9.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast10.sh

# this model has smaller number of positive samples
#sbatch --nodes=1 --time=03:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast1.sh


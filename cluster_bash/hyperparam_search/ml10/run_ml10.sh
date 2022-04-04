#!/usr/bin/env bash

# Try with  baseline VAEs
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer1.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer2.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer3.sh
#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEmultilayer4.sh

# Try with different sample sizes
#sbatch --nodes=1 --time=96:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner VAEcontrast_samples1.sh
#sbatch --nodes=1 --time=96:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_samples2.sh
#sbatch --nodes=1 --time=96:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_samples3.sh
#sbatch --nodes=1 --time=96:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_samples4.sh


# Formal tuning
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning1.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning2.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning3.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning4.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning5.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning6.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning7.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning8.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning9.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning10.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning11.sh
sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_tuning12.sh

#sbatch --nodes=1 --time=96:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast_finetuning1.sh

# Try with different temperature tau



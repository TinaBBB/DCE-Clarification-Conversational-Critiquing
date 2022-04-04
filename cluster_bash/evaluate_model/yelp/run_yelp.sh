#!/usr/bin/env bash

sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast1.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast2.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast3.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast4.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast5.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast6.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast7.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast8.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEcontrast9.sh
sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner --mail-user=tina.shen@mail.utoronto.ca --mail-type=ALL VAEcontrast10.sh


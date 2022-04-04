#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/bayesian-critiquing-recommender


python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise0.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise3.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise5.config --top_items 15 --top_users 8000


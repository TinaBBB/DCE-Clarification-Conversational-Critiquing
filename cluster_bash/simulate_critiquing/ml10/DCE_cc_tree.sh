#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings

# Experiment 2. DCE-Tree
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise3_distributional.config --top_items 15 --top_users 8000

# Experiment 3. personalized & non-personalized clarification
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise0_distributional.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise0_distributionalNoUserEmbed.config --top_items 15 --top_users 8000

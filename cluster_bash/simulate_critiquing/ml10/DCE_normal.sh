#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings

# Experiment 1. DCE-normal
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise10_normal.config --top_items 15 --top_users 8000 -expNum Exp1_Pure_Critique --methodName DCE-normal

# Experiment 3. without clarification
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise0_normal.config --top_items 15 --top_users 8000 --expNum Exp3_Clarification_Performance --methodName normal
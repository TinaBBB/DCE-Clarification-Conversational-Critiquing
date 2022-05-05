#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings


# Experiment 1. DCE-expert
python simulate_yelp.py --saved_model models_DCE-VAE/VAEmultilayer_contrast3.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg10_noise0_expert.config --top_items 10


#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings

# Experiment 2. DCE-Tree
python simulate_yelp.py --saved_model models_DCE-VAE/VAEmultilayer_contrast3.pt --data_name yelp --data_dir fold0 --conf sim_abs_diff_neg100_noise0_distributional.config --top_items 10 --expNum Exp2_Clarification_Critique --methodName DCE-Tree

# Experiment 3. personalized & non-personalized clarification
python simulate_yelp.py --saved_model models_DCE-VAE/VAEmultilayer_contrast3.pt --data_name yelp --data_dir fold0 --conf sim_abs_diff_neg100_noise0_distributional.config --top_items 10  --expNum Exp3_Clarification_Performance --methodName distributional
python simulate_yelp.py --saved_model models_DCE-VAE/VAEmultilayer_contrast3.pt --data_name yelp --data_dir fold0 --conf sim_abs_diff_neg100_noise0_distributionalNoUserEmbed.config --top_items 10  --expNum Exp3_Clarification_Performance --methodName distributionalNoUserEmbed
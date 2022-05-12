#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/DCE-Clarification-Conversational-Critiquing

# Experiment 2. BK-NN
python simulate.py --saved_model models_VAE/VAE_multilayer1.pt --data_name ml10 --data_dir fold0 --conf sim_abs_diff_neg1_noise3_NN.config --top_items 15 --top_users 8000 --expNum Exp2_Clarification_Critique --methodName BK-NN
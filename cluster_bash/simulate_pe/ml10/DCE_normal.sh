#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings
#cd ~/code/vae-pe


python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise0_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise3_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise5_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise8_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise10_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg10_noise0_normal.config --top_items 15 --top_users 8000
python simulate.py --saved_model models_DCE-VAE/VAEcontrast1.pt --data_name ml10_SIGIR --data_dir fold0 --conf sim_abs_diff_neg100_noise0_normal.config --top_items 15 --top_users 8000
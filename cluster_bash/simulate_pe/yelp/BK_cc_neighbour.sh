#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings
#cd ~/code/vae-pe

# NN KAV clarification
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise0_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg0_noise0_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise3_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise5_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise8_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise10_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg10_noise0_NN.config --top_items 10
python simulate_yelp.py --saved_model models_VAE/VAEmultilayer1.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg100_noise0_NN.config --top_items 10
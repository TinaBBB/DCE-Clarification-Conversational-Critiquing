#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/lijiaru/projects/def-ssanner/lijiaru/VAE_uncertainty_word_embeddings

python model_evaluate.py --model_name VAEmultilayer_contrast --data_name yelp_SIGIR --conf VAEcontrast5.config --log_dir VAEcontrast5 --top_items 10 --rating_threshold 3

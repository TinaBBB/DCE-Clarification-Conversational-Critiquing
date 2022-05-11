#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings

python model_evaluate.py --model_name VAEmultilayer_contrast --data_name ml10_SIGIR --conf VAEmultilayer_contrast1.config --log_dir VAEmultilayer_contrast1 --top_items 15 --top_users 8000





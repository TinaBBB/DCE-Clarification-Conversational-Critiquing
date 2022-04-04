#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/lijiaru/projects/def-ssanner/lijiaru/VAE_uncertainty_word_embeddings

python model_evaluate.py --model_name VAEmultilayer_contrast --data_name yelp_SIGIR --data_dir fold0 --conf VAEmultilayer_contrast3.config --log_dir VAEmultilayer_contrast3 --top_items 10

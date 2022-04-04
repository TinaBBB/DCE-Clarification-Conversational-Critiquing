#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/lijiaru/projects/def-ssanner/lijiaru/VAE_uncertainty_word_embeddings

python model_evaluate.py --model_name VAEmultilayer --data_name yelp_SIGIR --data_dir fold0 --conf VAE_multilayer1.config --log_dir VAE_multilayer1 --top_items 10

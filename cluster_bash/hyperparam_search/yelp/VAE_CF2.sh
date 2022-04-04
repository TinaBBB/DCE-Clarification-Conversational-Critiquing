#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/lijiaru/projects/def-ssanner/lijiaru/VAE_uncertainty_word_embeddings

python hp_search.py --model_name VAEmultilayer --data_name yelp_SIGIR --conf VAEmultilayer2.config --fold_name fold0_valid

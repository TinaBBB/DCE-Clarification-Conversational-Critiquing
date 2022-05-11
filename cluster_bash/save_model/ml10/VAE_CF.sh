#!/usr/bin/env bash
module load python/3.7
source ~/vae_uncertainty/bin/activate

cd /home/tinashen/projects/def-ssanner/tinashen/VAE_uncertainty_word_embeddings

python model_save.py --model_name VAEmultilayer --data_name ml10_SIGIR --data_dir fold0 --conf VAEmultilayer1.config --log_dir VAEmultilayer1 --top_items 15 --top_users 8000



# WWW'22 Distributional Contrastive Embedding for Clarification-based Conversational Critiquing

Authors: 
[Tianshu Shen](https://tinabbb.github.io/), 
[Zheda Mai](https://zheda-mai.github.io/), 
[Ga Wu](https://wuga214.github.io/), 
[Scott Sanner](https://d3m.mie.utoronto.ca/)

All the compiling scripts can be found under `./cluster_bash/`, where experiments were ran using Compute Canada  

# Hyperparameter tuning:
* configuration folder: `./conf_hp_search/` <br>
For example, to run the hyperparameter-tuning script for the model using one of the configurations, run the following:

```
python hp_search.py --model_name VAEmultilayer_contrast --data_name yelp_SIGIR --conf VAEcontrast_tuning1.config --fold_name fold0 --top_items 10
```

# Model Evaluation:
* configuration folder: `./conf/` <br>
To run model evaluation for a set of hyperparameters:
```
python model_evaluate.py --model_name VAEmultilayer_contrast --data_name yelp_SIGIR --conf VAEcontrast1.config --log_dir VAEcontrast1 --top_items 10 --rating_threshold 3
```

# Save Model: 
* configuration folder: `./conf/` <br>
To save the model after the model evaluation step:
```
python model_save.py --model_name VAEmultilayer_contrast --data_name yelp_SIGIR --data_dir fold0 --conf VAEmultilayer_contrast2.config --log_dir VAEmultilayer_contrast2 --top_items 10
```


# Simulate Critiquing:
* configuration folder: `./conf_simulate/`
All the configurations for different critiquing/clarification-based critiquing tasks are listed under the specified folder. For example, to run the simulation task for DCE's experiment critiquing scenario using the Yelp dataset:
```
python simulate_yelp.py --saved_model models_DCE-VAE/VAEmultilayer_contrast3.pt --data_name yelp_SIGIR --data_dir fold0 --conf sim_abs_diff_neg1_noise0_expert.config --top_items 10
```

# Citation
Please cite:

```
@article{shen2022distributional,
    title={Distributional Contrastive Embedding for Clarification-based Conversational Critiquing},
    author={Shen, Tianshu and Mai, Zheda and Wu, Ga and Sanner, Scott},
    year={2022}
  }
```

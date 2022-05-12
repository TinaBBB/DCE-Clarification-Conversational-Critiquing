import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.Dataset import Dataset
from utils.HierarchicalClustering import HierarchicalClustering
from utils.KAVgenerator import KAVgenerator
from utils.Simulator import Simulator
from utils.io import save_dataframe_csv

DATA_PATH = Path("./data")
MODEL_PATH = Path("./saves")
CONFIG_PATH = Path("./conf_simulate")
TABLE_PATH = "./tables"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ml10')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--saved_model', type=str, default='VAE_beta_multilayer.pt')
    parser.add_argument('--conf', type=str, default='sim_abs_diff_neg1_noise0_expert.config')
    parser.add_argument('--top_items', type=int, default=15, help='used to indicate top labels for each item')
    parser.add_argument('--kernel_method', type=str, default='BK', help='kernel method for computation')
    parser.add_argument('--expNum', type=str, help='experiment number for the paper')
    parser.add_argument('--methodName', type=str, help='method name for plotting the figure')
    parser.add_argument('--top_users', type=int, help='if cuting the matrix with top user numbers')
    parser.add_argument('--rejection_threshold', type=int, default=3,
                        help='rejection of threshold times will turn off the clarification stage')
    parser.add_argument('--rating_threshold', type=float, default=1,
                        help='used to indicate user liked items for generating uk matrices')
    parser.add_argument('--save_examples', action='store_true',
                        help='whether storing the simulation examples')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = "{}/{}/{}/".format(DATA_PATH, p.data_name, p.data_dir)
    model_dir = MODEL_PATH / p.data_name / p.saved_model
    config_dir = CONFIG_PATH / p.data_name / p.conf
    table_dir = "{}/{}_experiment_results/{}/".format(TABLE_PATH, p.data_name, p.expNum)
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)
    print(config_dir)

    with open(config_dir) as f:
        conf = json.load(f)

    # load model
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    item_embeddings = model.decoder.weight.detach().numpy()

    # load data
    dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold,
                      top_users=p.top_users)

    # generate keyphrase embedding
    k = KAVgenerator(dataset.train_item_keyphrase_matrix, item_embeddings, 1)
    keyphrase_embeddings = k.get_all_mean_kav(20, 20)

    # generate keyphrase hierarchies based on distributional embeddings, not generated for VAE baseline
    if 'distributional' not in conf['clarification_type']:
        keyphrase_hierarchies = None
    else:
        keyphrase_hierarchies = HierarchicalClustering(model, dataset)
        keyphrase_hierarchies.getKeyphraseIndexHierarchy(kernel_method=p.kernel_method)

    # load df
    experiment_df = pd.DataFrame(columns=['user_id', 'step', 'critiquing_keyphrase', 'critique_polarity',
                                          'clarification_keyphrase', 'clarification_polarity',
                                          'accept_clarification'])

    conf['rejection_threshold'] = p.rejection_threshold
    conf['kernel_method'] = p.kernel_method

    s = Simulator(dataset=dataset, model=model, keyphrase_embeddings=keyphrase_embeddings,
                  keyphrase_hierarchies=keyphrase_hierarchies,
                  item_keyphrase_matrix=dataset.train_item_keyphrase_matrix,
                  sim_conf=conf, experiment_df=experiment_df)
    r = s.simulate_hr()
    save_dataframe_csv(pd.DataFrame(r), table_dir, p.methodName)

    # also save experiment df table
    if p.save_examples:
        save_dataframe_csv(s.experiment_df, table_dir, p.methodName + '_examples')

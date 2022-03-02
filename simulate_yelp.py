
import argparse
import json
import numpy as np
import pandas as pd
import torch
from anytree.exporter import DotExporter
from pathlib import Path
from utils.Dataset import Dataset
from utils.HierarchicalClustering import HierarchicalClustering
from utils.KAVgenerator import KAVgenerator
from utils.Simulator import Simulator
from utils.io import save_dataframe_csv

DATA_PATH = "./data"
MODEL_PATH = Path("./saves")
CONFIG_PATH = Path("./conf_simulate")
TABLE_PATH = "./tables"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='yelp_SIGIR')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--saved_model', type=str, default='VAEcontrast8.pt')
    parser.add_argument('--conf', type=str, default='sim_abs_diff_neg1_noise0_expert.config')
    parser.add_argument('--top_items', type=int, default=10, help='used to indicate top labels for each item')
    parser.add_argument('--kernel_method', type=str, default='BK', help='kernel method for computation')
    parser.add_argument('--rejection_threshold', type=int, default=3,
                        help='rejection of threshold times will turn off the clarification stage')
    parser.add_argument('--rating_threshold', type=float, default=1,
                        help='used to indicate user liked items for generating uk matrices')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = "{}/{}/{}/".format(DATA_PATH, p.data_name, p.data_dir)
    model_dir = MODEL_PATH / p.data_name / p.saved_model
    config_dir = CONFIG_PATH / p.data_name / p.conf
    table_dir = "{}/{}/{}/{}/".format(TABLE_PATH, p.data_name, p.saved_model.split('.pt')[0], 'c-critiquing')
    print(config_dir)

    with open(config_dir) as f:
        conf = json.load(f)

    # load model
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    item_embeddings = model.decoder.weight.detach().numpy()

    # load data
    dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold)

    # generate keyphrase embedding
    k = KAVgenerator(dataset.train_item_keyphrase_matrix, item_embeddings, 20)
    keyphrase_embeddings = k.get_all_mean_kav(20, 20)

    # generate keyphrase hierarchies based on distributional embeddings, not generated for VAE baseline
    if 'distributional' not in conf['clarification_type']:
        keyphrase_hierarchies = None
    else:
        keyphrase_hierarchies = HierarchicalClustering(model, dataset, top_k_range=range(2, 3), lower_k_range=range(2, 3))  #
        keyphrase_hierarchies.getKeyphraseIndexHierarchy(kernel_method=p.kernel_method)

    # load df
    experiment_df = pd.DataFrame(columns=['user_id', 'step', 'critiquing_keyphrase', 'critique_polarity',
                               'clarification_keyphrase', 'clarification_polarity', 'accept_clarification'])

    conf['rejection_threshold'] = p.rejection_threshold
    dataset.train_item_keyphrase_matrix[dataset.train_item_keyphrase_matrix < 20] = 0
    conf['kernel_method'] = p.kernel_method

    s = Simulator(dataset=dataset, model=model, keyphrase_embeddings=keyphrase_embeddings,
                  keyphrase_hierarchies=keyphrase_hierarchies, item_keyphrase_matrix=dataset.train_item_keyphrase_matrix,
                  sim_conf=conf, experiment_df=experiment_df)

    r = s.simulate_hr()
    save_dataframe_csv(pd.DataFrame(r), table_dir, p.conf.split(".")[0])

    # also save experiment df table
    save_dataframe_csv(s.experiment_df, table_dir, p.conf.split('.config')[0]+'_examples')
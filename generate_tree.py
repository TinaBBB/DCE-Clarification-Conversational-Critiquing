import argparse
import collections
import json
import os

import numpy as np
import pandas as pd
import torch
import torchtext
from anytree.exporter import DotExporter
from pathlib import Path

from utils.Dataset import Dataset
from utils.HierarchicalClustering import HierarchicalClustering
from utils.io import save_dataframe_csv

DATA_PATH = "./data"
MODEL_PATH = Path("./saves")
CONFIG_PATH = Path("./conf_simulate")
TABLE_PATH = "./tables"


def get_children(node, c_list):
    for child in node.children:
        c_list.append(child)
        get_children(child, c_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--data_dir', type=str, default='fold0')
    parser.add_argument('--saved_model', type=str, default='models_DCE-VAE/VAEmultilayer_contrast2.pt')
    parser.add_argument('--conf', type=str, default='sim_abs_diff_neg1_noise0_distributional.config')
    parser.add_argument('--top_items', type=int, default=10, help='used to indicate top labels for each item')
    parser.add_argument('--top_users', type=int, help='if cuting the matrix with top user numbers')
    parser.add_argument('--kernel_method', type=str, default='ELK', help='kernel method for computation')
    parser.add_argument('--rating_threshold', type=float, default=1,
                        help='used to indicate user liked items for generating uk matrices')
    parser.add_argument('--seed', type=int, default=201231)
    p = parser.parse_args()

    np.random.seed(p.seed)
    torch.random.manual_seed(p.seed)

    data_dir = "{}/{}/{}/".format(DATA_PATH, p.data_name, p.data_dir)
    # model_dir = MODEL_PATH / p.data_name / (p.data_dir + '-' + p.saved_model)
    model_dir = MODEL_PATH / p.data_name / p.saved_model
    config_dir = CONFIG_PATH / p.data_name / p.conf
    table_dir = "experiments/cluster_distances/{}/".format(p.data_name)
    with open(config_dir) as f:
        conf = json.load(f)
    # load model
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    item_embeddings = model.decoder.weight.detach().numpy()

    # 1. load data
    dataset = Dataset(data_dir=data_dir, top_keyphrases=p.top_items, rating_threshold=p.rating_threshold,
                      top_users=p.top_users)

    # 2. generate keyphrase hierarchies based on distributional embeddings, not generated for VAE baseline
    if conf['clarification_type'] != 'distributional':
        keyphrase_hierarchies = None
    else:
        keyphrase_hierarchies = HierarchicalClustering(model, dataset)
        keyphrase_hierarchies.getKeyphraseHierarchy(kernel_method=p.kernel_method)

        # Save KKT figure
        save_figure_dir = 'experiments/kkt_figures/{}/'.format(p.data_name)
        if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)

        DotExporter(keyphrase_hierarchies.root_node, indent=2,
                    nodeattrfunc=lambda node: "shape=plaintext, width=.01, height=.01, margin=0.01, fontsize=20").to_picture(
            "{}tree_{}.pdf".format(save_figure_dir, p.kernel_method))

        print('KKT figure saved to', save_figure_dir + 'tree_{}.pdf'.format(p.kernel_method))

    # 3. get inter- and intra- cluster distances
    # Get the glove embedding keys
    # The first time you run this will download a ~823MB file
    glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus
                                  dim=50)  # embedding size = 100
    gloveEmbed_dict = {key: glove[key] for key in keyphrase_hierarchies.node_dict.keys()}

    # start queue with the root node
    queue = collections.deque([keyphrase_hierarchies.root_node])
    intraCluster_dist = []
    interCluster_dist = []

    while queue:
        size = len(queue)
        for index in range(size):
            node = queue.popleft()
            children_list = list(node.children)

            # loop through the children node
            for firstChild_idx, child in enumerate(children_list):
                queue.append(child)

                # S_bar
                S_bar = gloveEmbed_dict[child.name]
                # not able to retrieve the glove embedding for this cluster's centroid node
                if sum(S_bar) == torch.zeros([]):
                    continue

                # 1. calculate intra-cluster distance, child's all children
                all_children = []
                get_children(child, all_children)
                # get children glove embeddings
                stack_list = [gloveEmbed_dict[child_node.name] \
                              for child_node in all_children if
                              sum(gloveEmbed_dict[child_node.name]) != torch.zeros([])]
                if not stack_list:
                    continue
                subChildren_glove = torch.stack(stack_list, dim=0)
                temp_intraDist = [torch.norm(s_glove - S_bar) for s_glove in subChildren_glove]
                intraCluster_dist.append(np.mean(temp_intraDist))

                # 2. calculate inter-cluster distance
                for secondChild_idx in range(firstChild_idx + 1, len(children_list)):
                    other_child = children_list[secondChild_idx]
                    if other_child == child: continue
                    # T_bar
                    T_bar = gloveEmbed_dict[other_child.name]

                    if sum(T_bar) == torch.zeros([]): continue
                    all_children_others = []
                    get_children(other_child, all_children_others)
                    # get children glove embeddings
                    stack_list = [gloveEmbed_dict[child_node.name] \
                                  for child_node in all_children_others if
                                  sum(gloveEmbed_dict[child_node.name]) != torch.zeros([])]
                    if not stack_list:
                        continue
                    other_subChildren_glove = torch.stack(stack_list)
                    temp_interDist = [torch.norm(s_glove - T_bar) for s_glove in subChildren_glove] + [torch.norm(t_glove - S_bar) for
                                                                                                       t_glove in other_subChildren_glove]

                    interCluster_dist.append(np.mean(temp_interDist))

    # get statistics for the distance results
    intraCluster_dist = [np.average(intraCluster_dist), 1.96 * np.std(intraCluster_dist) / np.sqrt(len(intraCluster_dist))]
    interCluster_dist = [np.average(interCluster_dist), 1.96 * np.std(interCluster_dist) / np.sqrt(len(interCluster_dist))]
    print('intra distance: {}@{}'.format(intraCluster_dist[0], intraCluster_dist[1]))
    print('inter distance: {}@{}'.format(interCluster_dist[0], interCluster_dist[1]))

    # save statistics
    stats_dict = {'conf': p.conf,
                  'data_dir': p.data_dir,
                  'data_name': p.data_name,
                  'kernel_method': p.kernel_method,
                  'rating_threshold': p.rating_threshold,
                  'saved_model': p.saved_model,
                  'seed': p.seed,
                  'top_items': p.top_items,
                  'top_users': p.top_users,
                  'intra_distance': str(intraCluster_dist),
                  'inter_distance': str(interCluster_dist)},

    save_dataframe_csv(pd.DataFrame(data=stats_dict, index=[0]), table_dir, 'intraInter_distances')

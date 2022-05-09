import numpy as np
import torch
import warnings
from kneed import KneeLocator
from sklearn.cluster import KMeans
from utils.Tools import kernel_selection

warnings.filterwarnings('ignore')

# for tree
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


# model keyphrase gaussian distributions in
# hierarchical clustering results (tree) out

class HierarchicalClustering(object):
    def __init__(self, model, dataset, n_init=50, max_iter=500, random_state=42,
                 top_k_range=range(2, 11), lower_k_range=range(2, 7)):
        self.model = model

        # will get a dictionary for {"keyphrase": node}
        self.node_dict = {}
        # self.node_dict_k = {}   #just investigating for fun, would cause memory
        self.dataset = dataset
        self.kmeans_kwargs = {
            "init": "k-means++",
            "n_init": n_init,
            "max_iter": max_iter,
            "random_state": random_state}
        self.top_k_range = top_k_range
        self.lower_k_range = lower_k_range
        self.root_node = None
        # self.root_node_k = None  # investigating for fun, woudl cause memory

    # Get hierarchy for keyphrase indexes
    def getKeyphraseIndexHierarchy(self, kernel_method):
        mean_embeddings_k = self.model.keyphrase_mu.weight.data
        var_embeddings_k = torch.exp(self.model.keyphrase_log_var.weight.data)
        self.root_node = self.recursClustering(parent_node=None, mean_embeddings=mean_embeddings_k,
                                               var_embeddings=var_embeddings_k,
                                               keyphrase_idx=np.array(range(len(self.dataset.keyphrase_2_idx))),
                                               kernel_method=kernel_method)

    # Get hierarchy for keyphrase names
    def getKeyphraseHierarchy(self, kernel_method):
        mean_embeddings_k = self.model.keyphrase_mu.weight.data
        var_embeddings_k = torch.exp(self.model.keyphrase_log_var.weight.data)
        keyphrase_array = np.array(
            [self.dataset.idx_2_keyphrase[idx] for idx in range(len(self.dataset.keyphrase_2_idx.keys()))])
        self.root_node = self.recursClustering(parent_node=None, mean_embeddings=mean_embeddings_k,
                                               var_embeddings=var_embeddings_k,
                                               keyphrase_idx=keyphrase_array,
                                               kernel_method=kernel_method)

    def recursClustering(self, parent_node, mean_embeddings, var_embeddings, keyphrase_idx, kernel_method):
        # pass down a parent node, and keyphrase embedding data for clustering
        '''
            parent_node: parent node for this cluster of data passed in
            mean_embeddings, var_embeddings: the gaussian embeddings for the keyphrases
            keyphrase_idx: the real keyphrase idx in the corpus for the current cluster

        '''
        # Do this thing for each layer
        # 1. find best number of clusters (use mean)
        # 2. find kemans (use mean )
        # 3. get labels for each cluster
        # 4. find the representing keyphrase (use mean and var)
        # 5. child clusters enter recursion

        assert len(keyphrase_idx) == len(mean_embeddings)
        assert len(keyphrase_idx) == mean_embeddings.shape[0]
        assert mean_embeddings.shape == var_embeddings.shape

        # bottom condition
        if len(keyphrase_idx) <= 2:
            for k_idx in keyphrase_idx:  # assign parent node for the buttom children
                self.node_dict[k_idx] = Node(k_idx, parent=parent_node)
            return

        '''1-3'''
        num_clusters = self.find_num_cluster(
            set_k_range(mean_embeddings, self.top_k_range if parent_node is None else self.lower_k_range),
            mean_embeddings)
        kmeans = self.fit_kmeans(num_clusters, mean_embeddings)
        cluster_labels = kmeans.labels_

        '''4 find the representing keyphrase'''
        # Get K * K kernel matrix
        kernel_matrix = self.compute_kernel_matrix(kernel_method, mean_embeddings, var_embeddings)
        avg_kernel = np.mean(kernel_matrix, axis=1)
        temp_idx = np.where(avg_kernel == max(avg_kernel))[0]  # will get a reference index
        represent_keyphrase_idx = keyphrase_idx[temp_idx][0]  # get real keyphrase index
        # represent_keyphrase_name = self.idx_2_keyphrase[represent_keyphrase_idx]  # get real keyphrase name value

        if parent_node is None:  # first layer
            self.node_dict = {represent_keyphrase_idx: Node(represent_keyphrase_idx)}
        else:  # lower layer
            self.node_dict[represent_keyphrase_idx] = Node(represent_keyphrase_idx, parent=parent_node)

        # Get representing keyphrase out of further clustering - easiest operation now
        for child_cluster in range(num_clusters):
            # locate current cluster data
            meanEmbed_, varEmbed_, keyphraseIdx_ = split_cluster_data(cluster_labels, child_cluster, mean_embeddings, \
                                                                      var_embeddings, keyphrase_idx)
            if represent_keyphrase_idx in keyphraseIdx_:
                if len(keyphraseIdx_) == 1:
                    continue
                # get rid of the representing keyphrases
                del_idx = np.where(keyphraseIdx_ == represent_keyphrase_idx)[0][0]
                meanEmbed_ = torch.cat([meanEmbed_[0:del_idx], meanEmbed_[del_idx + 1:]])
                varEmbed_ = torch.cat([varEmbed_[0:del_idx], varEmbed_[del_idx + 1:]])
                keyphraseIdx_ = np.delete(keyphraseIdx_, del_idx)

                # need to take care of deleting the only keyphrase corner case
                assert len(keyphraseIdx_) == meanEmbed_.shape[0]
                assert len(keyphraseIdx_) == varEmbed_.shape[0]

            # pass down parent_node, embeddings, and keyphrase indexs
            self.recursClustering(self.node_dict[represent_keyphrase_idx],
                                  meanEmbed_, varEmbed_, keyphraseIdx_, kernel_method=kernel_method)

        # maybe here return the top root node
        if parent_node is None:
            return self.node_dict[represent_keyphrase_idx]
        else:
            return

    # Find the best clustering number
    def find_num_cluster(self, k_range, data):
        num_clusters = None
        sse = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        try:
            kl = KneeLocator(
                k_range, sse, curve="convex", direction="decreasing")
            num_clusters = kl.elbow
        except:
            # print('setting cluster numbers')
            pass

        if num_clusters is None:
            if len(k_range) > 1:
                num_clusters = k_range[-1]
            else:
                num_clusters = 2

        return num_clusters

    def fit_kmeans(self, num_clusters, data):
        kmeans = KMeans(n_clusters=num_clusters, **self.kmeans_kwargs)
        kmeans.fit(data)
        # print('iterations required to cluster:', kmeans.n_iter_)
        return kmeans

    # method to compute a K * K kernel matrix, will be symmetric
    def compute_kernel_matrix(self, kernel_method, mu, var):
        assert mu.shape == var.shape
        kernel_matrix = torch.zeros(mu.shape[0], mu.shape[0])

        # Methods that encounter the parameters of the Gaussian distributions
        if kernel_method in ['ELK', 'BK', 'W2', 'MB']:
            # computing the upper matrix, looping till the 2nd last index
            for row_idx in range(mu.shape[0] - 1):
                col_idx = list(range(row_idx + 1, mu.shape[0]))
                row_mu = torch.repeat_interleave(mu[row_idx].reshape(1, -1), repeats=len(col_idx), dim=0)
                row_var = torch.repeat_interleave(var[row_idx].reshape(1, -1), repeats=len(col_idx), dim=0)
                col_mu = mu[col_idx]
                col_var = var[col_idx]
                row_kernel = kernel_selection(kernel_method, row_mu, col_mu, row_var, col_var)

                kernel_matrix[row_idx, row_idx + 1: mu.shape[0]] = row_kernel

            kernel_matrix = np.array(kernel_matrix)
            kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(np.diag(kernel_matrix))
            return kernel_matrix
        else:
            return None


def split_cluster_data(cluster_labels, cluster_num, mean_embeddings, var_embeddings, keyphrase_idx):
    # find keyphrases in current cluster
    cluster_keyphrase_idx = np.where(cluster_labels == cluster_num)[0]

    # find keyphrase embeddings in current cluster
    mean_ = mean_embeddings[cluster_keyphrase_idx]
    var_ = var_embeddings[cluster_keyphrase_idx]
    keyphrase_ = keyphrase_idx[cluster_keyphrase_idx]

    return mean_, var_, keyphrase_


def set_k_range(cluster_data, k_range):
    k_max = min(len(cluster_data) - 1, k_range[-1])
    if k_max < 2:
        return None
    else:
        k_range_lower = range(2, k_max)
        return k_range_lower

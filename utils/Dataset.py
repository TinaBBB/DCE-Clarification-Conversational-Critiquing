import json
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from pathlib import Path
from utils.io import load_numpy_csr


class Dataset:
    def __init__(self, data_dir, top_keyphrases, rating_threshold, top_users=None):
        print('Read data from %s' % data_dir)
        self.data_dir = data_dir
        self.data_name = self.data_dir.split('/')[-2]
        glob_data_dir = self.data_dir.split('fold')[0]
        self.train_matrix, self.raw_test_matrix, self.test_matrix, self.uk_train, self.uk_test, \
        self.train_item_keyphrase_matrix, self.ik_label_matrix \
            = self.load_data(data_dir,
                             top_keyphrases,
                             rating_threshold,
                             top_users)

        self.num_users, self.num_items = self.train_matrix.shape
        self.num_keyphrases = self.uk_train.shape[1]

        # log user's rating frequency (from UI)
        binary_ui_train = np.zeros(self.train_matrix.shape)
        binary_ui_train = sp.csr_matrix(binary_ui_train)
        binary_ui_train[self.train_matrix > 0] = 1

        binary_uk_train = np.zeros(self.uk_train.shape)
        binary_uk_train = sp.csr_matrix(binary_uk_train)
        binary_uk_train[self.uk_train > 0] = 1

        self.log_freq_array = np.array(np.log1p(binary_ui_train.sum(axis=1)))  # [num_users * 1]

        # log user's keyphrase frequency (from UK)
        self.log_freq_array_keyphrase = np.array(np.log1p(binary_uk_train.sum(axis=1)))

        # get keyphrases' user and keyphrase log frequency information
        binary_ik_train = np.zeros(self.ik_label_matrix.shape)
        binary_ik_train = sp.csr_matrix(binary_ik_train)
        binary_ik_train[self.ik_label_matrix > 0] = 1

        self.log_freq_array_ku = np.array(np.log1p((binary_uk_train.T).sum(axis=1)))
        self.log_freq_array_ki = np.array(np.log1p((binary_ik_train.T).sum(axis=1)))

        self.idx_2_keyphrase, self.keyphrase_2_idx = self.load_idx_keyphrase_dic(glob_data_dir)

    def load_data(self, data_path, top_keyphrases, rating_threshold, top_users):
        # load npz files, we're binarizing the data for this task
        with open(Path(data_path) / 'tr_data.pkl', 'rb') as f:
            train_matrix = pickle.load(f)
            train_matrix[train_matrix > 0] = 1

        with open(Path(data_path) / 'te_data.pkl', 'rb') as f:
            test_matrix = pickle.load(f)
            raw_test_matrix = test_matrix.copy()
            test_matrix[test_matrix > 0] = 1

        # shrinking down the data size
        if top_users:
            train_matrix = train_matrix[:top_users, :]
            raw_test_matrix = raw_test_matrix[:top_users, :]
            test_matrix = test_matrix[:top_users, :]

        df_tags = pd.read_csv(str(Path(data_path) / 'tr_tags.csv'))
        rows, cols, values = df_tags.item, df_tags.tag, np.ones(len(df_tags))
        ik_matrix = sp.csr_matrix((values, (rows, cols)), dtype='float64', shape=(train_matrix.shape[1],
                                                                                  len(df_tags.tag.unique())))
        IK_binary = label_IK(ik_matrix, top_keyphrases=top_keyphrases)
        uk_train = label_UK(train_matrix, IK_binary, rating_threshold=1)
        uk_test = label_UK(test_matrix, IK_binary, rating_threshold=1)

        return train_matrix, raw_test_matrix, test_matrix, uk_train, uk_test, ik_matrix, IK_binary

    def load_idx_keyphrase_dic(self, data_path):
        keyphrase_2_idx = json.load(open(data_path + "tag_id_dict.json"))
        idx_2_keyphrase = {int(v): k for k, v in keyphrase_2_idx.copy().items()}
        return idx_2_keyphrase, keyphrase_2_idx

    def eval_data(self):
        return self.train_matrix, self.test_matrix

    def eval_data_uk(self):
        return self.uk_train, self.uk_test

    def all_data(self):
        return self.train_matrix

    def all_uk(self):
        return self.uk_train

    def __str__(self):
        # return string representation of 'Dataset' class
        # print(Dataset) or str(Dataset)
        ret = '======== [Dataset] ========\n'
        ret += 'Number of users : %d\n' % self.num_users
        ret += 'Number of items : %d\n' % self.num_items
        ret += 'Number of Keyphrases : %d\n' % self.num_keyphrases
        ret += 'None-zero training entries: %d\n' % self.train_matrix.nnz
        ret += 'None-zero testing entries: %d\n' % self.test_matrix.nnz
        ret += '\n'
        return ret


def label_IK(ik_matrix, top_keyphrases):
    IK_binary = ik_matrix.toarray()
    num_items, num_keyphrases = IK_binary.shape

    # generate top k=10 labels for each item according to frequency number
    for item in range(num_items):
        item_keyphrase = IK_binary[item]
        nonzero_keyphrases_index = item_keyphrase.nonzero()[0]
        nonzero_keyphrases_frequency = item_keyphrase[nonzero_keyphrases_index]

        # sort to get the top candidate keyphrases to label each item
        candidate_index = nonzero_keyphrases_index[np.argsort(-nonzero_keyphrases_frequency)[:top_keyphrases]]
        binarized_keyphrase = np.zeros(num_keyphrases)
        binarized_keyphrase[candidate_index] = 1
        IK_binary[item] = binarized_keyphrase

    return sp.csr_matrix(IK_binary)


def label_UK(ui_matrix, ik_matrix, rating_threshold):
    """
    ui_matrix: original user-item rating matrix, explicit
    ik_matrix: labeled-item matrix
    rating_threshold: binarized user-item matrix
    """

    # get binarized rating data, treat as topical preference
    ui_matrix_binary = np.zeros(ui_matrix.shape)
    ui_matrix_binary = sp.csr_matrix(ui_matrix_binary)

    # rating threshold
    ui_matrix_binary[ui_matrix >= rating_threshold] = 1
    ui_matrix_binary.eliminate_zeros()

    UK_matrix = ui_matrix_binary @ ik_matrix
    UK_matrix.eliminate_zeros()
    assert UK_matrix.shape[0] == ui_matrix.shape[0]
    assert UK_matrix.shape[1] == ik_matrix.shape[1]

    # return the generated UK matrix, not necessarily binary
    return UK_matrix

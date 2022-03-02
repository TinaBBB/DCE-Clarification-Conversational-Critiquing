import numpy as np
import torch
# update_posterior_logistic
from tqdm import tqdm

from .SimulatorUtils import get_select_query_func, get_clarification_query_func, update_posterior
from .Tools import compute_cosSimilarity, kernel_selection


class Simulator:
    def __init__(self, dataset, model, keyphrase_embeddings, keyphrase_hierarchies, item_keyphrase_matrix, sim_conf,
                 experiment_df):
        self.dataset = dataset
        self.model = model
        self.experiment_df = experiment_df  # dataframe for user study
        self.keyphrase_embeddings = keyphrase_embeddings
        self.keyphrase_hierarchies = keyphrase_hierarchies
        self.embedding_size = keyphrase_embeddings.shape[1]
        self.item_keyphrase_matrix = item_keyphrase_matrix
        self.steps = sim_conf['steps']
        self.select_query = get_select_query_func[sim_conf['query_type']]
        self.rejection_threshold = sim_conf['rejection_threshold']
        self.clarification_query = get_clarification_query_func[sim_conf['clarification_type']]
        self.clarification_type = sim_conf['clarification_type']
        self.critique_surrogate = sim_conf['surrogate_critique']
        self.rating_threshold = sim_conf['rating_threshold']
        self.keyphrase_threshold = sim_conf['keyphrase_threshold']
        self.diff = sim_conf['diff']
        self.response_noise = sim_conf['response_noise']
        self.k = sim_conf['k']
        self.pos_prec = sim_conf['pos_prec']
        self.neg_prec = sim_conf['neg_prec']
        self.sim_conf = sim_conf
        self.kernel_method = self.sim_conf['kernel_method']
        self.surrogate_candidate_num = 15 if 'ml10' in self.dataset.data_dir else 15
        if 'distributional' in self.clarification_type:
            self.kernel_matrix = self.compute_kernel_matrix()

    # the general function to perform the simulation
    def simulate_hr(self):
        result = {'HR@{}'.format(_k): [] for _k in self.k}
        result['Steps'] = list(range(self.steps + 1))

        # threshold on the ui matrix,
        test_matrix = self.dataset.raw_test_matrix >= self.rating_threshold
        item_keyphrase_matrix = self.item_keyphrase_matrix > 0
        print(np.sum(item_keyphrase_matrix))
        users, items = np.nonzero(test_matrix)
        print(len(items))

        # get positive items with total keyphrase mentionings >= threshold
        pos = (item_keyphrase_matrix.sum(axis=1) >= self.keyphrase_threshold).nonzero()[0]
        print('positive_items:', len(pos))
        print('keyphrase_threshold:', self.keyphrase_threshold)
        print('IK shape:', item_keyphrase_matrix.shape)
        mask = np.isin(items, pos)
        items = items[mask]
        users = users[mask]

        # try to use top ones or random sample some for faster evaluation
        # Sample users to check performance
        # pass in the sample number parameter here
        sample_num = 5000 if 'ml10' in self.dataset.data_dir else 5000  # using 5000 when reporting the table
        sample_index = np.random.choice(len(items), sample_num)
        items = items[sample_index]
        users = users[sample_index]

        # t = 0
        for u, i in tqdm(zip(users, items), total=len(items)):
            r = self.simulate_user_hr(u, i)
            for _k in self.k:
                result['HR@{}'.format(_k)].append(r[_k])

        for _k in self.k:
            avg = np.mean(result['HR@{}'.format(_k)], axis=0)
            ci = 1.96 * np.std(result['HR@{}'.format(_k)], axis=0) / np.sqrt(len(items))
            result['HR@{}'.format(_k)] = avg
            result['HR@{}_CI'.format(_k)] = ci

        return result

    def simulate_user_hr(self, user_id, target_item_id):
        result = {_k: [] for _k in self.k}
        asked_queries = []  # store indexes of keyphrase asked in prevous step to avoid redundant query
        clarification_rejection_count = 0
        # for thompson sampling, success and failure hit number
        surrogate_S = np.zeros(self.keyphrase_embeddings.shape[0], dtype=np.int)
        surrogate_F = np.zeros(self.keyphrase_embeddings.shape[0], dtype=np.int)
        rnd = np.random.RandomState(7)

        # current user preference embedding state
        mu, S = self.get_mu_S(user_id)
        # prec_y = np.array(np.linalg.norm(0.01/(S+1e-6)))

        # get positive and negative entries here
        neg = np.min(mu.T @ self.keyphrase_embeddings.T)
        pos = np.max(mu.T @ self.keyphrase_embeddings.T)
        _, relevant_keyphrases = np.nonzero(self.item_keyphrase_matrix[target_item_id])
        # initial hr
        pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
        top_item = pred_sorted_items[:10]
        # get the initial item and their corresponding ranking
        initial_item_rank_dict = {item_idx: rank for rank, item_idx in enumerate(pred_sorted_items)}
        for _k in self.k:
            result[_k].append(hr_k(pred_sorted_items, target_item_id, _k))

        for j in range(1, self.steps + 1):

            # get top 5 recommended items and their original ranking

            experiment_log = {'user_id': user_id, 'step': j, 'target_item_id': target_item_id,
                              'accept_clarification': False}

            # record top recommended items and their original rankings, for use case study
            for idx in range(5):
                experiment_log['top_{}_recommendation'.format(idx)] = '{} ({})'.format(top_item[idx],
                                                                                       initial_item_rank_dict[
                                                                                           top_item[idx]])
                # for each step:
            #   1. system queries for critiquing, and for clarification
            #   2. get user responses for critiquing and clarification in one step, decision if replacement
            #   3. update user belief
            #   4. compute HR

            '''1. generate query candidates'''
            # 1.1 get critiquing query and user responses towards it
            sorted_query_candidates = self.select_query(
                item_keyphrase_matrix=self.item_keyphrase_matrix,
                items=top_item,
                target_item=target_item_id
            )
            # remove redundant queries, and get the first element for critiquing
            reduns = np.isin(sorted_query_candidates, asked_queries).nonzero()[0]
            # query_idx = relevant_keyphrases[j-1]
            query_idx = np.delete(sorted_query_candidates, reduns)[0]
            expert_query_idx = query_idx
            experiment_log['expert_keyphrase'] = self.dataset.idx_2_keyphrase[query_idx]

            # 1.1.1 replace the original critiquing surrogate under the case where clarification_type is not none
            if self.critique_surrogate:
                # flag for if surrogate was used as the final ciritquing keyphrase
                # compute similarity with other corpus keyphrases
                # [1 * k] similarity score vector
                # the method to choose the keyphrase is key

                # method 1. compute word similarity directly with other keyphrases
                simScore_vector = compute_cosSimilarity(self.dataset.train_item_keyphrase_matrix.T[query_idx],
                                                        self.dataset.train_item_keyphrase_matrix.T)

                # candidate sorting
                sorted_surrogate_candidates = np.argsort(-simScore_vector)[
                                              1:self.surrogate_candidate_num]  # including itself
                reduns = np.isin(sorted_surrogate_candidates, asked_queries).nonzero()[0]
                sorted_surrogate_candidates = np.delete(sorted_surrogate_candidates, reduns)

                # select query index as surrogate, using beta distribution
                # get success and failure data, for user using that surrogate word
                S_candidate = surrogate_S[sorted_surrogate_candidates]
                F_candidate = surrogate_F[sorted_surrogate_candidates]
                beta_prob = np.array([rnd.beta(S_candidate[idx] + 1, F_candidate[idx] + 1) for idx in
                                      range(len(sorted_surrogate_candidates))])

                if self.clarification_type == 'distributional' or self.clarification_type == 'none':
                    query_idx = sorted_surrogate_candidates[np.argmax(beta_prob)]
                else:  # saving the original methodology
                    # query_idx = sorted_surrogate_candidates[0]
                    query_idx = sorted_surrogate_candidates[np.argmax(beta_prob)]
                surrogate_query_idx = query_idx

                # method 2. compute ranking of keyphrases for the critiqued keyphrase using kernel computation
                # simScore_vector = self.kernel_matrix[query_idx]
                # replace

            # print(query_idx)
            experiment_log['critiquing_keyphrase'] = self.dataset.idx_2_keyphrase[query_idx]

            # 1.2 get clarification query and user responses towards the clarification query
            # also validating the clarification criteria
            if self.clarification_type != 'none' and clarification_rejection_count < self.rejection_threshold:
                clarification_query_candidates = self.clarification_query(
                    item_keyphrase_matrix=self.item_keyphrase_matrix,
                    keyphrase_embeddings=self.keyphrase_embeddings,
                    query_index=query_idx,
                    keyphrase_hierarchies=self.keyphrase_hierarchies
                )

                # not to clarify on the same word
                clarification_query_candidates = np.delete(clarification_query_candidates,
                                                           np.where(clarification_query_candidates == query_idx))

                reduns = np.isin(clarification_query_candidates, asked_queries).nonzero()[0]
                try:
                    clarification_query_idx = np.delete(clarification_query_candidates, reduns)
                    if self.clarification_type != 'distributional' or len(clarification_query_idx) == 1:
                        clarification_query_idx = clarification_query_idx[0]
                        # distributional but no interaction with the user's embedding
                    elif self.clarification_type == 'distributionalNoUserEmbed':
                        kk_score = self.compute_kk_kernel_matrix(self.kernel_method, surrogate_query_idx,
                                                                 clarification_query_idx)
                        clarification_query_idx = clarification_query_idx[np.argmax(kk_score)]
                    else:  # distributional, select the keyphrase that's most dissimilar with user embedding
                        uk_score = self.compute_uk_kernel_matrix(self.kernel_method, clarification_query_idx, user_id)
                        # select the clarification keyphrase that's furthest from the user representation
                        clarification_query_idx = clarification_query_idx[np.argmax(uk_score)]
                except:
                    clarification_query_idx = query_idx
            else:
                # no clarification step
                clarification_query_idx = query_idx

            # reveals the expert keyphrase missed in the previous step...
            if clarification_query_idx == expert_query_idx:
                query_idx = expert_query_idx

            experiment_log['clarification_keyphrase'] = self.dataset.idx_2_keyphrase[clarification_query_idx]

            '''2. check user responses towards critiquing and clarification'''
            # 2.1 the critiquing decision (pos/neg) should be fixed for the critiquing decision
            s = np.random.uniform()
            if s < self.response_noise:
                y = pos if np.random.uniform() > 0.5 else neg
                response2clarification = pos if np.random.uniform() > 0.5 else neg
            else:
                y = pos if np.isin(query_idx, relevant_keyphrases) else neg
                response2clarification = pos if np.isin(clarification_query_idx, relevant_keyphrases) else neg

            # 2.2 set critiquing intent, precision for the likelihood, fixed based on critiquing decision.
            prec_y = self.pos_prec if y == pos else self.neg_prec
            experiment_log['critique_polarity'] = 'pos' if y == pos else 'neg'
            experiment_log['clarification_polarity'] = 'pos' if response2clarification == pos else 'neg'

            # 2.3 replacement decision, if clarification responses consistent with critiquing response,
            # replace phrase being critiqued
            if response2clarification == y:
                query_idx = clarification_query_idx
                experiment_log['accept_clarification'] = True
            else:
                clarification_rejection_count += 1

            # mark surrogate success, for thompson sampling, success if continue usage, failure if not
            if self.critique_surrogate and surrogate_query_idx == query_idx:
                surrogate_S[surrogate_query_idx] += 1
            elif self.critique_surrogate and surrogate_query_idx != query_idx:
                surrogate_F[surrogate_query_idx] += 1

            # 2.4 get keyphrase embedding based on final query decision
            x = self.keyphrase_embeddings[int(query_idx)][:, np.newaxis]

            '''3. update user belief'''
            mu, S = update_posterior(x, y, mu, S, prec_y)

            # new HR
            pred_sorted_items, _ = self.get_user_preds_using_mu(mu, user_id)
            top_item = pred_sorted_items[:10]

            for _k in self.k:
                result[_k].append(hr_k(pred_sorted_items, target_item_id, _k))

            asked_queries.append(query_idx)

            self.experiment_df = self.experiment_df.append(experiment_log, ignore_index=True)

        return result

    def get_user_preds_using_mu(self, user_mu, user_id=None):
        """
        user_mu: hidden_dim by 1
        """

        _mu = torch.FloatTensor(user_mu.T)  # 1 by hidden_dim

        with torch.no_grad():
            preds = self.model.decoder(_mu)

        preds = np.asarray(preds).reshape(-1)

        if user_id is not None:
            _, user_input = np.nonzero(self.dataset.train_matrix[user_id])
            preds[user_input] = -np.inf

        sorted_pred_items = preds.argsort()[::-1]
        sorted_pred_ratings = preds[sorted_pred_items]
        return sorted_pred_items, sorted_pred_ratings

    def get_user_item_pred(self, user_id, item_id):
        user_input = self.dataset.train_matrix[user_id]
        i = torch.FloatTensor(user_input.toarray()).to(torch.device('cpu'))
        with torch.no_grad():
            preds = self.model.forward(i).cpu().numpy().reshape(-1)

        return preds[item_id]

    def get_mu_S(self, user_id):
        user_input = self.dataset.train_matrix[user_id]
        i = torch.FloatTensor(user_input.toarray()).to(torch.device('cpu'))
        with torch.no_grad():
            mu, logvar = self.model.get_mu_logvar(i)
            std = self.model.logvar2std(logvar)
        mu, std = mu.numpy().T, std.numpy()

        return mu, np.diagflat(std * std)

    # compute kernel matrix for all the keyphrases
    def compute_kernel_matrix(self):

        mu = self.model.keyphrase_mu.weight.data
        var = torch.exp(self.model.keyphrase_log_var.weight.data)
        assert mu.shape == var.shape
        kernel_matrix = torch.zeros(mu.shape[0], mu.shape[0])

        # Methods that encounter the parameters of the Gaussian distributions
        if self.kernel_method in ['ELK', 'BK', 'W2', 'MB']:
            # computing the upper matrix, looping till the 2nd last index
            for row_idx in range(mu.shape[0] - 1):
                col_idx = list(range(row_idx + 1, mu.shape[0]))
                row_mu = torch.repeat_interleave(mu[row_idx].reshape(1, -1), repeats=len(col_idx), dim=0)
                row_var = torch.repeat_interleave(var[row_idx].reshape(1, -1), repeats=len(col_idx), dim=0)
                col_mu = mu[col_idx]
                col_var = var[col_idx]
                row_kernel = (-1) * kernel_selection(self.kernel_method, row_mu, col_mu, row_var, col_var)

                kernel_matrix[row_idx, row_idx + 1: mu.shape[0]] = row_kernel

            kernel_matrix = np.array(kernel_matrix)
            kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(np.diag(kernel_matrix))
            return kernel_matrix
        else:
            return None

    def compute_uk_kernel_matrix(self, kernel_method, keyphrase_id, user_id):
        # keyphrase
        keyphrase_mu = self.model.keyphrase_mu.weight.data
        kephrase_var = torch.exp(self.model.keyphrase_log_var.weight.data)
        assert keyphrase_mu.shape == kephrase_var.shape
        # take keyphrase embeddings
        mu_k = keyphrase_mu[keyphrase_id]
        var_k = kephrase_var[keyphrase_id]

        # user
        user_input = self.dataset.train_matrix[user_id]
        i = torch.FloatTensor(user_input.toarray()).to(torch.device('cpu'))
        with torch.no_grad():
            mu_u, logvar = self.model.get_mu_logvar(i)
            var_u = torch.exp(logvar)

        repeated_mu_u = torch.repeat_interleave(mu_u.reshape(1, -1), repeats=len(keyphrase_id), dim=0)
        repeated_var_u = torch.repeat_interleave(var_u.reshape(1, -1), repeats=len(keyphrase_id), dim=0)
        uk_score = kernel_selection(kernel_method, repeated_mu_u, mu_k, repeated_var_u, var_k)
        return uk_score

    def compute_kk_kernel_matrix(self, kernel_method, anchor_keyphrase_id, keyphrase_ids):
        # keyphrase
        keyphrase_mu = self.model.keyphrase_mu.weight.data
        kephrase_var = torch.exp(self.model.keyphrase_log_var.weight.data)
        assert keyphrase_mu.shape == kephrase_var.shape
        # take keyphrase embeddings
        mu_k = keyphrase_mu[keyphrase_ids]
        var_k = kephrase_var[keyphrase_ids]

        mu_anchor = keyphrase_mu[anchor_keyphrase_id]
        va_anchor = keyphrase_mu[anchor_keyphrase_id]

        repeated_mu_anchor = torch.repeat_interleave(mu_anchor.reshape(1, -1), repeats=len(keyphrase_ids), dim=0)
        repeated_var_anchor = torch.repeat_interleave(va_anchor.reshape(1, -1), repeats=len(keyphrase_ids), dim=0)
        kk_score = kernel_selection(kernel_method, repeated_mu_anchor, mu_k, repeated_var_anchor, var_k)
        return kk_score


def hr_k(preds, target, k):
    if target in set(preds[:k]):
        return 1
    else:
        return 0

import numpy as np
import torch
from utils.Tools import kernel_selection


class Evaluator:
    def __init__(self, rec_atK, explain_atK):
        self.rec_atK = rec_atK  # a list of the topK indecies
        self.rec_maxK = max(self.rec_atK)
        self.explain_atK = explain_atK
        self.explain_maxK = max(self.explain_atK)

        self.global_metrics = {
            "R-Precision": r_precision,
            "NDCG": ndcg
        }

        self.local_metrics = {
            "Precision": precisionk,
            "Recall": recallk,
            "MAP": average_precisionk,
            "NDCG": ndcg
        }

        self.global_metrics_embeddings = {
            "UK_R-Precision": r_precision,
            "UK_NDCG": ndcg
        }

        self.local_metrics_embeddings = {
            "UK_NDCG": ndcg,
            "UK_Precision": precisionk,
            "UK_Recall": recallk,
            "UK_MAP": average_precisionk
        }

    # evaluate Gaussian embeddings, explanations and keyphraes relationships
    def evaluate_embeddings(self, model, train_matrix_uk, test_matrix_uk,
                            mu_user, var_user, ndcg_only, data_name, analytical=False):
        """
        Args:
            model: passed in model, e.g., VAE, VAE-contrast
            train_matrix_uk: test matrix of UK
            test_matrix_uk: test matrix of UK
            mu_user: mean embedding for users with all historical item entires known
            var_user: sigma embedding for users with all historical item entires known
            analytical: False if getting the confidence interval value

        Returns: a dictionary of metric scores
        """
        # switch to evaluation mode
        model.eval()
        model.before_evaluate()

        mu_user = torch.from_numpy(mu_user).to(model.device)
        var_user = torch.from_numpy(var_user).to(model.device)

        assert mu_user.shape[0] == train_matrix_uk.shape[0]
        assert torch.all(torch.gt(var_user, torch.zeros(size=var_user.size()).to(var_user.device)))

        with torch.no_grad():
            keyphrase_mean_embeddings = model.keyphrase_mu.weight.data
            keyphrase_var_embeddings = torch.exp(model.keyphrase_log_var.weight.data)

        # Get corresponding keyphrases predictions
        predicted_uk = self.kernel_predict(train_matrix_uk, test_matrix_uk, mu_user, var_user,
                                           keyphrase_mean_embeddings, keyphrase_var_embeddings,
                                           model.kernel_method, model.temperature_tau_u, data_name)

        uk_results = self.evaluation(predicted_uk, test_matrix_uk, eval_type='embeddings', ndcg_only=ndcg_only,
                                     analytical=analytical)

        return uk_results

    def kernel_predict(self, train_matrix, test_matrix, mu_anchor, var_anchor, mu_samples, var_samples,
                       kernel_method, temperature, data_name):

        # maxK = self.explain_maxK
        pos_entries = train_matrix.tolil().rows  # array of lists, to not consider
        # ground_entries = test_matrix.tolil().rows

        prediction = []
        for i in range(pos_entries.shape[0]):  # for each user

            # skipping the negative keyphrase cases
            if len(pos_entries[i]) == 0:
                prediction.append(np.zeros(self.explain_maxK, dtype=np.float32))

            else:
                unk_entries = list(set(range(train_matrix.shape[1])))
                mu_anchor_i = torch.repeat_interleave(mu_anchor[i].reshape(1, -1),
                                                      repeats=len(unk_entries), dim=0).to(mu_anchor.device)
                var_anchor_i = torch.repeat_interleave(var_anchor[i].reshape(1, -1),
                                                       repeats=len(unk_entries), dim=0).to(mu_anchor.device)

                # corresponding unknown keyphrases' embeddings
                mu_sample_i = mu_samples[unk_entries]
                var_sample_i = var_samples[unk_entries]

                assert mu_anchor_i.shape == mu_sample_i.shape
                assert var_anchor_i.shape == var_sample_i.shape

                # Becomes the predictions
                kernel = torch.divide(kernel_selection(kernel_method, mu_anchor_i,
                                                       mu_sample_i, var_anchor_i,
                                                       var_sample_i), temperature)

                # check kernel shape correspondence
                assert kernel.shape[0] == len(unk_entries)

                # select argmax
                top_index = (torch.argsort(kernel, dim=-1, descending=True)[:self.explain_maxK]).cpu().data.numpy()
                top_predict = np.array(unk_entries)[top_index]

                prediction.append(top_predict)

        # predicted item indecies
        predicted_items = prediction.copy()
        assert len(predicted_items) == train_matrix.shape[0]
        return predicted_items

    def evaluate_recommendations(self, model, input_matrix, test_matrix, mse_only, ndcg_only, test_batch_size,
                                 analytical=False):
        model.eval()
        # operations before evaluation, does not perform for VAE models
        model.before_evaluate()

        # get prediction data, in matrix form
        # get prediction data, in matrix form, not masking, for recommendation results
        pred_matrix = model.predict(input_matrix)
        assert pred_matrix.shape == input_matrix.shape
        RMSE = round(np.sqrt(np.mean((input_matrix.toarray() - pred_matrix) ** 2)), 4)

        if mse_only:
            recommendation_results = {"RMSE": (RMSE, 0)}
        else:
            # get predicted item index
            prediction = []
            # get prediction data, in matrix form, not masking, for recommendation results
            num_users = pred_matrix.shape[0]

            # Prediction section
            for user_index in range(num_users):
                vector_prediction = pred_matrix[user_index]
                vector_train = input_matrix[user_index]

                if len(vector_train.nonzero()[0]) > 0:
                    vector_predict = sub_routine(vector_prediction, vector_train, topK=self.rec_maxK)
                else:
                    vector_predict = np.zeros(self.rec_maxK, dtype=np.float32)

                prediction.append(vector_predict)

            # predicted item indecies
            predicted_items = prediction.copy()
            recommendation_results = self.evaluation(predicted_items, test_matrix, eval_type='recommendations',
                                                     ndcg_only=ndcg_only, analytical=analytical)
            recommendation_results["RMSE"] = (RMSE, 0)
        return recommendation_results

    # function to perform evaluation on metrics
    def evaluation(self, predicted_items, test_matrix, eval_type, ndcg_only, analytical=False):
        if eval_type == 'recommendations' and ndcg_only:
            local_metrics = None
            global_metrics = {"NDCG": ndcg}
            atK = self.rec_atK
        elif eval_type == 'recommendations' and not ndcg_only:
            local_metrics = self.local_metrics
            global_metrics = self.global_metrics
            atK = self.rec_atK
        elif eval_type == 'embeddings' and ndcg_only:
            local_metrics = None
            global_metrics = {"UK_NDCG": ndcg}
            atK = self.explain_atK
        elif eval_type == 'embeddings' and not ndcg_only:
            local_metrics = self.local_metrics_embeddings
            global_metrics = self.global_metrics_embeddings
            atK = self.explain_atK
        else:
            raise NotImplementedError("Please select proper evaluation type, current choice: %s" % eval_type)

        num_users = test_matrix.shape[0]

        # evaluation section
        output = dict()

        # The @K metrics
        if local_metrics:
            for k in atK:
                results = {name: [] for name in local_metrics.keys()}

                for user_index in range(num_users):
                    vector_predict = predicted_items[user_index][:k]
                    if len(vector_predict.nonzero()[0]) > 0:
                        vector_true_dense = test_matrix[user_index].nonzero()[1]

                        if vector_true_dense.size > 0:  # only if length of validation set is not 0
                            hits = np.isin(vector_predict, vector_true_dense)
                            for name in local_metrics.keys():
                                results[name].append(local_metrics[name](vector_true_dense=vector_true_dense,
                                                                         vector_predict=vector_predict,
                                                                         hits=hits))

                results_summary = dict()
                if analytical:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = results[name]
                else:
                    for name in local_metrics.keys():
                        results_summary['{0}@{1}'.format(name, k)] = (np.average(results[name]),
                                                                      1.96 * np.std(results[name]) / np.sqrt(
                                                                          len(results[name])))
                output.update(results_summary)

        # The global metrics
        results = {name: [] for name in global_metrics.keys()}
        for user_index in range(num_users):
            vector_predict = predicted_items[user_index]

            if len(vector_predict.nonzero()[0]) > 0:
                vector_true_dense = test_matrix[user_index].nonzero()[1]
                hits = np.isin(vector_predict, vector_true_dense)

                if vector_true_dense.size > 0:
                    for name in global_metrics.keys():
                        results[name].append(global_metrics[name](vector_true_dense=vector_true_dense,
                                                                  vector_predict=vector_predict, hits=hits))
        results_summary = dict()
        if analytical:
            for name in global_metrics.keys():
                results_summary[name] = results[name]
        else:
            for name in global_metrics.keys():
                results_summary[name] = (
                    np.average(results[name]), 1.96 * np.std(results[name]) / np.sqrt(len(results[name])))
        output.update(results_summary)

        return output


def sub_routine(vector_predict, vector_train, topK):
    train_index = vector_train.nonzero()[1]

    # take the top recommended items
    candidate_index = np.argpartition(-vector_predict, topK + len(train_index))[:topK + len(train_index)]
    vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]

    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]


def recallk(vector_true_dense, hits, **_unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_true_dense)


def precisionk(vector_predict, hits, **_unused):
    hits = len(hits.nonzero()[0])
    return float(hits) / len(vector_predict)


def average_precisionk(vector_predict, hits, **_unused):
    precisions = np.cumsum(hits, dtype=np.float32) / range(1, len(vector_predict) + 1)
    return np.mean(precisions)


def r_precision(vector_true_dense, vector_predict, **_unused):
    vector_predict_short = vector_predict[:len(vector_true_dense)]
    hits = len(np.isin(vector_predict_short, vector_true_dense).nonzero()[0])
    return float(hits) / len(vector_true_dense)


def _dcg_support(size):
    arr = np.arange(1, size + 1) + 1
    return 1. / np.log2(arr)


def ndcg(vector_true_dense, vector_predict, hits):
    idcg = np.sum(_dcg_support(len(vector_true_dense)))
    dcg_base = _dcg_support(len(vector_predict))
    dcg_base[np.logical_not(hits)] = 0
    dcg = np.sum(dcg_base)
    return dcg / idcg

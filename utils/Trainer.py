import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
from sklearn.manifold import TSNE
from utils.Table import Table
from utils.Tools import generate_sample_embeddings
from utils.io import pickle_dump


class Trainer:
    def __init__(self, dataset, model, evaluator, logger, conf, experiment=None,
                 plot_graph=False, run_samples=False):
        self.dataset = dataset
        self.train_matrix, self.test_matrix = self.dataset.eval_data()
        self.uk_train, self.uk_valid = self.dataset.eval_data_uk()
        self.model = model
        self.evaluator = evaluator
        self.logger = logger
        self.conf = conf
        self.experiment = experiment
        self.plot_graphs = plot_graph
        self.run_samples = run_samples

        self.num_epochs = conf.num_epochs
        self.lr = conf.learning_rate
        self.batch_size = conf.batch_size
        self.test_batch_size = conf.test_batch_size

        self.early_stop = conf.early_stop
        self.patience = conf.patience
        self.endure = 0
        self.skip_eval = conf.skip_eval

        self.best_epoch = -1
        self.best_score = None
        self.best_params = None
        self.best_rec_score = None
        self.best_uk_score = None

        # save the best keyphrase embeddings during best epochs
        self.mean_embeddings = None
        self.stdev_embeddings = None

        # for use case, save selected user and keyphrase embeddings during training
        self.sampled_user_idx = 799  # a single number
        self.sample_user_embeddings = None
        self.sample_user_embeddings_std = None
        self.sampled_keyphrase_idx = [225, 429, 674]  # a list of keyphrase idx
        self.sampled_keyphrase_embeddings = None
        self.sampled_keyphrase_embeddings_std = None
        self.sampled_epoch = [10, 50, 100, 200, 250, 300]  # a list of training epochs to sample
        self.score_comparison_df = pd.DataFrame(columns=['MSE', 'uk_rprec', 'epoch'])

    def train(self):
        self.logger.info(self.conf)

        # pass module parameters to the optimizer
        if len(list(self.model.parameters())) > 0:
            optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)
        else:
            optimizer = None

        # create table for logging
        score_table = Table(table_name='Scores')

        for epoch in range(1, self.num_epochs + 1):
            # train for an epoch
            epoch_start = time.time()
            loss = self.model.train_one_epoch(train_matrix=self.train_matrix,
                                              uk_matrix=self.uk_train,
                                              uk_test=self.uk_valid,
                                              optimizer=optimizer,
                                              batch_size=self.batch_size,
                                              verbose=False,
                                              experiment=self.experiment)  # verbose/printing false

            # log epoch loss
            if self.experiment: self.experiment.log_metric(name='epoch_loss', value=loss, epoch=epoch)

            train_elapsed = time.time() - epoch_start

            # 1. not skipping evaluation, evaluate at every 10 epochs after the 50th epoch
            # 2. skipping evaluation, evaluate at the end of all epochs
            if (not self.skip_eval and epoch >= 50 and epoch % 10 == 0) or (
                    self.skip_eval and epoch == self.num_epochs):
                if not self.skip_eval and self.early_stop:  # get scores during training only
                    # recommendation performance
                    rec_score = self.evaluator.evaluate_recommendations(self.model, self.train_matrix,
                                                                        self.test_matrix,
                                                                        mse_only=False,
                                                                        ndcg_only=True, analytical=False,
                                                                        test_batch_size=self.test_batch_size)

                else:  # At the end of training epochs, during evaluation
                    # recommendation performance
                    rec_score = self.evaluator.evaluate_recommendations(self.model, self.train_matrix,
                                                                        self.test_matrix, mse_only=False,
                                                                        ndcg_only=False, analytical=False,
                                                                        test_batch_size=self.test_batch_size)

                # score we want to check during training
                score = {"Loss": float(loss),
                         "RMSE": rec_score['RMSE'][0]}
                if "NDCG" in rec_score.keys():
                    score['NDCG'] = rec_score['NDCG'][0]

                score_str = ' '.join(['%s=%.4f' % (m, score[m]) for m in score])
                epoch_elapsed = time.time() - epoch_start

                self.logger.info('[Epoch %3d/%3d, epoch time: %.2f, train_time: %.2f] %s' % (
                    epoch, self.num_epochs, epoch_elapsed, train_elapsed, score_str))

                # log for comet ml, per 10 epochs
                if self.experiment:
                    self.experiment.log_metric(name='RMSE', value=score['RMSE'], \
                                               epoch=epoch)
                    if "NDCG" in rec_score.keys():
                        self.experiment.log_metric(name='NDCG', value=score['NDCG'],
                                                   epoch=epoch)
                # update if ...
                standard = 'NDCG'
                if self.best_score is None or score[standard] > self.best_score[standard]:
                    self.best_epoch = epoch
                    self.best_score = score
                    self.best_rec_score = rec_score
                    self.best_params = self.model.parameters()

                    self.endure = 0

                    # log stats plot, every 50 epoch is enough
                    if self.plot_graphs and epoch >= 50 and epoch % 50 == 0:
                        self.log_stats_plot(epoch)
                else:
                    self.endure += 10
                    if self.early_stop and self.endure >= self.patience:
                        print('Early Stop Triggered...')
                        break

        # log plot at the end of training, and log last epoch embeddings
        if self.plot_graphs:
            self.log_stats_plot(epoch)
            plt.clf()
            plt.cla()
            plt.close()

        print('Training Finished.')
        score_table.add_row('Best at epoch %d' % self.best_epoch, self.best_score)
        self.logger.info(score_table.to_string())

    # create scatter plot for user embedding values
    def create_scatter(self, embedding, axis_value):
        log_freq = self.dataset.log_freq_array
        log_freq_keyphrase = self.dataset.log_freq_array_keyphrase
        avg_stdev = np.mean(embedding, axis=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax1.scatter(log_freq, avg_stdev)
        ax1.set_xlabel('log total rating frequency')
        ax1.set_ylabel('avg_{}'.format(axis_value))

        ax2.scatter(log_freq_keyphrase, avg_stdev)
        ax2.set_xlabel('log total keyphrase frequency')
        ax2.set_ylabel('avg_{}'.format(axis_value))

        fig.suptitle('Rating frequencies & keyphrase mentioning frequencies vs. averaged {}'.format(axis_value))

        return fig

    def create_scatter_keyhrase(self, axis_value):
        if axis_value == 'stdev':
            embedding = np.array(torch.exp(0.5 * self.model.keyphrase_log_var.weight.data))
        elif axis_value == 'mean':
            embedding = np.array(self.model.keyphrase_mu.weight.data)
        else:
            raise NotImplementedError('Choose appropriate embedding parameter. (current input: %s)' % axis_value)

        log_freq_ku = self.dataset.log_freq_array_ku
        log_freq_ki = self.dataset.log_freq_array_ki

        avg_embedding = np.mean(embedding, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
        ax1.scatter(log_freq_ku, avg_embedding)
        ax1.set_xlabel('log user mentioning frequency')
        ax1.set_ylabel('avg_{}'.format(axis_value))

        ax2.scatter(log_freq_ki, avg_embedding)
        ax2.set_xlabel('log item labeled frequency')
        ax2.set_ylabel('avg_{}'.format(axis_value))

        fig.suptitle('User mention frequencies & item labeled frequencies vs. averaged {}'.format(axis_value))
        return fig

    def log_stats_plot(self, epoch_num):
        mean_embedding, stdev_embedding = self.get_mu_S()

        stats_figure = self.create_scatter(stdev_embedding, 'stdev')

        stats_figure_k = self.create_scatter_keyhrase('stdev')

        self.experiment.log_figure(figure_name='stats_fig_' + str(epoch_num), figure=stats_figure, overwrite=True)

        self.experiment.log_figure(figure_name='stats_fig_keyphrase_' + str(epoch_num), figure=stats_figure_k,
                                   overwrite=True)
        plt.clf()
        plt.cla()
        plt.close()

    def get_mu_S(self):
        input_matrix = self.dataset.all_data()
        i = torch.FloatTensor(input_matrix.toarray()).to(self.model.device)
        with torch.no_grad():
            mu, logvar = self.model.get_mu_logvar(i)
            std = self.model.logvar2std(logvar)
        mu, std = mu.cpu().data.numpy(), std.cpu().data.numpy()

        return mu, std

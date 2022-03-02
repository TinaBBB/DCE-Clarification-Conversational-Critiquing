"""
Combining two frameworks, VAE-CF[1] and Gaussian word embedding[2]
[1] Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018.
https://arxiv.org/pdf/1802.05814

[2] Vilnis, Luke, and Andrew McCallum. "Word representations via gaussian embedding." ICLR 2015.
https://arxiv.org/abs/1412.6623
github: https://github.com/schelotto/Gaussian_Word_Embedding

Note the model setting is for :
1. Explicit Rating Data
2. Observation_std set to 1 since Explicit ratings are in range 1-5
3. Reconstruction losses are MASKED RMSE, i.e., unknown entries are treated to be unobserved
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils.Tools import AverageMeter, logsumexp, gaussian_nll
from utils.Tools import kernel_selection, activation_map, sampling, softclip


class VAEmultilayer_contrast(BaseModel):
    def __init__(self, model_conf, num_users, num_items, num_keyphrases, device,
                 observation_std=0.01):
        super(VAEmultilayer_contrast, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_keyphrases = num_keyphrases
        self.hidden_dim = model_conf.hidden_dim
        self.act = model_conf.act
        self.sparse_normalization = model_conf.sparse_normalization
        self.dropout_ratio = model_conf.dropout_ratio
        self.weight_decay = model_conf.weight_decay
        self.weighted_recon = model_conf.weighted_recon
        # number of positive and negative samples for users and keyphrases
        self.pos_uk_num = model_conf.pos_uk_num
        self.neg_uk_num = model_conf.neg_uk_num
        self.pos_kk_num = model_conf.pos_kk_num
        self.neg_kk_num = model_conf.neg_kk_num
        self.kernel_method = model_conf.kernel_method
        self.temperature_tau_u = model_conf.temperature_tau_u  # tau for similarity measure, set to 1 as default
        self.temperature_tau_k = model_conf.temperature_tau_k
        self.total_anneal_steps = model_conf.total_anneal_steps
        self.anneal_cap = model_conf.anneal_cap
        self.observation_std = observation_std

        # contrastive loss hyper-parameters
        if model_conf.use_default_hp:
            self.hp_contrastive_u = 1 / (self.pos_uk_num)
        else:
            self.hp_contrastive_u = model_conf.hp_contrastive_u
        self.hp_contrastive_k = self.hp_contrastive_u * self.num_keyphrases / self.pos_kk_num
        # self.max_var = model_conf.max_var
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(self.num_items, self.hidden_dim * 4))
        self.encoder.append(activation_map(self.act))
        self.encoder.append(nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2))
        for layer in self.encoder:
            if 'weight' in dir(layer):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

        # Create embeddings for keyphrases
        self.keyphrase_mu = nn.Embedding(num_keyphrases, self.hidden_dim)
        self.keyphrase_log_var = nn.Embedding(num_keyphrases, self.hidden_dim)
        torch.nn.init.xavier_uniform_(self.keyphrase_mu.weight)
        torch.nn.init.xavier_uniform_(self.keyphrase_log_var.weight)

        self.anneal = 0.
        self.update_count = 0
        self.device = device
        self.to(self.device)

    def forward(self, rating_matrix):
        # encoder
        mu_q, logvar_q = self.get_mu_logvar(rating_matrix)
        std_q = self.logvar2std(logvar_q)
        eps = torch.randn_like(std_q)  # reparametrization trick
        sampled_z = mu_q + self.training * eps * std_q  # apply reparameterization if in training mode?

        output = self.decoder(sampled_z)  # pass through the decoder

        # clamping values of keyphrases embeddings
        # for p in self.keyphrase_log_var.parameters():
        #     p.data.clamp_(None, math.log(self.max_var))
        #     p.data.clamp_(None, None)

        if self.training:
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # not averaged yet
            kl_loss = -0.5 * torch.mean(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
            return output, kl_loss, mu_q, logvar_q
        else:  # evaluation mode
            return output

    def get_mu_logvar(self, rating_matrix):
        # apply dropout during training time
        if self.training and self.dropout_ratio > 0:
            rating_matrix = F.dropout(rating_matrix, p=self.dropout_ratio) * (1 - self.dropout_ratio)

        if self.sparse_normalization:
            deno = torch.sum(rating_matrix > 0, axis=1, keepdim=True) + 1e-5
            rating_matrix = rating_matrix / deno

        h = rating_matrix
        for layer in self.encoder:  # pass through encoder layer
            h = layer(h)
        mu_q = h[:, :self.hidden_dim]
        logvar_q = h[:, self.hidden_dim:]  # log sigmod^2

        # logvar_q = softclip(logvar_q, -6)

        return mu_q, logvar_q

    def logvar2std(self, logvar):
        return torch.exp(0.5 * logvar)  # sigmod

    def train_one_epoch(self, train_matrix, uk_matrix, uk_test,
                        optimizer, batch_size, verbose, experiment):

        # initialize average meters
        vae_avgmeter = AverageMeter()
        usercl_avgmeter = AverageMeter()
        keyphrasecl_avgmeter = AverageMeter()
        # uk_pos_avgmeter = AverageMeter()
        # uk_neg_avgmeter = AverageMeter()
        # kk_pos_avgmeter = AverageMeter()
        # kk_neg_avgmeter = AverageMeter()

        self.train()

        num_training = train_matrix.shape[0]  # number of users
        num_batches = int(np.ceil(num_training / batch_size))

        # permutation of user and keyphrase indecies
        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):

            # To remove when training is stable TODO
            for layer in self.encoder:
                if 'weight' in dir(layer) and torch.isnan(layer.weight).any():
                    print('at batch: ', b)
                    print('debugging')

            optimizer.zero_grad()

            # get batch UI matrix for training VAE-CF
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)

            # contrastive losses, use batch_idx and batch_idx_k to select pos/neg samples
            # get batch Keyphrase id
            # if (b + 1) * batch_size_k >= num_training_keyphrase:
            #     batch_idx_k = perm_k[b * batch_size_k:]
            # else:
            #     batch_idx_k = perm_k[b * batch_size_k: (b + 1) * batch_size_k]

            '''sampling keyphrases for users and keyphrases'''
            # Avoided self-sampling for both POSITIVE (input matrix) and NEGATIVE (sampling method)
            # returns array of nonzero column lists [batch_size * num_samples] if all batch users have keyphrase records

            # filter out users who don't have keyphrase entries
            uk_records = uk_matrix[batch_idx].tolil().rows
            uk_records = np.array([len(value) for value in uk_records])
            u_sample = np.where(uk_records > 1)[0]  # sample idx for curent batch to compute loss
            # pos_uk, neg_uk with shape [len(u_sample) * self.pos_uk_num] and [len(u_sample) * self.neg_uk_num]
            # pos_kk, neg_kk with shape [len(u_sample) * self.pos_uk_num, self.pos_kk_num]
            # and [len(u_sample) * self.pos_uk_num, self.neg_kk_num]

            # sampling_start = time.time()
            pos_uk, neg_uk, pos_kk, neg_kk = sampling(idx=batch_idx[u_sample], matrix=uk_matrix,
                                                      uk_pos_num=self.pos_uk_num, uk_neg_num=self.neg_uk_num,
                                                      kk_pos_num=self.pos_kk_num,
                                                      kk_neg_num=self.neg_kk_num)
            # sampling_elapsed = time.time() - sampling_start
            # print('train_time: %.2f' % sampling_elapsed)

            # used for assignment of beta value
            if self.total_anneal_steps > 0:
                self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                self.anneal = self.anneal_cap

            '''get predictions, kl losses, and the user embeddings'''
            pred_matrix, kl_loss, mu_q, logvar_q = self.forward(batch_matrix)

            '''Gaussian log-likelihood loss'''
            mask = batch_matrix != 0
            sigma = self.observation_std * torch.ones([], device=pred_matrix.device)
            # recon_loss = torch.sum(gaussian_nll(pred_matrix, sigma, batch_matrix) * mask) / torch.sum(mask)
            recon_loss = gaussian_nll(pred_matrix * mask, sigma, batch_matrix * mask)

            # for the unobserved entries
            mask0 = batch_matrix == 0
            sigma0 = self.observation_std * torch.ones([], device=pred_matrix.device)
            # recon_loss0 = torch.sum(gaussian_nll(pred_matrix, sigma0, batch_matrix) * mask0) / torch.sum(mask0)
            recon_loss0 = gaussian_nll(pred_matrix * mask0, sigma0, batch_matrix * mask0)

            # recon_loss = gaussian_nll(pred_matrix, sigma, batch_matrix)

            '''compute contrastive loss'''
            '''1. User-keyphrase contrastive loss'''
            # repeat to have same number as the pos & neg samples
            mu_user_p = torch.repeat_interleave(mu_q[u_sample], repeats=self.pos_uk_num, dim=0)
            #.to(self.device)
            var_user_p = torch.repeat_interleave(torch.exp(logvar_q[u_sample]), repeats=self.pos_uk_num, dim=0)
            #.to(self.device)

            mu_user_n = torch.repeat_interleave(mu_q[u_sample], repeats=self.neg_uk_num, dim=0)
            #.to(self.device)
            var_user_n = torch.repeat_interleave(torch.exp(logvar_q[u_sample]), repeats=self.neg_uk_num, dim=0)
            #.to(self.device)

            # get positive and negatie keyphrase samples for users
            mu_user_pos = (self.keyphrase_mu.weight[pos_uk.ravel()])\
                # .to(self.device)
            var_user_pos = (torch.exp(self.keyphrase_log_var.weight[pos_uk.ravel()]))\
                # .to(self.device)

            mu_user_neg = (self.keyphrase_mu.weight[neg_uk.ravel()])\
                # .to(self.device)
            var_user_neg = (torch.exp(self.keyphrase_log_var.weight[neg_uk.ravel()]))\
                # .to(self.device)

            # [(batch_size*num_pos_samples) * hidden_dim]
            assert mu_user_p.shape == mu_user_pos.shape
            assert mu_user_p.shape == var_user_p.shape

            # compute the kernel between the anchoring point with their corresponding sample each
            # [(batch_size*num_pos_samples),]
            pos_uk_kernel = torch.divide(
                kernel_selection(self.kernel_method, mu_user_p, mu_user_pos, var_user_p, var_user_pos),
                self.temperature_tau_u)
            neg_uk_kernel = torch.divide(
                kernel_selection(self.kernel_method, mu_user_n, mu_user_neg, var_user_n, var_user_neg),
                self.temperature_tau_u)

            assert mu_user_p.shape[0] == pos_uk_kernel.shape[0]
            assert mu_user_n.shape[0] == neg_uk_kernel.shape[0]

            # compute contrastive loss
            log_prob_user = pos_uk_kernel - np.log(self.num_keyphrases / (self.neg_uk_num + self.pos_uk_num)) \
                            - logsumexp(pos_uk_kernel, neg_uk_kernel, self.pos_uk_num, self.neg_uk_num)

            # user_contrastive_loss = torch.sum(log_prob_user) / (
            #         self.num_samples_u * len(batch_idx))  # double summation and averaging
            user_contrastive_loss = torch.sum(log_prob_user) / len(u_sample)

            '''2. Keyphrase-keyphrase contrastive loss'''
            mu_keyphrase = self.keyphrase_mu.weight[[pos_uk.ravel()]].to(self.device)
            logvar_keyphrase = self.keyphrase_log_var.weight[[pos_uk.ravel()]].to(self.device)

            # repeat to have same number as the pos & neg samples [(batch_size * self.pos_uk_num * self.pos_kk_num) * hidden_dim]
            mu_keyphrase_p = torch.repeat_interleave(mu_keyphrase, repeats=self.pos_kk_num, dim=0)\
                # .to(self.device)
            var_keyphrase_p = torch.repeat_interleave(torch.exp(logvar_keyphrase), repeats=self.pos_kk_num, dim=0)\
                # .to(self.device)

            # [(batch_size * self.pos_uk_num * self.neg_kk_num) * hidden_dim]
            mu_keyphrase_n = torch.repeat_interleave(mu_keyphrase, repeats=self.neg_kk_num, dim=0)\
                # .to(self.device)
            var_keyphrase_n = torch.repeat_interleave(torch.exp(logvar_keyphrase), repeats=self.neg_kk_num, dim=0)\
                # .to(self.device)

            # get positive and negative keyphrase samples for keyphrases
            mu_keyphrase_pos = (self.keyphrase_mu.weight[pos_kk.ravel()])\
                # .to(self.device)
            var_keyphrase_pos = (torch.exp(self.keyphrase_log_var.weight[pos_kk.ravel()]))\
                # .to(self.device)

            mu_keyphrase_neg = (self.keyphrase_mu.weight[neg_kk.ravel()])\
                # .to(self.device)
            var_keyphrase_neg = (torch.exp(self.keyphrase_log_var.weight[neg_kk.ravel()]))\
                # .to(self.device)

            # [(batch_size*num_pos_samples),]
            assert mu_keyphrase_p.shape == mu_keyphrase_pos.shape
            assert mu_keyphrase_p.shape == var_keyphrase_p.shape

            # [(batch_size*num_pos_samples),]
            pos_kk_kernel = torch.divide(
                kernel_selection(self.kernel_method, mu_keyphrase_p, mu_keyphrase_pos, var_keyphrase_p,
                                 var_keyphrase_pos),
                self.temperature_tau_k)
            neg_kk_kernel = torch.divide(
                kernel_selection(self.kernel_method, mu_keyphrase_n, mu_keyphrase_neg, var_keyphrase_n,
                                 var_keyphrase_neg),
                self.temperature_tau_k)

            assert mu_keyphrase_p.shape[0] == pos_kk_kernel.shape[0]
            assert mu_keyphrase_n.shape[0] == neg_kk_kernel.shape[0]

            # compute contrastive loss
            log_prob_keyphrase = (pos_kk_kernel - np.log(self.num_keyphrases / (self.neg_kk_num + self.pos_kk_num))
                                  - logsumexp(pos_kk_kernel, neg_kk_kernel,
                                              self.pos_kk_num, self.neg_kk_num))\

            keyphrase_contrastive_loss = torch.sum(log_prob_keyphrase) / len(u_sample)

            # keyphrase_contrastive_loss = torch.sum(
            #     log_prob_keyphrase) / (self.num_samples_k * len(batch_idx_k))  # double summation and averaging
            # l2 norm regularization, also regularizing the keyphrases' stdev embeddings
            l2_reg = torch.tensor(0., requires_grad=True)
            for layer in self.encoder:
                if 'weight' in dir(layer):
                    l2_reg = l2_reg + torch.norm(layer.weight)

            l2_reg = l2_reg + torch.norm(self.decoder.weight)
            l2_reg = l2_reg + torch.norm(self.keyphrase_mu.weight)
            l2_reg = l2_reg + torch.norm(torch.exp(0.5 * self.keyphrase_log_var.weight))

            # (recon_loss + kl_loss * self.anneal) / torch.sum(mask) \
            # + self.weighted_recon * recon_loss0\

            batch_loss = recon_loss + self.weighted_recon * recon_loss0\
                         + kl_loss * self.anneal \
                         - self.hp_contrastive_u * (user_contrastive_loss) \
                         - self.hp_contrastive_k * (keyphrase_contrastive_loss) \
                         + self.weight_decay * l2_reg

            # record if experiment exists
            if experiment:
                vae_avgmeter.update(recon_loss + self.weighted_recon * recon_loss0\
                         + kl_loss * self.anneal, 1)
                usercl_avgmeter.update(-(user_contrastive_loss * self.hp_contrastive_u), 1)
                keyphrasecl_avgmeter.update(-(keyphrase_contrastive_loss * self.hp_contrastive_k), 1)
                # uk_pos_avgmeter.update(torch.mean(pos_uk_kernel.detach()), 1)
                # uk_neg_avgmeter.update(torch.mean(neg_uk_kernel.detach()), 1)
                # kk_pos_avgmeter.update(torch.mean(pos_kk_kernel.detach()), 1)
                # kk_neg_avgmeter.update(torch.mean(neg_kk_kernel.detach()), 1)

            # Backward keyphrase cl only
            # individual_loss = -(keyphrase_contrastive_loss * self.hp_contrastive_k)
            # individual_loss.backward()

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

            loss += batch_loss
            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))

        if experiment:
            experiment.log_metric(name='VAE_loss',
                                  value=vae_avgmeter.avg())
            experiment.log_metric(name='User_CL',
                                  value=usercl_avgmeter.avg())
            experiment.log_metric(name='Keyphrase_CL',
                                  value=keyphrasecl_avgmeter.avg())
            # experiment.log_metric(name='UK_positive',
            #                       value=uk_pos_avgmeter.avg())
            # experiment.log_metric(name='UK_negative',
            #                       value=uk_neg_avgmeter.avg())
            # experiment.log_metric(name='KK_positive',
            #                       value=kk_pos_avgmeter.avg())
            # experiment.log_metric(name='KK_negative',
            #                       value=kk_neg_avgmeter.avg())

        return loss.detach()

    def predict(self, input_matrix):
        '''
        Args:
            input_matrix: a input UI matrix
        Returns:
            pred_matrix: a predicted UI matrix
        '''
        with torch.no_grad():
            input_batch_matrix = torch.FloatTensor(input_matrix.toarray()).to(self.device)
            pred_batch_matrix = self.forward(input_batch_matrix).cpu().numpy()

        return pred_batch_matrix

    # def predict(self, input_matrix, test_matrix, test_batch_size):
    #     total_preds = []
    #     total_ys = []
    #     with torch.no_grad():
    #         num_data = input_matrix.shape[0]
    #         num_batches = int(np.ceil(num_data / test_batch_size))
    #         perm = list(range(num_data))
    #         for b in range(num_batches):
    #             if (b + 1) * test_batch_size >= num_data:
    #                 batch_idx = perm[b * test_batch_size:]
    #             else:
    #                 batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
    #
    #             input_batch_matrix = torch.FloatTensor(input_matrix[batch_idx].toarray()).to(self.device)
    #             test_batch_matrix = torch.FloatTensor(test_matrix[batch_idx].toarray())
    #
    #             pred_batch_matrix = self.forward(input_batch_matrix).cpu().numpy()
    #
    #             preds = pred_batch_matrix[test_batch_matrix != 0]
    #             ys = test_batch_matrix[test_batch_matrix != 0]
    #             if len(ys) > 0:
    #                 total_preds.append(preds)
    #                 total_ys.append(ys)
    #
    #     total_preds = np.concatenate(total_preds)
    #     total_ys = np.concatenate(total_ys)
    #
    #     return total_preds, total_ys
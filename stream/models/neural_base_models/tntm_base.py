import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

from ..abstract_helper_models.inference_networks import InferenceNetwork
from .ctm_base import CTMBase


class TNTMBase(CTMBase):

    #@override
    def __init__(
            self,
            dataset,
            mus_init : torch.Tensor,
            L_lower_init: torch.Tensor,
            log_diag_init: torch.Tensor,
            word_embeddings_projected: torch.Tensor,
            n_topics: int = 50,
            encoder_dim: int = 128,
            inference_type="zeroshot",
            dropout: float = 0.1,
            inference_activation = nn.Softplus(),
            n_layers_inference_network: int = 1,
    ):
        """
            Initialize the topic model parameters.

            Parameters
            ----------
            dataset : object
                The dataset containing bag-of-words (BoW) and embeddings.
            mus_init : torch.Tensor
                Initial value for the topic means. Shape: (n_topics, vocab_size).
            L_lower_init : torch.Tensor
                Initial value for the lower triangular matrix. Shape: (n_topics, vocab_size, vocab_size).
            log_diag_init : torch.Tensor
                Initial value for the diagonal of the covariance matrix (log of the diagonal). Shape: (n_topics, vocab_size).
            word_embeddings_projected : torch.Tensor
                Projected word embeddings. Shape: (vocab_size, encoder_dim).
            n_topics : int, optional
                Number of topics, by default 50.
            encoder_dim : int, optional
                Dimension of the encoder, by default 200.
            inference_type : str, optional
                Type of inference, either "combined", "zeroshot", or "avitm". By default "zeroshot".
            dropout : float, optional
                Dropout rate, by default 0.1.
            inference_activation : nn.Module, optional
                Activation function for inference, by default nn.Softplus().
            n_layers_inference_network : int, optional
                Number of layers in the inference network, by default 3.
        """
        super().__init__(dataset = dataset, n_topics = n_topics, encoder_dim = encoder_dim, dropout = dropout)

        self.mus = nn.Parameter(mus_init)   #create topic means as learnable paramter
        self.L_lower = nn.Parameter(L_lower_init)   # factor of covariance per topic
        self.log_diag = nn.Parameter(log_diag_init)  # summand for diagonal of covariance
        self.word_embeddings_projected = torch.tensor(word_embeddings_projected)

        emb_dim = word_embeddings_projected.shape[1]

        self.vocab_size = dataset.bow.shape[1]
        self.n_topics = n_topics
        self.encoder_dim = encoder_dim
        self.inference_activation = inference_activation
        self.inference_type = inference_type
        self.dropout = dropout

        assert self.mus.shape == (n_topics, emb_dim), f"Shape of mus is {self.mus.shape} but expected {(n_topics, emb_dim)}"
        assert self.L_lower.shape == (n_topics, emb_dim, emb_dim), f"Shape of L_lower is {self.L_lower.shape} but expected {(n_topics, emb_dim, emb_dim)}"
        assert self.log_diag.shape == (n_topics, emb_dim), f"Shape of log_diag is {self.log_diag.shape} but expected {(n_topics, emb_dim)}"
        assert word_embeddings_projected.shape == (self.vocab_size, emb_dim), f"Shape of word_embeddings_projected is {word_embeddings_projected.shape} but expected {(self.vocab_size, emb_dim)}"

        contextual_embed_size = dataset.embeddings.shape[1]

        self.mu2 = nn.Parameter(
            torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        )
        self.var2 = nn.Parameter(
            torch.as_tensor(
                (
                    ((1.0 / self.a) * (1 - (2.0 / n_topics))).T
                    + (1.0 / (n_topics * n_topics)) * np.sum(1.0 / self.a, 1)
                ).T
            )
        )

        self.inference_network = InferenceNetwork(
            input_size=self.vocab_size,
            bert_size=contextual_embed_size,
            output_size=n_topics,
            hidden_sizes=[encoder_dim]*n_layers_inference_network,
            activation=inference_activation,
            dropout=dropout,
            inference_type=inference_type,
        )

    def calc_log_beta(self):
        """
        Calculate the log of beta given self.mus, self.L_lower, and self.log_diag.
        """

        diag = torch.exp(self.log_diag)

        normal_dis_lis = [LowRankMultivariateNormal(mu, cov_factor= lower, cov_diag = D) for mu, lower, D in zip(self.mus, self.L_lower, diag)]
        log_probs = torch.zeros(self.n_topics, self.vocab_size)

        for i, dis in enumerate(normal_dis_lis):
            log_probs[i] = dis.log_prob(self.word_embeddings_projected)
        return log_probs

    def get_beta(self):
        """
        Get the beta distribution given self.mus, self.L_lower, and self.log_diag.
        """

        log_beta = self.calc_log_beta()
        return torch.exp(log_beta)
    #@override
    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : dict
            Input data containing 'bow' and 'embedding'.

        Returns
        ------
        log_recon : torch.Tensor
            The log of the reconstruction.
        posterior_mean : torch.Tensor
            The mean of the variational posterior.
        posterior_logvar : torch.Tensor
            The log variance of the variational posterior.
        """
        theta, posterior_mean, posterior_logvar = self.get_theta(x)

        log_beta = self.calc_log_beta()



        # prodLDA vs LDA
        # use numerical trick to compute log(beta @ theta )
        log_theta = torch.nn.LogSoftmax(dim=-1)(theta)        #calculate log theta = log_softmax(theta_hat)
        A = log_beta + log_theta.unsqueeze(-1)               #calculate (log (beta @ theta))[i] = (log (exp(log_beta) @ exp(log_theta)))[i] = log(\sum_k exp (log_beta[i,k] + log_theta[k]))
        log_recon = torch.logsumexp(A, dim = 1)

        return log_recon, posterior_mean, posterior_logvar


    def loss_function(self, x_bow, log_recon, posterior_mean, posterior_logvar):
        """
        Computes the reconstruction and KL divergence loss.

        Parameters
        ----------
        x_bow: torch.Tensor
            Bag-of-words data.
        log_recon : torch.Tensor
            The log of the reconstruction for x_bow
        posterior_mean : torch.Tensor
            The mean of the variational posterior.
        posterior_logvar : torch.Tensor
            The log variance of the variational posterior

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
         #Negative log-likelihood:  - (u^d)^T @ log(beta @ \theta^d)
        NL = -(x_bow * log_recon).sum(1)

        prior_mean = self.mu2
        prior_var = self.var2

        #KLD between variational posterior p(\theta|d) and prior p(\theta)
        posterior_var = posterior_logvar.exp()
        prior_mean = prior_mean.expand_as(posterior_mean)
        prior_var = prior_var.expand_as(posterior_mean)
        prior_logvar = torch.log(prior_var)

        var_division = posterior_var / prior_var

        diff = posterior_mean - prior_mean
        diff_term = diff*diff / prior_var
        logvar_division = prior_logvar - posterior_logvar


        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.n_topics)

        loss = (NL + KLD).mean()
        return loss

    def compute_loss(self, x):
        """
        Computes the loss for the model.

        Parameters
        ----------
        x : dict
            Input data containing 'bow' and 'embedding'.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        x_bow = x['bow']
        log_recon, posterior_mean, posterior_logvar = self.forward(x)
        loss = self.loss_function(x_bow, log_recon, posterior_mean, posterior_logvar)
        return loss








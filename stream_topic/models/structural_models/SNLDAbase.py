import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..abstract_helper_models.inference_networks import InferenceNetwork
from ...NAM.NAM import NeuralAdditiveModel
from torch.distributions import Normal, Dirichlet


class StructuralNeuralLDABase(nn.Module):

    def __init__(
        self,
        dataset,
        n_topics=10,
        encoder_dim=128,
        dropout=0.1,
        inference_activation=nn.Softplus(),
        theta_dependence_on_covariates=True,
        beta_dependence_on_covariates=True,
        hidden_units_theta=[128, 128, 64],
        hidden_units_beta=[128, 128],
        lambda_kl=1.0,
        lambda_covariate=1.0,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.vocab_size = dataset.bow.shape[1]
        self.lambda_kl = lambda_kl
        self.lambda_covariate = lambda_covariate
        self.theta_dependence_on_covariates = theta_dependence_on_covariates
        self.beta_dependence_on_covariates = beta_dependence_on_covariates

        # Priors for theta and beta
        self.theta_prior = Normal(0, 1)
        self.beta_prior = Dirichlet(torch.ones(self.vocab_size))

        self.inference_network = InferenceNetwork(
            input_size=self.vocab_size,
            bert_size=None,
            output_size=n_topics,
            hidden_sizes=[encoder_dim],
            activation=inference_activation,
            dropout=dropout,
            inference_type="avitm",
        )

        if theta_dependence_on_covariates:
            self.theta_nam = NeuralAdditiveModel(
                input_size=dataset.features.shape[1],
                output_size=2 * n_topics,
                hidden_units=hidden_units_theta,
                feature_dropout=dropout,
                hidden_dropout=dropout,
                activation="ReLU",
                out_activation=None,
            )

        if beta_dependence_on_covariates:
            self.beta_nam = NeuralAdditiveModel(
                input_size=dataset.features.shape[1],
                output_size=n_topics * self.vocab_size,
                hidden_units=hidden_units_beta,
                feature_dropout=dropout,
                hidden_dropout=dropout,
                activation="ReLU",
                out_activation=None,
            )

        self.beta_mean = nn.Parameter(torch.empty(n_topics, self.vocab_size))
        self.beta_logvar = nn.Parameter(torch.empty(n_topics, self.vocab_size))
        nn.init.xavier_uniform_(self.beta_mean)
        nn.init.xavier_uniform_(self.beta_logvar)

        self.covariate_predictor = nn.Linear(n_topics, dataset.features.shape[1])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_theta(self, x):
        mu, logvar = self.inference_network(x)
        if self.theta_dependence_on_covariates:
            adjustments = self.theta_nam(x["features"]).view(-1, 2 * self.n_topics)
            mu += adjustments[:, : self.n_topics]
            logvar += adjustments[:, self.n_topics :]

        theta = F.softmax(self.reparameterize(mu, logvar), dim=1)
        return theta, mu, logvar

    def get_beta(self, covariates=None):
        beta_mu = self.beta_mean
        beta_logvar = self.beta_logvar
        if self.beta_dependence_on_covariates and covariates is not None:
            adjustments = self.beta_nam(covariates).view(self.n_topics, self.vocab_size)
            beta_mu += adjustments

        beta = F.softmax(self.reparameterize(beta_mu, beta_logvar), dim=1)
        return beta

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        if self.beta_dependence_on_covariates:
            beta = self.get_beta(covariates=x["features"])
        else:
            beta = self.get_beta()

        word_dist = torch.matmul(theta, beta)
        return word_dist, mu, logvar, theta, beta

    def compute_loss(self, x):
        word_dist, mu, logvar, theta, beta = self.forward(x)
        x_bow = x["bow"]

        # Reconstruction Loss
        recon_loss = -(x_bow * (word_dist + 1e-10).log()).sum(axis=1).mean()

        # KL Divergence Loss
        kl_theta = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(axis=1).mean()
        kl_beta = torch.distributions.kl.kl_divergence(
            Dirichlet(beta), self.beta_prior
        ).mean()

        # Auxiliary Covariate Prediction Loss
        covariate_preds = self.covariate_predictor(theta)
        covariate_loss = F.mse_loss(covariate_preds, x["features"])

        # Total Loss
        total_loss = (
            recon_loss
            + self.lambda_kl * (kl_theta + kl_beta)
            + self.lambda_covariate * covariate_loss
        )
        return total_loss

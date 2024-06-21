"""PyTorch class for feed foward AVITM network."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .encoding_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):
    """AVITM Network."""

    def __init__(
        self,
        input_size,
        bert_size,
        infnet,
        n_components=10,
        model_type="prodLDA",
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        topic_prior_mean=0.0,
        topic_prior_variance=None,
        topic_perturb=1,
    ):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
            topic_prior_mean: double, mean parameter of the prior
            topic_prior_variance: double, variance parameter of the prior
        """
        super().__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert (
            isinstance(n_components, int) or isinstance(n_components, np.int64)
        ) and n_components > 0, "n_components must be type int > 0."
        assert model_type in ["prodLDA", "LDA"], "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be type tuple."
        assert activation in [
            "softplus",
            "relu",
            "sigmoid",
            "tanh",
            "leakyrelu",
            "rrelu",
            "elu",
            "selu",
        ], (
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu',"
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        )
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(
            topic_prior_mean, float
        ), "topic_prior_mean must be type float"
        # and topic_prior_variance >= 0, \
        # assert isinstance(topic_prior_variance, float), \
        #    "topic prior_variance must be type float"

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        self.neg_method = True
        self.topic_perturb = topic_perturb

        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation
            )
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation
            )
        else:
            raise Exception(
                "Missing infnet parameter, options are zeroshot and combined"
            )
        if torch.cuda.is_available():
            self.inf_net = self.inf_net.cuda()
        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        # self.topic_prior_mean = topic_prior_mean
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        if topic_prior_variance is None:
            topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor([topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    @staticmethod
    def perturb(x):
        """Add Gaussian noise."""
        eps = torch.randn_like(x)
        return eps.add_(x)

    @staticmethod
    def perturbTopK(x, k):
        _, kidx = x.topk(k=k, dim=1)
        y = x.clone()
        y[torch.arange(y.size(0))[:, None], kidx] = 0.0
        return y

    @staticmethod
    def perturbTheta(x, k):
        x_new = DecoderNetwork.perturbTopK(x, k)
        x_new = x_new / x_new.sum(dim=-1).unsqueeze(1)
        return x_new

    def forward(self, x, x_bert):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

        topic_doc = theta
        # theta = self.drop_theta(theta)

        if self.neg_method:
            theta_neg = DecoderNetwork.perturbTheta(theta, self.topic_perturb)

        # prodLDA vs LDA
        if self.model_type == "prodLDA":
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
            )

            if self.neg_method:
                word_dist_neg = F.softmax(
                    self.beta_batchnorm(torch.matmul(theta_neg, self.beta)), dim=1
                )

            topic_word = self.beta
            # word_dist: batch_size x input_size
            self.topic_word_matrix = self.beta
        elif self.model_type == "LDA":
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)

            topic_word = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
            if self.neg_method:
                word_dist_neg = torch.matmul(theta_neg, beta)

        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist,
            topic_word,
            topic_doc,
            word_dist_neg,
        )

    def get_theta(self, x, x_bert):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )

            return theta

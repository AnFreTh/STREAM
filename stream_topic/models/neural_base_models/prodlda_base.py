import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..abstract_helper_models.inference_networks import InferenceNetwork


class ProdLDABase(nn.Module):
    """
    Product of Experts Latent Dirichlet Allocation (ProdLDA) model.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    n_topics : int, optional
        The number of topics (default is 50).
    encoder_dim : int, optional
        The number of units in the encoder (default is 200).
    dropout : float, optional
        The dropout rate (default is 0.4).
    """

    def __init__(
        self,
        dataset,
        n_topics=10,
        encoder_dim=128,
        dropout=0.1,
        inference_activation=nn.Softplus(),
        rescale_loss=False,
        rescale_factor=1e-2,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.vocab_size = dataset.bow.shape[1]

        self.a = 1 * np.ones((1, n_topics)).astype(np.float32)
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
            bert_size=None,
            output_size=n_topics,
            hidden_sizes=[encoder_dim],
            activation=inference_activation,
            dropout=dropout,
            inference_type="avitm",
        )

        self.rescale_loss = rescale_loss
        self.rescale_factor = rescale_factor

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.mean_bn = nn.BatchNorm1d(n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(n_topics))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(
            n_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.logvar_bn.weight.data.copy_(torch.ones(n_topics))
        self.logvar_bn.weight.requires_grad = False

        self.beta_batchnorm = nn.BatchNorm1d(
            self.vocab_size, eps=0.001, momentum=0.001, affine=True
        )
        self.beta_batchnorm.weight.data.copy_(torch.ones(self.vocab_size))
        self.beta_batchnorm.weight.requires_grad = False

        self.theta_drop = nn.Dropout(dropout)

        self.beta = nn.Parameter(torch.empty(n_topics, self.vocab_size))
        nn.init.xavier_uniform_(self.beta)

    def get_beta(self):
        """
        Returns the beta parameter.

        Returns
        -------
        torch.Tensor
            The beta parameter.
        """
        return self.beta

    def get_theta(self, x, only_theta=False):
        """
        Computes the theta (document-topic distribution).

        Parameters
        ----------
        x : dict
            Input data containing 'bow' and 'embedding'.
        only_theta : bool, optional
            If True, returns only theta, by default False.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensors
            Theta if only_theta is True, otherwise theta, mu, and sigma.
        """
        mu, sigma = self.inference_network(x)
        theta = F.softmax(self.reparameterize(mu, sigma), dim=1)

        theta = self.theta_drop(theta)

        if only_theta:
            return theta
        else:
            return theta, mu, sigma

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent variables.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the latent variables.
        logvar : torch.Tensor
            The log variance of the latent variables.

        Returns
        -------
        torch.Tensor
            The reparameterized latent variables.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def forward(self, x):
        """
        Perform a forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        dict
            A dictionary containing the loss.
        """

        theta, mu, logvar = self.get_theta(x)
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
        )
        return word_dist, mu, logvar

    def compute_loss(self, x):
        """
        Compute the loss for the model.

        Parameters
        ----------
        x : dict
            The input data containing the 'bow' key.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        word_dist, mu, logvar = self.forward(x)
        x = x["bow"]
        loss = self.loss_function(x, word_dist, mu, logvar)
        return loss

    def loss_function(self, x, recon_x, mu, logvar):
        """
        Calculate the loss function.

        Parameters
        ----------
        x : torch.Tensor
            The original input tensor.
        recon_x : torch.Tensor
            The reconstructed input tensor.
        mu : torch.Tensor
            The mean of the latent variables.
        logvar : torch.Tensor
            The log variance of the latent variables.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * (
            (var_division + diff_term + logvar_division).sum(axis=1) - self.n_topics
        )
        loss = (recon_loss + KLD).mean()
        return loss

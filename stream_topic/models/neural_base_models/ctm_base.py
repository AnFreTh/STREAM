import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..abstract_helper_models.inference_networks import InferenceNetwork


class CTMBase(nn.Module):
    """
    CTMBase is a neural network-based topic modeling class that supports both ProdLDA and LDA models.

    Parameters
    ----------
    dataset : object
        The dataset containing bag-of-words (BoW) and embeddings.
    n_topics : int, optional
        Number of topics, by default 50.
    encoder_dim : int, optional
        Dimension of the encoder, by default 200.
    dropout : float, optional
        Dropout rate, by default 0.1.
    inference_type : str, optional
        Type of inference, either "combined" or "zeroshot", by default "combined".
    inference_activation : nn.Module, optional
        Activation function for inference, by default nn.Softplus().
    model_type : str, optional
        Type of model, either "ProdLDA" or "LDA", by default "ProdLDA".
    rescale_loss : bool, optional
        Whether to rescale the loss, by default False.
    rescale_factor : float, optional
        Factor to rescale the loss, by default 1e-2.

    Attributes
    ----------
    model_type : str
        Type of model.
    vocab_size : int
        Size of the vocabulary.
    n_topics : int
        Number of topics.
    a : np.ndarray
        Array of ones for topic prior.
    mu2 : nn.Parameter
        Mean parameter for the prior.
    var2 : nn.Parameter
        Variance parameter for the prior.
    inference_network : InferenceNetwork
        Inference network.
    rescale_loss : bool
        Whether to rescale the loss.
    rescale_factor : float
        Factor to rescale the loss.
    mean_bn : nn.BatchNorm1d
        Batch normalization for the mean.
    logvar_bn : nn.BatchNorm1d
        Batch normalization for the log variance.
    beta_batchnorm : nn.BatchNorm1d
        Batch normalization for beta.
    theta_drop : nn.Dropout
        Dropout for theta.
    beta : nn.Parameter
        Beta parameter.

    Methods
    -------
    get_beta()
        Returns the beta parameter.
    get_theta(x, only_theta=False)
        Computes the theta (document-topic distribution).
    reparameterize(mu, logvar)
        Reparameterizes the distribution for the reparameterization trick.
    forward(x)
        Forward pass through the network.
    compute_loss(x)
        Computes the loss for the model.
    loss_function(x, recon_x, mu, logvar)
        Computes the reconstruction and KL divergence loss.
    """

    def __init__(
        self,
        dataset,
        n_topics=50,
        encoder_dim=128,
        dropout=0.1,
        inference_type="combined",
        inference_activation=nn.Softplus(),
        model_type="ProdLDA",
        rescale_loss=False,
        rescale_factor=1e-2,
    ):
        super().__init__()

        assert model_type in ["ProdLDA", "LDA"]
        self.model_type = model_type

        self.vocab_size = dataset.bow.shape[1]
        contextual_embed_size = dataset.embeddings.shape[1]
        self.n_topics = n_topics

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
            bert_size=contextual_embed_size,
            output_size=n_topics,
            hidden_sizes=[encoder_dim],
            activation=inference_activation,
            dropout=dropout,
            inference_type=inference_type,
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
        return self.beta  # .weight.T

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
        Reparameterizes the distribution for the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the distribution.
        logvar : torch.Tensor
            Log variance of the distribution.

        Returns
        -------
        torch.Tensor
            Reparameterized sample.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : dict
            Input data containing 'bow' and 'embedding'.

        Returns
        -------
        tuple of torch.Tensors
            Word distribution, mu, and logvar.
        """

        theta, mu, logvar = self.get_theta(x)

        # prodLDA vs LDA
        if self.model_type == "ProdLDA":
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
            )
        elif self.model_type == "LDA":
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            word_dist = torch.matmul(theta, beta)

        return word_dist, mu, logvar

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
        recon_x, mu, logvar = self.forward(x)
        loss = self.loss_function(x["bow"], recon_x, mu, logvar)
        if self.rescale_loss:
            loss *= self.rescale_factor

        return loss

    def loss_function(self, x, recon_x, mu, logvar):
        """
        Computes the reconstruction and KL divergence loss.

        Parameters
        ----------
        x : torch.Tensor
            Original input data.
        recon_x : torch.Tensor
            Reconstructed data.
        mu : torch.Tensor
            Mean of the distribution.
        logvar : torch.Tensor
            Log variance of the distribution.

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

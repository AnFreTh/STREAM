import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..abstract_helper_models.inference_networks import InferenceNetwork
from ...NAM.NAM import NeuralAdditiveModel
from typing import List


class StructuralCTMBase(nn.Module):
    """
    StructuralCTMBase is a neural network-based topic modeling class that supports both ProdLDA and LDA models and uses a NAM to model theta.

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
        feature_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        hidden_units: List[int] = [128, 128, 64],
        rescale_loss=False,
        rescale_factor: float = 1e-2,
        beta_dependence_on_covariates: bool = False,
        nam_model_variance: bool = False,
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

        self.nam_model_variance = nam_model_variance
        if self.nam_model_variance:
            output_size = 2 * n_topics
        else:
            output_size = n_topics

        self.nam = NeuralAdditiveModel(
            input_size=dataset.features.shape[1],
            output_size=output_size,
            hidden_units=hidden_units,
            feature_dropout=feature_dropout,
            hidden_dropout=hidden_dropout,
            activation="ReLU",
            out_activation=None,
        )

        if beta_dependence_on_covariates:
            self.beta_nam = NeuralAdditiveModel(
                input_size=dataset.features.shape[1],
                output_size=n_topics * self.vocab_size,
                hidden_units=hidden_units,
                feature_dropout=feature_dropout,
                hidden_dropout=hidden_dropout,
                activation="ReLU",
                out_activation=None,
            )
            self.beta_dependence_on_covariates = True
        else:
            self.beta_dependence_on_covariates = False

    def get_beta(self, covariates=None):
        """
        Returns the beta parameter. If covariate dependence is enabled,
        computes beta based on covariates.

        Parameters
        ----------
        covariates : torch.Tensor, optional
            Covariates to condition beta on, by default None.

        Returns
        -------
        torch.Tensor
            The beta parameter.
        """
        if self.beta_dependence_on_covariates and covariates is not None:
            # Compute covariate-dependent beta using NAM
            beta_adjustments = self.beta_nam(covariates).view(
                self.n_topics, self.vocab_size
            )
            beta = self.beta + beta_adjustments
        else:
            beta = self.beta

        return beta

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
        features = x["features"]
        mu, sigma = self.inference_network(x)
        feature_preds = self.nam(features).squeeze()
        if self.nam_model_variance:
            half = int(mu.shape[1])
            feature_mu = feature_preds[:, 0:half]
            feature_var = feature_preds[:, half:]
            mu += feature_mu
            sigma += feature_var
        else:
            mu += feature_preds
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

        if self.beta_dependence_on_covariates:
            beta = self.get_beta(covariates=x["features"])
        else:
            beta = self.get_beta()

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

    def plotting_preds(self, datamodule):
        """
        Generate predictions using the datamodule's predict dataloader.

        Parameters
        ----------
        datamodule : LightningDataModule
            The PyTorch Lightning datamodule.

        Returns
        -------
        tuple of lists of np.ndarray
            Two or three lists:
            - Feature outputs for mu
            - (Optional) Feature outputs for var if self.nam_model_variance is True
            - Corresponding feature values from the input batch
        """
        # Prepare the datamodule (if required)
        datamodule.setup(stage="predict")

        # Get the predict dataloader
        dataloader = datamodule.predict_dataloader()

        # Put the model in evaluation mode
        self.eval()

        all_mu = []  # List to store mu predictions for each feature
        all_var = (
            [] if self.nam_model_variance else None
        )  # Store var predictions only if enabled
        all_features = []  # List to store corresponding feature values for each feature

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"]
                feature_mu = []  # Temporary list for this batch's mu outputs
                feature_var = (
                    [] if self.nam_model_variance else None
                )  # Temporary list for var outputs
                batch_features = []  # Temporary list for feature inputs

                for i, nn in enumerate(self.nam.feature_nns):
                    feature_input = features[:, i : i + 1]  # Extract single feature
                    feature_output = nn(feature_input)  # Forward pass

                    if self.nam_model_variance:
                        half = feature_output.shape[1] // 2
                        mu = feature_output[:, :half]  # First half for mu
                        var = feature_output[:, half:]  # Second half for var

                        feature_mu.append(mu)
                        feature_var.append(var)
                    else:
                        feature_mu.append(feature_output)  # Only store mu

                    batch_features.append(feature_input)

                all_mu.append(feature_mu)
                if self.nam_model_variance:
                    all_var.append(feature_var)
                all_features.append(batch_features)

        # Combine all batches into numpy arrays
        feature_mu_outputs = [
            torch.cat([batch_mu[i] for batch_mu in all_mu], dim=0).cpu().numpy()
            for i in range(len(self.nam.feature_nns))
        ]
        feature_values = [
            torch.cat([batch_features[i] for batch_features in all_features], dim=0)
            .cpu()
            .numpy()
            for i in range(len(self.nam.feature_nns))
        ]

        if self.nam_model_variance:
            feature_var_outputs = [
                torch.cat([batch_var[i] for batch_var in all_var], dim=0).cpu().numpy()
                for i in range(len(self.nam.feature_nns))
            ]
            return feature_mu_outputs, feature_var_outputs, feature_values

        return feature_mu_outputs, feature_values

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...NAM.NAM import NeuralAdditiveModel
from typing import List


class StructuralETMBase(nn.Module):
    """
    An implementation of the Structural Embedded Topic Model (SETM).

    Parameters
    ----------
    dataset : Dataset
        The dataset containing the bag-of-words (bow) matrix.
    embed_size : int, optional
        The size of the word embeddings (default is 128).
    n_topics : int, optional
        The number of topics (default is 10).
    en_units : int, optional
        The number of units in the encoder (default is 256).
    dropout : float, optional
        The dropout rate (default is 0.0).
    pretrained_WE : ndarray, optional
        Pretrained word embeddings (default is None).
    train_WE : bool, optional
        Whether to train the word embeddings (default is True).
    """

    def __init__(
        self,
        dataset,
        embed_size: int = 64,
        n_topics: int = 10,
        encoder_dim: int = 128,
        dropout: float = 0.1,
        train_WE: bool = True,
        encoder_activation: callable = nn.ReLU(),
        activation: callable = nn.ReLU(),
        feature_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        hidden_units: List[int] = [128, 128, 64],
        pretrained_WE=None,
    ):
        super().__init__()

        vocab_size = dataset.bow.shape[1]

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(
            torch.randn((n_topics, self.word_embeddings.shape[1]))
        )

        self.vocab_encoder = nn.Sequential(
            nn.Linear(vocab_size, encoder_dim),
            encoder_activation,
            nn.Linear(encoder_dim, encoder_dim),
            encoder_activation,
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 2 * n_topics),
        )

        self.nam = NeuralAdditiveModel(
            input_size=dataset.features.shape[1],
            output_size=2 * n_topics,
            hidden_units=hidden_units,
            feature_dropout=feature_dropout,
            hidden_dropout=hidden_dropout,
            activation="ReLU",
            out_activation=None,
        )

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

    def encode(self, x, features):
        """
        Encode the input into latent variables.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple of torch.Tensor
            The mean and log variance of the latent variables.
        """
        vocab_preds = self.vocab_encoder(x)
        feature_preds = self.nam(features).squeeze()
        vals = vocab_preds + feature_preds
        half = int(vocab_preds.shape[1] / 2)
        mu = vals[:, 0:half]
        var = vals[:, half:]
        return mu, var

    def get_theta(self, x, features, only_theta=False):
        """
        Get the topic proportions (theta).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        only_theta : bool, optional
            Whether to return only theta (default is False).

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The topic proportions. If `only_theta` is False, also returns the mean and log variance of the latent variables.
        """
        norm_x = x / (x.sum(1, keepdim=True) + 1e-6)
        mu, logvar = self.encode(norm_x, features)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        if only_theta:
            return theta
        else:
            return theta, mu, logvar

    def get_beta(self):
        """
        Get the topic-word distribution (beta).

        Returns
        -------
        torch.Tensor
            The topic-word distribution.
        """
        beta = F.softmax(
            torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1
        )
        return beta

    def forward(self, x):
        """
        Perform a forward pass of the model.

        Parameters
        ----------
        x : dict
            The input data containing the 'bow' key.

        Returns
        -------
        tuple of torch.Tensor
            The reconstructed bag-of-words, the mean and log variance of the latent variables.
        """
        bow = x["bow"]
        features = x["features"]
        theta, mu, logvar = self.get_theta(bow, features)
        beta = self.get_beta()
        recon_x = torch.matmul(theta, beta)
        return recon_x, mu, logvar

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
        recon_x, mu, logvar = self.forward(x)
        x = x["bow"]
        loss = self.loss_function(x, recon_x, mu, logvar)
        return loss * 1e-02

    def loss_function(self, x, recon_x, mu, logvar):
        """
        Calculate the loss function.

        Parameters
        ----------
        x : torch.Tensor
            The original bag-of-words.
        recon_x : torch.Tensor
            The reconstructed bag-of-words.
        mu : torch.Tensor
            The mean of the latent variables.
        logvar : torch.Tensor
            The log variance of the latent variables.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)
        loss = (recon_loss + KLD).mean()
        return loss

    def get_complete_theta_mat(self, x):
        """
        Get the complete matrix of topic proportions.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The topic proportions.
        """
        theta, mu, logvar = self.get_theta(x)
        return theta

    def plotting_preds(self, datamodule):
        """
        Generate predictions using the datamodule's predict dataloader.

        Parameters
        ----------
        model : LightningModule
            The PyTorch Lightning model.
        datamodule : LightningDataModule
            The PyTorch Lightning datamodule.

        Returns
        -------
        tuple of np.ndarray
            All predictions (mu, var) for the dataset.
        """
        # Prepare the datamodule (if required)
        datamodule.setup(stage="predict")

        # Get the predict dataloader
        dataloader = datamodule.predict_dataloader()

        # Put the model in evaluation mode
        self.eval()

        all_mu = []
        all_var = []

        with torch.no_grad():
            for batch in dataloader:
                # Pass the batch through the model's prediction method
                features = batch["features"]
                preds = self.nam(features).squeeze()
                half = int(preds.shape[1] / 2)
                mu = preds[:, 0:half]
                var = preds[:, half:]

                # Collect predictions
                all_mu.append(mu)
                all_var.append(var)

        # Concatenate all predictions into single tensors
        all_mu = torch.cat(all_mu, dim=0).cpu().numpy()
        all_var = torch.cat(all_var, dim=0).cpu().numpy()

        return all_mu, all_var

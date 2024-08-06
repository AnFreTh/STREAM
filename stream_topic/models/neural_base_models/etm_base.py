import torch
import torch.nn as nn
import torch.nn.functional as F


class ETMBase(nn.Module):
    """
    An implementation of the Embedded Topic Model (ETM).

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
        embed_size: int = 128,
        n_topics: int = 10,
        encoder_dim: int = 256,
        dropout: float = 0.1,
        pretrained_WE=None,
        train_WE: bool = True,
        encoder_activation: callable = nn.ReLU(),
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

        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, encoder_dim),
            encoder_activation,
            nn.Linear(encoder_dim, encoder_dim),
            encoder_activation,
            nn.Dropout(dropout),
        )

        self.fc21 = nn.Linear(encoder_dim, n_topics)
        self.fc22 = nn.Linear(encoder_dim, n_topics)

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

    def encode(self, x):
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
        e1 = self.encoder1(x)
        return self.fc21(e1), self.fc22(e1)

    def get_theta(self, x, only_theta=False):
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
        mu, logvar = self.encode(norm_x)
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
        x = x["bow"]
        theta, mu, logvar = self.get_theta(x)
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

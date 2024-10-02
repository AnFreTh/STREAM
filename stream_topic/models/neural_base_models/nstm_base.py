import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.sinkhorn_loss import sinkhorn_loss


class NSTMBase(nn.Module):
    """
    Neural Topic Model via Optimal Transport (NSTM). Based on the paper presented at ICLR 2021 by
    He Zhao, Dinh Phung, Viet Huynh, Trung Le, and Wray Buntine.

    This model learns topic embeddings using an encoder and leverages optimal transport
    via the Sinkhorn loss for topic and word distributions.

    Parameters
    ----------
    dataset : Dataset
        A dataset object containing the bag-of-words (BoW) matrix used for training.
    n_topics : int, optional
        The number of topics to be learned by the model, by default 50.
    encoder_dim : int, optional
        The dimension of the encoder's hidden layer, by default 128.
    dropout : float, optional
        The dropout rate for the encoder, by default 0.1.
    pretrained_WE : numpy.ndarray, optional
        Pretrained word embeddings as a numpy array. If None, the embeddings will be randomly initialized, by default None.
    train_WE : bool, optional
        Whether to fine-tune (train) the word embeddings during model training, by default True.
    encoder_activation : callable, optional
        The activation function for the encoder, by default nn.ReLU().
    embed_size : int, optional
        The size of the word embedding vectors, by default 256.
    recon_loss_weight : float, optional
        The weight given to the reconstruction loss, by default 0.07.
    sinkhorn_alpha : float, optional
        The scaling factor for the Sinkhorn loss, by default 20.

    Attributes
    ----------
    recon_loss_weight : float
        The weight of the reconstruction loss in the final loss computation.
    sinkhorn_alpha : float
        The scaling factor applied to the Sinkhorn loss for optimal transport.
    encoder : nn.Sequential
        The neural network that encodes bag-of-words input into topic distribution.
    word_embeddings : nn.Parameter
        The word embeddings matrix, either pretrained or initialized randomly.
    topic_embeddings : nn.Parameter
        The matrix of learned topic embeddings.

    Methods
    -------
    get_beta():
        Computes the normalized topic-word distribution matrix.
    get_theta(x):
        Computes the topic distribution (theta) for the input BoW vector.
    forward(x):
        Executes the forward pass, returning the topic distribution, topic-word distribution, and the transport cost matrix.
    compute_loss(x):
        Computes the overall loss, combining reconstruction and Sinkhorn losses.
    """

    def __init__(
        self,
        dataset,
        n_topics: int = 50,
        encoder_dim: int = 128,
        dropout: float = 0.1,
        pretrained_WE=None,
        train_WE: bool = True,
        encoder_activation: callable = nn.ReLU(),
        embed_size: int = 256,
        recon_loss_weight=0.07,
        sinkhorn_alpha=20,
    ):
        """
        Initializes the Neural Topic Model.

        Parameters
        ----------
        dataset : Dataset
            A dataset object containing the BoW matrix as `dataset.bow`.
        n_topics : int, optional
            Number of topics to be learned, by default 50.
        encoder_dim : int, optional
            Hidden dimension size for the encoder, by default 128.
        dropout : float, optional
            Dropout rate for regularization in the encoder, by default 0.1.
        pretrained_WE : np.ndarray, optional
            Pretrained word embeddings (optional), by default None.
        train_WE : bool, optional
            Whether the word embeddings are trainable, by default True.
        encoder_activation : callable, optional
            Activation function for the encoder layers, by default nn.ReLU().
        embed_size : int, optional
            Size of the word embeddings, by default 256.
        recon_loss_weight : float, optional
            Weight of the reconstruction loss, by default 0.07.
        sinkhorn_alpha : float, optional
            Scaling factor for the Sinkhorn loss, by default 20.
        """
        super().__init__()

        vocab_size = dataset.bow.shape[1]

        self.recon_loss_weight = recon_loss_weight
        self.sinkhorn_alpha = sinkhorn_alpha

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, encoder_dim),
            encoder_activation,
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, n_topics),
            nn.BatchNorm1d(n_topics),
        )

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(
                torch.randn((vocab_size, embed_size)) * 1e-03
            )

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(
            torch.randn((n_topics, self.word_embeddings.shape[1])) * 1e-03
        )

    def get_beta(self):
        """
        Computes the normalized topic-word distribution matrix (beta) by taking the dot product
        of the normalized topic embeddings and word embeddings.

        Returns
        -------
        torch.Tensor
            The topic-word distribution matrix of shape (n_topics, vocab_size).
        """
        word_embedding_norm = F.normalize(self.word_embeddings)
        topic_embedding_norm = F.normalize(self.topic_embeddings)
        beta = torch.matmul(topic_embedding_norm, word_embedding_norm.T)
        return beta

    def get_theta(self, x):
        """
        Computes the document-topic distribution (theta) for a given bag-of-words input using the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the bag-of-words (BoW) data of shape (batch_size, vocab_size).

        Returns
        -------
        torch.Tensor
            The document-topic distribution of shape (batch_size, n_topics).
        """
        theta = self.encoder(x)
        theta = F.softmax(theta, dim=-1)
        return theta

    def forward(self, x):
        """
        Performs the forward pass of the model, which computes the document-topic distribution (theta),
        the topic-word distribution (beta), and the optimal transport distance matrix (M).

        Parameters
        ----------
        x : dict
            A dictionary containing the input bag-of-words tensor under the key "bow".

        Returns
        -------
        tuple
            A tuple containing:
            - theta (torch.Tensor): Document-topic distribution of shape (batch_size, n_topics).
            - beta (torch.Tensor): Topic-word distribution of shape (n_topics, vocab_size).
            - M (torch.Tensor): Distance matrix of shape (n_topics, vocab_size).
        """
        x = x["bow"]
        theta = self.get_theta(x)
        beta = self.get_beta()
        M = 1 - beta
        return theta, beta, M

    def compute_loss(self, x):
        """
        Computes the total loss for a given input by combining the reconstruction loss and the Sinkhorn loss.

        Parameters
        ----------
        x : dict
            A dictionary containing the input bag-of-words tensor under the key "bow".

        Returns
        -------
        torch.Tensor
            The total loss, averaged over the batch.
        """
        theta, beta, M = self.forward(x)
        sh_loss = sinkhorn_loss(
            M, theta.T, F.softmax(x["bow"], dim=-1).T, lambda_sh=self.sinkhorn_alpha
        )
        recon = F.softmax(torch.matmul(theta, beta), dim=-1)
        recon_loss = -(x["bow"] * recon.log()).sum(axis=1)

        loss = self.recon_loss_weight * recon_loss + sh_loss
        loss = loss.mean()
        return loss

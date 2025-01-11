import torch
import torch.nn as nn
import torch.nn.functional as F


class DETMBase(nn.Module):
    """
    An implementation of the Dynamic Embedded Topic Model (DETM).

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
        eta_hidden_size:int = 128,
        nlayers_eta:int=2,
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
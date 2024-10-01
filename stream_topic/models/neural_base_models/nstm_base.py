import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.sinkhorn_loss import sinkhorn_loss


class NSTMBase(nn.Module):
    """
    Neural Topic Model via Optimal Transport. ICLR 2021

    He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine.
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
        word_embedding_norm = F.normalize(self.word_embeddings)
        topic_embedding_norm = F.normalize(self.topic_embeddings)
        beta = torch.matmul(topic_embedding_norm, word_embedding_norm.T)
        return beta

    def get_theta(self, x):
        theta = self.encoder(x)
        theta = F.softmax(theta, dim=-1)
        return theta

    def forward(self, x):
        x = x["bow"]
        theta = self.get_theta(x)
        beta = self.get_beta()
        M = 1 - beta
        return theta, beta, M

    def compute_loss(self, x):
        theta, beta, M = self.forward(x)
        sh_loss = sinkhorn_loss(
            M, theta.T, F.softmax(x["bow"], dim=-1).T, lambda_sh=self.sinkhorn_alpha
        )
        recon = F.softmax(torch.matmul(theta, beta), dim=-1)
        recon_loss = -(x["bow"] * recon.log()).sum(axis=1)

        loss = self.recon_loss_weight * recon_loss + sh_loss
        loss = loss.mean()
        return loss

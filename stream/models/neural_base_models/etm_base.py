import torch
import torch.nn as nn
import torch.nn.functional as F


class ETMBase(nn.Module):
    def __init__(
        self,
        dataset,
        embed_size=128,
        n_topics=10,
        en_units=256,
        dropout=0.0,
        pretrained_WE=None,
        train_WE=True,
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
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc21 = nn.Linear(en_units, n_topics)
        self.fc22 = nn.Linear(en_units, n_topics)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = self.encoder1(x)
        return self.fc21(e1), self.fc22(e1)

    def get_theta(self, x, only_theta=False):
        norm_x = x / (x.sum(1, keepdim=True) + 1e-6)
        mu, logvar = self.encode(norm_x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        if only_theta:
            return theta
        else:
            return theta, mu, logvar

    def get_beta(self):
        beta = F.softmax(
            torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1
        )
        return beta

    def forward(self, x):
        x = x["bow"]
        theta, mu, logvar = self.get_theta(x)
        beta = self.get_beta()
        recon_x = torch.matmul(theta, beta)
        return recon_x, mu, logvar

    def compute_loss(self, x):
        recon_x, mu, logvar = self.forward(x)
        x = x["bow"]
        loss = self.loss_function(x, recon_x, mu, logvar)
        return loss * 1e-02

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)
        loss = (recon_loss + KLD).mean()
        return loss

    def get_complete_theta_mat(self, x):
        theta, mu, logvar = self.get_theta(x)
        return theta

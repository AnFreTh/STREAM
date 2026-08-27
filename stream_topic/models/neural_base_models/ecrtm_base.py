import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ECR(nn.Module):
    """Embedding Clustering Regularization using optimal transport."""
    
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):
        device = M.device
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        u = (torch.ones_like(a) / a.size()[0]).to(device)

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1                     # Python float — keeps the `while err > stopThr`
        cpt = 0                     # compare on-host so it doesn't sync the GPU
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                # `.item()` here syncs ONLY every 50 iters (when err actually
                # updates), not every iter via the tensor->bool convert in the
                # while condition. Value passed to the compare is exact.
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf')).item()

        transp = u * (K * v.T)
        loss_ECR = torch.sum(transp * M) * self.weight_loss_ECR
        return loss_ECR


class ECRTMBase(nn.Module):
    """
    ECRTM: Effective Neural Topic Modeling with Embedding Clustering Regularization.
    
    Reference: Xiaobao Wu et al. ICML 2023
    """
    
    def __init__(
        self,
        dataset,
        n_topics=50,
        encoder_dim=200,
        dropout=0.0,
        embed_size=200,
        beta_temp=0.2,
        weight_loss_ECR=250.0,  # official ECRTM default (was 100.0)
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=1000,
        pretrained_WE=None,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.vocab_size = dataset.bow.shape[1]
        self.beta_temp = beta_temp

        # Prior parameters
        self.a = 1 * np.ones((1, n_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / n_topics))).T + (1.0 / (n_topics * n_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False
        # var2 is a fixed prior; cache its log once so forward() doesn't recompute
        # self.var2.log() every step. Non-trainable buffer; bit-identical.
        self.register_buffer("_log_var2", self.var2.detach().log())

        # Encoder
        self.fc11 = nn.Linear(self.vocab_size, encoder_dim)
        self.fc12 = nn.Linear(encoder_dim, encoder_dim)
        self.fc21 = nn.Linear(encoder_dim, n_topics)
        self.fc22 = nn.Linear(encoder_dim, n_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        # Batch normalization
        self.mean_bn = nn.BatchNorm1d(n_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(n_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(self.vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        # Embeddings
        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(self.vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((n_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        # ECR module
        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        return theta, mu, logvar

    def get_theta(self, x, only_theta=False):
        if isinstance(x, dict):
            input = x["bow"]
        else:
            input = x
        theta, mu, logvar = self.encode(input)
        if only_theta:
            return theta
        return theta, mu, logvar

    def forward(self, x):
        if isinstance(x, dict):
            input = x["bow"]
        else:
            input = x

        from ...utils.fast_mode import fast_mode
        fast = fast_mode()

        theta, mu, logvar = self.encode(input)

        if fast:
            # Fp-equivalent: compute the (topic x word) distance ONCE and share it
            # between the beta softmax and the ECR cost. Same forward value; grads
            # reassociate at ~1e-7 relative (float32 epsilon). ALSO fuse softmax+log
            # into log_softmax to avoid one exp+log round-trip.
            dist = self.pairwise_euclidean_distance(
                self.topic_embeddings, self.word_embeddings
            )
            beta = F.softmax(-dist / self.beta_temp, dim=0)
            log_recon = F.log_softmax(
                self.decoder_bn(torch.matmul(theta, beta)), dim=-1
            )
            recon_loss = -(input * log_recon).sum(axis=1).mean()
            cost = dist
        else:
            beta = self.get_beta()
            recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
            recon_loss = -(input * recon.log()).sum(axis=1).mean()
            cost = self.pairwise_euclidean_distance(
                self.topic_embeddings, self.word_embeddings
            )

        # KL divergence
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self._log_var2 - logvar  # Tier A cached buffer
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.n_topics)
        KLD = KLD.mean()

        loss_TM = recon_loss + KLD
        loss_ECR = self.ECR(cost)

        loss = loss_TM + loss_ECR
        return loss

    def compute_loss(self, x):
        return self.forward(x)

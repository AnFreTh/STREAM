import torch
from torch import nn
import torch.nn.functional as F


def pairwise_euclidean_distance(x, y):
    """Compute pairwise Euclidean distance between two tensors."""
    x_norm = (x**2).sum(1, keepdim=True)
    y_norm = (y**2).sum(1)
    cost = x_norm + y_norm - 2 * torch.mm(x, y.t())
    return cost.clamp(min=0)  # Numerical stability


class ETP(nn.Module):
    """Entropic Transport Plan module for optimal transport."""

    def __init__(
        self,
        sinkhorn_alpha,
        init_a_dist=None,
        init_b_dist=None,
        OT_max_iter=5000,
        stop_thr=0.5e-2,
    ):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stop_thr = stop_thr
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        M = pairwise_euclidean_distance(x, y)
        device = M.device

        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0], device=device) / M.shape[0]).unsqueeze(1)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1], device=device) / M.shape[1]).unsqueeze(1)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        # Adaptive threshold based on problem size
        if self.stop_thr is None:
            stop_thr = 1e-3 / max(M.shape[0], M.shape[1])
        else:
            stop_thr = self.stop_thr

        log_a = torch.log(a + 1e-30)
        log_b = torch.log(b + 1e-30)

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        log_K = -M * self.sinkhorn_alpha

        # Sinkhorn iterations in log-domain
        err = 1
        cpt = 0
        while err > self.stop_thr and cpt < self.OT_max_iter:
            # LOG-DOMAIN UPDATE: log(v) = log(b) - logsumexp(log(K^T) + log(u), dim=0)
            # This is equivalent to: v = b / (K^T @ u) but numerically stable
            log_Ku = log_K.T + log_u.T  # Shape: (m, n)
            log_v = log_b - torch.logsumexp(log_Ku, dim=1).unsqueeze(1)  # Shape: (m, 1)

            # LOG-DOMAIN UPDATE: log(u) = log(a) - logsumexp(log(K) + log(v^T), dim=1)
            # This is equivalent to: u = a / (K @ v) but numerically stable
            log_Kv = log_K + log_v.T  # Shape: (n, m)
            log_u = log_a - torch.logsumexp(log_Kv, dim=1).unsqueeze(1)  # Shape: (n, 1)

            cpt += 1
            if cpt % 50 == 1:
                # Absorb current scalings
                log_K = log_K + log_u + log_v.T
                # Reset scalings
                log_u = torch.zeros_like(log_a)
                log_v = torch.zeros_like(log_b)

                err = self.check_convergence(log_K, log_u, log_v, a, b)

        # Convert final results back to linear domain for compatibility
        u = torch.exp(log_u)  # Shape: (n, 1)
        v = torch.exp(log_v)  # Shape: (m, 1)
        K = torch.exp(log_K)  # Shape: (n, m)

        # Compute transport plan: P = diag(u) K diag(v)
        transp = u * (K * v.T)  # Shape: (n, m)

        # Compute transport cost: <P, M>
        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp

    def check_convergence(self, log_K, log_u, log_v, a, b):
        """
        Check convergence by verifying the marginal constraints using absolute error.
        The transport plan is P = diag(u) @ K @ diag(v)
        We need: P @ 1 = a and P^T @ 1 = b
        """
        with torch.no_grad():
            # Row constraint: (u ⊙ (K @ v)) should equal a
            # In log domain: log(u) + log(K @ v) should equal log(a)
            log_Kv = log_K + log_v.T  # Shape: (n, m)
            log_Kv_sum = torch.logsumexp(
                log_Kv, dim=1, keepdim=True
            )  # log(K @ v), Shape: (n, 1)
            log_row_sums = log_u + log_Kv_sum  # log(u ⊙ (K @ v)), Shape: (n, 1)
            row_sums = torch.exp(log_row_sums)  # Shape: (n, 1)

            # Column constraint: (v ⊙ (K^T @ u)) should equal b
            # In log domain: log(v) + log(K^T @ u) should equal log(b)
            log_Ku = log_K.T + log_u.T  # Shape: (m, n)
            log_Ku_sum = torch.logsumexp(
                log_Ku, dim=1, keepdim=True
            )  # log(K^T @ u), Shape: (m, 1)
            log_col_sums = log_v + log_Ku_sum  # log(v ⊙ (K^T @ u)), Shape: (m, 1)
            col_sums = torch.exp(log_col_sums)  # Shape: (m, 1)

            # Compute absolute errors
            row_err = torch.abs(row_sums - a).sum()

            col_err = torch.abs(col_sums - b).sum()

            return max(row_err.item(), col_err.item())


class FAStopicBase(nn.Module):
    """FASTopic: Fast Adaptive Sparse Topic Model base module."""

    def __init__(
        self,
        dataset,
        n_topics: int = 20,
        theta_temp: float = 1.0,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        normalize_embeddings: bool = False,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.normalize_embeddings = normalize_embeddings

        self.epsilon = 1e-12

        # Get dataset dimensions
        self.vocab_size = dataset.bow.shape[1]
        self.embed_size = dataset.embeddings.shape[1]

        # Store training embeddings for theta computation
        self.register_buffer(
            "train_doc_embeddings",
            torch.tensor(dataset.embeddings, dtype=torch.float32),
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(
        self, _fitted: bool = False, pre_vocab: list = None, vocab: list = None
    ):
        topic_embeddings = F.normalize(
            nn.init.trunc_normal_(torch.empty((self.n_topics, self.embed_size)))
        )
        topic_weights = (torch.ones(self.n_topics) / self.n_topics).unsqueeze(1)

        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)

        word_embeddings = F.normalize(
            nn.init.trunc_normal_(torch.empty(self.vocab_size, self.embed_size))
        )
        word_weights = (torch.ones(self.vocab_size) / self.vocab_size).unsqueeze(1)

        self.word_embeddings = nn.Parameter(word_embeddings)
        self.word_weights = nn.Parameter(word_weights)

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    def get_beta(self):
        with torch.no_grad():
            _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
            beta = transp_TW * transp_TW.shape[0]
            return beta

    def get_theta(self, x, only_theta=False):
        """Compute document-topic distribution."""
        with torch.no_grad():
            doc_embeddings = x["embedding"]
            topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
            train_doc_embeddings = self.train_doc_embeddings.to(doc_embeddings.device)

            dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
            train_dist = pairwise_euclidean_distance(
                train_doc_embeddings, topic_embeddings
            )

            exp_dist = torch.exp(-dist / self.theta_temp)
            exp_train_dist = torch.exp(-train_dist / self.theta_temp)

            theta = exp_dist / (exp_train_dist.sum(0))
            theta = theta / theta.sum(1, keepdim=True)

            return theta

    def forward(self, x):
        """Forward pass."""
        doc_embeddings = x["embedding"]
        train_bow = x["bow"]

        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        recon = torch.matmul(theta, beta)
        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

        loss = loss_DSR + loss_ETP

        return loss

    def compute_loss(self, x):
        """Compute loss for training."""
        return self.forward(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple MLP block with residual connection."""

    def __init__(self, in_features, out_features, activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        if self.in_features == self.out_features:
            out = self.fc2(self.activation(self.fc1(x)))
            return self.activation(self.bn(x + out))
        else:
            x = self.fc1(x)
            out = self.fc2(self.activation(x))
            return self.activation(self.bn(x + out))


class SawETMBase(nn.Module):
    """
    SawETM: Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network.

    Reference: Zhibin Duan et al. ICML 2021
    """

    def __init__(
        self,
        dataset,
        n_topics=None,
        n_topics_list=None,
        embed_size=100,
        hidden_size=256,
        pretrained_WE=None,
    ):
        super().__init__()

        # Handle both n_topics and n_topics_list for compatibility
        if n_topics_list is None:
            if n_topics is not None:
                n_topics_list = [
                    n_topics,
                    max(10, n_topics // 2),
                    max(5, n_topics // 4),
                ]
            else:
                n_topics_list = [50, 36, 12]

        self.vocab_size = dataset.bow.shape[1]

        # Register constants as buffers (will move with model to device)
        self.register_buffer("gam_prior", torch.tensor(1.0, dtype=torch.float))
        self.register_buffer("real_min", torch.tensor(1e-30, dtype=torch.float))
        self.register_buffer("theta_max", torch.tensor(1000.0, dtype=torch.float))
        self.register_buffer("wei_shape_min", torch.tensor(1e-1, dtype=torch.float))
        self.register_buffer("wei_shape_max", torch.tensor(100.0, dtype=torch.float))

        # Hyperparameters (reverse order for bottom-up)
        self.num_topics_list = n_topics_list[::-1]
        self.num_hiddens_list = [hidden_size] * len(self.num_topics_list)
        self.num_layers = len(self.num_topics_list)

        # Word embeddings
        if pretrained_WE is not None:
            self.rho = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.rho = nn.Parameter(
                torch.empty(self.vocab_size, embed_size).normal_(std=0.02)
            )

        # Topic embeddings for different layers
        self.alpha = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.num_topics_list[n], embed_size).normal_(std=0.02)
                )
                for n in range(self.num_layers)
            ]
        )

        # Deterministic encoder
        self.h_encoder = nn.ModuleList()
        for n in range(self.num_layers):
            if n == 0:
                self.h_encoder.append(
                    ResBlock(self.vocab_size, self.num_hiddens_list[n])
                )
            else:
                self.h_encoder.append(
                    ResBlock(self.num_hiddens_list[n - 1], self.num_hiddens_list[n])
                )

        # Variational encoder
        self.q_theta = nn.ModuleList()
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.q_theta.append(
                    nn.Linear(self.num_hiddens_list[n], 2 * self.num_topics_list[n])
                )
            else:
                self.q_theta.append(
                    nn.Linear(
                        self.num_hiddens_list[n] + self.num_topics_list[n],
                        2 * self.num_topics_list[n],
                    )
                )

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min))

    def reparameterize(self, shape, scale, sample_num=50):
        """Reparameterization for Weibull distribution."""
        shape = shape.unsqueeze(0).repeat(sample_num, 1, 1)
        scale = scale.unsqueeze(0).repeat(sample_num, 1, 1)
        eps = torch.rand_like(shape, dtype=torch.float)
        samples = scale * torch.pow(-self.log_max(1 - eps), 1 / shape)
        return torch.clamp(samples.mean(0), self.real_min.item(), self.theta_max.item())

    def kl_weibull_gamma(self, wei_shape, wei_scale, gam_shape, gam_scale):
        """KL divergence between Weibull and Gamma distributions."""
        euler_mascheroni_c = torch.tensor(
            0.5772, dtype=torch.float, device=wei_shape.device
        )
        t1 = torch.log(wei_shape) + torch.lgamma(gam_shape)
        t2 = -gam_shape * torch.log(wei_scale * gam_scale)
        t3 = euler_mascheroni_c * (gam_shape / wei_shape - 1) - 1
        t4 = gam_scale * wei_scale * torch.exp(torch.lgamma(1 + 1 / wei_shape))
        return (t1 + t2 + t3 + t4).sum(1).mean()

    def get_nll(self, x, x_reconstruct):
        """Negative Poisson log-likelihood."""
        log_likelihood = (
            self.log_max(x_reconstruct) * x - torch.lgamma(1.0 + x) - x_reconstruct
        )
        return -torch.sum(log_likelihood, dim=1, keepdim=False).mean()

    def get_phis(self):
        """Factor loading matrices via sawtooth connection."""
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                phi = torch.softmax(
                    torch.mm(self.rho, self.alpha[n].transpose(0, 1)), dim=0
                )
            else:
                phi = torch.softmax(
                    torch.mm(self.alpha[n - 1].detach(), self.alpha[n].transpose(0, 1)),
                    dim=0,
                )
            phis.append(phi)
        return phis

    def get_beta(self):
        """Get topic-word distributions for all layers."""
        beta_list = []
        phis = self.get_phis()
        last_beta = None

        for layer_id, phi in enumerate(phis):
            if layer_id == 0:
                last_beta = phi.T
            else:
                last_beta = torch.matmul(phi.T, last_beta)
            beta_list.append(last_beta)

        return beta_list[::-1]

    def get_theta(self, x, only_theta=False):
        """Get document-topic distributions."""
        if isinstance(x, dict):
            input_x = x["bow"]
        else:
            input_x = x

        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](input_x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        phis = self.get_phis()

        ks = []
        lambs = []
        thetas = []
        phi_by_theta_list = []

        for n in range(self.num_layers - 1, -1, -1):
            if n == self.num_layers - 1:
                joint_feat = hidden_feats[n]
            else:
                joint_feat = torch.cat((hidden_feats[n], phi_by_theta_list[0]), dim=1)

            k, lamb = torch.chunk(F.softplus(self.q_theta[n](joint_feat)), 2, dim=1)
            k = torch.clamp(k, self.wei_shape_min.item(), self.wei_shape_max.item())
            lamb = torch.clamp(lamb, self.real_min.item())

            if self.training:
                lamb = lamb / torch.exp(torch.lgamma(1 + 1 / k))
                theta = (
                    self.reparameterize(k, lamb, sample_num=3)
                    if n == 0
                    else self.reparameterize(k, lamb)
                )
            else:
                theta = torch.min(lamb, self.theta_max)

            phi_by_theta = torch.mm(theta, phis[n].t())
            phi_by_theta_list.insert(0, phi_by_theta)
            thetas.insert(0, theta)
            lambs.insert(0, lamb)
            ks.insert(0, k)

        if only_theta:
            return thetas[::-1]

        # Always return all 4 values for consistency
        return ks, lambs, phi_by_theta_list, thetas

    def forward(self, x):
        """Forward pass."""
        if isinstance(x, dict):
            input_x = x["bow"]
        else:
            input_x = x

        ks, lambs, phi_by_theta_list, thetas = self.get_theta(x)

        nll = self.get_nll(input_x, phi_by_theta_list[0])

        kl_loss = []
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                kl_loss.append(
                    self.kl_weibull_gamma(
                        ks[n], lambs[n], self.gam_prior, self.gam_prior
                    )
                )
            else:
                kl_loss.append(
                    self.kl_weibull_gamma(
                        ks[n], lambs[n], phi_by_theta_list[n + 1], self.gam_prior
                    )
                )

        nelbo = nll + sum(kl_loss)
        return nelbo

    def compute_loss(self, x):
        return self.forward(x)

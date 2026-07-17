import torch
import torch.nn as nn
import torch.nn.functional as F
from .sawetm_base import SawETMBase


class PoincareBall:
    """Poincare ball manifold for hyperbolic embeddings."""
    
    def __init__(self):
        self.name = 'PoincareBall'
        self.eps = 1e-5
        self.min_norm = 1e-15
        self.max_norm = 1e15
        
    def clip(self, x):
        return torch.clamp(x, min=self.min_norm, max=self.max_norm)
    
    def truncate_c(self, c):
        return torch.clamp(c, min=-1e5, max=-1e-5)
    
    def proj(self, x, c):
        c = self.truncate_c(c)
        x_norm = self.clip(x.norm(dim=-1, keepdim=True, p=2))
        max_norm = (1 - self.eps) / c.abs().sqrt()
        cond = x_norm > max_norm
        projected = x / x_norm * max_norm
        return torch.where(cond, projected, x)
    
    def expmap0(self, v, c):
        c = self.truncate_c(c)
        v_norm = self.clip(v.norm(p=2, dim=-1, keepdim=True))
        gamma = self._tanc(v_norm, c) * v / v_norm
        return gamma
    
    def dist(self, x, y, c):
        c = self.truncate_c(c)
        return 2.0 * self._artanc(self._mobius_add(-x, y, c).norm(p=2, dim=-1), c)
    
    def _tanc(self, x, c):
        """Unified tangent function."""
        return 1 / c.abs().sqrt() * torch.tanh(torch.clamp(x * c.abs().sqrt(), min=-15, max=15))
    
    def _artanc(self, x, c):
        """Unified inverse tangent function."""
        return 1 / c.abs().sqrt() * torch.atanh(torch.clamp(x * c.abs().sqrt(), min=-1 + 1e-7, max=1 - 1e-7))
    
    def _mobius_add(self, x, y, c):
        c = self.truncate_c(c)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2
        return num / self.clip(denom)


class HyperMinerBase(SawETMBase):
    """
    HyperMiner: Topic Taxonomy Mining with Hyperbolic Embedding.
    
    Reference: Yishi Xu et al. NeurIPS 2022
    """
    
    def __init__(
        self,
        dataset,
        n_topics=None,
        n_topics_list=None,
        embed_size=50,
        hidden_size=300,
        pretrained_WE=None,
        manifold="PoincareBall",
        clip_r=None,
        curvature=-0.01,
    ):
        super().__init__(
            dataset=dataset,
            n_topics=n_topics,
            n_topics_list=n_topics_list,
            embed_size=embed_size,
            hidden_size=hidden_size,
            pretrained_WE=pretrained_WE,
        )
        
        self.manifold = PoincareBall()
        
        if curvature is not None:
            self.register_buffer('curvature', torch.tensor([curvature]))
        else:
            self.curvature = nn.Parameter(torch.tensor([-1.0]))
        
        self.clip_r = clip_r
    
    def feat_clip(self, x):
        if self.clip_r is None:
            return x
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        cond = x_norm > self.clip_r
        projected = x / x_norm * self.clip_r
        return torch.where(cond, projected, x)
    
    def get_phis(self):
        """Factor loading matrices using hyperbolic distance."""
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                hyp_rho = self.manifold.proj(
                    self.manifold.expmap0(self.rho, self.curvature), self.curvature)
                hyp_alpha = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_rho.unsqueeze(1), hyp_alpha.unsqueeze(0), self.curvature), dim=0)
            else:
                hyp_alpha1 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n - 1], self.curvature), self.curvature)
                hyp_alpha2 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_alpha1.unsqueeze(1).detach(), hyp_alpha2.unsqueeze(0), self.curvature), dim=0)
            phis.append(phi)
        return phis

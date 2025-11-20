import math
import torch
import torch.nn as nn


class LDAHead(nn.Module):
    """Fixed-mean LDA classifier with spherical covariance and trainable priors."""

    def __init__(self, C, D):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        self.C = C
        self.D = D
        dtype = torch.get_default_dtype()
        mu = self._regular_simplex_vertices(C, D, dtype=dtype)
        pairwise_dist = math.sqrt(2.0 * C / (C - 1))
        scale = 6.0 / pairwise_dist
        mu = mu * scale
        self.register_buffer('mu', mu)
        self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @staticmethod
    def _regular_simplex_vertices(C, D, dtype):
        """Construct vertices of a regular simplex centered at the origin."""
        eye = torch.eye(C, dtype=dtype)
        centered = eye - eye.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh.transpose(-2, -1)[:, :C - 1]
        simplex = centered @ basis
        simplex = simplex / simplex.norm(dim=1, keepdim=True)
        if D > C - 1:
            pad = torch.zeros(C, D - (C - 1), dtype=dtype)
            simplex = torch.cat([simplex, pad], dim=1)
        return simplex

    @property
    def cov_diag(self):
        """Return diagonal of the spherical covariance matrix."""
        return torch.exp(self.log_cov).repeat(self.D)

    def forward(self, z):
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        var = torch.exp(self.log_cov).to(z.dtype)
        log_det = self.D * self.log_cov.to(z.dtype)
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

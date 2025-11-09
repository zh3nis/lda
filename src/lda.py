import torch
import torch.nn as nn

def classwise_mean_cov(z, y, C, eps=1e-5):
    B, D = z.shape

    # Build one-hot encodings so we can aggregate per-class statistics.
    y_onehot = torch.zeros(B, C, device=z.device, dtype=z.dtype)
    y_onehot.scatter_(1, y.view(-1, 1), 1.0)

    class_counts = y_onehot.sum(dim=0) + eps
    class_prior = (class_counts / class_counts.sum()).to(z.dtype)

    # Mean of embeddings for each class.
    class_sums = y_onehot.t() @ z
    class_mean = class_sums / class_counts.unsqueeze(1)

    # Covariance of each class (center, mask unrelated examples, then accumulate).
    centered = z.unsqueeze(1) - class_mean.unsqueeze(0)
    centered_masked = centered * y_onehot.unsqueeze(-1)
    class_cov = torch.einsum('bcd,bce->cde', centered_masked, centered) / class_counts.view(C, 1, 1)

    # Pooled covariance weighted by class prior.
    pooled_cov = (class_prior.view(C, 1, 1) * class_cov).sum(dim=0)
    pooled_cov = 0.5 * (pooled_cov + pooled_cov.transpose(0, 1))
    pooled_cov = pooled_cov + torch.eye(D, device=z.device, dtype=z.dtype) * eps

    return class_mean, pooled_cov, class_prior

class LDAHead(nn.Module):
    """Linear Discriminant Analysis classifier with running statistics."""

    def __init__(self, C, D, ema=0.9):
        super().__init__()
        self.C = C
        self.D = D
        self.m = ema
        self.register_buffer('mu_ema', torch.zeros(C, D))
        self.register_buffer('cov_ema', torch.eye(D))
        self.register_buffer('prior_ema', torch.full((C,), 1.0 / C))
        self.register_buffer('init', torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _update(self, mu, cov, prior):
        if not bool(self.init.item()):
            self.mu_ema.copy_(mu)
            self.cov_ema.copy_(cov)
            self.prior_ema.copy_(prior)
            self.init.fill_(1)
            return

        momentum = self.m
        self.mu_ema.mul_(momentum).add_(mu, alpha=1 - momentum)
        self.cov_ema.mul_(momentum).add_(cov, alpha=1 - momentum)
        self.prior_ema.mul_(momentum).add_(prior, alpha=1 - momentum)

    def _precision_from_covariance(self, cov, eps=1e-6, max_tries=5):
        # Symmetrize to avoid numerical drift before factorization.
        cov = 0.5 * (cov + cov.transpose(0, 1))
        try:
            chol = torch.linalg.cholesky(cov)
            return torch.cholesky_inverse(chol)
        except RuntimeError:
            eye = torch.eye(self.D, device=cov.device, dtype=cov.dtype)
            jitter = eps
            for _ in range(max_tries):
                cov_jitter = cov + jitter * eye
                try:
                    chol = torch.linalg.cholesky(cov_jitter)
                    return torch.cholesky_inverse(chol)
                except RuntimeError:
                    jitter *= 10.0
        # Fallback to pseudo-inverse if Cholesky keeps failing.
        return torch.linalg.pinv(cov)

    def forward(self, z, y=None):
        if self.training and (y is not None):
            mu, cov, prior = classwise_mean_cov(z, y, self.C)
            self._update(mu, cov, prior)
        else:
            mu, cov, prior = self.mu_ema, self.cov_ema, self.prior_ema
        
        precision = self._precision_from_covariance(cov)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        proj = torch.matmul(diff, precision)
        m2 = (proj * diff).sum(-1)
        return prior.log().unsqueeze(0) - 0.5*m2  # logits


class TrainableLDAHead(nn.Module):
    """LDA-style classifier where the statistics are learned end-to-end."""

    def __init__(self, C, D, eps=1e-4):
        super().__init__()
        self.C = C
        self.D = D
        self.eps = eps
        self.mu = nn.Parameter(torch.zeros(C, D))
        self.prior_logits = nn.Parameter(torch.zeros(C))
        self.precision_factor = nn.Parameter(torch.eye(D))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mu, mean=0.0, std=1.0)
        nn.init.zeros_(self.prior_logits)
        nn.init.eye_(self.precision_factor)

    def _precision(self):
        # Enforce positive-definiteness via AA^T + eps I.
        factor = self.precision_factor
        precision = factor @ factor.transpose(0, 1)
        precision = 0.5 * (precision + precision.transpose(0, 1))
        eye = torch.eye(self.D, device=factor.device, dtype=factor.dtype)
        return precision + self.eps * eye

    def forward(self, z, y=None):
        prior = torch.softmax(self.prior_logits, dim=0)
        precision = self._precision()
        diff = z.unsqueeze(1) - self.mu.unsqueeze(0)
        proj = torch.matmul(diff, precision)
        m2 = (proj * diff).sum(-1)
        return prior.log().unsqueeze(0) - 0.5 * m2

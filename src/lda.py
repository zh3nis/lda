import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def joint_nll(self, z, y, update_stats=False):
        """
        Supervised joint negative log-likelihood using the current LDA statistics.
        Mirrors LDAHeadMLE.joint_nll so callers can share the same loss code.
        """
        if y is None:
            raise ValueError("joint_nll requires target labels y.")

        if update_stats and self.training:
            mu, cov, prior = classwise_mean_cov(z, y, self.C)
            self._update(mu, cov, prior)
        else:
            mu, cov, prior = self.mu_ema, self.cov_ema, self.prior_ema

        precision = self._precision_from_covariance(cov)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        proj = torch.matmul(diff, precision)
        m2 = (proj * diff).sum(-1)

        sign, logdet = torch.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("Covariance matrix must be positive definite for joint_nll.")

        normalizer = self.D * torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype)) + logdet
        logpdf = -0.5 * (normalizer + m2)

        log_prior = prior.clamp_min(1e-12).log()
        ll = log_prior[y] + logpdf.gather(1, y.view(-1, 1)).squeeze(1)
        return -ll.mean()


class LDAHeadMLE(nn.Module):
    """
    Trainable LDA head (shared covariance across classes).
    Parameters:
      - mu:            [C, D] class means
      - L_raw:         [D, D] unconstrained; we form L = tril(L_raw) with positive diag
      - logits_prior:  [C]    unnormalized class-prior logits
    Forward returns logits: log pi_c - 0.5 * (x - mu_c)^T Σ^{-1} (x - mu_c)
    (log|Σ| term is class-independent in LDA and omitted).
    """

    def __init__(self, C: int, D: int, init_scale: float = 0.1, jitter: float = 1e-3):
        super().__init__()
        self.C, self.D = C, D
        self.jitter = jitter

        # Learnable class means
        self.mu = nn.Parameter(torch.randn(C, D) * init_scale)

        # Learnable pooled covariance via Cholesky factor L (lower-triangular)
        # Start near identity so optimization is stable.
        L0 = torch.eye(D)
        self.L_raw = nn.Parameter(L0.clone())

        # Learnable class priors (as logits)
        self.logits_prior = nn.Parameter(torch.zeros(C))

        # One-time init flag
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def _chol_factor(self) -> torch.Tensor:
        """
        Build a valid lower-triangular Cholesky factor L from L_raw:
        - take tril
        - make diag strictly positive via softplus + jitter
        """
        L = torch.tril(self.L_raw)
        d = torch.diagonal(L, 0, -2, -1)
        d_pos = F.softplus(d) + self.jitter
        L = L - torch.diag_embed(d) + torch.diag_embed(d_pos)
        return L  # [D, D]

    def logpdf_per_class(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log N(z | mu_c, Σ) for every class.
        Returns: [B, C]
        """
        B, D = z.shape
        L = self._chol_factor()                       # [D, D]
        mu = self.mu                                  # [C, D]

        # Mahalanobis term via triangular solve: u = L^{-1}(z - mu_c)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)       # [B, C, D]
        rhs  = diff.reshape(B * self.C, D).T          # [D, B*C]
        u    = torch.linalg.solve_triangular(L, rhs, upper=False)  # [D, B*C]
        m2   = (u * u).sum(dim=0).reshape(B, self.C)  # [B, C]

        # log|Σ| from Cholesky: |Σ| = (prod diag(L))^2
        logdet_Sigma = 2.0 * torch.log(torch.diagonal(L)).sum()    # scalar

        # log N = -0.5 * (D log 2π + log|Σ| + m2)
        return -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
                       + logdet_Sigma + m2)

    def joint_nll(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Supervised joint negative log-likelihood:
        NLL = - E[ log π_{y} + log N(z | μ_y, Σ) ]
        z: [B, D], y: [B] int64
        """
        logpdf = self.logpdf_per_class(z)                         # [B, C]
        log_prior = F.log_softmax(self.logits_prior, dim=0)       # [C]
        # pick per-sample true-class terms
        ll = log_prior[y] + logpdf.gather(1, y.view(-1,1)).squeeze(1)  # [B]
        return -ll.mean()
    
    def forward(self, z: torch.Tensor, y: torch.Tensor = None, init_if_needed: bool = True) -> torch.Tensor:
        # Note: for NLL training you won't use these logits in the loss.
        B, D = z.shape
        L = self._chol_factor()
        mu = self.mu
        log_prior = F.log_softmax(self.logits_prior, dim=0)

        diff = z.unsqueeze(1) - mu.unsqueeze(0)              # [B, C, D]
        rhs  = diff.reshape(B * self.C, D).T
        u    = torch.linalg.solve_triangular(L, rhs, upper=False)
        m2   = (u * u).sum(dim=0).reshape(B, self.C)

        logits = log_prior.unsqueeze(0) - 0.5 * m2
        return logits

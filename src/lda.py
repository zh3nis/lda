import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd


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


class FisherLDAHead(nn.Module):
    """
    Fixed-mean spherical LDA head trained with Fisher's discriminant criterion.

    Forward returns the negative Fisher ratio for a labeled batch. Use `logits`
    for evaluation-time class logits identical to LDAHead.
    """

    def __init__(self, C, D, fisher_eps=1e-8, prior_strength=0.5):
        super().__init__()
        if D < C - 1:
            raise ValueError(f"D must be at least C-1 to embed the simplex (got C={C}, D={D}).")
        if not (0.0 <= prior_strength <= 1.0):
            raise ValueError(f"prior_strength must be in [0,1] (got {prior_strength}).")
        self.C = C
        self.D = D
        self.fisher_eps = fisher_eps
        self.prior_strength = prior_strength
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
        return torch.exp(self.log_cov).repeat(self.D)

    def logits(self, z):
        """Compute LDA logits for evaluation (same form as LDAHead.forward)."""
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        var = torch.exp(self.log_cov).to(z.dtype)
        log_det = self.D * self.log_cov.to(z.dtype)
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)

    def forward(self, z, y):
        """
        Return negative Fisher ratio for the batch.

        z: (B, D) embeddings
        y: (B,) labels in [0, C)
        """
        if y is None:
            raise ValueError("Labels y must be provided to compute the Fisher criterion.")
        if y.dim() != 1:
            raise ValueError(f"Expected y to be 1D (got shape {tuple(y.shape)}).")
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched batch sizes: z has {z.shape[0]}, y has {y.shape[0]}.")

        dtype = z.dtype
        device = z.device
        mu = self.mu.to(device=device, dtype=dtype)
        var = torch.exp(self.log_cov.to(device=device, dtype=dtype))

        # Within-class scatter measured by Mahalanobis distance to fixed class means.
        diff = z - mu[y]
        within = (diff.pow(2).sum(dim=1) / var).mean()

        # Between-class scatter of the fixed means around their prior-weighted centroid.
        counts = torch.bincount(y, minlength=self.C).to(device=device, dtype=dtype)
        total = counts.sum().clamp_min(1.0)
        data_pi = counts / total
        learned_pi = torch.softmax(self.prior_logits.to(device=device, dtype=dtype), dim=0)
        pi = self.prior_strength * learned_pi + (1.0 - self.prior_strength) * data_pi

        overall_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
        centered = mu - overall_mu
        between_per_class = centered.pow(2).sum(dim=1) / var
        between = (pi * between_per_class).sum()

        fisher_ratio = between / (within + self.fisher_eps)
        return -fisher_ratio



class TrainableLDAHead(nn.Module):
    """LDA classifier with trainable class means, spherical covariance, and trainable priors."""

    def __init__(self, C, D):
        super().__init__()
        self.C = C
        self.D = D
        dtype = torch.get_default_dtype()
        # Start class means from a normal distribution instead of a fixed simplex layout.
        self.mu = nn.Parameter(torch.randn(C, D, dtype=dtype) * 6.0 / math.sqrt(2*D))
        #self.mu = nn.Parameter(torch.zeros(C, D, dtype=dtype))
        self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @property
    def cov_diag(self):
        return torch.exp(self.log_cov).repeat(self.D)

    def forward(self, z):
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        log_cov = self.log_cov.to(z.dtype)
        var = torch.exp(log_cov)
        log_det = self.D * log_cov
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 / var + log_det)


class TrainableMDAHead(nn.Module):
    """MDA classifier with per-class Gaussian mixtures and shared spherical covariance."""

    def __init__(self, C, D, K=2):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be positive (got K={K}).")
        self.C = C
        self.D = D
        self.K = K
        dtype = torch.get_default_dtype()
        scale = 6.0 / math.sqrt(2 * D)
        self.mu = nn.Parameter(torch.randn(C, K, D, dtype=dtype) * scale)
        self.log_cov = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))
        self.mix_logits = nn.Parameter(torch.zeros(C, K, dtype=dtype))

    @property
    def cov_diag(self):
        return torch.exp(self.log_cov).repeat(self.D)

    def forward(self, z):
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1).unsqueeze(1) - mu.unsqueeze(0)
        m2 = (diff * diff).sum(-1)
        log_cov = self.log_cov.to(z.dtype)
        var = torch.exp(log_cov)
        log_det = self.D * log_cov
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        log_mix = torch.log_softmax(self.mix_logits, dim=1)
        log_comp = log_mix.unsqueeze(0) - 0.5 * (m2 / var + log_det)
        log_like = torch.logsumexp(log_comp, dim=2)
        return log_prior.unsqueeze(0) + log_like


class DiagTrainableLDAHead(nn.Module):
    """LDA classifier with trainable means, diagonal covariance, and trainable priors."""

    def __init__(self, C, D):
        super().__init__()
        self.C = C
        self.D = D
        dtype = torch.get_default_dtype()
        # Start class means from a normal distribution instead of a fixed simplex layout.
        self.mu = nn.Parameter(torch.randn(C, D, dtype=dtype) * 6.0 / math.sqrt(2*D))
        self.log_cov_diag = nn.Parameter(torch.zeros(D, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    @property
    def cov_diag(self):
        return torch.exp(self.log_cov_diag)

    def forward(self, z):
        mu = self.mu.to(z.dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        log_cov_diag = self.log_cov_diag.to(z.dtype)
        var = torch.exp(log_cov_diag)
        m2 = (diff * diff / var).sum(-1)
        log_det = log_cov_diag.sum()
        log_prior = torch.log_softmax(self.prior_logits, dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


class FullCovLDAHead(nn.Module):
    """LDA classifier with trainable means, full shared covariance, and trainable priors."""

    def __init__(self, C, D, min_scale=1e-4):
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be positive (got D={D}).")
        self.C = C
        self.D = D
        self.min_scale = min_scale
        dtype = torch.get_default_dtype()
        self.mu = nn.Parameter(torch.zeros(C, D, dtype=dtype))
        #self.mu = nn.Parameter(torch.randn(C, D, dtype=dtype) * 6.0 / math.sqrt(2*D))
        self.raw_tril = nn.Parameter(torch.zeros(D, D, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    def _get_cholesky(self, dtype, device):
        raw = torch.tril(self.raw_tril.to(device=device, dtype=dtype))
        diag = torch.diagonal(raw, 0)
        safe_diag = F.softplus(diag) + self.min_scale
        L = raw - torch.diag(diag) + torch.diag(safe_diag)
        return L

    @property
    def covariance(self):
        """Full covariance matrix Sigma = L L^T."""
        L = self._get_cholesky(self.raw_tril.dtype, self.raw_tril.device)
        return L @ L.transpose(-2, -1)

    def forward(self, z):
        dtype = z.dtype
        device = z.device
        mu = self.mu.to(device=device, dtype=dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        L = self._get_cholesky(dtype, device)
        diff_flat = diff.reshape(-1, self.D).transpose(0, 1)
        solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
        m2 = (solved * solved).sum(dim=0).reshape(z.shape[0], self.C)
        log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
        log_prior = torch.log_softmax(self.prior_logits.to(device=device, dtype=dtype), dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


class FisherFullCovLDAHead(nn.Module):
    """
    Full-covariance LDA head trained via Fisher's discriminant criterion.

    Forward returns a scalar loss (-Fisher ratio) that pushes class means apart
    relative to the shared covariance using the current batch. Use `logits(z)`
    to obtain class logits identical to FullCovLDAHead for evaluation.
    """

    def __init__(self, C, D, min_scale=1e-4, fisher_eps=1e-8, prior_strength=0.5):
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be positive (got D={D}).")
        if not (0.0 <= prior_strength <= 1.0):
            raise ValueError(f"prior_strength must be in [0,1] (got {prior_strength}).")
        self.C = C
        self.D = D
        self.min_scale = min_scale
        self.fisher_eps = fisher_eps
        self.prior_strength = prior_strength
        dtype = torch.get_default_dtype()
        self.mu = nn.Parameter(torch.zeros(C, D, dtype=dtype))
        self.raw_tril = nn.Parameter(torch.zeros(D, D, dtype=dtype))
        self.prior_logits = nn.Parameter(torch.zeros(C, dtype=dtype))

    def _get_cholesky(self, dtype, device):
        raw = torch.tril(self.raw_tril.to(device=device, dtype=dtype))
        diag = torch.diagonal(raw, 0)
        safe_diag = F.softplus(diag) + self.min_scale
        L = raw - torch.diag(diag) + torch.diag(safe_diag)
        return L

    def logits(self, z):
        """Compute class logits (same form as FullCovLDAHead) for evaluation."""
        dtype = z.dtype
        device = z.device
        mu = self.mu.to(device=device, dtype=dtype)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        L = self._get_cholesky(dtype, device)
        diff_flat = diff.reshape(-1, self.D).transpose(0, 1)
        solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
        m2 = (solved * solved).sum(dim=0).reshape(z.shape[0], self.C)
        log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
        log_prior = torch.log_softmax(self.prior_logits.to(device=device, dtype=dtype), dim=0)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)

    def forward(self, z, y):
        """
        Return negative Fisher ratio for the batch.

        z: (B, D) embeddings
        y: (B,) labels in [0, C)
        """
        if y is None:
            raise ValueError("Labels y must be provided to compute the Fisher criterion.")
        if y.dim() != 1:
            raise ValueError(f"Expected y to be 1D (got shape {tuple(y.shape)}).")
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched batch sizes: z has {z.shape[0]}, y has {y.shape[0]}.")

        dtype = z.dtype
        device = z.device
        mu = self.mu.to(device=device, dtype=dtype)
        L = self._get_cholesky(dtype, device)

        # Within-class scatter measured by Mahalanobis distance to the class mean.
        diff = z - mu[y]
        diff_flat = diff.transpose(0, 1)
        solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
        within = (solved * solved).sum(dim=0).mean()

        # Between-class scatter: weighted spread of class means around the global mean.
        counts = torch.bincount(y, minlength=self.C).to(device=device, dtype=dtype)
        total = counts.sum().clamp_min(1.0)
        data_pi = counts / total
        learned_pi = torch.softmax(self.prior_logits.to(device=device, dtype=dtype), dim=0)
        pi = self.prior_strength * learned_pi + (1.0 - self.prior_strength) * data_pi

        overall_mu = (pi.unsqueeze(1) * mu).sum(dim=0)
        centered = mu - overall_mu
        centered_flat = centered.transpose(0, 1)
        centered_solved = torch.linalg.solve_triangular(L, centered_flat, upper=False)
        between_per_class = (centered_solved * centered_solved).sum(dim=0)
        between = (pi * between_per_class).sum()

        fisher_ratio = between / (within + self.fisher_eps)
        return -fisher_ratio


class BatchwiseFullCovLDAHead(nn.Module):
    """
    LDA head that re-estimates priors/means/shared covariance on each batch and
    maintains an EMA of those estimates. Only the encoder is trained; the head
    parameters are treated as plug-in statistics.
    """

    def __init__(self, C, D, ema_beta=0.9, min_scale=1e-4, prior_floor=1e-6):
        super().__init__()
        if D < 1:
            raise ValueError(f"D must be positive (got D={D}).")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError(f"ema_beta must be in [0,1) (got {ema_beta}).")
        self.C = C
        self.D = D
        self.ema_beta = ema_beta
        self.min_scale = min_scale
        self.prior_floor = prior_floor
        dtype = torch.get_default_dtype()
        self.register_buffer("ema_pi", torch.full((C,), 1.0 / C, dtype=dtype))
        self.register_buffer("ema_mu", torch.zeros(C, D, dtype=dtype))
        self.register_buffer("ema_cov", torch.eye(D, dtype=dtype))

    @torch.no_grad()
    def _reestimate_from_batch(self, z, y):
        """
        Re-estimate pi, mu, Sigma from a labeled batch and update EMA buffers.
        """
        if y is None:
            raise ValueError("Labels y must be provided during training for batchwise re-estimation.")
        if y.dim() != 1:
            raise ValueError(f"Expected y to be 1D (got shape {tuple(y.shape)}).")
        if z.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched batch sizes: z has {z.shape[0]}, y has {y.shape[0]}.")

        device = z.device
        dtype = z.dtype
        m = float(z.shape[0])
        one_hot = F.one_hot(y, num_classes=self.C).to(dtype=dtype)
        counts = one_hot.sum(dim=0)  # (C,)
        pi_hat = counts / m

        sum_z = one_hot.transpose(0, 1) @ z  # (C, D)
        safe_counts = counts.clamp_min(1.0).unsqueeze(1)
        mu_hat = sum_z / safe_counts

        # For classes absent in the batch, keep previous means to avoid dividing by zero noise.
        present = counts > 0
        mu_hat = torch.where(present.unsqueeze(1), mu_hat, self.ema_mu.to(device=device, dtype=dtype))

        # Shared covariance across all classes.
        mu_per_sample = mu_hat[y]
        diff = z - mu_per_sample
        cov_hat = (diff.transpose(0, 1) @ diff) / m
        cov_hat = cov_hat + torch.eye(self.D, device=device, dtype=dtype) * self.min_scale

        beta = self.ema_beta
        pi_tilde = beta * self.ema_pi.to(device=device, dtype=dtype) + (1.0 - beta) * pi_hat
        mu_tilde = beta * self.ema_mu.to(device=device, dtype=dtype) + (1.0 - beta) * mu_hat
        cov_tilde = beta * self.ema_cov.to(device=device, dtype=dtype) + (1.0 - beta) * cov_hat

        self.ema_pi.copy_(pi_tilde)
        self.ema_mu.copy_(mu_tilde)
        self.ema_cov.copy_(cov_tilde)

    def forward(self, z, y=None):
        device = z.device
        dtype = z.dtype
        if self.training:
            self._reestimate_from_batch(z.detach(), y.detach() if y is not None else None)

        pi = self.ema_pi.to(device=device, dtype=dtype)
        mu = self.ema_mu.to(device=device, dtype=dtype)
        cov = self.ema_cov.to(device=device, dtype=dtype)

        # Compute log likelihoods under N(mu_c, cov) with a shared covariance.
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        # Add a small diagonal in case cov drifts toward singular.
        cov_safe = cov + torch.eye(self.D, device=device, dtype=dtype) * self.min_scale
        L = torch.linalg.cholesky(cov_safe)
        diff_flat = diff.reshape(-1, self.D).transpose(0, 1)
        solved = torch.linalg.solve_triangular(L, diff_flat, upper=False)
        m2 = (solved * solved).sum(dim=0).reshape(z.shape[0], self.C)
        log_det = 2.0 * torch.log(torch.diagonal(L)).sum()

        pi_safe = torch.clamp(pi, min=self.prior_floor)
        log_prior = torch.log(pi_safe)
        return log_prior.unsqueeze(0) - 0.5 * (m2 + log_det)


class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        loss = self.loss(X, target)

        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size
        return loss


class LDALossFunction(Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, X, target):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]
        p_star = torch.exp(X)
        loss = X.mean(dim=1) #need to change to normal loss
        p_star.scatter_add_(1, target.unsqueeze(1), torch.full_like(p_star, -1))
        ctx.save_for_backward(p_star)
        return loss
    
    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        p_star,  = ctx.saved_tensors 
        grad = grad_output.unsqueeze(1) * p_star
        ret = (grad,)

        # pad with as many Nones as needed
        return ret + (None,) * (1 + 1)

def lda_loss(X, target):
    return LDALossFunction.apply(X, target)


class LDALoss(_GenericLoss):
    def __init__(self, ignore_index=-100, reduction='elementwise_mean'):
        super(LDALoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return lda_loss(X, target)



def dnll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    lambda_reg: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    Applicable to any generative classifier with class-wise
    (unnormalized) log-density or log-joint scores.

    Args:
        input:  Tensor (N, C) of class scores δ_c(x).
        target: LongTensor (N,) with class indices in [0, C-1].
        lambda_reg: float ≥ 0, strength of discriminative penalty.
        reduction: "none" | "mean" | "sum".

    Returns:
        Loss reduced according to `reduction`.
    """
    # NLL part: -δ_y(x)
    nll = -input.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)

    # Discriminative penalty: λ * ∑_c exp(δ_c(x))
    reg = lambda_reg * input.exp().sum(dim=1)               # (N,)

    loss = nll + reg                                        # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class DNLLLoss(nn.Module):
    r"""
    DNLL: Discriminative Negative Log-Likelihood

        L(x, y) = -input_y(x) + λ * sum_c exp(input_c(x))

    A drop-in loss module similar to nn.CrossEntropyLoss, but designed
    for generative classifiers whose outputs are log-density scores.
    """
    def __init__(self, lambda_reg: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dnll_loss(
            input=input,
            target=target,
            lambda_reg=self.lambda_reg,
            reduction=self.reduction,
        )


def logistic_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    One-vs-all logistic loss over all classes.

        L(x, y) = -log σ(δ_y(x)) - ∑_{c≠y} log σ(-δ_c(x))

    Equivalent to summing binary cross-entropy terms for each class with a
    one-hot target.

    Args:
        input:  Tensor (N, C) of class scores δ_c(x).
        target: LongTensor (N,) with class indices in [0, C-1].
        reduction: "none" | "mean" | "sum".

    Returns:
        Loss reduced according to `reduction`.
    """
    one_hot = F.one_hot(target, num_classes=input.shape[1]).type_as(input)  # (N, C)
    per_class = F.binary_cross_entropy_with_logits(input, one_hot, reduction="none")  # (N, C)
    loss = per_class.sum(dim=1)  # (N,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class LogisticLoss(nn.Module):
    r"""
    One-vs-all logistic loss over all classes.

        L(x, y) = -log σ(δ_y(x)) - ∑_{c≠y} log σ(-δ_c(x))
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return logistic_loss(
            input=input,
            target=target,
            reduction=self.reduction,
        )

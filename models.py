"""
Model architectures for LDA classification and segmentation experiments.

Includes:
- Classification: Encoder, SoftmaxClassifier, SimplexLDAClassifier, TrainableLDAClassifier
- Tabular: MLPEncoder, SoftmaxTabular, SimplexLDATabular, TrainableLDATabular
- Segmentation: UNetEncoder, SoftmaxSegmentation, SimplexLDASegmentation, TrainableLDASegmentation
"""
import torch
import torch.nn as nn

from src.lda import LDAHead, TrainableLDAHead


# ==============================================================================
# Tabular Classification Models (MLP-based)
# ==============================================================================
class MLPEncoder(nn.Module):
    """MLP encoder for tabular data."""

    def __init__(self, input_dim: int, dim: int, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SoftmaxTabular(nn.Module):
    """Standard softmax classifier for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, dim: int, hidden_dims=None, dropout=0.3):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, dim, hidden_dims, dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class SimplexLDATabular(nn.Module):
    """LDA classifier with fixed simplex geometry for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, dim: int, hidden_dims=None, dropout=0.3):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, dim, hidden_dims, dropout)
        self.head = LDAHead(num_classes, dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class TrainableLDATabular(nn.Module):
    """LDA classifier with trainable class means and full covariance for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, dim: int, hidden_dims=None, dropout=0.3):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, dim, hidden_dims, dropout)
        self.head = TrainableLDAHead(num_classes, dim, cov_type="full")

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class TrainableLDASphericalTabular(nn.Module):
    """LDA classifier with trainable class means and spherical covariance for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, dim: int, hidden_dims=None, dropout=0.3):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, dim, hidden_dims, dropout)
        self.head = TrainableLDAHead(num_classes, dim, cov_type="spherical")

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# ==============================================================================
# Image Classification Models
# ==============================================================================
class Encoder(nn.Module):
    """Simple CNN encoder for image classification."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class SoftmaxClassifier(nn.Module):
    """Standard softmax classifier."""

    def __init__(self, in_channels: int, num_classes: int, dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class SimplexLDAClassifier(nn.Module):
    """LDA classifier with fixed simplex geometry."""

    def __init__(self, in_channels: int, num_classes: int, dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, dim)
        self.head = LDAHead(num_classes, dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class TrainableLDAClassifier(nn.Module):
    """LDA classifier with trainable class means and full covariance."""

    def __init__(self, in_channels: int, num_classes: int, dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, dim)
        self.head = TrainableLDAHead(num_classes, dim, cov_type="full")

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class TrainableLDASphericalClassifier(nn.Module):
    """LDA classifier with trainable class means and spherical covariance."""

    def __init__(self, in_channels: int, num_classes: int, dim: int):
        super().__init__()
        self.encoder = Encoder(in_channels, dim)
        self.head = TrainableLDAHead(num_classes, dim, cov_type="spherical")

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


# ==============================================================================
# Segmentation Models
# ==============================================================================
class ConvBlock(nn.Module):
    """Double convolution block for UNet."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleFCNEncoder(nn.Module):
    """Simple fully convolutional encoder without skip connections.
    
    A lightweight alternative to UNet, suitable for small scenes like Indian Pines.
    Uses dilated convolutions to increase receptive field without downsampling.
    """

    def __init__(self, in_ch: int, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_ch

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    """Small UNet encoder-decoder that outputs a feature map."""

    def __init__(self, in_ch: int, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.up2 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up0 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec0 = ConvBlock(base_ch * 2, base_ch)

        self.out_channels = base_ch

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e1], dim=1)
        d0 = self.dec0(d0)

        return d0


class SoftmaxSegmentation(nn.Module):
    """Standard softmax segmentation model with UNet backbone."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.head = nn.Conv2d(self.encoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.encoder(x))


class SimplexLDASegmentation(nn.Module):
    """LDA segmentation model with fixed simplex geometry."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = LDAHead(num_classes, self.dim)

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits


class TrainableLDASegmentation(nn.Module):
    """LDA segmentation model with trainable class means and full covariance."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = TrainableLDAHead(num_classes, self.dim, cov_type="full")

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits


class TrainableLDASphericalSegmentation(nn.Module):
    """LDA segmentation model with trainable class means and spherical covariance."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = UNetEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = TrainableLDAHead(num_classes, self.dim, cov_type="spherical")

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits


# ==============================================================================
# Simple FCN Segmentation Models (for small scenes like Indian Pines)
# ==============================================================================
class SoftmaxFCNSegmentation(nn.Module):
    """Simple FCN segmentation with softmax head."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = SimpleFCNEncoder(in_ch, base_ch)
        self.head = nn.Conv2d(self.encoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.encoder(x))


class SimplexLDAFCNSegmentation(nn.Module):
    """Simple FCN segmentation with LDA head (fixed simplex)."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = SimpleFCNEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = LDAHead(num_classes, self.dim)

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits


class TrainableLDAFCNSegmentation(nn.Module):
    """Simple FCN segmentation with trainable LDA head (full covariance)."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = SimpleFCNEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = TrainableLDAHead(num_classes, self.dim, cov_type="full")

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits


class TrainableLDASphericalFCNSegmentation(nn.Module):
    """Simple FCN segmentation with trainable LDA head (spherical covariance)."""

    def __init__(self, in_ch: int, num_classes: int, base_ch: int = 32):
        super().__init__()
        self.encoder = SimpleFCNEncoder(in_ch, base_ch)
        self.num_classes = num_classes
        self.dim = num_classes - 1
        self.proj = nn.Conv2d(self.encoder.out_channels, self.dim, kernel_size=1)
        self.lda_head = TrainableLDAHead(num_classes, self.dim, cov_type="spherical")

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.proj(z)
        b, d, h, w = z_embed.shape
        z_flat = z_embed.permute(0, 2, 3, 1).reshape(b * h * w, d)
        logits_flat = self.lda_head(z_flat)
        logits = logits_flat.view(b, h, w, self.num_classes).permute(0, 3, 1, 2)
        return logits

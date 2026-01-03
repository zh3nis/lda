"""
Visualize PCA embeddings for LDA vs Softmax models.

Usage:
    python visualize_embeddings.py --dataset EuroSAT --epochs 20
    python visualize_embeddings.py --dataset IndianPines --data-dir ./indian_pines --epochs 20
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from sklearn.decomposition import PCA

from src.lda import LDAHead, TrainableLDAHead, DNLLLoss
from utils import seed_all


# ==============================================================================
# Encoders and Models
# ==============================================================================
class Encoder(nn.Module):
    """CNN encoder for classification."""
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.proj(torch.flatten(self.features(x), 1))


class SegEncoder(nn.Module):
    """CNN encoder for segmentation."""
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_classifier(encoder_cls, head_type, in_ch, num_classes, is_seg=False):
    """Factory for classification/segmentation models."""
    dim = num_classes - 1
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder_cls(in_ch, dim)
            self.num_classes = num_classes
            if head_type == "softmax":
                self.head = nn.Conv2d(dim, num_classes, 1) if is_seg else nn.Linear(dim, num_classes)
            elif head_type == "simplex":
                self.head = LDAHead(num_classes, dim)
            else:  # trainable
                self.head = TrainableLDAHead(num_classes, dim, cov_type="full")
            self.is_seg = is_seg
            self.is_lda = head_type != "softmax"

        def forward(self, x):
            z = self.encoder(x)
            if self.is_seg and self.is_lda:
                B, D, H, W = z.shape
                z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, D)
                logits = self.head(z_flat)
                return logits.view(B, H, W, self.num_classes).permute(0, 3, 1, 2)
            return self.head(z)

        def get_embeddings(self, x):
            return self.encoder(x)

    return Model()


# ==============================================================================
# Data Loading
# ==============================================================================
def load_eurosat(data_root: str, batch_size: int):
    from torchgeo.datasets import EuroSAT

    class Wrapper(torch.utils.data.Dataset):
        def __init__(self, ds, transform):
            self.ds, self.transform = ds, transform
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            s = self.ds[i]
            img = s["image"] if isinstance(s, dict) else s[0]
            label = s["label"] if isinstance(s, dict) else s[1]
            if not torch.is_tensor(img):
                img = T.functional.to_tensor(img)
            return self.transform(img.float()), int(label)

    ds = EuroSAT(root=f"{data_root}/EuroSAT", download=True)
    n_train = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(42))
    transform = T.Compose([T.Resize(64), T.Normalize([0.5]*13, [0.5]*13)])
    
    return (DataLoader(Wrapper(train_ds, transform), batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(Wrapper(test_ds, transform), batch_size=batch_size, shuffle=False, num_workers=4),
            13, 10)


def load_indian_pines(data_dir: str, batch_size: int, crop_size: int = 32):
    from scipy.io import loadmat

    data = loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))
    gt = loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))
    img = data['indian_pines_corrected'].astype(np.float32).transpose(2, 0, 1)
    labels = gt['indian_pines_gt'].astype(np.int64)

    # Normalize
    mean, std = img.mean(axis=(1, 2), keepdims=True), img.std(axis=(1, 2), keepdims=True) + 1e-6
    img = (img - mean) / std

    class HSDataset(torch.utils.data.Dataset):
        def __init__(self, img, labels, n_samples, crop_size):
            self.img, self.labels, self.n, self.cs = img, labels, n_samples, crop_size
            self.h, self.w = labels.shape
        def __len__(self):
            return self.n
        def __getitem__(self, _):
            for _ in range(20):
                y, x = np.random.randint(0, self.h - self.cs), np.random.randint(0, self.w - self.cs)
                lab = self.labels[y:y+self.cs, x:x+self.cs]
                if (lab > 0).sum() > self.cs * self.cs * 0.1:
                    break
            return (torch.from_numpy(self.img[:, y:y+self.cs, x:x+self.cs].copy()),
                    torch.from_numpy(lab.copy()))

    return (DataLoader(HSDataset(img, labels, 1000, crop_size), batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(HSDataset(img, labels, 200, crop_size), batch_size=batch_size, shuffle=False, num_workers=0),
            img.shape[0], int(labels.max()) + 1)


# ==============================================================================
# Training
# ==============================================================================
def train_epoch(model, loader, optimizer, loss_fn, device, is_seg=False, ignore_index=0):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        
        if is_seg:
            B, C, H, W = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
            y = y.reshape(-1)
            mask = y != ignore_index
            if mask.sum() == 0:
                continue
            logits, y = logits[mask], y[mask]
        
        optimizer.zero_grad()
        loss_fn(logits, y).backward()
        optimizer.step()


# ==============================================================================
# Visualization
# ==============================================================================
def collect_embeddings(models, loader, device, is_seg=False, max_pixels=5000):
    """Collect embeddings from all models."""
    for m in models:
        m.eval()
    
    embeddings = [[] for _ in models]
    labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            for i, m in enumerate(models):
                z = m.get_embeddings(x_dev)
                if is_seg:
                    z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
                embeddings[i].append(z.cpu().numpy())
            
            y_np = y.reshape(-1).numpy() if is_seg else y.numpy()
            labels.append(y_np)
            
            if is_seg and sum(len(l) for l in labels) > max_pixels:
                break
    
    embeddings = [np.concatenate(e) for e in embeddings]
    labels = np.concatenate(labels)
    
    if is_seg:
        # Filter out background and limit pixels
        mask = labels > 0
        embeddings = [e[mask][:max_pixels] for e in embeddings]
        labels = labels[mask][:max_pixels]
    
    return embeddings, labels


def plot_pca_embeddings(embeddings_list, labels_list, num_classes_list, save_path, dataset_names, is_seg_list):
    """Plot PCA embeddings for all models and datasets in a 2x3 grid."""
    titles = ["Softmax (PCA)", "Simplex LDA (PCA)", "Trainable LDA (PCA)"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    print("Running PCA on embeddings...")
    for row, (embeddings, labels, num_classes, name, is_seg) in enumerate(
            zip(embeddings_list, labels_list, num_classes_list, dataset_names, is_seg_list)):
        cmap = plt.colormaps.get_cmap('tab20' if num_classes > 10 else 'tab10')
        start_class = 1 if is_seg else 0
        
        for col, (title, z) in enumerate(zip(titles, embeddings)):
            ax = axes[row, col]
            z_pca = PCA(n_components=2, random_state=42).fit_transform(z)
            for c in range(start_class, num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    ax.scatter(z_pca[mask, 0], z_pca[mask, 1], c=[cmap(c % 20)], 
                              s=8 if is_seg else 15, alpha=0.5, label=f"{c}")
            ax.set_title(f"{name}: {title}", fontsize=12)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.legend(loc='upper right', fontsize=5, ncol=3)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================
def train_and_collect(encoder_cls, train_loader, test_loader, in_ch, num_classes, device, epochs, is_seg=False):
    """Train models and collect embeddings."""
    models = [make_classifier(encoder_cls, h, in_ch, num_classes, is_seg=is_seg) 
              for h in ["softmax", "simplex", "trainable"]]
    models = [m.to(device) for m in models]
    
    ce_loss = nn.CrossEntropyLoss(ignore_index=0) if is_seg else nn.CrossEntropyLoss()
    dnll_loss = DNLLLoss(lambda_reg=1.0)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    losses = [ce_loss, dnll_loss, dnll_loss]

    for epoch in range(1, epochs + 1):
        for model, opt, loss_fn in zip(models, optimizers, losses):
            train_epoch(model, train_loader, opt, loss_fn, device, is_seg=is_seg)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}")

    return collect_embeddings(models, test_loader, device, is_seg)


def main():
    parser = argparse.ArgumentParser(description="Visualize PCA embeddings")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--data-dir", type=str, default="./indian_pines")
    parser.add_argument("--out-dir", type=str, default="./plots")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    # EuroSAT
    print("Training on EuroSAT...")
    train_loader, test_loader, in_ch, num_classes = load_eurosat(args.data_root, args.batch_size)
    print(f"EuroSAT: {num_classes} classes, {in_ch} channels")
    euro_emb, euro_labels = train_and_collect(Encoder, train_loader, test_loader, in_ch, num_classes, device, args.epochs)

    # Indian Pines
    print("\nTraining on Indian Pines...")
    train_loader, test_loader, in_ch, num_classes_ip = load_indian_pines(args.data_dir, args.batch_size)
    print(f"Indian Pines: {num_classes_ip} classes, {in_ch} channels")
    ip_emb, ip_labels = train_and_collect(SegEncoder, train_loader, test_loader, in_ch, num_classes_ip, device, args.epochs, is_seg=True)

    # Plot combined
    os.makedirs(args.out_dir, exist_ok=True)
    plot_pca_embeddings(
        [euro_emb, ip_emb], [euro_labels, ip_labels], [num_classes, num_classes_ip],
        os.path.join(args.out_dir, "pca_comparison.png"),
        ["EuroSAT", "Indian Pines"], [False, True]
    )


if __name__ == "__main__":
    main()

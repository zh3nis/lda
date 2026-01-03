"""
Shared utilities for LDA classification and segmentation experiments.
"""
import os
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ==============================================================================
# Seeding and I/O
# ==============================================================================
def seed_all(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_json(path: str, obj: Any):
    """Save object to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ==============================================================================
# Metrics
# ==============================================================================
@torch.no_grad()
def compute_segmentation_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int = None
) -> Tuple[float, float]:
    """Compute pixel accuracy and mean IoU for segmentation."""
    pred = logits.argmax(dim=1)

    if ignore_index is not None:
        valid = masks != ignore_index
        pred = pred[valid]
        masks = masks[valid]
        start_cls = 1 if ignore_index == 0 else 0
    else:
        pred = pred.flatten()
        masks = masks.flatten()
        start_cls = 0

    if masks.numel() == 0:
        return 0.0, 0.0

    correct = (pred == masks).sum().item()
    acc = float(correct) / float(masks.numel())

    ious: List[float] = []
    for cls in range(start_cls, num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_c = pred == cls
        mask_c = masks == cls
        inter = torch.logical_and(pred_c, mask_c).sum().item()
        union = torch.logical_or(pred_c, mask_c).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    miou = float(sum(ious) / max(len(ious), 1))
    return acc, miou


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = None
) -> Dict[str, float]:
    """Evaluate segmentation model."""
    model.eval()
    total_acc = 0.0
    total_iou = 0.0
    batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        acc, miou = compute_segmentation_metrics(logits, y, num_classes, ignore_index)
        total_acc += acc
        total_iou += miou
        batches += 1
    if batches == 0:
        return {"acc": 0.0, "miou": 0.0}
    return {"acc": total_acc / batches, "miou": total_iou / batches}


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate classification model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return float(correct) / max(total, 1)


# ==============================================================================
# Loss Wrappers
# ==============================================================================
class SegmentationLossWrapper(nn.Module):
    """Wrapper to apply a classification loss to segmentation outputs."""

    def __init__(self, base_loss, ignore_index=None):
        super().__init__()
        self.base_loss = base_loss
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        b, c, h, w = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(b * h * w, c)
        target_flat = target.reshape(b * h * w)
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            logits_flat = logits_flat[mask]
            target_flat = target_flat[mask]
        return self.base_loss(logits_flat, target_flat)


# ==============================================================================
# Training Loops
# ==============================================================================
def train_segmentation_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int = None
) -> Dict[str, float]:
    """Train segmentation model for one epoch."""
    model.train()
    total_acc = 0.0
    total_iou = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc, miou = compute_segmentation_metrics(logits.detach(), y, num_classes, ignore_index)
        total_acc += acc
        total_iou += miou
        batches += 1

    if batches == 0:
        return {"acc": 0.0, "miou": 0.0}
    return {"acc": total_acc / batches, "miou": total_iou / batches}


def train_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """Train classification model for one epoch, return training accuracy."""
    model.train()
    total = 0
    total_correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
    return float(total_correct) / max(total, 1)


# ==============================================================================
# Plotting Utilities
# ==============================================================================
def mean_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and 95% confidence interval."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * std / np.sqrt(max(n, 1))
    return mean, ci


def plot_classification_results(results_root: str, out_path: str = None, models: List[str] = None):
    """
    Plot classification experiment results with confidence intervals.
    
    Args:
        results_root: Directory containing dataset folders with seed_* subdirs
        out_path: Output path for the plot (default: results_root/results.png)
        models: List of model names to plot (default: softmax, simplex_lda, trainable_lda)
    """
    if models is None:
        models = ["softmax", "simplex_lda", "trainable_lda"]
    if out_path is None:
        out_path = os.path.join(results_root, "results.png")

    datasets = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    datasets.sort()

    n_datasets = len(datasets)
    if n_datasets == 0:
        print(f"No datasets found in {results_root}")
        return

    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    colors = {"softmax": "C0", "simplex_lda": "C1", "trainable_lda": "C2"}
    linestyles = {"softmax": "-", "simplex_lda": "--", "trainable_lda": "-."}

    for idx, dataset in enumerate(datasets):
        ds_dir = os.path.join(results_root, dataset)
        seed_dirs = [d for d in os.listdir(ds_dir) if d.startswith("seed_")]
        seed_dirs.sort()

        histories = []
        for sd in seed_dirs:
            path = os.path.join(ds_dir, sd, "acc_history.json")
            if os.path.exists(path):
                histories.append(load_json(path))

        if not histories:
            print(f"No results for {dataset}")
            continue

        epochs = histories[0]["epochs"]
        stacked = {m: {"train": [], "test": []} for m in models}

        for run in histories:
            hist = run["history"]
            for m in models:
                if m in hist:
                    stacked[m]["train"].append(hist[m]["train_acc"])
                    stacked[m]["test"].append(hist[m]["test_acc"])

        for m in models:
            stacked[m]["train"] = np.array(stacked[m]["train"])
            stacked[m]["test"] = np.array(stacked[m]["test"])

        x = np.arange(1, epochs + 1)

        # Train accuracy
        ax = axes[0, idx]
        for m in models:
            if stacked[m]["train"].size > 0:
                mean, ci = mean_ci(stacked[m]["train"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title(f"{dataset} — Train")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Test accuracy
        ax = axes[1, idx]
        for m in models:
            if stacked[m]["test"].size > 0:
                mean, ci = mean_ci(stacked[m]["test"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title(f"{dataset} — Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_segmentation_results(results_root: str, out_path: str = None, models: List[str] = None):
    """
    Plot segmentation experiment results with confidence intervals.
    Creates a separate plot for each dataset.
    
    Args:
        results_root: Directory containing dataset folders with seed_* subdirs
        out_path: Ignored (plots saved per dataset)
        models: List of model names to plot
    """
    if models is None:
        models = ["softmax", "simplex_lda", "trainable_lda"]

    datasets = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    datasets.sort()

    if len(datasets) == 0:
        print(f"No datasets found in {results_root}")
        return

    colors = {"softmax": "C0", "simplex_lda": "C1", "trainable_lda": "C2"}
    linestyles = {"softmax": "-", "simplex_lda": "--", "trainable_lda": "-."}

    for dataset in datasets:
        ds_dir = os.path.join(results_root, dataset)
        seed_dirs = [d for d in os.listdir(ds_dir) if d.startswith("seed_")]
        seed_dirs.sort()

        histories = []
        for sd in seed_dirs:
            path = os.path.join(ds_dir, sd, "metrics_history.json")
            if os.path.exists(path):
                histories.append(load_json(path))

        if not histories:
            print(f"No results for {dataset}")
            continue

        epochs = histories[0]["epochs"]
        
        # Determine metric keys (val vs test)
        sample_hist = histories[0]["history"]["softmax"]
        use_val = "val_miou" in sample_hist
        miou_key = "val_miou" if use_val else "test_miou"
        acc_key = "val_acc" if use_val else "test_acc"

        stacked = {m: {"train_miou": [], miou_key: [], "train_acc": [], acc_key: []} for m in models}

        for run in histories:
            hist = run["history"]
            for m in models:
                if m in hist:
                    stacked[m]["train_miou"].append(hist[m]["train_miou"])
                    stacked[m][miou_key].append(hist[m][miou_key])
                    stacked[m]["train_acc"].append(hist[m]["train_acc"])
                    stacked[m][acc_key].append(hist[m][acc_key])

        for m in models:
            for k in stacked[m]:
                stacked[m][k] = np.array(stacked[m][k])

        x = np.arange(1, epochs + 1)

        # Create figure for this dataset
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{dataset} Segmentation Results", fontsize=14)

        # Train mIoU
        ax = axes[0, 0]
        for m in models:
            if stacked[m]["train_miou"].size > 0:
                mean, ci = mean_ci(stacked[m]["train_miou"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title("Train mIoU")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mIoU")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Val/Test mIoU
        ax = axes[0, 1]
        for m in models:
            if stacked[m][miou_key].size > 0:
                mean, ci = mean_ci(stacked[m][miou_key])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title("Val mIoU" if use_val else "Test mIoU")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mIoU")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Train Accuracy
        ax = axes[1, 0]
        for m in models:
            if stacked[m]["train_acc"].size > 0:
                mean, ci = mean_ci(stacked[m]["train_acc"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title("Train Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Val/Test Accuracy
        ax = axes[1, 1]
        for m in models:
            if stacked[m][acc_key].size > 0:
                mean, ci = mean_ci(stacked[m][acc_key])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title("Val Accuracy" if use_val else "Test Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(ds_dir, f"{dataset.lower()}_segmentation_results.png")
        plt.savefig(plot_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")


def save_segmentation_comparison_plot(
    models_dict: Dict[str, nn.Module],
    get_sample_fn,  # Callable that takes idx and returns (raw_img_np, img_tensor, mask_gt)
    indices: List[int],
    save_path: str,
    device: torch.device,
    num_classes: int,
    rgb_bands: List[int] = None
):
    """
    Save a comparison plot for segmentation models.
    
    Args:
        models_dict: Dict of model_name -> model
        get_sample_fn: Function that takes an index and returns (raw_rgb_np, img_tensor, mask_gt_np)
        indices: List of sample indices to plot
        save_path: Output path for the plot
        device: Torch device
        num_classes: Number of classes
        rgb_bands: Optional RGB band indices for hyperspectral data
    """
    for model in models_dict.values():
        model.eval()

    n = len(indices)
    n_cols = 2 + len(models_dict)
    fig, axes = plt.subplots(n, n_cols, figsize=(2.5 * n_cols, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Input", "Ground Truth"] + list(models_dict.keys())
    norm = Normalize(vmin=0, vmax=num_classes - 1)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(num_classes)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw_rgb, img_tensor, mask_gt = get_sample_fn(idx)

            # Input
            axes[row, 0].imshow(np.clip(raw_rgb, 0, 1))
            axes[row, 0].axis("off")

            # Ground truth
            axes[row, 1].imshow(mask_gt, cmap=cmap, norm=norm)
            axes[row, 1].axis("off")

            # Model predictions
            img_input = img_tensor.unsqueeze(0).to(device)
            for col, (name, model) in enumerate(models_dict.items(), start=2):
                pred_mask = model(img_input).argmax(dim=1).squeeze(0).cpu().numpy()
                axes[row, col].imshow(pred_mask, cmap=cmap, norm=norm)
                axes[row, col].axis("off")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Tabular Classification Plotting
# ==============================================================================
def plot_tabular_results(results_root: str, out_path: str = None, models: List[str] = None):
    """Plot tabular classification results with confidence intervals."""
    if models is None:
        models = ["softmax", "simplex_lda", "trainable_lda"]
    if out_path is None:
        out_path = os.path.join(results_root, "results.png")

    datasets = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    datasets.sort()

    n_datasets = len(datasets)
    if n_datasets == 0:
        print(f"No datasets found in {results_root}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"softmax": "C0", "simplex_lda": "C1", "trainable_lda": "C2"}
    linestyles = {"softmax": "-", "simplex_lda": "--", "trainable_lda": "-."}

    for idx, dataset in enumerate(datasets):
        ds_dir = os.path.join(results_root, dataset)
        seed_dirs = [d for d in os.listdir(ds_dir) if d.startswith("seed_")]
        seed_dirs.sort()

        histories = []
        for sd in seed_dirs:
            path = os.path.join(ds_dir, sd, "acc_history.json")
            if os.path.exists(path):
                histories.append(load_json(path))

        if not histories:
            print(f"No results for {dataset}")
            continue

        epochs = histories[0]["epochs"]
        stacked = {m: {"train": [], "test": []} for m in models}

        for run in histories:
            hist = run["history"]
            for m in models:
                if m in hist:
                    stacked[m]["train"].append(hist[m]["train_acc"])
                    stacked[m]["test"].append(hist[m]["test_acc"])

        for m in models:
            stacked[m]["train"] = np.array(stacked[m]["train"])
            stacked[m]["test"] = np.array(stacked[m]["test"])

        x = np.arange(1, epochs + 1)

        # Train accuracy
        ax = axes[0]
        for m in models:
            if stacked[m]["train"].size > 0:
                mean, ci = mean_ci(stacked[m]["train"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title(f"{dataset} — Train")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Test accuracy
        ax = axes[1]
        for m in models:
            if stacked[m]["test"].size > 0:
                mean, ci = mean_ci(stacked[m]["test"])
                ax.plot(x, mean, label=m, color=colors[m], linestyle=linestyles[m])
                ax.fill_between(x, mean - ci, mean + ci, color=colors[m], alpha=0.2)
        ax.set_title(f"{dataset} — Test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_tabular_embeddings(models_dict: Dict[str, nn.Module], dataloader: DataLoader,
                            num_classes: int, device: torch.device, save_path: str,
                            max_batches: int = 10):
    """Plot 2D PCA embeddings for tabular models in 1x3 layout without legend."""
    for model in models_dict.values():
        model.eval()

    # Collect embeddings
    all_embeds = {k: [] for k in models_dict}
    labels_list = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            for name, model in models_dict.items():
                z = model.encoder(x).cpu()
                all_embeds[name].append(z)
            labels_list.append(y)
            if i >= max_batches - 1:
                break

    y = torch.cat(labels_list)

    # 1x3 layout
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, models_dict.items()):
        z = torch.cat(all_embeds[name])
        z0 = z - z.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(z0, q=2)
        z2 = z0 @ V[:, :2]

        for c in range(num_classes):
            idx = y == c
            ax.scatter(z2[idx, 0].numpy(), z2[idx, 1].numpy(), s=8, alpha=0.6)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(name)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embeddings plot: {save_path}")

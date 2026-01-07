"""
Classification experiments with LDA heads on TorchGeo datasets.

Datasets: EuroSAT, RESISC45, UCMerced

Usage:
    python train_classification.py --data-root ./data --epochs 30 --seeds 42 123 456
    python train_classification.py --plot-only --results-root ./runs/classification
"""
import os
import time
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

from torchgeo.datasets import EuroSAT, RESISC45, UCMerced

from src.lda import DNLLLoss
from models import (
    SoftmaxClassifier, SimplexLDAClassifier, TrainableLDAClassifier,
    TrainableLDASphericalClassifier,
)
from utils import (
    seed_all,
    save_json,
    train_classification_epoch,
    evaluate_classification,
    plot_classification_results,
)


# ==============================================================================
# Dataset Configuration
# ==============================================================================
DATASETS = {
    "EuroSAT": EuroSAT,
    "RESISC45": RESISC45,
    "UCMerced": UCMerced,
}


# ==============================================================================
# Data Utilities
# ==============================================================================
class TorchGeoClassificationWrapper(Dataset):
    """Wrapper for TorchGeo classification datasets."""

    def __init__(self, base_ds, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        if isinstance(sample, dict):
            img = sample["image"]
            label = sample["label"]
        else:
            img, label = sample

        if not torch.is_tensor(img):
            img = T.functional.to_tensor(img)
        img = img.float()

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)


def make_dataloaders(
    dataset_cls,
    name: str,
    root: str,
    batch_size: int,
    num_workers: int,
    train_frac: float
) -> Dict[str, Any]:
    """Create train/test dataloaders for a classification dataset."""
    ds_root = f"{root}/{name}"
    base_ds = dataset_cls(root=ds_root, download=True, transforms=None)

    # Infer input channels
    sample = base_ds[0]
    if isinstance(sample, dict):
        img = sample["image"]
    else:
        img, _ = sample
    if not torch.is_tensor(img):
        img = T.functional.to_tensor(img)
    in_channels = img.shape[0]

    mean = [0.5] * in_channels
    std = [0.5] * in_channels

    train_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Resize(128),
        T.RandomHorizontalFlip(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Resize(128),
        T.Normalize(mean, std),
    ])

    n = len(base_ds)
    n_train = int(train_frac * n)
    n_test = n - n_train
    train_base, test_base = random_split(
        base_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds = TorchGeoClassificationWrapper(train_base, transform=train_transform)
    test_ds = TorchGeoClassificationWrapper(test_base, transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(getattr(base_ds, "classes"))

    return {
        "name": name,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "in_channels": in_channels,
    }


# ==============================================================================
# Main Experiment
# ==============================================================================
ALL_MODELS = ["softmax", "simplex_lda", "trainable_lda", "trainable_lda_spherical"]


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select models to train
    models_to_train = args.models if args.models else ALL_MODELS
    print(f"Models to train: {models_to_train}")

    # Prepare dataloaders for selected datasets
    if args.dataset:
        selected_datasets = {args.dataset: DATASETS[args.dataset]}
    else:
        selected_datasets = DATASETS

    datasets_info = []
    for name, cls in selected_datasets.items():
        info = make_dataloaders(
            cls,
            name,
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_frac=args.train_frac,
        )
        datasets_info.append(info)
        print(f"{name}: classes={info['num_classes']}, in_channels={info['in_channels']}")

    results_root = os.path.abspath(args.results_root)
    os.makedirs(results_root, exist_ok=True)

    # Train models for each dataset and seed
    for info in datasets_info:
        name = info["name"]
        train_loader = info["train_loader"]
        test_loader = info["test_loader"]
        C = info["num_classes"]
        D = C - 1  # LDA embedding dimension
        in_ch = info["in_channels"]

        for seed in args.seeds:
            seed_dir = os.path.join(results_root, name, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            out_path = os.path.join(seed_dir, "acc_history.json")

            # Load existing history if exists
            if os.path.exists(out_path):
                import json
                with open(out_path) as f:
                    existing_data = json.load(f)
                hist = existing_data.get("history", {})
            else:
                hist = {}

            print(f"\n=== {name} | seed={seed} ===")
            seed_all(seed)

            ce_loss = nn.CrossEntropyLoss().to(device)
            dnll_loss = DNLLLoss(lambda_reg=1.0).to(device)

            # Model configs: (name, class, loss_fn)
            model_configs = {
                "softmax": (SoftmaxClassifier, ce_loss),
                "simplex_lda": (SimplexLDAClassifier, dnll_loss),
                "trainable_lda": (TrainableLDAClassifier, dnll_loss),
                "trainable_lda_spherical": (TrainableLDASphericalClassifier, dnll_loss),
            }

            models_dir = os.path.join(seed_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            for model_name in models_to_train:
                if model_name not in model_configs:
                    print(f"Unknown model: {model_name}")
                    continue

                # Skip if already trained and not overwriting
                if model_name in hist and not args.overwrite:
                    print(f"  Skipping {model_name} (cached)")
                    continue

                model_cls, loss_fn = model_configs[model_name]
                model = model_cls(in_ch, C, D).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                hist[model_name] = {"train_acc": [], "test_acc": []}

                for epoch in range(1, args.epochs + 1):
                    tr_acc = train_classification_epoch(model, train_loader, optimizer, loss_fn, device)
                    te_acc = evaluate_classification(model, test_loader, device)
                    hist[model_name]["train_acc"].append(tr_acc)
                    hist[model_name]["test_acc"].append(te_acc)

                    if epoch % 10 == 0 or epoch == args.epochs:
                        print(f"  [{model_name}] Epoch {epoch:02d}: train={tr_acc:.3f} test={te_acc:.3f}")

                # Save model
                torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pth"))

            # Save history
            save_json(out_path, {
                "seed": seed,
                "epochs": args.epochs,
                "history": hist,
                "dataset": name,
                "timestamp": time.time(),
            })
            print(f"Saved {out_path}")

    # Plot results
    print("\nGenerating plots...")
    plot_classification_results(results_root)


def main():
    parser = argparse.ArgumentParser(description="LDA Classification Experiments on TorchGeo Datasets")
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--results-root", type=str, default="./runs/classification", help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASETS.keys()),
                        help="Dataset to use (default: all)")
    parser.add_argument("--models", type=str, nargs="*", default=None, choices=ALL_MODELS,
                        help="Models to train (default: all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached runs")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")

    args = parser.parse_args()

    if args.plot_only:
        plot_classification_results(args.results_root)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()

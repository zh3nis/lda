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
from models import SoftmaxClassifier, SimplexLDAClassifier, TrainableLDAClassifier
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
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

            if os.path.exists(out_path) and not args.overwrite:
                print(f"Skipping {name} seed={seed} (cached)")
                continue

            print(f"\n=== {name} | seed={seed} ===")
            seed_all(seed)

            # Build models
            softmax_model = SoftmaxClassifier(in_ch, C, D).to(device)
            simplex_model = SimplexLDAClassifier(in_ch, C, D).to(device)
            trainable_model = TrainableLDAClassifier(in_ch, C, D).to(device)

            opt_soft = torch.optim.Adam(softmax_model.parameters(), lr=args.lr)
            opt_simp = torch.optim.Adam(simplex_model.parameters(), lr=args.lr)
            opt_trn = torch.optim.Adam(trainable_model.parameters(), lr=args.lr)

            ce_loss = nn.CrossEntropyLoss().to(device)
            dnll_loss = DNLLLoss(lambda_reg=1.0).to(device)

            hist = {
                "softmax": {"train_acc": [], "test_acc": []},
                "simplex_lda": {"train_acc": [], "test_acc": []},
                "trainable_lda": {"train_acc": [], "test_acc": []},
            }

            for epoch in range(1, args.epochs + 1):
                # Train all models
                tr_soft = train_classification_epoch(softmax_model, train_loader, opt_soft, ce_loss, device)
                te_soft = evaluate_classification(softmax_model, test_loader, device)

                tr_simp = train_classification_epoch(simplex_model, train_loader, opt_simp, dnll_loss, device)
                te_simp = evaluate_classification(simplex_model, test_loader, device)

                tr_trn = train_classification_epoch(trainable_model, train_loader, opt_trn, dnll_loss, device)
                te_trn = evaluate_classification(trainable_model, test_loader, device)

                # Record history
                hist["softmax"]["train_acc"].append(tr_soft)
                hist["softmax"]["test_acc"].append(te_soft)
                hist["simplex_lda"]["train_acc"].append(tr_simp)
                hist["simplex_lda"]["test_acc"].append(te_simp)
                hist["trainable_lda"]["train_acc"].append(tr_trn)
                hist["trainable_lda"]["test_acc"].append(te_trn)

                print(
                    f"[Epoch {epoch:02d}] "
                    f"Softmax tr={tr_soft:.3f} te={te_soft:.3f} | "
                    f"Simplex tr={tr_simp:.3f} te={te_simp:.3f} | "
                    f"Trainable tr={tr_trn:.3f} te={te_trn:.3f}"
                )

            save_json(out_path, {
                "seed": seed,
                "epochs": args.epochs,
                "history": hist,
                "dataset": name,
                "timestamp": time.time(),
            })
            print(f"Saved {out_path}")

            # Save models
            models_dir = os.path.join(seed_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            torch.save(softmax_model.state_dict(), os.path.join(models_dir, "softmax.pth"))
            torch.save(simplex_model.state_dict(), os.path.join(models_dir, "simplex_lda.pth"))
            torch.save(trainable_model.state_dict(), os.path.join(models_dir, "trainable_lda.pth"))
            print(f"Saved models to {models_dir}")

    # Plot results
    print("\nGenerating plots...")
    plot_classification_results(results_root)


def main():
    parser = argparse.ArgumentParser(description="LDA Classification Experiments on TorchGeo Datasets")
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--results-root", type=str, default="./runs/classification", help="Output directory")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASETS.keys()),
                        help="Dataset to use (default: all)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
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

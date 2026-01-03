"""
Tabular classification experiments with LDA heads.

Usage:
    python train_tabular.py --data ./datasets/LU22_tabular.csv --epochs 30 --seeds 42 123 456
    python train_tabular.py --plot-only --results-root ./runs/tabular
"""
import os
import time
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.lda import DNLLLoss
from models import SoftmaxTabular, SimplexLDATabular, TrainableLDATabular
from utils import (
    seed_all,
    save_json,
    train_classification_epoch,
    evaluate_classification,
    plot_tabular_results,
    plot_tabular_embeddings,
)


# ==============================================================================
# Dataset
# ==============================================================================
class TabularDataset(Dataset):
    """PyTorch Dataset for tabular classification data."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_tabular_data(data_path: str, label_col: str, test_size: float = 0.2, seed: int = 42):
    """Load and preprocess tabular data."""
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != label_col]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "scaler": scaler,
    }


# ==============================================================================
# Main Experiment
# ==============================================================================
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_root = os.path.abspath(args.results_root)
    os.makedirs(results_root, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    for seed in args.seeds:
        seed_dir = os.path.join(results_root, dataset_name, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        out_path = os.path.join(seed_dir, "acc_history.json")

        if os.path.exists(out_path) and not args.overwrite:
            print(f"Skipping {dataset_name} seed={seed} (cached)")
            continue

        print(f"\n=== {dataset_name} | seed={seed} ===")
        seed_all(seed)

        # Load data
        data = load_tabular_data(args.data, args.label_col, args.test_size, seed)
        
        train_ds = TabularDataset(data["X_train"], data["y_train"])
        test_ds = TabularDataset(data["X_test"], data["y_test"])
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        C = data["num_classes"]
        input_dim = data["input_dim"]
        D = C  # Embedding dimension

        print(f"Classes: {C}, Features: {input_dim}, Train: {len(train_ds)}, Test: {len(test_ds)}")

        # Build models
        softmax_model = SoftmaxTabular(input_dim, C, D, args.hidden_dims, args.dropout).to(device)
        simplex_model = SimplexLDATabular(input_dim, C, D, args.hidden_dims, args.dropout).to(device)
        trainable_model = TrainableLDATabular(input_dim, C, D, args.hidden_dims, args.dropout).to(device)

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
            tr_soft = train_classification_epoch(softmax_model, train_loader, opt_soft, ce_loss, device)
            te_soft = evaluate_classification(softmax_model, test_loader, device)

            tr_simp = train_classification_epoch(simplex_model, train_loader, opt_simp, dnll_loss, device)
            te_simp = evaluate_classification(simplex_model, test_loader, device)

            tr_trn = train_classification_epoch(trainable_model, train_loader, opt_trn, dnll_loss, device)
            te_trn = evaluate_classification(trainable_model, test_loader, device)

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
            "dataset": dataset_name,
            "timestamp": time.time(),
        })
        print(f"Saved {out_path}")

        # Save models for embedding plots
        models_dir = os.path.join(seed_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(softmax_model.state_dict(), os.path.join(models_dir, "softmax.pt"))
        torch.save(simplex_model.state_dict(), os.path.join(models_dir, "simplex_lda.pt"))
        torch.save(trainable_model.state_dict(), os.path.join(models_dir, "trainable_lda.pt"))

        # Plot embeddings for this seed
        plot_tabular_embeddings(
            {"Softmax": softmax_model, "Simplex LDA": simplex_model, "Trainable LDA": trainable_model},
            train_loader, C, device,
            save_path=os.path.join(seed_dir, "embeddings.png")
        )

    # Plot results
    print("\nGenerating plots...")
    plot_tabular_results(results_root)


def main():
    parser = argparse.ArgumentParser(description="LDA Tabular Classification Experiments")
    parser.add_argument("--data", type=str, default="data/LU22_tabular.csv", help="Path to CSV data file")
    parser.add_argument("--label-col", type=str, default="Label_clas", help="Label column name")
    parser.add_argument("--results-root", type=str, default="./runs/tabular", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached runs")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")

    args = parser.parse_args()

    if args.plot_only:
        plot_tabular_results(args.results_root)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()

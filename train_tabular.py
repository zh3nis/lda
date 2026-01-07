"""
Tabular classification experiments with LDA heads.

Usage:
    python train_tabular.py --data ./datasets/LU22_tabular.csv --epochs 30 --seeds 42 123 456 --dnll-lambda 1.0
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
from models import SoftmaxTabular, SimplexLDATabular, TrainableLDATabular, TrainableLDASphericalTabular
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
ALL_MODELS = ["softmax", "simplex_lda", "trainable_lda", "trainable_lda_spherical"]


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select models to train
    models_to_train = args.models if args.models else ALL_MODELS
    print(f"Models to train: {models_to_train}")

    results_root = os.path.abspath(args.results_root)
    os.makedirs(results_root, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    for seed in args.seeds:
        seed_dir = os.path.join(results_root, dataset_name, f"seed_{seed}")
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

        ce_loss = nn.CrossEntropyLoss().to(device)
        dnll_loss = DNLLLoss(lambda_reg=args.dnll_lambda).to(device)

        # Model configs: (class, loss_fn)
        model_configs = {
            "softmax": (SoftmaxTabular, ce_loss),
            "simplex_lda": (SimplexLDATabular, dnll_loss),
            "trainable_lda": (TrainableLDATabular, dnll_loss),
            "trainable_lda_spherical": (TrainableLDASphericalTabular, dnll_loss),
        }

        models_dir = os.path.join(seed_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        trained_models = {}
        for model_name in models_to_train:
            if model_name not in model_configs:
                print(f"Unknown model: {model_name}")
                continue

            # Skip if already trained and not overwriting
            if model_name in hist and not args.overwrite:
                print(f"  Skipping {model_name} (cached)")
                continue

            model_cls, loss_fn = model_configs[model_name]
            model = model_cls(input_dim, C, D, args.hidden_dims, args.dropout).to(device)
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
            torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pt"))
            trained_models[model_name] = model

        save_json(out_path, {
            "seed": seed,
            "epochs": args.epochs,
            "history": hist,
            "dataset": dataset_name,
            "dnll_lambda": args.dnll_lambda,
            "timestamp": time.time(),
        })
        print(f"Saved {out_path}")

        # Plot embeddings for this seed - load ALL saved models for complete plot
        name_map = {"softmax": "Softmax", "simplex_lda": "Simplex LDA", 
                   "trainable_lda": "Trainable LDA", "trainable_lda_spherical": "Trainable LDA (Sph)"}
        plot_models = {}
        for model_name in ALL_MODELS:
            model_path = os.path.join(models_dir, f"{model_name}.pt")
            if os.path.exists(model_path):
                model_cls, _ = model_configs[model_name]
                model = model_cls(input_dim, C, D, args.hidden_dims, args.dropout).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                plot_models[name_map.get(model_name, model_name)] = model
        
        if plot_models:
            plot_tabular_embeddings(
                plot_models, train_loader, C, device,
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
    parser.add_argument("--models", type=str, nargs="*", default=None, choices=ALL_MODELS,
                        help="Models to train (default: all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached runs")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")
    parser.add_argument("--dnll-lambda", dest="dnll_lambda", type=float, default=1.0,
                        help="DNLL loss lambda regularization strength (default: 1.0)")

    args = parser.parse_args()

    if args.plot_only:
        plot_tabular_results(args.results_root)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()

"""
Calibration experiments for classification models.

Generates confidence histograms and reliability diagrams similar to
Guo et al. "On Calibration of Modern Neural Networks" (2017).

Usage:    python calibration_experiment.py --dataset RESISC45 --seed 42    python calibration_experiment.py --dataset EuroSAT --seed 42
"""
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torchgeo.datasets import EuroSAT, RESISC45, UCMerced

from models import SoftmaxClassifier, SimplexLDAClassifier, TrainableLDAClassifier


DATASETS = {
    "EuroSAT": EuroSAT,
    "RESISC45": RESISC45,
    "UCMerced": UCMerced,
}


# ==============================================================================
# Data Utilities (from train_classification.py)
# ==============================================================================
class TorchGeoWrapper:
    """Simple wrapper for TorchGeo datasets."""

    def __init__(self, base_ds, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        img = sample["image"] if isinstance(sample, dict) else sample[0]
        label = sample["label"] if isinstance(sample, dict) else sample[1]

        if not torch.is_tensor(img):
            img = T.functional.to_tensor(img)
        img = img.float()

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)


def get_test_loader(data_root: str, dataset_name: str = "RESISC45", batch_size: int = 128):
    """Create test dataloader for specified dataset."""
    dataset_cls = DATASETS[dataset_name]
    ds_root = f"{data_root}/{dataset_name}"
    base_ds = dataset_cls(root=ds_root, download=True, transforms=None)

    # Get input channels
    sample = base_ds[0]
    img = sample["image"] if isinstance(sample, dict) else sample[0]
    if not torch.is_tensor(img):
        img = T.functional.to_tensor(img)
    in_channels = img.shape[0]

    mean = [0.5] * in_channels
    std = [0.5] * in_channels

    test_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Resize(128),
        T.Normalize(mean, std),
    ])

    # Use same split as training (seed=42)
    n = len(base_ds)
    n_train = int(0.8 * n)
    n_test = n - n_train
    _, test_base = random_split(
        base_ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    test_ds = TorchGeoWrapper(test_base, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(base_ds.classes)
    return test_loader, num_classes, in_channels


# ==============================================================================
# Calibration Metrics
# ==============================================================================
def get_predictions(model, loader, device):
    """Get predictions, confidences, and labels from model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    return confidences, accuracies, predictions, labels


def compute_calibration(confidences, accuracies, n_bins=15):
    """Compute calibration metrics (ECE and per-bin stats)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.mean()
        bin_counts.append(in_bin.sum())

        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_acc - avg_conf)
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
        else:
            bin_accs.append(0)
            bin_confs.append(0)

    return {
        "ece": ece * 100,  # percentage
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "bin_edges": bin_boundaries,
    }


# ==============================================================================
# Plotting
# ==============================================================================
def plot_calibration(results_dict, save_path, dataset_name="RESISC45"):
    """
    Plot confidence histograms and reliability diagrams for all models.
    
    Similar to Figure 1 in Guo et al. (2017).
    """
    models = list(results_dict.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 6))
    
    for i, model_name in enumerate(models):
        res = results_dict[model_name]
        confidences = res["confidences"]
        accuracies = res["accuracies"]
        cal = res["calibration"]
        
        n_bins = len(cal["bin_accs"])
        bin_edges = cal["bin_edges"]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / n_bins
        
        # Top row: Confidence histogram
        ax = axes[0, i]
        ax.hist(confidences, bins=bin_edges, density=True, alpha=0.7, color="blue", edgecolor="black")
        
        # Add accuracy and avg confidence lines
        avg_acc = accuracies.mean()
        avg_conf = confidences.mean()
        ax.axvline(avg_acc, color="gray", linestyle="--", linewidth=1.5, label=f"Accuracy")
        ax.axvline(avg_conf, color="gray", linestyle=":", linewidth=1.5, label=f"Avg. conf")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel("Confidence")
        ax.set_ylabel("% of Samples")
        ax.set_title(f"{model_name}\n{dataset_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Reliability diagram
        ax = axes[1, i]
        
        # Plot bars for accuracy
        ax.bar(bin_centers, cal["bin_accs"], width=bin_width * 0.9, 
               alpha=0.7, color="blue", edgecolor="black", label="Outputs")
        
        # Plot gap (difference between confidence and accuracy)
        gaps = cal["bin_confs"] - cal["bin_accs"]
        # Only show positive gaps (overconfident)
        for j, (center, acc, gap) in enumerate(zip(bin_centers, cal["bin_accs"], gaps)):
            if gap > 0 and cal["bin_counts"][j] > 0:
                ax.bar(center, gap, bottom=acc, width=bin_width * 0.9,
                       alpha=0.3, color="red", edgecolor="red", hatch="//")
        
        # Add diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7)
        
        # Add ECE text
        ax.text(0.05, 0.9, f"ECE={cal['ece']:.1f}%", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        
        # Custom legend for bottom plot
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, edgecolor="black", label="Outputs"),
            Patch(facecolor="red", alpha=0.3, edgecolor="red", hatch="//", label="Gap"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot: {save_path}")


def plot_calibration_all_datasets(all_results, save_path):
    """
    Plot calibration for all datasets in a compact grid.
    
    Layout: 6 rows × 3 columns
    - Rows 0,2,4: Confidence histograms for each dataset
    - Rows 1,3,5: Reliability diagrams for each dataset
    - Columns: Softmax, Simplex LDA, Trainable LDA
    """
    from matplotlib.patches import Patch
    
    # Fixed order: EuroSAT, UCMerced, RESISC45
    dataset_order = ["EuroSAT", "UCMerced", "RESISC45"]
    datasets = [d for d in dataset_order if d in all_results]
    models = list(all_results[datasets[0]].keys())
    n_datasets = len(datasets)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_datasets * 2, n_models, figsize=(3.5 * n_models, 2.2 * n_datasets * 2))
    
    for d_idx, dataset_name in enumerate(datasets):
        results_dict = all_results[dataset_name]
        
        for m_idx, model_name in enumerate(models):
            res = results_dict[model_name]
            confidences = res["confidences"]
            accuracies = res["accuracies"]
            cal = res["calibration"]
            
            n_bins = len(cal["bin_accs"])
            bin_edges = cal["bin_edges"]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = 1.0 / n_bins
            
            # ---- Confidence histogram (even rows: 0, 2, 4) ----
            ax = axes[d_idx * 2, m_idx]
            ax.hist(confidences, bins=bin_edges, density=True, alpha=0.7, color="mediumpurple", edgecolor="black", linewidth=0.5)
            
            avg_acc = accuracies.mean()
            avg_conf = confidences.mean()
            ax.axvline(avg_acc, color="gray", linestyle="--", linewidth=1.5)
            ax.axvline(avg_conf, color="gray", linestyle=":", linewidth=1.5)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.grid(True, alpha=0.3)
            
            # Title only on first row
            if d_idx == 0:
                ax.set_title(model_name, fontsize=11)
            
            # Dataset label on left
            if m_idx == 0:
                ax.set_ylabel(f"{dataset_name}\n% of Samples", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            
            # X-axis label only on last histogram row before reliability
            ax.set_xticklabels([])
            
            # ---- Reliability diagram (odd rows: 1, 3, 5) ----
            ax = axes[d_idx * 2 + 1, m_idx]
            
            ax.bar(bin_centers, cal["bin_accs"], width=bin_width, 
                   alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
            
            gaps = cal["bin_confs"] - cal["bin_accs"]
            for j, (center, acc, gap) in enumerate(zip(bin_centers, cal["bin_accs"], gaps)):
                if gap > 0 and cal["bin_counts"][j] > 0:
                    ax.bar(center, gap, bottom=acc, width=bin_width,
                           alpha=0.4, color="salmon", edgecolor="darkred", linewidth=0.5, hatch="//")
            
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.7)
            
            # ECE text
            ax.text(0.05, 0.92, f"ECE={cal['ece']:.1f}%", transform=ax.transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Y-axis label only on left column
            if m_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=9)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            
            # X-axis label only on bottom row
            if d_idx == n_datasets - 1:
                ax.set_xlabel("Confidence", fontsize=9)
            else:
                ax.set_xticklabels([])
    
    # Add legend at the bottom
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, edgecolor="black", label="Accuracy"),
        Patch(facecolor="salmon", alpha=0.4, edgecolor="darkred", hatch="//", label="Gap (overconfident)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Accuracy"),
        plt.Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5, label="Avg. confidence"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, 
               bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined calibration plot: {save_path}")


# ==============================================================================
# Main
# ==============================================================================
def get_available_seeds(dataset_name):
    """Find all available seeds with trained models."""
    base_dir = f"./runs/classification/{dataset_name}"
    seeds = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith("seed_"):
                seed = int(d.split("_")[1])
                model_path = os.path.join(base_dir, d, "models", "softmax.pth")
                if os.path.exists(model_path):
                    seeds.append(seed)
    return sorted(seeds)


def run_single_dataset(dataset_name, data_root, seed, device):
    """Run calibration for a single dataset and seed."""
    test_loader, num_classes, in_channels = get_test_loader(data_root, dataset_name)
    D = num_classes - 1

    models_dir = os.path.join(f"./runs/classification/{dataset_name}", f"seed_{seed}", "models")
    
    model_configs = [
        ("Softmax", SoftmaxClassifier, "softmax.pth"),
        ("Simplex LDA", SimplexLDAClassifier, "simplex_lda.pth"),
        ("Trainable LDA", TrainableLDAClassifier, "trainable_lda.pth"),
    ]

    results = {}
    for name, cls, filename in model_configs:
        model = cls(in_channels, num_classes, D).to(device)
        model_path = os.path.join(models_dir, filename)
        model.load_state_dict(torch.load(model_path, map_location=device))

        confidences, accuracies, preds, labels = get_predictions(model, test_loader, device)
        calibration = compute_calibration(confidences, accuracies, n_bins=15)

        results[name] = {
            "confidences": confidences,
            "accuracies": accuracies,
            "predictions": preds,
            "labels": labels,
            "calibration": calibration,
        }

    return results, num_classes, in_channels


def run_dataset_averaged(dataset_name, data_root, device):
    """Run calibration for a dataset, averaged across all available seeds."""
    seeds = get_available_seeds(dataset_name)
    print(f"{dataset_name}: Found seeds {seeds}")
    
    if not seeds:
        raise ValueError(f"No trained models found for {dataset_name}")
    
    all_seed_results = []
    for seed in seeds:
        results, _, _ = run_single_dataset(dataset_name, data_root, seed, device)
        all_seed_results.append(results)
    
    # Average results across seeds
    models = list(all_seed_results[0].keys())
    averaged_results = {}
    
    for model_name in models:
        # Collect across seeds
        all_confidences = np.concatenate([r[model_name]["confidences"] for r in all_seed_results])
        all_accuracies = np.concatenate([r[model_name]["accuracies"] for r in all_seed_results])
        
        # Average ECE and bin stats
        avg_ece = np.mean([r[model_name]["calibration"]["ece"] for r in all_seed_results])
        avg_bin_accs = np.mean([r[model_name]["calibration"]["bin_accs"] for r in all_seed_results], axis=0)
        avg_bin_confs = np.mean([r[model_name]["calibration"]["bin_confs"] for r in all_seed_results], axis=0)
        avg_bin_counts = np.mean([r[model_name]["calibration"]["bin_counts"] for r in all_seed_results], axis=0)
        bin_edges = all_seed_results[0][model_name]["calibration"]["bin_edges"]
        
        averaged_results[model_name] = {
            "confidences": all_confidences,
            "accuracies": all_accuracies,
            "calibration": {
                "ece": avg_ece,
                "bin_accs": avg_bin_accs,
                "bin_confs": avg_bin_confs,
                "bin_counts": avg_bin_counts,
                "bin_edges": bin_edges,
            },
        }
        print(f"  {model_name}: Avg Accuracy={all_accuracies.mean()*100:.1f}%, Avg ECE={avg_ece:.1f}%")
    
    return averaged_results


def main():
    parser = argparse.ArgumentParser(description="Calibration experiments")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="RESISC45", choices=list(DATASETS.keys()) + ["all"],
                        help="Dataset to use (or 'all' for combined plot)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed (default: average all available)")
    parser.add_argument("--output-dir", type=str, default="./plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "all":
        # Run all datasets and create combined plot
        all_results = {}
        for dataset_name in DATASETS.keys():
            print(f"\n=== {dataset_name} ===")
            if args.seed is not None:
                results, _, _ = run_single_dataset(dataset_name, args.data_root, args.seed, device)
                all_results[dataset_name] = results
            else:
                all_results[dataset_name] = run_dataset_averaged(dataset_name, args.data_root, device)
        
        save_path = os.path.join(args.output_dir, "all_calibration.png")
        plot_calibration_all_datasets(all_results, save_path)
    else:
        # Single dataset
        if args.seed is not None:
            results, _, _ = run_single_dataset(args.dataset, args.data_root, args.seed, device)
        else:
            print(f"\n=== {args.dataset} ===")
            results = run_dataset_averaged(args.dataset, args.data_root, device)
        save_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_calibration.png")
        plot_calibration(results, save_path, args.dataset)


if __name__ == "__main__":
    main()

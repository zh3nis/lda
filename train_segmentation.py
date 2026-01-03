"""
Segmentation experiments with LDA heads.

Datasets:
  - Hyperspectral: IndianPines, Salinas (from .mat files)
  - TorchGeo: LoveDA

Usage:
    # Run Indian Pines experiment
    python train_segmentation.py --dataset IndianPines --epochs 30
    
    # Run Salinas experiment
    python train_segmentation.py --dataset Salinas --epochs 30
    
    # Run TorchGeo dataset (LoveDA or GID15)
    python train_segmentation.py --dataset LoveDA --epochs 30
    
    # Plot existing results only
    python train_segmentation.py --plot-only --results-root ./runs/segmentation
"""
import os
import time
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.lda import DNLLLoss
from models import (
    SoftmaxSegmentation, SimplexLDASegmentation, TrainableLDASegmentation,
    SoftmaxFCNSegmentation, SimplexLDAFCNSegmentation, TrainableLDAFCNSegmentation,
)
from utils import (
    seed_all,
    save_json,
    train_segmentation_epoch,
    evaluate_segmentation,
    SegmentationLossWrapper,
    plot_segmentation_results,
    save_segmentation_comparison_plot,
)


IGNORE_INDEX = 0  # Background class for hyperspectral datasets


# ==============================================================================
# Hyperspectral Dataset Utilities (Indian Pines, Salinas)
# ==============================================================================
def load_hyperspectral_mat(data_dir: str, dataset: str, train_ratio: float = 0.7) -> Dict[str, Any]:
    """Load hyperspectral dataset from .mat files."""
    from scipy.io import loadmat
    
    if dataset == "IndianPines":
        data_mat = loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))
        gt_mat = loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))
        img = data_mat['indian_pines_corrected'].astype(np.float32).transpose(2, 0, 1)
        labels = gt_mat['indian_pines_gt'].astype(np.int64)
        classes = [
            "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
            "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat",
            "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
        ]
    elif dataset == "Salinas":
        data_mat = loadmat(os.path.join(data_dir, 'Salinas_corrected.mat'))
        gt_mat = loadmat(os.path.join(data_dir, 'Salinas_gt.mat'))
        img = data_mat['salinas_corrected'].astype(np.float32).transpose(2, 0, 1)
        labels = gt_mat['salinas_gt'].astype(np.int64)
        classes = [
            "Background", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
            "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained",
            "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            "Vinyard_untrained", "Vinyard_vertical_trellis"
        ]
    else:
        raise ValueError(f"Unknown hyperspectral dataset: {dataset}")

    # Create mask and split
    mask = (labels > 0).astype(np.uint8)
    valid_indices = np.where(mask > 0)
    num_valid = len(valid_indices[0])

    perm = np.random.permutation(num_valid)
    n_train = int(num_valid * train_ratio)

    train_indices = (valid_indices[0][perm[:n_train]], valid_indices[1][perm[:n_train]])
    val_indices = (valid_indices[0][perm[n_train:]], valid_indices[1][perm[n_train:]])

    train_mask = np.zeros_like(mask)
    train_mask[train_indices] = 1

    val_mask = np.zeros_like(mask)
    val_mask[val_indices] = 1

    return {
        "img": img,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "classes": classes,
    }


def compute_band_stats(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for each spectral band."""
    valid = mask > 0
    means, stds = [], []
    for c in range(img.shape[0]):
        vals = img[c][valid]
        means.append(float(vals.mean()))
        stds.append(float(vals.std() + 1e-6))
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


class HyperspectralRandomCrop(Dataset):
    """Random crop dataset for hyperspectral images."""

    def __init__(self, img: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                 crop_size: int, num_samples: int, mean: np.ndarray, std: np.ndarray,
                 hflip_prob: float = 0.5, vflip_prob: float = 0.5):
        self.img = img
        self.labels = labels
        self.mask = mask
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.h, self.w = labels.shape

    def __len__(self):
        return self.num_samples

    def _sample_coords(self):
        for _ in range(20):
            y = np.random.randint(0, self.h - self.crop_size + 1)
            x = np.random.randint(0, self.w - self.crop_size + 1)
            if self.mask[y:y + self.crop_size, x:x + self.crop_size].sum() > 0:
                return y, x
        return 0, 0

    def __getitem__(self, idx):
        y, x = self._sample_coords()
        img_patch = self.img[:, y:y + self.crop_size, x:x + self.crop_size].copy()
        label_patch = self.labels[y:y + self.crop_size, x:x + self.crop_size].copy()
        mask_patch = self.mask[y:y + self.crop_size, x:x + self.crop_size]
        label_patch[mask_patch == 0] = IGNORE_INDEX

        if np.random.rand() < self.hflip_prob:
            img_patch = img_patch[:, :, ::-1].copy()
            label_patch = label_patch[:, ::-1].copy()
        if np.random.rand() < self.vflip_prob:
            img_patch = img_patch[:, ::-1, :].copy()
            label_patch = label_patch[::-1, :].copy()

        img_norm = (img_patch - self.mean) / self.std
        return torch.from_numpy(img_norm).float(), torch.from_numpy(label_patch).long()


class HyperspectralSlidingWindow(Dataset):
    """Sliding window dataset for hyperspectral images (validation)."""

    def __init__(self, img: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                 crop_size: int, stride: int, mean: np.ndarray, std: np.ndarray,
                 skip_empty: bool = True):
        self.img = img
        self.labels = labels
        self.mask = mask
        self.crop_size = crop_size
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

        h, w = labels.shape
        positions: List[Tuple[int, int]] = []
        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                m_patch = mask[y:y + crop_size, x:x + crop_size]
                if skip_empty and m_patch.sum() == 0:
                    continue
                positions.append((y, x))
        if not positions:
            positions.append((0, 0))
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        y, x = self.positions[idx]
        img_patch = self.img[:, y:y + self.crop_size, x:x + self.crop_size]
        label_patch = self.labels[y:y + self.crop_size, x:x + self.crop_size].copy()
        mask_patch = self.mask[y:y + self.crop_size, x:x + self.crop_size]
        label_patch[mask_patch == 0] = IGNORE_INDEX
        img_norm = (img_patch - self.mean) / self.std
        return torch.from_numpy(img_norm).float(), torch.from_numpy(label_patch).long()


def make_hyperspectral_loaders(data_dir: str, dataset: str, batch_size: int, num_workers: int,
                                crop_size: int, train_samples: int, val_stride: int,
                                train_ratio: float) -> Dict[str, Any]:
    """Create dataloaders for hyperspectral datasets."""
    data = load_hyperspectral_mat(data_dir, dataset, train_ratio)
    img = data["img"]
    labels = data["labels"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]

    band_mean, band_std = compute_band_stats(img, train_mask)

    train_ds = HyperspectralRandomCrop(img, labels, train_mask, crop_size, train_samples,
                                        band_mean, band_std, hflip_prob=0.5, vflip_prob=0.5)
    val_ds = HyperspectralSlidingWindow(img, labels, val_mask, crop_size, val_stride, band_mean, band_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    num_classes = int(labels.max()) + 1

    return {
        "name": dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_dataset": val_ds,
        "num_classes": num_classes,
        "in_channels": img.shape[0],
        "ignore_index": IGNORE_INDEX,
        "classes": data.get("classes", None),
        "raw_img": img,
        "labels_full": labels,
    }


# ==============================================================================
# TorchGeo Dataset Utilities (LoveDA, GID15)
# ==============================================================================
TORCHGEO_CONFIGS = {
    "LoveDA": {
        "num_classes": 8,
        "ignore_index": 0,
        "splits": {"train": "train", "val": "val"},
        "in_channels": 3,
    },
}


class TorchGeoSegmentationTransform:
    """Apply paired transforms to image/mask."""

    def __init__(self, resize: int = 256, hflip_prob: float = 0.5, mean: float = 0.5, std: float = 0.5):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = sample["image"]
        mask = sample["mask"]

        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        img = TF.resize(img, self.resize, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask.unsqueeze(0), self.resize, interpolation=InterpolationMode.NEAREST).squeeze(0)

        if torch.rand(1).item() < self.hflip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.normalize(img.float(), [self.mean] * img.shape[0], [self.std] * img.shape[0])
        mask = mask.long()

        sample["image"] = img
        sample["mask"] = mask
        return sample


class TorchGeoSegmentationDataset(Dataset):
    """Wrapper for TorchGeo segmentation datasets."""

    def __init__(self, base_ds, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample["image"].float(), sample["mask"].long()


def make_torchgeo_loaders(dataset_name: str, data_root: str, batch_size: int,
                          num_workers: int) -> Dict[str, Any]:
    """Create dataloaders for TorchGeo segmentation datasets."""
    from torchgeo.datasets import LoveDA
    
    dataset_classes = {"LoveDA": LoveDA}
    
    config = TORCHGEO_CONFIGS[dataset_name]
    dataset_cls = dataset_classes[dataset_name]
    ds_root = f"{data_root}/{dataset_name}"

    train_split = config["splits"]["train"]
    val_split = config["splits"]["val"]

    train_base = dataset_cls(root=ds_root, split=train_split, download=True, transforms=None)
    val_base = dataset_cls(root=ds_root, split=val_split, download=True, transforms=None)

    train_transform = TorchGeoSegmentationTransform(resize=256, hflip_prob=0.5)
    test_transform = TorchGeoSegmentationTransform(resize=256, hflip_prob=0.0)

    train_ds = TorchGeoSegmentationDataset(train_base, transform=train_transform)
    val_ds = TorchGeoSegmentationDataset(val_base, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return {
        "name": dataset_name,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_dataset": val_ds,
        "base_dataset": val_base,
        "num_classes": config["num_classes"],
        "in_channels": config["in_channels"],
        "ignore_index": config["ignore_index"],
    }


# ==============================================================================
# Main Experiment
# ==============================================================================
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine dataset type and create loaders
    hyperspectral_datasets = ["IndianPines", "Salinas"]
    torchgeo_datasets = list(TORCHGEO_CONFIGS.keys())

    if args.dataset in hyperspectral_datasets:
        info = make_hyperspectral_loaders(
            data_dir=args.data_dir,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            crop_size=args.crop_size,
            train_samples=args.train_samples,
            val_stride=args.val_stride,
            train_ratio=args.train_ratio,
        )
    elif args.dataset in torchgeo_datasets:
        info = make_torchgeo_loaders(
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Options: {hyperspectral_datasets + torchgeo_datasets}")

    results_root = os.path.abspath(args.results_root)
    os.makedirs(results_root, exist_ok=True)

    name = info["name"]
    train_loader = info["train_loader"]
    val_loader = info["val_loader"]
    val_dataset = info["val_dataset"]
    C = info["num_classes"]
    in_ch = info["in_channels"]
    ignore_idx = info["ignore_index"]

    print(f"Dataset: {name}, Classes: {C}, Input channels: {in_ch}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for seed in args.seeds:
        seed_all(seed)
        seed_dir = os.path.join(results_root, name, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        models_dir = os.path.join(seed_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        softmax_path = os.path.join(models_dir, "softmax.pth")
        simplex_path = os.path.join(models_dir, "simplex_lda.pth")
        trainable_path = os.path.join(models_dir, "trainable_lda.pth")
        out_path = os.path.join(seed_dir, "metrics_history.json")

        models_exist = all(os.path.exists(p) for p in [softmax_path, simplex_path, trainable_path])
        training_cached = os.path.exists(out_path) and not args.overwrite

        if training_cached:
            print(f"\n=== {name} | seed={seed} (cached) ===")
        else:
            print(f"\n=== {name} | seed={seed} ===")

        # Build models - use simple FCN for Indian Pines, UNet for others
        if args.dataset == "IndianPines":
            softmax_model = SoftmaxFCNSegmentation(in_ch, C, base_ch=args.base_channels).to(device)
            simplex_model = SimplexLDAFCNSegmentation(in_ch, C, base_ch=args.base_channels).to(device)
            trainable_model = TrainableLDAFCNSegmentation(in_ch, C, base_ch=args.base_channels).to(device)
        else:
            softmax_model = SoftmaxSegmentation(in_ch, C, base_ch=args.base_channels).to(device)
            simplex_model = SimplexLDASegmentation(in_ch, C, base_ch=args.base_channels).to(device)
            trainable_model = TrainableLDASegmentation(in_ch, C, base_ch=args.base_channels).to(device)

        if (models_exist or training_cached) and not args.overwrite:
            print("Loading cached models...")
            softmax_model.load_state_dict(torch.load(softmax_path, map_location=device))
            simplex_model.load_state_dict(torch.load(simplex_path, map_location=device))
            trainable_model.load_state_dict(torch.load(trainable_path, map_location=device))
        else:
            # Train models
            opt_soft = torch.optim.Adam(softmax_model.parameters(), lr=args.lr)
            opt_simp = torch.optim.Adam(simplex_model.parameters(), lr=args.lr)
            opt_trn = torch.optim.Adam(trainable_model.parameters(), lr=args.lr)

            ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_idx).to(device)
            dnll_wrapped = SegmentationLossWrapper(DNLLLoss(lambda_reg=1.0), ignore_index=ignore_idx).to(device)

            hist = {
                "softmax": {"train_acc": [], "train_miou": [], "val_acc": [], "val_miou": []},
                "simplex_lda": {"train_acc": [], "train_miou": [], "val_acc": [], "val_miou": []},
                "trainable_lda": {"train_acc": [], "train_miou": [], "val_acc": [], "val_miou": []},
            }

            for epoch in range(1, args.epochs + 1):
                # Train
                tr_soft = train_segmentation_epoch(softmax_model, train_loader, opt_soft, ce_loss, device, C, ignore_idx)
                val_soft = evaluate_segmentation(softmax_model, val_loader, device, C, ignore_idx)

                tr_simp = train_segmentation_epoch(simplex_model, train_loader, opt_simp, dnll_wrapped, device, C, ignore_idx)
                val_simp = evaluate_segmentation(simplex_model, val_loader, device, C, ignore_idx)

                tr_trn = train_segmentation_epoch(trainable_model, train_loader, opt_trn, dnll_wrapped, device, C, ignore_idx)
                val_trn = evaluate_segmentation(trainable_model, val_loader, device, C, ignore_idx)

                # Record
                for model_name, tr, val in [("softmax", tr_soft, val_soft),
                                             ("simplex_lda", tr_simp, val_simp),
                                             ("trainable_lda", tr_trn, val_trn)]:
                    hist[model_name]["train_acc"].append(tr["acc"])
                    hist[model_name]["train_miou"].append(tr["miou"])
                    hist[model_name]["val_acc"].append(val["acc"])
                    hist[model_name]["val_miou"].append(val["miou"])

                print(
                    f"[Epoch {epoch:02d}] "
                    f"Softmax tr={tr_soft['miou']:.3f} val={val_soft['miou']:.3f} | "
                    f"Simplex tr={tr_simp['miou']:.3f} val={val_simp['miou']:.3f} | "
                    f"Trainable tr={tr_trn['miou']:.3f} val={val_trn['miou']:.3f}"
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
            torch.save(softmax_model.state_dict(), softmax_path)
            torch.save(simplex_model.state_dict(), simplex_path)
            torch.save(trainable_model.state_dict(), trainable_path)
            print(f"Saved models to {models_dir}")

        # Generate comparison plot
        comparison_path = os.path.join(seed_dir, "comparison.png")
        models_dict = {
            "Softmax": softmax_model,
            "Simplex LDA": simplex_model,
            "Trainable LDA": trainable_model,
        }

        # Define sample getter based on dataset type
        if args.dataset in hyperspectral_datasets:
            raw_img = info["raw_img"]
            labels_full = info["labels_full"]
            rgb_bands = [50, 27, 17]

            def get_sample(idx):
                y, x = val_dataset.positions[idx]
                cs = val_dataset.crop_size
                raw_patch = raw_img[:, y:y + cs, x:x + cs]
                rgb = raw_patch[rgb_bands]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                rgb_np = np.moveaxis(rgb, 0, 2)
                img_tensor, _ = val_dataset[idx]
                # Use full, unmasked labels for visualization
                mask_gt_full = labels_full[y:y + cs, x:x + cs]
                return rgb_np, img_tensor, mask_gt_full.astype(np.int64)

            indices = list(range(0, len(val_dataset), max(1, len(val_dataset) // args.num_plot_samples)))[:args.num_plot_samples]
        else:
            base_dataset = info.get("base_dataset")

            def get_sample(idx):
                raw_sample = base_dataset[idx]
                raw_img_t = raw_sample["image"]
                if torch.is_tensor(raw_img_t):
                    raw_img_np = raw_img_t.permute(1, 2, 0).numpy()
                else:
                    raw_img_np = np.array(raw_img_t)
                if raw_img_np.max() > 1:
                    raw_img_np = raw_img_np.astype(np.float32) / 255.0
                raw_img_t = torch.from_numpy(raw_img_np).permute(2, 0, 1)
                raw_img_resized = TF.resize(raw_img_t, 256, interpolation=InterpolationMode.BILINEAR)
                raw_rgb = raw_img_resized.permute(1, 2, 0).numpy()[:, :, :3]
                img_tensor, mask_gt = val_dataset[idx]
                return raw_rgb, img_tensor, mask_gt.numpy()

            indices = [11, 55, 111][:args.num_plot_samples]

        save_segmentation_comparison_plot(models_dict, get_sample, indices, comparison_path, device, C)
        print(f"Saved comparison plot to {comparison_path}")

    # Generate aggregate plots
    print("\nGenerating plots...")
    plot_segmentation_results(results_root)


def main():
    parser = argparse.ArgumentParser(description="LDA Segmentation Experiments")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="IndianPines",
                        choices=["IndianPines", "Salinas", "LoveDA"],
                        help="Dataset to use")
    
    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to data directory containing .mat files")
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root directory for TorchGeo datasets")
    parser.add_argument("--results-root", type=str, default="./runs/segmentation")
    
    # Training params
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--base-channels", type=int, default=32)
    
    # Hyperspectral-specific
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-stride", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    
    # Other
    parser.add_argument("--num-plot-samples", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots")

    args = parser.parse_args()

    if args.plot_only:
        plot_segmentation_results(args.results_root)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()

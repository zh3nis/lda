"""Run a latent dimension sweep on CIFAR-100 and save metrics/plots to disk."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.lda import LDAHead


def build_loaders(data_root: pathlib.Path, batch_size: int, test_batch_size: int, num_workers: int):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    pin_memory = torch.cuda.is_available()

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100(root=data_root, train=True, transform=train_tfm, download=True)
    test_ds = datasets.CIFAR100(root=data_root, train=False, transform=test_tfm, download=True)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_ld = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_ld, test_ld


class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class DeepLDA(nn.Module):
    def __init__(self, C: int, D: int):
        super().__init__()
        self.encoder = Encoder(D)
        self.head = LDAHead(C, D)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


class SoftmaxHead(nn.Module):
    def __init__(self, D: int, C: int):
        super().__init__()
        self.linear = nn.Linear(D, C)

    def forward(self, z):
        return self.linear(z)


class DeepClassifier(nn.Module):
    def __init__(self, C: int, D: int):
        super().__init__()
        self.encoder = Encoder(D)
        self.head = SoftmaxHead(D, C)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ok = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ok += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return ok / tot


def train_single(model, loss_fn, opt, train_ld, test_ld, device, epochs: int):
    train_acc = []
    test_acc = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = acc_sum = n_sum = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                acc_sum += (pred == y).sum().item()
                n_sum += y.size(0)
                loss_sum += loss.item() * y.size(0)

        tr_acc = acc_sum / n_sum
        te_acc = evaluate(model, test_ld, device)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        print(f"[{epoch:02d}] train loss={loss_sum/n_sum:.4f} acc={tr_acc:.4f} | test acc={te_acc:.4f}")

    return {"train_acc": train_acc, "test_acc": test_acc, "final_test": test_acc[-1]}


def run_sweep(name, Ds, make_model, loss_fn, epochs, train_ld, test_ld, device):
    results = {}
    for D in Ds:
        print(f"=== {name} with D={D} ===")
        model = make_model(D).to(device)
        opt = torch.optim.Adam(model.parameters())
        results[D] = train_single(model, loss_fn, opt, train_ld, test_ld, device, epochs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def extract_final(results, Ds):
    return [results[D]["final_test"] for D in Ds]


def save_results_json(path: pathlib.Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to {path}")


def save_plot(path: pathlib.Path, Ds, lda_results, softmax_results):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(Ds, extract_final(lda_results, Ds), marker='o', label='DeepLDA')
    plt.plot(Ds, extract_final(softmax_results, Ds), marker='s', label='Softmax')
    plt.xlabel('Latent dimension D')
    plt.ylabel('Final test accuracy')
    plt.title('CIFAR-100 final test accuracy vs D (D > C-1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 latent dimension sweep (DeepLDA vs Softmax)")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[150, 200, 250, 300, 350, 400, 450, 500],
                        help="latent dimensions D to try (space-separated)")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs per model")
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("./data"), help="dataset root directory")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("plots"), help="where to save JSON and plot")
    parser.add_argument("--batch-size", type=int, default=256, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="test batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device =", device)

    train_ld, test_ld = build_loaders(args.data_root, args.batch_size, args.test_batch_size, args.workers)

    Ds = args.latent_dims
    lda_results = run_sweep(
        name="DeepLDA",
        Ds=Ds,
        make_model=lambda D: DeepLDA(C=100, D=D),
        loss_fn=nn.NLLLoss(),
        epochs=args.epochs,
        train_ld=train_ld,
        test_ld=test_ld,
        device=device,
    )

    softmax_results = run_sweep(
        name="Softmax",
        Ds=Ds,
        make_model=lambda D: DeepClassifier(C=100, D=D),
        loss_fn=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        train_ld=train_ld,
        test_ld=test_ld,
        device=device,
    )

    output_dir = args.output_dir
    metrics_path = output_dir / "cifar100_latent_dim_sweep.json"
    plot_path = output_dir / "cifar100_latent_dim_sweep.png"

    payload = {
        "latent_dims": Ds,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "DeepLDA": lda_results,
        "Softmax": softmax_results,
    }
    save_results_json(metrics_path, payload)
    save_plot(plot_path, Ds, lda_results, softmax_results)

    for name, results in [("DeepLDA", lda_results), ("Softmax", softmax_results)]:
        print(f"{name} final test accuracy by D:")
        for D in Ds:
            print(f"D={D:3d}: final test acc = {results[D]['final_test']:.4f}")


if __name__ == "__main__":
    main()

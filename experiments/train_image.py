"""Training script for Fashion-MNIST and CIFAR-10 with Deep LDA and Softmax heads."""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lda import LDAHead


class ConvEncoder(nn.Module):
    """Simple ConvNet encoder for Fashion-MNIST and CIFAR-10."""
    def __init__(self, in_channels, embedding_dim, hidden_channels=[32, 64]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels[1], embedding_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder for CIFAR-10."""
    def __init__(self, embedding_dim):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)


class SoftmaxHead(nn.Module):
    """Standard softmax classifier."""
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, z, y=None):
        return self.fc(z)


def train_epoch(model, head, loader, optimizer, device, head_type):
    model.train()
    head.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        z = model(x)
        logits = head(z, y)
        loss = nn.functional.cross_entropy(logits, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    total_loss, correct, total = 0, 0, 0
    all_logits, all_targets = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            logits = head(z, y)
            loss = nn.functional.cross_entropy(logits, y)
            
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    nll = nn.functional.cross_entropy(all_logits, all_targets).item()
    
    return total_loss / total, correct / total, nll


def get_embeddings_2d(model, head, loader, device, num_classes):
    """Extract 2D embeddings for visualization."""
    model.eval()
    head.eval()
    all_z, all_y = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            all_z.append(z.cpu())
            all_y.append(y)
    
    all_z = torch.cat(all_z).numpy()
    all_y = torch.cat(all_y).numpy()
    
    # Get class means and covariance if LDA
    class_means = None
    if isinstance(head, LDAHead):
        class_means = head.mu_ema.cpu().numpy()
        cov = head.cov_ema.cpu().numpy()
    else:
        cov = None
    
    return all_z, all_y, class_means, cov


def plot_2d_embeddings(z, y, class_means, cov, title, save_path, num_classes):
    """Plot 2D embeddings with class means and covariance ellipses."""
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for c in range(num_classes):
        mask = y == c
        plt.scatter(z[mask, 0], z[mask, 1], c=[colors[c]], alpha=0.5, s=10, label=f'Class {c}')
    
    if class_means is not None and cov is not None:
        # Plot class means
        plt.scatter(class_means[:, 0], class_means[:, 1], c='black', marker='X', s=200, 
                   edgecolors='white', linewidths=2)
        
        # Plot covariance ellipses (90% confidence)
        from scipy.stats import chi2
        scale = np.sqrt(chi2.ppf(0.9, df=2))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        width, height = 2 * scale * np.sqrt(eigenvalues)
        
        for c in range(num_classes):
            ellipse = Ellipse(xy=class_means[c], width=width, height=height, angle=angle,
                            facecolor=colors[c], alpha=0.2, edgecolor=colors[c], linewidth=2)
            plt.gca().add_patch(ellipse)
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, dataset, head, encoder, embed_dim, seed):
    """Plot training loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training and Test Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [acc * 100 for acc in history['test_acc']], 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training and Test Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # NLL curve
    axes[2].plot(epochs, history['test_nll'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('NLL', fontsize=11)
    axes[2].set_title('Test Negative Log-Likelihood', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.suptitle(f'{dataset.upper()} - {head.upper()} ({encoder}) - d={embed_dim}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join('figures', f'{dataset}_{head}_{encoder}_d{embed_dim}_curves_seed{seed}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['fashion_mnist', 'cifar10'])
    parser.add_argument('--head', type=str, required=True, choices=['softmax', 'lda'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--encoder', type=str, default='convnet', choices=['convnet', 'resnet18'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ema', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--debug', action='store_true', help='Fast debug mode: 5 epochs, subset of data')
    parser.add_argument('--fast', action='store_true', help='Fast training: reduce epochs by half')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision (faster on modern GPUs)')
    args = parser.parse_args()
    
    # Debug mode: override settings for fast testing
    if args.debug:
        args.epochs = 5
        args.num_workers = 0
        print("=" * 60)
        print("DEBUG MODE: Running with 5 epochs and reduced data")
        print("=" * 60)
    
    # Fast mode: reduce epochs
    if args.fast and not args.debug:
        args.epochs = max(args.epochs // 2, 10)
        print("=" * 60)
        print(f"FAST MODE: Reducing epochs to {args.epochs}")
        print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Load dataset
    if args.dataset == 'fashion_mnist':
        transform = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
        hidden_channels = [32, 64]
    else:  # cifar10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
        in_channels, num_classes = 3, 10
        hidden_channels = [64, 128]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    
    # Debug mode: use subset of data
    if args.debug:
        from torch.utils.data import Subset
        train_indices = list(range(min(1000, len(train_dataset))))
        test_indices = list(range(min(500, len(test_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"Using {len(train_dataset)} train samples, {len(test_dataset)} test samples")
    
    # Create model
    if args.encoder == 'resnet18' and args.dataset == 'cifar10':
        encoder = ResNetEncoder(args.embed_dim).to(device)
    else:
        encoder = ConvEncoder(in_channels, args.embed_dim, hidden_channels).to(device)
    
    if args.head == 'softmax':
        head = SoftmaxHead(args.embed_dim, num_classes).to(device)
    else:
        head = LDAHead(num_classes, args.embed_dim, ema=args.ema).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    if args.amp:
        if torch.cuda.is_available():
            print("Using Automatic Mixed Precision (AMP)")
        else:
            print("AMP requested but CUDA not available, using FP32")
    
    # Training loop
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_nll': []}
    
    for epoch in range(args.epochs):
        # Train
        encoder.train()
        head.train()
        total_loss, correct, total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    z = encoder(x)
                    logits = head(z, y)
                    loss = nn.functional.cross_entropy(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z = encoder(x)
                logits = head(z, y)
                loss = nn.functional.cross_entropy(logits, y)
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Evaluate
        test_loss, test_acc, test_nll = evaluate(encoder, head, test_loader, device)
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_nll'].append(test_nll)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f'Epoch {epoch+1}/{args.epochs} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | NLL: {test_nll:.4f}')
    
    # In debug mode, print final results immediately
    if args.debug:
        print("\n" + "="*60)
        print("DEBUG RUN COMPLETE")
        print(f"Final test accuracy: {test_acc:.4f}")
        print(f"Final test NLL: {test_nll:.4f}")
        print("="*60)
        return
    
    # Plot training curves
    plot_training_curves(history, args.dataset, args.head, args.encoder, args.embed_dim, args.seed)
    
    # Save results
    result_file = os.path.join(args.save_dir, f'{args.dataset}_{args.head}_d{args.embed_dim}_seed{args.seed}.txt')
    with open(result_file, 'w') as f:
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Head: {args.head}\n')
        f.write(f'Encoder: {args.encoder}\n')
        f.write(f'Embedding dim: {args.embed_dim}\n')
        f.write(f'Best test accuracy: {best_acc:.4f}\n')
        f.write(f'Final test accuracy: {test_acc:.4f}\n')
        f.write(f'Final test NLL: {test_nll:.4f}\n')
    
    print(f'\nResults saved to {result_file}')
    print(f'Best test accuracy: {best_acc:.4f}')
    print(f'Final test accuracy: {test_acc:.4f}')
    print(f'Final test NLL: {test_nll:.4f}')
    
    # Generate 2D visualization if embed_dim == 2
    if args.embed_dim == 2:
        z, y, means, cov = get_embeddings_2d(encoder, head, test_loader, device, num_classes)
        title = f'{args.dataset.upper()} - {args.head.upper()} (2D)'
        save_path = os.path.join('figures', f'{args.dataset}_{args.head}_2d_seed{args.seed}.png')
        plot_2d_embeddings(z, y, means, cov, title, save_path, num_classes)
        print(f'2D visualization saved to {save_path}')


if __name__ == '__main__':
    main()

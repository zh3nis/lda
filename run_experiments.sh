#!/bin/bash
# Run all image experiments for the paper

echo "Running Fashion-MNIST experiments..."

echo "Fashion-MNIST Softmax (d=128)"
python experiments/train_image.py --dataset fashion_mnist --head softmax --embed_dim 128 --epochs 100 --seed 42

echo "Fashion-MNIST LDA (d=128)"
python experiments/train_image.py --dataset fashion_mnist --head lda --embed_dim 128 --epochs 100 --seed 42

echo "Fashion-MNIST Softmax (d=2)"
python experiments/train_image.py --dataset fashion_mnist --head softmax --embed_dim 2 --epochs 100 --seed 42

echo "Fashion-MNIST LDA (d=2)"
python experiments/train_image.py --dataset fashion_mnist --head lda --embed_dim 2 --epochs 100 --seed 42

echo "Running CIFAR-10 experiments..."

echo "CIFAR-10 ConvNet Softmax (d=128)"
python experiments/train_image.py --dataset cifar10 --head softmax --encoder convnet --embed_dim 128 --epochs 200 --seed 42

echo "CIFAR-10 ConvNet LDA (d=128)"
python experiments/train_image.py --dataset cifar10 --head lda --encoder convnet --embed_dim 128 --epochs 200 --seed 42

echo "CIFAR-10 ConvNet Softmax (d=2)"
python experiments/train_image.py --dataset cifar10 --head softmax --encoder convnet --embed_dim 2 --epochs 200 --seed 42

echo "CIFAR-10 ConvNet LDA (d=2)"
python experiments/train_image.py --dataset cifar10 --head lda --encoder convnet --embed_dim 2 --epochs 200 --seed 42

echo "CIFAR-10 ResNet-18 Softmax (d=128)"
python experiments/train_image.py --dataset cifar10 --head softmax --encoder resnet18 --embed_dim 128 --epochs 200 --seed 42

echo "CIFAR-10 ResNet-18 LDA (d=128)"
python experiments/train_image.py --dataset cifar10 --head lda --encoder resnet18 --embed_dim 128 --epochs 200 --seed 42
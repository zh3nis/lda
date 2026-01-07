# Deep Linear Discriminant Analysis with Discriminative NLL Loss

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)

**Authors:** Arman Bolatov, Rustem Kaliyev, Amantay Nurlanuly, Zhenisbek Assylbekov

## Abstract

We investigate Deep Linear Discriminant Analysis (DeepLDA) for remote sensing image classification and semantic segmentation. Unlike softmax heads that produce purely discriminative logits, DeepLDA heads combine learnable class-conditional Gaussian models with a discriminative training objective called Discriminative Negative Log-Likelihood (DNLL). We compare three architectures on hyperspectral segmentation benchmarks (Salinas AVIRIS, Indian Pines), high-resolution RGB segmentation (LoveDA), and scene classification datasets (EuroSAT, RESISC45, UCMerced): a baseline softmax classifier trained with cross-entropy, a simplex-constrained LDA head, and a trainable full-covariance LDA head. Both LDA variants achieve competitive accuracy and mean IoU while maintaining explicit generative structure that enables interpretation of learned class means and covariances.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Image Classification (EuroSAT, RESISC45, UCMerced)

```bash
python train_classification.py --dataset EuroSAT --epochs 100 --seeds 42 123 456
```

### Semantic Segmentation (Indian Pines, Salinas, LoveDA)

```bash
python train_segmentation.py --dataset IndianPines --epochs 30
python train_segmentation.py --dataset Salinas --epochs 30
python train_segmentation.py --dataset LoveDA --epochs 30
```

### Tabular Classification

```bash
python train_tabular.py --data ./data/LU22_tabular.csv --epochs 120 --seeds 42 123 456
```

## Project Structure

```
├── src/lda.py           # LDA heads: LDAHead, TrainableLDAHead, DNLLLoss
├── models.py            # CNN/MLP encoders + classification/segmentation models
├── utils.py             # Training, evaluation, and plotting utilities
├── train_classification.py
├── train_segmentation.py
├── train_tabular.py
└── visualize_embeddings.py
```

## License

See [LICENSE](LICENSE) for details.

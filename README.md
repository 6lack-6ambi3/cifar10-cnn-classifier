# CIFAR-10 Image Classifier — CNN from scratch

A convolutional neural network trained from scratch on CIFAR-10, built as part of
an Applied AI career portfolio. Includes a custom CNN architecture, full training
pipeline, and evaluation suite with confusion matrix and failure analysis.

---

## Project structure

```
├── data_loader.py      # Dataset, transforms, DataLoaders, EDA plots
├── train.py            # CIFARNet model, training loop, checkpointing
├── evaluate.py         # Confusion matrix, per-class accuracy, misclassified examples
├── data/               # CIFAR-10 auto-downloaded here (not committed)
├── best_model.pth      # Best checkpoint saved during training (not committed)
└── README.md
```

---

## Approach

### Dataset
CIFAR-10 — 60,000 32×32 RGB images across 10 classes (50k train / 10k val).
Pre-computed per-channel normalisation stats used (`mean=(0.4914, 0.4822, 0.4465)`,
`std=(0.2470, 0.2435, 0.2616)`) rather than generic ImageNet values.

### Augmentation (training only)
- Random horizontal flip
- Random crop with 4px padding

### Architecture — `CIFARNet`
Three VGG-style conv blocks followed by a two-layer classifier head.

| Stage    | Operation                              | Output shape     |
|----------|----------------------------------------|------------------|
| Block 1  | Conv(3→32) → BN → ReLU × 2 → MaxPool  | 32 × 16 × 16     |
| Block 2  | Conv(32→64) → BN → ReLU × 2 → MaxPool | 64 × 8 × 8       |
| Block 3  | Conv(64→128) → BN → ReLU × 2 → MaxPool| 128 × 4 × 4      |
| Head     | Flatten → Linear(2048→256) → Linear(256→10) | (B, 10)    |

Key design choices:
- `bias=False` on all conv layers (redundant when followed by BatchNorm)
- `Dropout2d(0.3)` after each block to regularise spatial features
- Raw logits output — no softmax (handled internally by `CrossEntropyLoss`)

### Training setup
| Hyperparameter   | Value                          |
|------------------|--------------------------------|
| Optimiser        | Adam, lr=1e-3, weight_decay=1e-4 |
| Loss             | CrossEntropyLoss, label_smoothing=0.1 |
| Scheduler        | CosineAnnealingLR (T_max=epochs) |
| Epochs           | 10                             |
| Batch size       | 32                             |
| Device           | Apple MPS (M-series GPU)       |

---

## Results

| Metric              | Score  |
|---------------------|--------|
| Val accuracy        | ~83%   |
| Strongest class     | automobile, ship, truck, frog |
| Weakest class       | cat, dog, bird (frequent mutual confusion) |

The `cat ↔ dog` and `automobile ↔ truck` confusions are the dominant failure modes —
consistent with the known difficulty of these pairs in CIFAR-10 and expected for a
model trained from scratch at this scale.

---

## What I'd improve

**Short term**
- Train for 30–50 epochs with a warmup scheduler — the 10-epoch run likely
  hasn't fully converged
- Add CutMix or MixUp augmentation for another 2–3% accuracy gain
- Tune dropout per block rather than using a single global value

**Medium term**
- Swap `CIFARNet` for a pretrained ResNet-18 (transfer learning) — expected jump
  from ~75% to ~92%+ with minimal code change
- Add a simple Flask or FastAPI wrapper so the model can serve predictions over HTTP

**Longer term**
- Apply the same pipeline to a domain-specific dataset (e.g. satellite image chips,
  matching prior work on the PlanesNet binary classification task)
- Explore knowledge distillation — compress a larger pretrained model into a
  lightweight one suitable for edge deployment (TFLite / ONNX on-device)

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision matplotlib scikit-learn pillow seaborn numpy
```

**macOS SSL fix** (Python 3.13 only, run once):
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

**Run in order:**
```bash
python data_loader.py   # download data + EDA plots
python train.py         # train and save best_model.pth
python evaluate.py      # confusion matrix + failure analysis
```

---

## .gitignore additions recommended

```
data/
best_model.pth
*.png
__pycache__/
.venv/
```

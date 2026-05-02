import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import val_loader, CLASSES
from train import CIFARNet, CKPT_PATH, DEVICE


# ── Load best checkpoint ───────────────────────────────────────
def load_model(ckpt_path=CKPT_PATH, device=DEVICE):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = CIFARNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"— val acc {checkpoint['val_acc']:.1%}")
    return model


# ── Collect all predictions ────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device=DEVICE):
    all_preds, all_labels, all_probs = [], [], []
    all_images = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_images.append(images.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_images),
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
    )


# ── Denormalise helper (same as data_loader.py) ────────────────
def denorm(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ── Plot 1: confusion matrix ───────────────────────────────────
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_pct,
        annot=True, fmt=".1f", cmap="YlGn",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.4, linecolor="white",
        ax=ax, cbar_kws={"label": "Row %"}
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Confusion matrix — row-normalised (%)", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → confusion_matrix.png")


# ── Plot 2: per-class accuracy bar chart ──────────────────────
def plot_per_class_accuracy(labels, preds):
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = ["#1D9E75" if a >= 70 else "#E24B4A" for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(CLASSES, per_class_acc, color=colors,
                  edgecolor="none", width=0.6)
    ax.axhline(per_class_acc.mean(), color="#888780",
               linestyle="--", linewidth=1, label=f"Mean {per_class_acc.mean():.1f}%")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy", fontsize=13)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → per_class_accuracy.png")


# ── Plot 3: misclassified examples ─────────────────────────────
def plot_misclassified(images, labels, preds, probs, n=20):
    wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]

    # Sort by confidence of the wrong prediction (most confident first)
    wrong_conf = probs[wrong_idx].max(dim=1).values
    sorted_idx = wrong_conf.argsort(descending=True)
    wrong_idx  = wrong_idx[sorted_idx[:n]]

    n_cols = 5
    n_rows = (len(wrong_idx) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 2.4))
    fig.suptitle("Most confidently wrong predictions", fontsize=13, y=1.01)

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, wrong_idx):
        img   = denorm(images[idx]).permute(1, 2, 0).numpy()
        true  = CLASSES[labels[idx]]
        pred  = CLASSES[preds[idx]]
        conf  = probs[idx].max().item() * 100

        ax.imshow(img)
        ax.set_title(
            f"true: {true}\npred: {pred} ({conf:.0f}%)",
            fontsize=8,
            color="#E24B4A"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → misclassified.png")


# ── Plot 4: top-k confidence distribution ─────────────────────
def plot_confidence_distribution(labels, preds, probs):
    correct_conf = probs[preds == labels].max(dim=1).values.numpy() * 100
    wrong_conf   = probs[preds != labels].max(dim=1).values.numpy() * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(correct_conf, bins=40, alpha=0.75,
            color="#1D9E75", label=f"Correct  (n={len(correct_conf):,})",
            edgecolor="none")
    ax.hist(wrong_conf,   bins=40, alpha=0.75,
            color="#E24B4A", label=f"Incorrect (n={len(wrong_conf):,})",
            edgecolor="none")
    ax.set_xlabel("Model confidence on top prediction (%)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution — correct vs incorrect", fontsize=13)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → confidence_distribution.png")


# ── Text report ────────────────────────────────────────────────
def print_report(labels, preds):
    overall_acc = (preds == labels).float().mean().item()
    print(f"\nOverall val accuracy : {overall_acc:.1%}\n")
    print(classification_report(
        labels.numpy(), preds.numpy(), target_names=CLASSES
    ))


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    # Install seaborn if needed: pip install seaborn
    model  = load_model()
    images, preds, labels, probs = get_predictions(model, val_loader)

    print_report(labels, preds)
    plot_confusion_matrix(labels, preds)
    plot_per_class_accuracy(labels, preds)
    plot_misclassified(images, labels, preds, probs)
    plot_confidence_distribution(labels, preds, probs)

    print("\nAll plots saved. Phase 4 complete.")
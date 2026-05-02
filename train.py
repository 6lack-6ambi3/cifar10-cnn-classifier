import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import train_loader, CLASSES, val_loader

DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu")

EPOCHS = 10
LR = 1e-3
CKPT_PATH = "best_model.pth"
print(f"Using device: {DEVICE}")

class CIFARNet(nn.Module):
    """
    Three conv blocks followed by two fully-connected layers.

    Block structure: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    Each block doubles the channel count and halves the spatial dims.

    Input : (B, 3, 32, 32)
    Output: (B, 10)  — raw logits, no softmax (CrossEntropyLoss handles it)
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()

        #Block 1: 3 → 32 channels, 32x32 → 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        #Block 2: 32 → 64 channels, 16x16 → 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        #Block 3: 64 → 128 channels, 8x8 → 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        #Classifier head: 128*4*4 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

# -Training helpers
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# -Training loop
def train(epochs=EPOCHS, lr=LR, device=DEVICE):
    model = CIFARNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [],
              "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'Epoch':>6}  {'Train loss':>10}  {'Train acc':>9}  "
          f"{'Val loss':>8}  {'Val acc':>7}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        current_lr = scheduler.get_last_lr()[0]
        marker = " *" if val_acc > best_val_acc else ""

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(), 
                "val_acc": val_acc,
            }, CKPT_PATH)

        print(f"{epoch:>6}  {tr_loss:10.4f}  {tr_acc*100:8.2f}%  "
              f"{val_loss:8.4f}  {val_acc*100:6.2f}% " 
              f"{current_lr:8.1e}{marker}")
    
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    return model, history

# ── Loss & accuracy curves ─────────────────────────────────────
def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="#1D9E75")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#E24B4A")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
             label="Train", color="#1D9E75")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
             label="Val",   color="#E24B4A")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("CIFARNet — training history", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → training_history.png")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train()
    plot_history(history)
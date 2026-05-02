import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

BATCH_SIZE = 32
DATA_DIR = './data'

CLASSES = ('airplane', 'automobile', 'bird', 'cat', "deer", 'dog', 'frog', 'horse', 'ship', 'truck')

#Transformations to apply to the images
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.4914, 0.4822, 0.4465),
                                          std=(0.2470, 0.2435, 0.2616)),])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)),
])

# Load the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transform)

val_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=False  # changed
)

# -Helper: denormalize for display
def denorm(tensor):
    """Undo CIFAR-10 normalization so pixel values are back in [0,1]"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

# -Plot1: Batch preview (4x 8 grid)
def show_batch(loader, n_col=8, n_row=4):
    images, labels = next(iter(loader))
    images = images[: n_col * n_row]
    labels = labels[: n_col * n_row]

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 1.4, n_row * 1.4))
    fig.suptitle("CIFAR-10 Sample Images", fontsize=13, y=1.01)

    for ax, img, lbl in zip(axes.flat, images, labels):
        ax.imshow(denorm(img).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[lbl], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cifar10_batch_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved batch_preview.png')

# -Plot2: Class distribution
def show_class_distribution(dataset):
    counts = Counter(dataset.targets)
    names = [CLASSES[i] for i in range(10)]
    values = [counts[i] for i in range(10)]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(names, values, color='#1D9E75', edgecolor='none', width=0.6)
    ax.set_title("CIFAR-10 Class Distribution - training set", fontsize=12)
    ax.set_ylabel('Images')
    ax.set_ylim(0, max(values) * 1.5)
    ax.spines[['top', 'right']].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 60, 
                f'{val:,}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('cifar10_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved class_distribution.png')

# -Plot 3: per-channel pixel intensity histogram
def show_pixel_histogram(loader, n_batches=5):
    """Sample a few batches and plot the pixel intensity distribution for each channel
    """
    pixels = {c: [] for c in ['R', 'G', 'B']}
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        for c, name in enumerate(['R', 'G', 'B']):
            pixels[name].append(imgs[:, c].flatten())

    colors = {"R": "#E24B4A", "G": "#1D9E75", "B": "#378ADD"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    fig.suptitle(f"Pixel Intensity Distribution - {n_batches} batches", fontsize=12)

    for ax, (name, batches) in zip(axes, pixels.items()):
        data = torch.cat(batches).numpy()
        ax.hist(data, bins=60, color=colors[name], alpha=0.85, edgecolor='none')
        ax.set_title(f'{name} channel')
        ax.set_xlabel("Normalised Value")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig('cifar10_pixel_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved pixel_histogram.png')

if __name__ == "__main__":
    print(f"Number of training samples: {len(train_dataset):,}")
    print(f"Number of validation samples: {len(val_dataset):,}")
    print(f"Batches per epoch: {len(train_loader):,}")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")
    
    show_batch(train_loader)
    show_class_distribution(train_dataset)
    show_pixel_histogram(train_loader)
"""Train a CNN on CIFAR-10.

CIFAR-10 downloads automatically to ./data the first time you run this.

Usage:
    py train.py                 # full run (default 15 epochs)
    py train.py --epochs 5
    py train.py --smoke-test    # 1 epoch on a tiny subset, just to verify it runs
"""
import argparse
import matplotlib
matplotlib.use("Agg")  # write plots to file, no GUI needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model import CIFARNet

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def get_loaders(batch_size, smoke_test):
    # Normalize with CIFAR-10 channel means/stds. Light augmentation on train.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    if smoke_test:
        # Use only a small slice so a full pass finishes in seconds.
        train_set = Subset(train_set, range(512))
        test_set = Subset(test_set, range(256))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size, args.smoke_test)
    model = CIFARNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"test loss {te_loss:.3f} acc {te_acc:.3f}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "cifar_cnn.pth")

    print(f"\nBest test accuracy: {best_acc:.3f}  (model saved to cifar_cnn.pth)")

    # Save training curves.
    epochs = range(1, args.epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig("training_curve.png", dpi=120)
    print("Training curve saved to training_curve.png")


if __name__ == "__main__":
    main()

"""
Training script for face image restoration (U-Net).
Run from project root: python train.py
"""
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset.face_dataset import FaceDataset
from models.unet import UNet
from losses.losees import get_loss, get_identity_loss
from utils.device import get_device


def save_curves(train_losses: list[float], val_losses: list[float]) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train L1 Loss")
    plt.plot(val_losses, label="Val L1 Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(config.RESULTS_DIR, "loss_curve.png"))
    plt.close()


@torch.no_grad()
def compute_l1_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for degraded, clean in loader:
        degraded = degraded.to(device)
        clean = clean.to(device)
        output = model(degraded)
        total += criterion(output, clean).item()
    return total / len(loader)


def main():
    device = get_device()

    use_align = getattr(config, "USE_ALIGNMENT", False)
    img_size = getattr(config, "IMG_SIZE", 128)
    train_dataset = FaceDataset(
        config.DATA_TRAIN_DIR, config.DESTRUCTION,
        use_alignment=use_align, img_size=img_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    val_dataset = FaceDataset(
        config.DATA_VAL_DIR, config.DESTRUCTION,
        use_alignment=use_align, img_size=img_size
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = get_loss()
    identity_loss_fn = (
        get_identity_loss(device) if getattr(config, "USE_IDENTITY_LOSS", False) else None
    )

    best_loss = float("inf")
    train_losses = []
    val_losses = []

    patience = getattr(config, "EARLY_STOPPING_PATIENCE", 5)
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0

        for degraded, clean in train_loader:
            degraded = degraded.to(device)
            clean = clean.to(device)

            output = model(degraded)
            loss = criterion(output, clean)
            if identity_loss_fn is not None:
                loss = loss + getattr(config, "IDENTITY_LOSS_WEIGHT", 0.1) * identity_loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        val_loss_epoch = compute_l1_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss_epoch)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train L1 Loss: {epoch_loss:.4f} | "
            f"Val L1 Loss: {val_loss_epoch:.4f}"
        )

        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            torch.save(model.state_dict(), config.MODEL_PATH)
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        save_curves(train_losses, val_losses)

    save_curves(train_losses, val_losses)
    print("Training finished.")


if __name__ == "__main__":
    main()

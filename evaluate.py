"""
Evaluation script: load best model, compute PSNR/SSIM/embed on val & test,
optionally on another dataset, and save triplet visualizations.
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

import config
from dataset.face_dataset import FaceDataset
from models.unet import UNet
from utils.visualize import save_triplet_grid
from utils.evaluator import evaluate_loader
from utils.device import get_device

CHECKPOINT_PATH = "/content/drive/MyDrive/FaceRestore/checkpoint.pth"



def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Evaluate face restoration model")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Run without loading model.pth (untrained model). Use to get triplet images when no checkpoint exists (e.g. in Colab before training).",
    )
    args = parser.parse_args()

    device = get_device()
    model = UNet().to(device)

    model_path = config.MODEL_PATH
    if not os.path.isfile(model_path):
        if args.no_checkpoint:
            print("Warning: '{}' not found. Using untrained model — metrics will be poor, but triplet images will be saved.".format(model_path))
        else:
            print("Error: trained model not found at '{}'.".format(model_path))
            print("In Colab you can either:")
            print("  1. Train first: run train.py (or your training notebook), then run evaluate.py again.")
            print("  2. Upload model.pth to this project folder (e.g. drag & drop in Colab file browser).")
            print("  3. Run with untrained model to still get image output: python evaluate.py --no-checkpoint")
            sys.exit(1)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    use_align = getattr(config, "USE_ALIGNMENT", False)
    img_size = getattr(config, "IMG_SIZE", 128)

    val_dataset = FaceDataset(
        config.DATA_VAL_DIR,
        config.DESTRUCTION,
        use_alignment=use_align,
        img_size=img_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    test_dataset = FaceDataset(
        config.DATA_TEST_DIR,
        config.DESTRUCTION,
        use_alignment=use_align,
        img_size=img_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Evaluate on validation and test (PSNR, SSIM, face embedding distance)
    val_psnr, val_ssim, val_embed = evaluate_loader(model, val_loader, device)
    test_psnr, test_ssim, test_embed = evaluate_loader(model, test_loader, device)

    print(
        "CelebA validation — PSNR: {:.2f} dB | SSIM: {:.4f} | Embed dist: {:.4f}".format(
            val_psnr, val_ssim, val_embed or 0
        )
    )
    print(
        "CelebA test       — PSNR: {:.2f} dB | SSIM: {:.4f} | Embed dist: {:.4f}".format(
            test_psnr, test_ssim, test_embed or 0
        )
    )

    # Generalization: evaluate on another face dataset (if present)
    if os.path.isdir(config.DATA_TEST_OTHER_DIR):
        other_dataset = FaceDataset(
            config.DATA_TEST_OTHER_DIR,
            config.DESTRUCTION,
            use_alignment=use_align,
            img_size=img_size,
        )
        other_loader = DataLoader(
            other_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )
        other_psnr, other_ssim, other_embed = evaluate_loader(
            model, other_loader, device
        )
        print(
            "Other (generalization) — PSNR: {:.2f} dB | SSIM: {:.4f} | Embed dist: {:.4f}".format(
                other_psnr, other_ssim, other_embed or 0
            )
        )
    else:
        print(
            "No '{}' folder — skip generalization eval.".format(
                config.DATA_TEST_OTHER_DIR
            )
        )

    # Visualize: degraded | restored | ground truth
    degraded, gt = next(iter(val_loader))
    degraded = degraded.to(device)
    gt = gt.to(device)
    with torch.no_grad():
        restored = model(degraded)
    save_triplet_grid(
        degraded, restored, gt, os.path.join(config.RESULTS_DIR, "val_triplet.png")
    )

    degraded, gt = next(iter(test_loader))
    degraded = degraded.to(device)
    gt = gt.to(device)
    with torch.no_grad():
        restored = model(degraded)
    save_triplet_grid(
        degraded, restored, gt, os.path.join(config.RESULTS_DIR, "test_triplet.png")
    )

    print("Triplet grids saved to {}.".format(config.RESULTS_DIR))


if __name__ == "__main__":
    main()

"""
Split a folder of face images into train / val / test for this project.

Usage:
  python prepare_data.py <path_to_images> [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42]

Example (CelebA):
  # After downloading CelebA, unzip img_align_celeba somewhere, then:
  python prepare_data.py /path/to/img_align_celeba

  # Custom split (e.g. 70% train, 15% val, 15% test):
  python prepare_data.py /path/to/img_align_celeba --train 0.7 --val 0.15 --test 0.15

Output:
  data/
    train/   <- training images
    val/     <- validation images
    test/    <- test images
  (data/test_other/ is for a *different* face dataset — put those images in manually for generalization eval)
"""

import os
import argparse
import shutil
import random

# Same extensions as FaceDataset
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def main():
    p = argparse.ArgumentParser(description="Split images into data/train, data/val, data/test")
    p.add_argument("source_dir", help="Folder containing face images (e.g. CelebA img_align_celeba)")
    p.add_argument("--train", type=float, default=0.8, help="Fraction for training (default 0.8)")
    p.add_argument("--val", type=float, default=0.1, help="Fraction for validation (default 0.1)")
    p.add_argument("--test", type=float, default=0.1, help="Fraction for test (default 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    p.add_argument("--out", type=str, default="data", help="Output base dir (default: data)")
    args = p.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("--train, --val, --test must sum to 1.0")

    if not os.path.isdir(args.source_dir):
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    files = [
        f for f in os.listdir(args.source_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    files = [os.path.join(args.source_dir, f) for f in files]

    if not files:
        raise SystemExit(f"No images found in {args.source_dir} (supported: {IMAGE_EXTENSIONS})")

    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val  # rest to test

    splits = [
        ("train", files[:n_train]),
        ("val", files[n_train : n_train + n_val]),
        ("test", files[n_train + n_val :]),
    ]

    for name, paths in splits:
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        for src in paths:
            dst = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dst)
        print(f"{name}: {len(paths)} images -> {out_dir}")

    print(f"\nDone. Total: {n} images. You can run train.ipynb now.")
    print("For generalization, put another face dataset in: data/test_other/")

if __name__ == "__main__":
    main()

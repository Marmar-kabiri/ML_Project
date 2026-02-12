# Face Image Restoration (ML Project)

Restore degraded face images (noise, low resolution, motion blur) using a U-Net and optional identity loss.

## Data: Where to Put It and How It Is Split

The code expects **already split** folders. There is no automatic split inside the training code.

### Expected folder structure

```
ML_Projectt/
  data/
    train/      ← training images (e.g. from CelebA)
    val/        ← validation images
    test/       ← test images (same distribution as train/val, for final metrics)
    test_other/ ← (optional) another face dataset to test generalization
```

- **train**: used for training the model.
- **val**: used for validation loss and early stopping during training.
- **test**: used only in evaluation (PSNR, SSIM, embedding distance).
- **test_other**: optional; put a different face dataset here to check generalization.

Any image format in `FaceDataset` is supported: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`.

### Option 1: Use the provided split script (recommended)

1. Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and unzip **img_align_celeba** (or use any folder of face images).
2. From the project root run:

```bash
# Default: 80% train, 10% val, 10% test (seed=42)
python prepare_data.py /path/to/img_align_celeba

# Custom split, e.g. 70% / 15% / 15%
python prepare_data.py /path/to/img_align_celeba --train 0.7 --val 0.15 --test 0.15 --seed 42
```

This creates `data/train`, `data/val`, and `data/test` and **copies** images into them (no symlinks).

### Option 2: Split manually

1. Put all your face images in one folder.
2. Manually (or with your own script) copy or move them into:
   - `data/train/`
   - `data/val/`
   - `data/test/`
3. Use any split you like (e.g. 80/10/10); the code does not care about the ratio.

### Generalization (optional)

- Put images from a **different** face dataset (e.g. LFW, or another subset) in **`data/test_other/`**.
- The evaluation notebook will run metrics on this folder if it exists, to report generalization.

## Quick start

1. Prepare data (see above) so that `data/train`, `data/val`, and `data/test` exist.
2. Install: `pip install -r requirements.txt`
3. Train: run `train.ipynb` (uses `config.py`: paths, batch size, degradation type, etc.).
4. Evaluate: run `evaluate.ipynb` (loads best model, computes PSNR/SSIM/embedding, saves triplet figures).

## Config

Edit `config.py` to change:

- `DATA_TRAIN_DIR`, `DATA_VAL_DIR`, `DATA_TEST_DIR`, `DATA_TEST_OTHER_DIR` if you use different paths.
- `DESTRUCTION`: `"noise"` (Gaussian), `"downsample"` (low-res), or e.g. `"blur"` (motion blur).
- `USE_ALIGNMENT`, `USE_IDENTITY_LOSS`, training hyperparameters, etc.

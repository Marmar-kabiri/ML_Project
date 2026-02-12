# Project Report — Face Image Restoration

**Topic:** Image restoration using a convolutional network  
**Project:** ML_Project — Face restoration pipeline with U-Net

*To generate a PDF: open this file in Cursor/VS Code and use "Markdown: Export to PDF" (or install the "Markdown PDF" extension), or run `pandoc Report_ML_Project.md -o Report_ML_Project.pdf`.*

---


## 1. Introduction

This project addresses **face image restoration**: reconstructing and improving the quality of face images using a convolutional architecture. The goal is to take a degraded image and produce an output close to the original high-quality image. The problem is framed as a **restoration** task with an emphasis on both **pixel quality** and **preserving face identity**. To support this, the project is organized in **modules** so that each part of the pipeline—data, model, loss, training, and evaluation—is implemented independently and can be extended or replaced without changing the rest of the code.

---

## 2. Module: `dataset/face_dataset.py`

The data pipeline assumes that face images (e.g. from **CelebA**) are roughly aligned so that spatial variation is reduced and the model can focus on restoring details such as eyes and mouth. The module `dataset/face_dataset.py` is responsible for **loading images** and **preparing them for training**.

- Images are **resized** to a configurable size (default 128×128) and **normalized** to the range [0, 1] via `ToTensor()`.
- **Data augmentation** is applied to the clean image: random horizontal flip (p=0.3), random rotation (±7°), and `ColorJitter` (brightness, contrast, saturation, hue). These changes are appearance-only and avoid altering identity, improving robustness to natural variations.
- The **degraded** version of the image is produced **on-the-fly** in `__getitem__`, so each epoch sees different degradation instances and the model does not overfit to a fixed degradation pattern.
- **Degradation type** is set by the `type` argument: `"noise"`, `"downsample"`, or `"motion_blur"` (chosen in `config.DESTRUCTION`).
- **Optional face alignment** (when `use_alignment=True`) uses `utils.alignment.align_face()` to detect the face, crop with padding, and resize; if detection fails, a center crop is used.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`.

---

## 3. Module: `utils/degradation.py`

Degradation logic is implemented in `utils/degradation.py`. Three types of degradation are supported:

1. **Gaussian noise** (`add_gaussian_noise`): Random standard deviation in a given range (default 0.2–0.5); noise is added then values are clipped to [0, 1].
2. **Downsampling** (`downsample`): Image is downsampled by a scale factor (e.g. 1.2 or 1.4), then upsampled back to the original size with bilinear interpolation, so high-frequency detail is lost.
3. **Motion blur** (`motion_blur`): A 1D line kernel is built in a random direction (horizontal, vertical, or diagonal), normalized, and applied with **depthwise convolution** per channel. Kernel size can be random (e.g. 5, 7, …, 15) or fixed.

Combining these degradations (via the dataset’s `type` and on-the-fly randomness) lets the model see diverse degradation and improves generalization.

---

## 4. Module: `models/unet.py`

The restoration architecture is implemented in `models/unet.py` as a **simplified U-Net**:

- **Encoder:** Two stages. First: two 3×3 conv layers (3→64 channels) + ReLU. Then max-pool (2×2) and two 3×3 conv layers (64→128) + ReLU.
- **Decoder:** One transposed convolution (128→64, stride 2) for upsampling, then **skip connection**: the upsampled feature map is concatenated with the first encoder output (64 channels) along the channel dimension (128 total). A final block of two 3×3 conv layers (128→64→3) produces the image; the last layer uses **Sigmoid** so outputs stay in [0, 1].

This structure preserves spatial information via skip connections and helps retain edges and face structure. The output is a 3-channel image in [0, 1].

---

## 5. Module: `losses/losees.py` and `losses/identity_loss.py`

- **Restoration loss** is defined in `losses/losees.py`. The function `get_loss()` returns **L1 loss** (PyTorch `nn.L1Loss()`). L1 is used so the model preserves details better than with L2, which tends to produce overly smooth images.
- **Identity loss** is implemented in `losses/identity_loss.py`. A pretrained face recognition model (**InceptionResnetV1**, VGGFace2) is loaded and frozen. For restored and ground-truth images, **face embeddings** are computed (with `fixed_image_standardization` and L2-normalization). The identity term is the **mean cosine distance** between these embeddings (i.e. \(1 - \text{cosine similarity}\)). Minimizing it encourages the restored image to match the identity of the clean image. `get_identity_loss(device)` in `losees.py` returns this module or `None` if the dependency is missing.

Using both **L1** and **identity loss** (with a configurable weight in `config.IDENTITY_LOSS_WEIGHT`) makes the network match the original both in pixel space and in face identity. The identity loss can be turned on/off via `config.USE_IDENTITY_LOSS`.

---

## 6. File: `train.py`

The training pipeline is implemented in `train.py`. Summary:

- **Data:** Training and validation datasets are built with `FaceDataset` using `config.DATA_TRAIN_DIR` and `config.DATA_VAL_DIR`, with optional alignment and image size from config. DataLoaders use `config.BATCH_SIZE` and shuffle for training.
- **Model and optimizer:** U-Net, Adam optimizer with learning rate from `config.LR`.
- **Loss:** Main criterion from `get_loss()` (L1). If `config.USE_IDENTITY_LOSS` is True, identity loss is added with weight `config.IDENTITY_LOSS_WEIGHT`.
- **Checkpointing:** A full checkpoint (model, optimizer, epoch, best validation loss) is saved to a fixed path; the best model state dict is also saved to `config.MODEL_PATH` for evaluation.
- **Validation:** After each epoch, L1 loss is computed on the validation set (no gradients). If validation loss improves, the checkpoint and best model are updated and early-stopping counter is reset.
- **Early stopping:** If validation loss does not improve for `config.EARLY_STOPPING_PATIENCE` consecutive epochs, training stops to reduce overfitting.
- **Curves:** Train and validation L1 losses are plotted and saved in `config.RESULTS_DIR` as `loss_curve.png`.

Training can be resumed from the checkpoint if it already exists.

---

## 7. Evaluation: `evaluate.ipynb` and `utils/evaluator.py`

- **`utils/evaluator.py`:** The function `evaluate_loader(model, loader, device, include_embedding=True)` runs the model in eval mode on a DataLoader and aggregates **PSNR**, **SSIM**, and (optionally) **face embedding distance** (cosine distance between restored and ground-truth embeddings). This keeps evaluation logic in one place for both training scripts and notebooks.
- **`evaluate.ipynb`:** Loads the best model from `config.MODEL_PATH`, builds validation and test datasets with the same degradation and alignment settings as training, and calls `evaluate_loader` to report PSNR, SSIM, and embedding distance on **validation** and **test**. If `config.DATA_TEST_OTHER_DIR` exists, it also evaluates on that dataset to assess **generalization** to another face dataset. Finally, it uses `utils/visualize.save_triplet_grid` to save grids of **degraded | restored | ground truth** for validation and test to `config.RESULTS_DIR` (e.g. `val_triplet.png`, `test_triplet.png`) for qualitative comparison.

---

## 8. Module: `utils/metrics.py` and `utils/visualize.py`

- **`utils/metrics.py`:** Defines **PSNR** (using MSE and 20·log10(1/√MSE) for data in [0,1]), **SSIM** (TorchMetrics `StructuralSimilarityIndexMeasure` with `data_range=1.0`), and **embedding_distance** (mean cosine distance between face embeddings from the same frozen InceptionResnetV1 backbone used in identity loss).
- **`utils/visualize.py`:** `save_triplet_grid(degraded, restored, gt, out_path, n=8)` stacks up to `n` triplets (degraded, restored, ground truth), concatenates them in one tensor, and saves a single image grid via `torchvision.utils.save_image` for inclusion in reports.

---

## 9. File: `config.py`

All main hyperparameters and paths are centralized in `config.py`:

- **Paths:** `DATA_TRAIN_DIR`, `DATA_VAL_DIR`, `DATA_TEST_DIR`, `DATA_TEST_OTHER_DIR`, `MODEL_PATH`, `RESULTS_DIR`.
- **Data:** `DESTRUCTION` (degradation type: `"noise"`, `"downsample"`, or `"motion_blur"`), `IMG_SIZE`, `USE_ALIGNMENT`.
- **Training:** `BATCH_SIZE`, `LR`, `EPOCHS`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`.
- **Identity loss:** `USE_IDENTITY_LOSS`, `IDENTITY_LOSS_WEIGHT`.

Changing behavior (e.g. degradation type, identity weight, early stopping) is done by editing config only, without modifying the core logic in data, model, loss, or training code.

---

## 10. Data Preparation: `prepare_data.py`

Before training, images must be split into train/val/test. The script `prepare_data.py` takes a folder of face images (e.g. CelebA `img_align_celeba`) and copies them into `data/train`, `data/val`, and `data/test` with configurable ratios (default 0.8 / 0.1 / 0.1). Usage:

```bash
python prepare_data.py <path_to_images> [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42] [--out data]
```

Optional `data/test_other/` can be populated manually with another face dataset for generalization evaluation.

---

## 11. Summary

The project is structured so that each part of the pipeline is a **separate module**: the dataset produces (degraded, clean) pairs; the model performs restoration; the loss module defines pixel and identity terms; the training script handles learning and checkpointing; and the evaluator and notebook report PSNR, SSIM, embedding distance, and visual grids. This keeps the code maintainable and makes it straightforward to add extensions such as perceptual loss, deep supervision, or other architectures, while `config.py` keeps all main settings in one place.

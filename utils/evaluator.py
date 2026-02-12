import torch
from utils.metrics import psnr, ssim, embedding_distance

@torch.no_grad()
def evaluate_loader(model, loader, device, include_embedding=True):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    embed_sum = 0.0
    n_batches = 0

    for degraded, gt in loader:
        degraded = degraded.to(device)
        gt = gt.to(device)
        restored = model(degraded)

        psnr_sum += psnr(restored, gt)
        ssim_sum += ssim(restored, gt)
        if include_embedding:
            embed_sum += embedding_distance(restored, gt, device)
        n_batches += 1

    avg_psnr = psnr_sum / n_batches
    avg_ssim = ssim_sum / n_batches
    avg_embed = embed_sum / n_batches if include_embedding else None
    return avg_psnr, avg_ssim, avg_embed
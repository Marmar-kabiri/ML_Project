import os
import torch
from torchvision.utils import save_image

def save_triplet_grid(degraded, restored, gt, out_path, n=8):
    # degraded/restored/gt: (B,C,H,W)
    degraded = degraded[:n].detach().cpu()
    restored = restored[:n].detach().cpu()
    gt = gt[:n].detach().cpu()

    grid = torch.cat([degraded, restored, gt], dim=0)  # 3n images
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path, nrow=n)
import torch
import torch.nn.functional as F
import random

def add_gaussian_noise(img, min_std=0.2, max_std=0.5):
    std = random.uniform(min_std, max_std)
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0., 1.)

def downsample(img, scale=1.2):
    h, w = img.shape[-2:]
    new_h = int(h / scale)
    new_w = int(w / scale)

    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
    img = F.interpolate(img, size=(h, w), mode='bilinear')
    return img.squeeze(0)

def motion_blur(img, kernel_size=None):
    # img: Tensor (C,H,W) in [0,1]
    if kernel_size is None:
        kernel_size = random.choice([5, 7, 9, 11, 13, 15])

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    c = img.shape[0]

    # Choose blur direction
    direction = random.choice(["h", "v", "d1", "d2"])  # horizontal, vertical, diag, anti-diag
    kernel = torch.zeros((k, k), device=img.device, dtype=img.dtype)

    if direction == "h":
        kernel[k // 2, :] = 1.0
    elif direction == "v":
        kernel[:, k // 2] = 1.0
    elif direction == "d1":
        kernel.fill_diagonal_(1.0)
    else:  # "d2"
        kernel = torch.fliplr(torch.eye(k, device=img.device, dtype=img.dtype))

    kernel = kernel / kernel.sum()

    # (1,1,k,k) -> expand to (C,1,k,k) for depthwise conv
    weight = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

    x = img.unsqueeze(0)  # (1,C,H,W)
    x = F.conv2d(x, weight, padding=k // 2, groups=c)
    return torch.clamp(x.squeeze(0), 0.0, 1.0)
    
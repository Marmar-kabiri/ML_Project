import torch
import math
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / torch.sqrt(mse))

_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

def ssim(pred, target):
    return _ssim(pred, target).item()

_face_backbone = None

def _get_face_backbone(device):
    global _face_backbone
    if _face_backbone is None:
        from facenet_pytorch import InceptionResnetV1
        _face_backbone = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in _face_backbone.parameters():
            p.requires_grad = False
    return _face_backbone

@torch.no_grad()
def embedding_distance(restored, target, device):
    """Mean cosine distance between face embeddings (0 = same identity, 2 = opposite)."""
    from facenet_pytorch import fixed_image_standardization
    import torch.nn.functional as F
    backbone = _get_face_backbone(device)
    x1 = fixed_image_standardization(restored)
    x2 = fixed_image_standardization(target)
    e1 = F.normalize(backbone(x1), p=2, dim=1)
    e2 = F.normalize(backbone(x2), p=2, dim=1)
    return (1.0 - (e1 * e2).sum(dim=1)).mean().item()
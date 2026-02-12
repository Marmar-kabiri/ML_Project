import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

class IdentityLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = fixed_image_standardization(x)
        emb = self.backbone(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, restored: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        e1 = self.embed(restored)
        e2 = self.embed(gt)
        return (1.0 - (e1 * e2).sum(dim=1)).mean()

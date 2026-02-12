import torch.nn as nn

def get_loss():
    return nn.L1Loss()

def get_identity_loss(device):
    try:
        from losses.identity_loss import IdentityLoss
        return IdentityLoss(device)
    except ImportError:
        return None  # facenet_pytorch not installed
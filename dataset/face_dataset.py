from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from utils.degradation import add_gaussian_noise, downsample, motion_blur
import random
from utils.augment import get_augment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

class FaceDataset(Dataset):
    def __init__(self, root, type, use_alignment=True, img_size=128):
        self.paths = [
            os.path.join(root, p)
            for p in os.listdir(root)
            if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
        ]
        self.type = type
        self.use_alignment = use_alignment
        self.img_size = img_size
        self.augment = get_augment()
        if use_alignment:
            from utils.alignment import align_face
            self.align_face = align_face
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.use_alignment:
            img = self.align_face(img, output_size=self.img_size)
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.paths[idx])
        img = self.augment(img)
        img = self.transform(img)

        degraded = img.clone()

        if self.type == "noise":
            degraded = add_gaussian_noise(degraded)
        elif self.type == "downsample":
            degraded = downsample(degraded, scale=random.choice([1.2, 1.4]))
        else:
            degraded = motion_blur(degraded, kernel_size=5)

        return degraded, img
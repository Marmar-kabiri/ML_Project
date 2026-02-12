"""Face detection and alignment using 5 landmarks (MTCNN)."""
from PIL import Image
import numpy as np

_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(keep_all=False, post_process=False)
    return _mtcnn

def align_face(pil_image, output_size=128, padding=0.2):
    """Detect face with 5 landmarks, crop with padding and resize. Fallback: center crop."""
    try:
        mtcnn = _get_mtcnn()
        boxes, _, landmarks = mtcnn.detect(pil_image, landmarks=True)
        if boxes is None or len(boxes) == 0:
            raise ValueError("No face")
        x1, y1, x2, y2 = boxes[0]
        w_box = x2 - x1
        h_box = y2 - y1
        x1 = max(0, x1 - padding * w_box)
        y1 = max(0, y1 - padding * h_box)
        x2 = min(pil_image.width, x2 + padding * w_box)
        y2 = min(pil_image.height, y2 + padding * h_box)
        crop = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
        return crop.resize((output_size, output_size), Image.BILINEAR)
    except Exception:
        w, h = pil_image.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return pil_image.crop((left, top, left + s, top + s)).resize((output_size, output_size), Image.BILINEAR)

import io
import hashlib
import logging
from PIL import Image

logger = logging.getLogger("papervision")

def optimize_image(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """
    Resizes the PIL Image if either width or height exceeds `max_side` (1024px by default).
    Preserves the original aspect ratio exactly.
    """
    try:
        w, h = image.size
        if w <= max_side and h <= max_side:
            return image

        if w > h:
            new_w = max_side
            new_h = max_side * h // w
        else:
            new_h = max_side
            new_w = max_side * w // h

        logger.debug(f"Optimizing image size from {w}x{h} to {new_w}x{new_h}")
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.warning(f"Failed to optimize image, returning original: {e}")
        return image

def compute_image_hash(image_bytes: bytes) -> str:
    """
    Generates a deterministic SHA-256 hex string of the raw image bytes.
    Used for filtering out repeated publisher logos, icons, and page-decorations.
    """
    return hashlib.sha256(image_bytes).hexdigest()

# FILE: kyc_system/utils/image_utils.py
"""
Image preprocessing helpers for the KYC Verification System.
All functions accept and return numpy arrays (BGR convention used by OpenCV)
unless otherwise noted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.logger import get_logger

log = get_logger(__name__)


class ImageReadError(Exception):
    """Raised when an image file cannot be read or decoded."""


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_image(path: Path | str) -> np.ndarray:
    """
    Load an image from disk and return it as a BGR numpy array.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        BGR image as ``np.ndarray`` with dtype ``uint8``.

    Raises:
        ImageReadError: If the file does not exist or cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise ImageReadError(f"Could not read image: {path} — file not found.")
    img = cv2.imread(str(path))
    if img is None:
        raise ImageReadError(f"Could not read image: {path} — unsupported format or corrupt file.")
    return img


def save_image(img: np.ndarray, path: Path | str) -> None:
    """
    Save a BGR numpy array to disk.

    Args:
        img:  BGR image array.
        path: Destination file path (parent directories must exist).

    Raises:
        ImageReadError: If the write operation fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), img)
    if not success:
        raise ImageReadError(f"Could not write image to: {path}")


# ─── Conversion helpers ───────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR numpy array to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB numpy array to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL Image (RGB mode)."""
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a BGR numpy array."""
    return rgb_to_bgr(np.array(pil_img.convert("RGB")))


# ─── Resize helpers ───────────────────────────────────────────────────────────

def resize_keep_aspect(
    img: np.ndarray,
    max_w: int,
    max_h: int,
    inter: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize *img* so that it fits within (*max_w*, *max_h*) while preserving
    the original aspect ratio.

    Args:
        img:   Input BGR image.
        max_w: Maximum output width in pixels.
        max_h: Maximum output height in pixels.
        inter: OpenCV interpolation flag (default ``cv2.INTER_AREA``).

    Returns:
        Resized BGR image.
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale >= 1.0:
        return img  # no upscaling needed
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=inter)


def center_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of *img* to the given *(width, height)*.

    Args:
        img:  Input image.
        size: Target ``(width, height)`` tuple.

    Returns:
        Cropped image or the original if already smaller.
    """
    h, w = img.shape[:2]
    tw, th = size
    if w < tw or h < th:
        return img
    x_start = (w - tw) // 2
    y_start = (h - th) // 2
    return img[y_start:y_start + th, x_start:x_start + tw]


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize_face(face: np.ndarray, target_size: Tuple[int, int] = (80, 80)) -> np.ndarray:
    """
    Resize and normalise a face crop for liveness / deepfake model input.

    Processing steps:
    1. Resize to *target_size*.
    2. Convert to float32.
    3. Scale pixel values to [0, 1].
    4. Standardise with ImageNet mean/std (channel-wise).

    Args:
        face:        BGR face crop.
        target_size: ``(width, height)`` that the model expects.

    Returns:
        Float32 np array shaped ``(H, W, 3)`` in RGB channel order.
    """
    resized = cv2.resize(face, target_size)
    rgb = bgr_to_rgb(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / std


def normalize_for_deepfake(face: np.ndarray) -> np.ndarray:
    """
    Resize and normalise a face crop for the EfficientNet deepfake classifier.

    Args:
        face: BGR face crop.

    Returns:
        Float32 np array shaped ``(299, 299, 3)`` in RGB channel order.
    """
    return normalize_face(face, target_size=(299, 299))


# ─── PIL / CTkImage helpers ───────────────────────────────────────────────────

def pil_thumbnail(pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """
    Return a thumbnail copy of *pil_img* that fits within (*max_w*, *max_h*).

    Args:
        pil_img: Source PIL Image.
        max_w:   Maximum output width.
        max_h:   Maximum output height.

    Returns:
        New PIL Image that has been down-scaled.
    """
    copy = pil_img.copy()
    copy.thumbnail((max_w, max_h), Image.LANCZOS)
    return copy


def draw_face_box(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw a bounding-box rectangle (and optional text label) on a BGR image.

    Args:
        img:       BGR image (will be drawn on in-place on a copy).
        bbox:      ``(x1, y1, x2, y2)`` pixel coordinates.
        color:     BGR colour tuple for the rectangle.
        thickness: Line thickness in pixels.
        label:     Optional text rendered above the box.

    Returns:
        New BGR image with the annotation drawn.
    """
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            out, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out

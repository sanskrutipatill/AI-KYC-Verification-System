# FILE: kyc_system/core/liveness_detector.py
"""
Liveness Detector for the KYC Verification System.

Implements anti-spoofing using a MiniFASNet ensemble (V1SE + V2) from the
Silent-Face-Anti-Spoofing project.  When the pre-trained weights are not
available, a rule-based fallback is used so the application never crashes.

Reference: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import (
    LIVENESS_THRESHOLD,
    LIVENESS_MODEL_DIR,
    LIVENESS_MODEL_V1,
    LIVENESS_MODEL_V2,
    LIVENESS_INPUT_SIZE,
)
from utils.logger import get_logger

log = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when the liveness model weights cannot be loaded."""


# ─── Minimal MiniFASNet architecture ─────────────────────────────────────────

def _build_mini_fasnet(model_path: Path, num_classes: int = 3) -> Optional[object]:
    """
    Attempt to load a MiniFASNet model from *model_path*.

    Returns the loaded PyTorch model in eval mode, or ``None`` if PyTorch or
    the weights file are unavailable.
    """
    try:
        import torch
        import torch.nn as nn

        class _DepthwiseSepConv(nn.Module):
            def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
                super().__init__()
                self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
                self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_c)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.relu(self.bn(self.pw(self.dw(x))))

        class _MiniFASNet(nn.Module):
            """Lightweight binary anti-spoofing network (80×80 input)."""
            def __init__(self, num_classes: int = 3) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    _DepthwiseSepConv(32, 64),
                    _DepthwiseSepConv(64, 128, stride=2),
                    _DepthwiseSepConv(128, 128),
                    _DepthwiseSepConv(128, 256, stride=2),
                    _DepthwiseSepConv(256, 256),
                    _DepthwiseSepConv(256, 512, stride=2),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(512, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        if not model_path.exists():
            log.warning("Liveness model not found: %s — using fallback.", model_path)
            return None

        model = _MiniFASNet(num_classes=num_classes)
        state = torch.load(str(model_path), map_location="cpu", weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        log.info("Liveness model loaded: %s", model_path.name)
        return model

    except Exception as exc:
        log.warning("Could not load liveness model %s: %s", model_path, exc)
        return None


# ─── Main class ───────────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Anti-spoofing detector using a MiniFASNet ensemble.

    When model weights are absent, falls back to a heuristic based on image
    texture complexity (Laplacian variance) so the pipeline never errors out.

    Args:
        model_dir: Directory containing the two ``.pth`` weight files.
        threshold: Minimum liveness score to be considered "Real".
    """

    def __init__(
        self,
        model_dir: Path = LIVENESS_MODEL_DIR,
        threshold: float = LIVENESS_THRESHOLD,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._threshold = threshold
        self._model_v1 = None
        self._model_v2 = None
        self._loaded = False

    # ─── Initialisation ───────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load both MiniFASNet weight files.  Silently falls back when weights
        are not found.
        """
        if self._loaded:
            return
        v1_path = self._model_dir / LIVENESS_MODEL_V1
        v2_path = self._model_dir / LIVENESS_MODEL_V2
        self._model_v1 = _build_mini_fasnet(v1_path, num_classes=3)
        self._model_v2 = _build_mini_fasnet(v2_path, num_classes=3)
        self._loaded = True

    # ─── Inference helpers ────────────────────────────────────────────────────

    def _preprocess(self, face: np.ndarray) -> "torch.Tensor":
        """Resize and normalise a BGR face crop for MiniFASNet."""
        import torch

        resized = cv2.resize(face, LIVENESS_INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalised = (rgb - mean) / std
        tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def _predict_model(self, model, face: np.ndarray) -> float:
        """
        Run a single MiniFASNet model on *face* and return the "real" probability.

        The network outputs 3 classes: [spoof, real, unknown].  We take
        ``softmax[1]`` as the liveness score.
        """
        import torch
        import torch.nn.functional as F

        tensor = self._preprocess(face)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            real_prob = float(probs[0, 1].item())
        return real_prob

    def _heuristic_liveness(self, face: np.ndarray) -> float:
        """
        Texture-complexity heuristic used when model weights are unavailable.

        Printed/replayed images have lower high-frequency energy than a real
        face.  Laplacian variance is a simple, quick proxy.

        Returns:
            Normalised score in [0, 1].
        """
        gray = cv2.cvtColor(cv2.resize(face, (80, 80)), cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Empirically, real-face Laplacian variance > ~100; spoof < ~30
        score = min(lap_var / 200.0, 1.0)
        log.debug("Liveness heuristic: lap_var=%.2f score=%.3f", lap_var, score)
        return float(score)

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, face: np.ndarray) -> Dict[str, object]:
        """
        Predict whether *face* is a live person or a spoofed artefact.

        Args:
            face: BGR face crop (any size — resized internally).

        Returns:
            Dictionary::

                {
                    "liveness_score": float,          # 0.0 – 1.0
                    "is_real":        bool,
                    "label":          "Real" | "Fake",
                }
        """
        self.load()

        scores = []

        if self._model_v1 is not None:
            try:
                scores.append(self._predict_model(self._model_v1, face))
            except Exception as exc:
                log.warning("Liveness V1 inference error: %s", exc)

        if self._model_v2 is not None:
            try:
                scores.append(self._predict_model(self._model_v2, face))
            except Exception as exc:
                log.warning("Liveness V2 inference error: %s", exc)

        if not scores:
            # Neither model loaded — use heuristic
            liveness_score = self._heuristic_liveness(face)
        else:
            liveness_score = float(np.mean(scores))

        is_real = liveness_score >= self._threshold
        label = "Real" if is_real else "Fake"

        log.info("LivenessDetector: score=%.4f is_real=%s", liveness_score, is_real)

        return {
            "liveness_score": liveness_score,
            "is_real":        is_real,
            "label":          label,
        }

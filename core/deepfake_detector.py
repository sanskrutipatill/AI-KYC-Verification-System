# FILE: kyc_system/core/deepfake_detector.py
"""
Deepfake Detector for the KYC Verification System.

Uses an EfficientNet-B4 binary classifier (via ``timm``) to estimate the
probability that a face image is AI-generated or manipulated.

When fine-tuned weights are absent from config.DEEPFAKE_MODEL_PATH, the
module loads the ImageNet-pretrained backbone and applies a sigmoid head.
The resulting probabilities will not be calibrated but provide a reasonable
structural check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from config import DEEPFAKE_THRESHOLD, DEEPFAKE_INPUT_SIZE, DEEPFAKE_MODEL_PATH
from utils.logger import get_logger

log = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when the deepfake model cannot be initialised."""


class DeepfakeDetector:
    """
    EfficientNet-B4 deepfake binary classifier.

    Input:  299×299 normalised RGB face tensor.
    Output: probability that the face is AI-generated (0 = real, 1 = fake).

    Args:
        model_path: Path to fine-tuned model weights (``.pth``).  If the
                    file does not exist, pretrained ImageNet weights are used.
        threshold:  Probability above which a face is flagged as a deepfake.
    """

    def __init__(
        self,
        model_path: Path = DEEPFAKE_MODEL_PATH,
        threshold: float = DEEPFAKE_THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path)
        self._threshold = threshold
        self._model = None
        self._loaded = False

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _build_model(self):
        """Construct the EfficientNet-B4 model with a binary classification head."""
        try:
            import timm
            import torch.nn as nn

            backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)
            in_features = backbone.num_features

            class _DeepfakeNet(nn.Module):
                """EfficientNet-B4 backbone + sigmoid binary head."""
                def __init__(self) -> None:
                    super().__init__()
                    self.backbone = backbone
                    self.head = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(in_features, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    feats = self.backbone(x)
                    return self.head(feats)

            return _DeepfakeNet()
        except ImportError as exc:
            raise ModelLoadError(
                "timm is not installed. Run:\n"
                "  pip install timm==0.9.7 torch==2.1.2"
            ) from exc

    def load(self) -> None:
        """
        Build the model and load weights if available.

        Silently falls back to pretrained ImageNet weights when the KYC
        fine-tuned weights file is not present.
        """
        if self._loaded:
            return
        try:
            import torch

            self._model = self._build_model()
            if self._model_path.exists():
                state = torch.load(str(self._model_path), map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self._model.load_state_dict(state, strict=False)
                log.info("DeepfakeDetector: loaded fine-tuned weights from %s", self._model_path)
            else:
                log.warning(
                    "DeepfakeDetector: no fine-tuned weights at %s — using ImageNet pretrained.",
                    self._model_path,
                )
            self._model.eval()
            self._loaded = True
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"DeepfakeDetector: could not build model — {exc}") from exc

    # ─── Inference helpers ────────────────────────────────────────────────────

    def _preprocess(self, face: np.ndarray) -> "torch.Tensor":
        """Resize and normalise face for EfficientNet input."""
        import torch

        target_h, target_w = DEEPFAKE_INPUT_SIZE
        resized = cv2.resize(face, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    # ─── Public API ───────────────────────────────────────────────────────────

    def detect(self, face: np.ndarray) -> Dict[str, object]:
        """
        Estimate the probability that *face* is a deepfake.

        Args:
            face: BGR face crop (any size — resized internally).

        Returns:
            Dictionary::

                {
                    "deepfake_prob":  float,   # 0.0 – 1.0
                    "is_deepfake":    bool,
                }
        """
        try:
            self.load()
        except ModelLoadError as exc:
            log.error("DeepfakeDetector.detect: %s", exc)
            return {"deepfake_prob": 0.0, "is_deepfake": False}

        try:
            import torch

            tensor = self._preprocess(face)
            with torch.no_grad():
                prob = float(self._model(tensor).squeeze().item())
        except Exception as exc:
            log.error("DeepfakeDetector inference error: %s", exc)
            prob = 0.0

        is_deepfake = prob >= self._threshold
        log.info("DeepfakeDetector: prob=%.4f is_deepfake=%s", prob, is_deepfake)

        return {
            "deepfake_prob": prob,
            "is_deepfake":   is_deepfake,
        }

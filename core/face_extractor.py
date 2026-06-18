# FILE: kyc_system/core/face_extractor.py
"""
Face Extractor for the KYC Verification System.

Uses InsightFace's ``buffalo_l`` model pack to detect and crop the largest
face from a document image.  The face crop is used downstream by the face
matcher, liveness detector, and deepfake detector.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class ModelLoadError(Exception):
    """Raised when InsightFace models cannot be loaded."""


class FaceNotFoundError(Exception):
    """Raised when no face is detected in the supplied image."""


# ─── Face Extractor ───────────────────────────────────────────────────────────

class FaceExtractor:
    """
    Detects and crops the largest face from a document image using InsightFace.

    The underlying ``FaceAnalysis`` model is lazily loaded on first use, so
    the GUI is not blocked at startup.

    Args:
        model_name: InsightFace model pack name (default ``"buffalo_l"``).
        det_size:   Detection input resolution as ``(width, height)``.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self._model_name = model_name
        self._det_size = det_size
        self._app = None  # lazy-loaded

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load InsightFace FaceAnalysis model if not already loaded."""
        if self._app is not None:
            return
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            self._app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=self._det_size, det_thresh=0.20)
            log.info("FaceExtractor: loaded model '%s'.", self._model_name)
        except ImportError as exc:
            raise ModelLoadError(
                "InsightFace is not installed. Run:\n"
                "  pip install insightface==0.7.3 onnxruntime==1.16.3"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load InsightFace model: {exc}") from exc

    # ─── Public API ───────────────────────────────────────────────────────────

    def extract_face(
        self, image: np.ndarray
    ) -> Dict[str, object]:
        """
        Detect all faces in *image* and return the largest one cropped out.

        Args:
            image: BGR numpy array (document scan or photo).

        Returns:
            Dictionary::

                {
                    "face_crop":  np.ndarray,   # BGR face crop
                    "bbox":       (x1,y1,x2,y2) # pixel coords
                    "confidence": float,         # detection score 0-1
                    "embedding":  np.ndarray,    # 512-d ArcFace embedding (if available)
                }

        Raises:
            FaceNotFoundError: If no face was detected in the image.
        """
        self._ensure_loaded()
        faces = self._app.get(image)

        if not faces:
            raise FaceNotFoundError("No face detected in document image.")

        # Select face with largest bounding-box area
        def _area(face) -> float:
            b = face.bbox.astype(int)
            return float((b[2] - b[0]) * (b[3] - b[1]))

        largest = max(faces, key=_area)
        x1, y1, x2, y2 = [max(0, int(v)) for v in largest.bbox]

        # Clamp bounding box to image dimensions
        h, w = image.shape[:2]
        x1, x2 = min(x1, w - 1), min(x2, w)
        y1, y2 = min(y1, h - 1), min(y2, h)

        face_crop = image[y1:y2, x1:x2]
        embedding = largest.embedding if hasattr(largest, "embedding") else None
        det_score = float(largest.det_score) if hasattr(largest, "det_score") else 1.0

        # InsightFace gender attribute: 0 = Female, 1 = Male
        detected_gender = None
        if hasattr(largest, "gender"):
            detected_gender = "MALE" if int(largest.gender) == 1 else "FEMALE"

        log.info(
            "FaceExtractor: detected %d face(s); using bbox=(%d,%d,%d,%d) score=%.3f gender=%s",
            len(faces), x1, y1, x2, y2, det_score, detected_gender,
        )

        return {
            "face_crop":  face_crop,
            "bbox":       (x1, y1, x2, y2),
            "confidence": det_score,
            "embedding":  embedding,
            "detected_gender": detected_gender,
        }

    def get_all_faces(self, image: np.ndarray) -> list:
        """
        Return raw InsightFace face objects for *image* (for advanced use).

        Args:
            image: BGR numpy array.

        Returns:
            List of InsightFace face objects (may be empty).
        """
        self._ensure_loaded()
        return self._app.get(image)

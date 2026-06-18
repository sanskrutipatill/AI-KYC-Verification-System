# FILE: kyc_system/core/face_matcher.py
"""
Face Matcher for the KYC Verification System.

Extracts ArcFace 512-d embeddings from both the document face AND the live
webcam capture (with automatic face-crop detection on both sides) then
computes cosine similarity.  The real InsightFace score is ALWAYS used —
no artificial inflation or clamping of scores below threshold.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import FACE_MATCH_THRESHOLD, FACE_MATCH_MODEL
from utils.logger import get_logger

log = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when InsightFace cannot be loaded."""


class FaceMatcher:
    """
    Computes ArcFace-based cosine similarity between two face images.

    Both images are processed through face-detection → crop → embedding
    so that full frames (e.g. webcam snapshots) are handled correctly.

    Args:
        model_name: InsightFace model pack (default ``"buffalo_l"``).
        threshold:  Cosine similarity threshold for a positive match.
    """

    def __init__(
        self,
        model_name: str = FACE_MATCH_MODEL,
        threshold: float = FACE_MATCH_THRESHOLD,
    ) -> None:
        self._model_name = model_name
        self._threshold  = threshold
        self._app        = None  # lazy-loaded

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the InsightFace FaceAnalysis model."""
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            self._app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.15)
            log.info("FaceMatcher: InsightFace model '%s' loaded.", self._model_name)
        except ImportError as exc:
            raise ModelLoadError(
                "InsightFace is not installed. Run:\n"
                "  pip install insightface==0.7.3 onnxruntime==1.16.3"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"FaceMatcher: model load failed — {exc}") from exc

    # ─── Face crop helper ─────────────────────────────────────────────────────

    def _crop_best_face(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect all faces in *img*, crop the largest, and return its embedding.

        Returns:
            (face_crop_bgr, embedding_or_None)
        """
        self._ensure_loaded()
        faces = self._app.get(img)
        if not faces:
            # Try with a slightly bigger detection threshold fallback
            return img, None  # return full image as-is; embedding will be None

        # Pick highest-confidence face
        best = max(faces, key=lambda f: float(f.det_score) if hasattr(f, "det_score") else 0.0)
        x1, y1, x2, y2 = [max(0, int(v)) for v in best.bbox]
        h, w = img.shape[:2]
        x2, y2 = min(x2, w), min(y2, h)
        # Add 10 % padding so the crop is not too tight
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = img[y1:y2, x1:x2]
        emb  = best.embedding.astype(np.float32) if hasattr(best, "embedding") and best.embedding is not None else None
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        return crop, emb

    # ─── Embedding extraction ─────────────────────────────────────────────────

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the ArcFace embedding for an image.

        First tries to detect and crop a face; if the model already
        computed the embedding during detection it is reused directly.

        Args:
            img: BGR numpy array (any size — resized internally).

        Returns:
            512-d float32 embedding vector, L2-normalised.

        Raises:
            ValueError: If no face is detected in *img*.
        """
        self._ensure_loaded()
        crop, emb = self._crop_best_face(img)
        if emb is not None:
            return emb   # already computed during detection

        # Embedding not available from detection pass — run on the crop
        faces = self._app.get(crop)
        if not faces:
            raise ValueError("No face detected; cannot compute ArcFace embedding.")
        best = max(faces, key=lambda f: float(f.det_score) if hasattr(f, "det_score") else 0.0)
        emb  = best.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ─── Cosine similarity ────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalised embedding vectors.

        Returns:
            Cosine similarity in the range [-1, 1].
        """
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    # ─── Public API ───────────────────────────────────────────────────────────

    def match(
        self,
        doc_face:  np.ndarray,
        live_face: np.ndarray,
    ) -> Dict[str, object]:
        """
        Compare the document face against the live-capture face.

        Both images are run through face-detection so passing a full webcam
        frame is fine — the matcher will automatically locate and use the
        best detected face in each image.

        Returns:
            {
                "score":      float,  # raw ArcFace cosine similarity (0–1)
                "matched":    bool,   # True if score >= threshold
                "percentage": int,    # score * 100 (rounded)
                "error":      str | None   # set when detection fails
            }
        """
        error = None
        try:
            emb_doc  = self.get_embedding(doc_face)
        except (ValueError, Exception) as exc:
            log.error("FaceMatcher: could not embed DOCUMENT face — %s", exc)
            return {"score": 0.0, "matched": False, "percentage": 0,
                    "error": f"Document face: {exc}"}

        try:
            emb_live = self.get_embedding(live_face)
        except (ValueError, Exception) as exc:
            log.error("FaceMatcher: could not embed LIVE face — %s", exc)
            return {"score": 0.0, "matched": False, "percentage": 0,
                    "error": f"Live face: {exc}"}

        sim         = self.cosine_similarity(emb_doc, emb_live)
        # Clamp negative cosine to 0 (cosmetic only — negative means opposite people)
        sim_clamped = max(0.0, min(1.0, sim))
        matched     = sim_clamped >= self._threshold

        log.info(
            "FaceMatcher: cosine=%.4f (clamped=%.4f) threshold=%.2f matched=%s",
            sim, sim_clamped, self._threshold, matched,
        )

        return {
            "score":      sim_clamped,
            "matched":    matched,
            "percentage": int(round(sim_clamped * 100)),
            "error":      error,
        }

    def match_from_embeddings(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> Dict[str, object]:
        """
        Compare two pre-computed embeddings directly (avoids re-running inference).

        Returns:
            Same structure as :meth:`match`.
        """
        sim         = self.cosine_similarity(emb1, emb2)
        sim_clamped = max(0.0, min(1.0, sim))
        matched     = sim_clamped >= self._threshold
        return {
            "score":      sim_clamped,
            "matched":    matched,
            "percentage": int(round(sim_clamped * 100)),
            "error":      None,
        }

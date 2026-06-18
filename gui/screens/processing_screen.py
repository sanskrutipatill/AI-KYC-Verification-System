# FILE: kyc_system/gui/screens/processing_screen.py
"""
Screen 4 — Processing Screen.

Shows an animated progress bar and a step-by-step status list while the
six AI modules run in a single background thread.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from config import COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING
from utils.logger import get_logger

log = get_logger(__name__)

_STEPS = [
    "Extracting document face…",
    "Running face matching…",
    "Liveness detection…",
    "Deepfake analysis…",
    "Document validation…",
    "Computing risk score…",
]


class ProcessingScreen(ctk.CTkFrame):
    """
    Animated processing screen (Step 4 of the KYC flow).

    Runs all six AI pipeline stages in a single daemon thread, updating
    the UI step list via ``after()`` callbacks as each stage completes.

    Args:
        master:      Parent CTk widget.
        on_done:     Callback invoked with the full results dict when
                     processing completes.
        **kwargs:    Forwarded to CTkFrame.
    """

    def __init__(
        self,
        master,
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_done = on_done
        self._step_labels: List[ctk.CTkLabel] = []
        self._results: Dict[str, Any] = {}
        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the processing screen widgets."""
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text="Verifying Identity",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, pady=(38, 6))

        ctk.CTkLabel(
            self,
            text="Please wait while we analyse your document and facial data",
            font=ctk.CTkFont(size=12),
            text_color=("gray45", "gray60"),
        ).grid(row=1, column=0, pady=(0, 26))

        # Spinner
        self._spinner = ctk.CTkProgressBar(self, mode="indeterminate", width=360)
        self._spinner.grid(row=2, column=0, pady=(0, 32))
        self._spinner.start()

        # Step list
        steps_frame = ctk.CTkFrame(self, corner_radius=12, fg_color=("gray92", "gray14"))
        steps_frame.grid(row=3, column=0, padx=60, pady=4, sticky="ew")

        for i, step_text in enumerate(_STEPS):
            row_frame = ctk.CTkFrame(steps_frame, fg_color="transparent")
            row_frame.grid(row=i, column=0, sticky="ew", padx=20, pady=8)
            row_frame.grid_columnconfigure(1, weight=1)

            icon_lbl = ctk.CTkLabel(
                row_frame,
                text="◻",
                font=ctk.CTkFont(size=18),
                width=28,
                text_color=("gray55", "gray50"),
            )
            icon_lbl.grid(row=0, column=0, padx=(0, 12))

            text_lbl = ctk.CTkLabel(
                row_frame,
                text=step_text,
                font=ctk.CTkFont(size=13),
                anchor="w",
                text_color=("gray40", "gray65"),
            )
            text_lbl.grid(row=0, column=1, sticky="w")

            self._step_labels.append((icon_lbl, text_lbl))

        self._status_msg = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_WARNING,
        )
        self._status_msg.grid(row=4, column=0, pady=(16, 8))

    # ─── Step update ──────────────────────────────────────────────────────────

    def _update_step(self, index: int, success: bool) -> None:
        """
        Mark a pipeline step as complete (✓ or ✗).

        Args:
            index:   Zero-based step index.
            success: ``True`` → green tick; ``False`` → red cross.
        """
        if index >= len(self._step_labels):
            return
        icon_lbl, text_lbl = self._step_labels[index]
        if success:
            icon_lbl.configure(text="✓", text_color=COLOR_SUCCESS)
            text_lbl.configure(text_color=("gray20", "gray85"))
        else:
            icon_lbl.configure(text="✗", text_color=COLOR_DANGER)
            text_lbl.configure(text_color=COLOR_DANGER)

    def _set_status(self, msg: str) -> None:
        """Update the bottom status message."""
        self._status_msg.configure(text=msg)

    # ─── Pipeline runner ──────────────────────────────────────────────────────

    def start_processing(
        self,
        doc_image_path: Path,
        live_face_path: Path,
        doc_type: str,
        ocr_fields: Dict[str, Any],
    ) -> None:
        """
        Launch all six AI stages in a background thread.

        Args:
            doc_image_path: Path to the document image.
            live_face_path: Path to the captured webcam face image.
            doc_type:       Document type key (e.g. ``"AADHAAR"``).
            ocr_fields:     Fields dict returned by the OCR engine.
        """
        # Reset step icons
        for icon_lbl, text_lbl in self._step_labels:
            icon_lbl.configure(text="◻", text_color=("gray55", "gray50"))
            text_lbl.configure(text_color=("gray40", "gray65"))
        self._status_msg.configure(text="")
        self._spinner.start()
        self._results = {}

        t = threading.Thread(
            target=self._pipeline,
            args=(doc_image_path, live_face_path, doc_type, ocr_fields),
            daemon=True,
        )
        t.start()

    def _pipeline(
        self,
        doc_image_path: Path,
        live_face_path: Path,
        doc_type: str,
        ocr_fields: Dict[str, Any],
    ) -> None:
        """The complete six-stage KYC pipeline (runs in background thread)."""
        import cv2
        from utils.image_utils import load_image, ImageReadError

        results: Dict[str, Any] = {
            "doc_type":    doc_type,
            "ocr_fields":  ocr_fields,
        }

        # ── Load images ──────────────────────────────────────────────────────
        try:
            doc_img  = load_image(doc_image_path)
            live_img = load_image(live_face_path)
        except ImageReadError as exc:
            self.after(0, self._set_status, f"Image error: {exc}")
            self.after(0, self._spinner.stop)
            return

        # ── Step 1: Extract document face + embedding ──────────────────────
        doc_face     = None
        doc_face_emb = None
        try:
            from core.face_extractor import FaceExtractor, FaceNotFoundError
            extractor = FaceExtractor()
            face_result = extractor.extract_face(doc_img)
            doc_face     = face_result["face_crop"]
            doc_face_emb = face_result.get("embedding")   # 512-d ArcFace emb
            results["doc_face"] = doc_face
            self.after(0, self._update_step, 0, True)
            log.info("Step 1: doc face extracted, emb=%s", "yes" if doc_face_emb is not None else "no")
        except Exception as exc:
            log.error("Step 1 failed: %s", exc)
            results["doc_face_error"] = str(exc)
            self.after(0, self._update_step, 0, False)

        # ── Extract live face + embedding (shared for Steps 2, 3 gender) ──
        live_face_crop = None
        live_face_emb  = None
        detected_gender = None
        try:
            from core.face_extractor import FaceExtractor
            live_extractor = FaceExtractor()
            live_result = live_extractor.extract_face(live_img)
            live_face_crop = live_result["face_crop"]
            live_face_emb  = live_result.get("embedding")
            detected_gender = live_result.get("detected_gender")
            log.info("Pipeline: detected gender from live face = %s", detected_gender)
        except Exception as exc:
            log.warning("Pipeline: could not extract live face — %s", exc)

        # ── Extract gender from OCR fields (document gender) ──────────────
        extracted_gender = None
        gender_field = ocr_fields.get("gender")
        if isinstance(gender_field, dict):
            extracted_gender = gender_field.get("value")
        elif isinstance(gender_field, str):
            extracted_gender = gender_field
        if extracted_gender:
            extracted_gender = extracted_gender.strip().upper()
        log.info("Pipeline: extracted gender from document = %s", extracted_gender)

        # ── Step 2: Face matching ───────────────────────────────────────────
        face_match_result = {"score": 0.0, "matched": False, "percentage": 0, "error": None}
        try:
            from core.face_matcher import FaceMatcher
            import numpy as np
            matcher = FaceMatcher()

            # ── Strategy: prefer pre-computed embeddings (avoids re-detecting
            #    tiny face crops which confuses the detector) ──────────────
            if doc_face_emb is not None and live_face_emb is not None:
                # Best path: both embeddings are already computed — just compare
                log.info("Step 2: using pre-computed embeddings for both faces.")
                face_match_result = matcher.match_from_embeddings(doc_face_emb, live_face_emb)
                face_match_result["error"] = None
            elif doc_face_emb is not None and live_face_emb is None:
                # Doc has emb; extract live from the full image
                log.info("Step 2: doc emb ready; extracting live emb from full image.")
                try:
                    emb_live = matcher.get_embedding(live_img)
                    face_match_result = matcher.match_from_embeddings(doc_face_emb, emb_live)
                    face_match_result["error"] = None
                except Exception as e2:
                    face_match_result["error"] = str(e2)
            else:
                # Fallback: run matcher on full images (it will detect & crop internally)
                img_doc  = doc_img   # use full doc image so detector can find the face
                img_live = live_img
                log.info("Step 2: no pre-computed embeddings — running full-image match.")
                face_match_result = matcher.match(img_doc, img_live)

            results["face_match"] = face_match_result
            match_ok  = face_match_result["matched"]
            match_err = face_match_result.get("error")
            if match_err:
                self.after(0, self._set_status, f"⚠️  Face detect: {match_err[:60]}")
            self.after(0, self._update_step, 1, match_ok)
        except Exception as exc:
            log.error("Step 2 failed: %s", exc)
            results["face_match"] = face_match_result
            self.after(0, self._set_status, f"❌  Face match error: {str(exc)[:60]}")
            self.after(0, self._update_step, 1, False)

        # ── Step 3: Liveness detection ─────────────────────────────────────
        liveness_result = {"liveness_score": 0.5, "is_real": True, "label": "Real"}
        try:
            from core.liveness_detector import LivenessDetector
            detector = LivenessDetector()
            face_for_live = doc_face if doc_face is not None else live_img
            liveness_result = detector.detect(live_img)
            results["liveness"] = liveness_result
            self.after(0, self._update_step, 2, liveness_result["is_real"])
        except Exception as exc:
            log.error("Step 3 failed: %s", exc)
            results["liveness"] = liveness_result
            self.after(0, self._update_step, 2, False)

        # ── Step 4: Deepfake detection ─────────────────────────────────────
        deepfake_result = {"deepfake_prob": 0.0, "is_deepfake": False}
        try:
            from core.deepfake_detector import DeepfakeDetector
            df_det = DeepfakeDetector()
            face_for_df = doc_face if doc_face is not None else live_img
            deepfake_result = df_det.detect(face_for_df)
            results["deepfake"] = deepfake_result
            self.after(0, self._update_step, 3, not deepfake_result["is_deepfake"])
        except Exception as exc:
            log.error("Step 4 failed: %s", exc)
            results["deepfake"] = deepfake_result
            self.after(0, self._update_step, 3, True)

        # ── Step 5: Document validation ────────────────────────────────────
        doc_val_result = {"score": 0.5, "is_valid": True, "fail_reasons": []}
        try:
            from core.document_validator import DocumentValidator
            validator = DocumentValidator()
            doc_val_result = validator.validate(doc_type, ocr_fields)
            results["doc_validation"] = doc_val_result
            self.after(0, self._update_step, 4, doc_val_result["is_valid"])
        except Exception as exc:
            log.error("Step 5 failed: %s", exc)
            results["doc_validation"] = doc_val_result
            self.after(0, self._update_step, 4, False)

        # ── Step 6: Risk score ─────────────────────────────────────────────
        risk_result = {"risk_score": 0.5, "status": "REVIEW", "reason": "Unable to compute score."}
        try:
            from core.risk_scorer import RiskScorer

            # Compute field completeness
            non_meta = {k: v for k, v in ocr_fields.items()
                        if k not in ("raw_text", "avg_confidence")}
            found = sum(
                1 for v in non_meta.values()
                if isinstance(v, dict) and v.get("value") is not None
            )
            completeness = found / max(len(non_meta), 1)

            scorer = RiskScorer()
            risk_result = scorer.score(
                face_match_score         = face_match_result["score"],
                liveness_score           = liveness_result["liveness_score"],
                ocr_confidence_avg       = ocr_fields.get("avg_confidence", 0.5),
                deepfake_prob            = deepfake_result["deepfake_prob"],
                doc_validation_score     = doc_val_result["score"],
                field_completeness_ratio = completeness,
                face_matched             = face_match_result["matched"],
                is_real                  = liveness_result["is_real"],
                is_deepfake              = deepfake_result["is_deepfake"],
                doc_valid                = doc_val_result["is_valid"],
                extracted_gender         = extracted_gender,
                detected_gender          = detected_gender,
            )
            results["risk"] = risk_result
            self.after(0, self._update_step, 5, True)
        except Exception as exc:
            log.error("Step 6 failed: %s", exc)
            results["risk"] = risk_result
            self.after(0, self._update_step, 5, False)

        self.after(0, self._spinner.stop)
        self.after(0, self._set_status, "Analysis complete!")
        self.after(200, self._finish, results)

    def _finish(self, results: Dict[str, Any]) -> None:
        """Invoke the completion callback on the main thread."""
        if self._on_done:
            self._on_done(results)

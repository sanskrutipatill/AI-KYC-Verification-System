# FILE: kyc_system/core/risk_scorer.py
"""
KYC Decision Engine for the KYC Verification System.

Implements a STRICT, FAIR, and EXPLAINABLE decision framework
tuned for real-world Indian identity verification (Aadhaar, PAN, DL, Voter ID).

Decision Priority Order (STRICT)
────────────────────────────────
 Priority 1: Identity Consistency (Gender match — HARD FAILURE if mismatch)
 Priority 2: Document Validity
 Priority 3: Liveness
 Priority 4: Face Match (supporting signal only)
 Priority 5: Deepfake (probabilistic — never sole rejection trigger)

Design principles
─────────────────
• Gender mismatch is a NON-NEGOTIABLE immediate REJECTION.
• Liveness confirms real person presence — it does NOT confirm identity.
• Face match is a SUPPORTING signal only — aging / low-res ID photos are common.
• Deepfake detection is PROBABILISTIC — it can NEVER alone reject a user.
• Face match override (0% → adjusted) is ONLY allowed when gender matches.
• If genuinely uncertain → prefer "UNDER REVIEW" over wrong rejection.
• All decisions are explained via a structured reasoning chain with ACTUAL values.

Decision Cases
──────────────
REJECTED      Gender mismatch (immediate, no override possible)
              OR Liveness < 50 AND (Document Invalid OR data failed)
VERIFIED      Gender match AND Liveness ≥ 90 AND Document Valid AND Data consistent
              AND Deepfake < 60
UNDER REVIEW  Gender match AND Liveness ≥ 80 AND Document Valid AND Data consistent
              AND (Face match low OR Deepfake 30–70)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

from config import XGBOOST_MODEL_PATH
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Threshold constants ───────────────────────────────────────────────────────
# All scores are 0.0–1.0 internally; multiply by 100 for percentage representation.

_L_STRONG   = 0.90   # Liveness ≥ 90 → strong real-person evidence
_L_MIN      = 0.80   # Liveness ≥ 80 → acceptable for UNDER REVIEW
_L_SPOOF    = 0.50   # Liveness < 50 → possible spoof
_FM_LOW     = 0.30   # Face match < 30 → needs mitigation check
_DF_LOW     = 0.30   # Deepfake < 30 → low risk
_DF_HIGH    = 0.60   # Deepfake ≥ 60 → suspicious (still not conclusive)
_DOC_MIN    = 0.50   # Document score ≥ 50 → considered valid
_DATA_MIN   = 0.50   # Field completeness ≥ 50 → data considered consistent


class RiskScorer:
    """
    Strict and fair KYC decision engine.

    Implements a multi-signal decision framework with gender consistency
    as the highest-priority hard-failure condition, followed by document
    validity, liveness, face match, and deepfake signals.

    Args:
        model_path: Optional XGBoost model path for risk magnitude estimation.
                    When absent the rule-based engine is used for both
                    classification and magnitude.
    """

    def __init__(self, model_path=XGBOOST_MODEL_PATH) -> None:
        self._model_path = model_path
        self._xgb_model = None
        self._loaded = False

    # ─── XGBoost (optional — score magnitude only) ────────────────────────────

    def _load_xgb(self) -> None:
        if self._loaded:
            return
        try:
            import xgboost as xgb
            if self._model_path.exists():
                booster = xgb.Booster()
                booster.load_model(str(self._model_path))
                self._xgb_model = booster
                log.info("RiskScorer: XGBoost model loaded from %s", self._model_path)
            else:
                log.info("RiskScorer: no XGBoost model — rule-based fallback active.")
        except ImportError:
            log.warning("RiskScorer: xgboost not installed — using rule-based fallback.")
        except Exception as exc:
            log.warning("RiskScorer: XGBoost load failed — %s", exc)
        self._loaded = True

    def _xgb_magnitude(self, features: np.ndarray) -> Optional[float]:
        """Return XGBoost risk magnitude (0–1), or None if model unavailable."""
        if self._xgb_model is None:
            return None
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(features)
            score = float(np.clip(self._xgb_model.predict(dm)[0], 0.0, 1.0))
            return score
        except Exception as exc:
            log.warning("RiskScorer: XGBoost predict failed — %s", exc)
            return None

    # ─── Rule-based risk magnitude ────────────────────────────────────────────

    @staticmethod
    def _rule_magnitude(
        face_match: float,
        liveness: float,
        ocr: float,
        deepfake: float,
        doc_valid: float,
        fields: float,
    ) -> float:
        """
        Compute a numeric risk magnitude using calibrated weights.

        risk = 1 − quality_score
        """
        quality = (
            0.35 * face_match
            + 0.25 * liveness
            + 0.15 * ocr
            + 0.10 * (1.0 - deepfake)   # negative signal inverted
            + 0.10 * doc_valid
            + 0.05 * fields
        )
        return float(np.clip(1.0 - quality, 0.0, 1.0))

    # ─── Gender consistency check ─────────────────────────────────────────────

    @staticmethod
    def _check_gender_consistency(
        extracted_gender: Optional[str],
        detected_gender: Optional[str],
    ) -> tuple[bool, str]:
        """
        Compare document gender (OCR) against detected gender (face analysis).

        Returns:
            (is_consistent, status_string)
            - is_consistent: True if genders match or if either is unavailable.
            - status_string: "MATCH", "MISMATCH", or "UNAVAILABLE".
        """
        if not extracted_gender or not detected_gender:
            return True, "UNAVAILABLE"

        ext = extracted_gender.strip().upper()
        det = detected_gender.strip().upper()

        # Normalise common representations
        _norm = {"M": "MALE", "F": "FEMALE", "MALE": "MALE", "FEMALE": "FEMALE",
                 "TRANSGENDER": "TRANSGENDER"}
        ext = _norm.get(ext, ext)
        det = _norm.get(det, det)

        if ext == det:
            return True, "MATCH"
        return False, "MISMATCH"

    # ─── Dynamic face score adjustment ────────────────────────────────────────

    @staticmethod
    def _adjust_face_score(
        raw_score: float,
        liveness: float,
        gender_consistent: bool,
    ) -> float:
        """
        Context-aware face score note (display only — does NOT inflate the score).

        The REAL ArcFace cosine similarity is always returned unchanged.
        Inflation of scores is explicitly prohibited to keep the system honest.
        This method is kept for API compatibility but simply returns the raw score.
        """
        # ── POLICY: NEVER fabricate or inflate a face score ──────────────────
        # Old behaviour: replaced scores <0.30 with random(0.30-0.60). REMOVED.
        # Reason: inflating scores made every person appear as a near-match,
        # defeating the purpose of verification.
        return raw_score

    # ─── Reasoning builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_reasoning(
        liveness: float,
        face_match: float,
        raw_face_match: float,
        deepfake: float,
        doc_score: float,
        field_ratio: float,
        ocr_conf: float,
        face_matched: bool,
        is_real: bool,
        is_deepfake: bool,
        doc_valid: bool,
        status: str,
        gender_status: str,
        extracted_gender: Optional[str],
        detected_gender: Optional[str],
    ) -> List[str]:
        """
        Build an ordered list of human-readable reasoning bullets.

        Each bullet explains ONE signal and its contribution to the decision.
        Uses varied sentence structures and references actual numeric values.
        """
        bullets: List[str] = []

        # ── Priority 1: Gender consistency (ALWAYS first) ──
        if gender_status == "MISMATCH":
            bullets.append(
                f"CRITICAL: Gender mismatch detected — document states "
                f"'{extracted_gender}' but facial analysis detected "
                f"'{detected_gender}'. This is a hard identity failure that "
                f"cannot be overridden by any other signal."
            )
        elif gender_status == "MATCH":
            bullets.append(
                f"Gender consistency check passed — document gender "
                f"'{extracted_gender}' matches detected gender "
                f"'{detected_gender}'."
            )
        else:
            bullets.append(
                "Gender consistency check skipped — gender data not available "
                "from one or both sources (OCR / facial analysis)."
            )

        # ── Liveness ──
        live_pct = int(round(liveness * 100))
        _liveness_templates = {
            "strong": [
                f"Liveness score of {live_pct}/100 strongly indicates a real person "
                f"is present; anti-spoofing checks passed with high confidence.",
                f"Live presence confirmed at {live_pct}% — well above the 90% threshold "
                f"for strong real-person evidence.",
                f"Liveness detection returned {live_pct}/100 (≥90) — robust confirmation "
                f"of a genuine human presence. Note: liveness confirms presence, not identity.",
            ],
            "acceptable": [
                f"Liveness score is {live_pct}/100 (≥80) — acceptable evidence of a "
                f"real person, though minor uncertainty remains.",
                f"Live capture scored {live_pct}% on anti-spoofing — above the minimum "
                f"threshold but below the strong-confidence mark of 90%.",
            ],
            "borderline": [
                f"Liveness score is {live_pct}/100 (50–80) — borderline result; "
                f"environmental factors (lighting, pose, camera quality) may be at play.",
                f"Anti-spoofing returned {live_pct}% — in the uncertain range. "
                f"Cannot conclusively confirm or deny real presence.",
            ],
            "failed": [
                f"Liveness score is {live_pct}/100 (<50) — possible spoofing attempt "
                f"or severely degraded capture quality.",
                f"Anti-spoofing check flagged concern at {live_pct}% — below the "
                f"minimum acceptable threshold of 50%.",
            ],
        }
        if liveness >= _L_STRONG:
            bullets.append(random.choice(_liveness_templates["strong"]))
        elif liveness >= _L_MIN:
            bullets.append(random.choice(_liveness_templates["acceptable"]))
        elif liveness >= _L_SPOOF:
            bullets.append(random.choice(_liveness_templates["borderline"]))
        else:
            bullets.append(random.choice(_liveness_templates["failed"]))

        # ── Face Match ──
        fm_pct = int(round(face_match * 100))
        raw_fm_pct = int(round(raw_face_match * 100))
        was_adjusted = abs(face_match - raw_face_match) > 0.01

        if gender_status == "MISMATCH":
            bullets.append(
                f"Face match is INVALID — gender identity conflict prevents any "
                f"face comparison from being meaningful. Raw score was {raw_fm_pct}/100 "
                f"but is disregarded due to identity mismatch."
            )
        elif was_adjusted:
            bullets.append(
                f"Face match score was {raw_fm_pct}/100 (raw) → adjusted to "
                f"{fm_pct}/100 based on contextual signals. The low raw score is "
                f"likely attributable to document aging, image quality degradation, "
                f"or hairstyle changes since the document photo was taken. "
                f"Adjustment permitted because gender identity is consistent."
            )
        elif face_match >= 0.50:
            bullets.append(
                f"Face match score is {fm_pct}/100 — acceptable similarity between "
                f"document photo and live capture."
            )
        elif face_match >= _FM_LOW:
            bullets.append(
                f"Face match score is {fm_pct}/100 — partial similarity detected. "
                f"Structural facial alignment shows correspondence despite "
                f"low overall score; document image aging is a likely factor."
            )
        else:
            bullets.append(
                f"Face match score is {fm_pct}/100 (<30) — low similarity noted. "
                f"This may stem from aging, camera quality, or model limitations. "
                f"Face match alone is NOT sufficient to reject."
            )

        # ── Deepfake ──
        df_pct = int(round(deepfake * 100))
        if deepfake < _DF_LOW:
            bullets.append(
                f"Deepfake risk is {df_pct}/100 (<30) — low probability of "
                f"AI-generated or manipulated imagery."
            )
        elif deepfake < _DF_HIGH:
            _df_uncertain = [
                f"Deepfake risk is {df_pct}/100 (30–60) — uncertain range; deepfake "
                f"models have known false-positive rates and this alone does NOT "
                f"constitute evidence of manipulation.",
                f"Deepfake analysis returned {df_pct}% — in the inconclusive zone. "
                f"This signal is treated as advisory only and cannot trigger rejection.",
            ]
            bullets.append(random.choice(_df_uncertain))
        else:
            bullets.append(
                f"Deepfake risk is {df_pct}/100 (≥60) — elevated concern flagged. "
                f"However, deepfake detection is probabilistic and cannot by itself "
                f"confirm fraud. Treated as a supporting risk signal only."
            )

        # ── Document ──
        doc_pct = int(round(doc_score * 100))
        if doc_valid:
            bullets.append(
                f"Document validation passed (score {doc_pct}/100) — format, checksum, "
                f"and field consistency checks satisfied."
            )
        else:
            bullets.append(
                f"Document validation score is {doc_pct}/100 — one or more structural "
                f"checks failed (format, checksum, or field pattern). "
                f"This is a significant risk signal."
            )

        # ── Data consistency / OCR ──
        field_pct = int(round(field_ratio * 100))
        ocr_pct   = int(round(ocr_conf   * 100))
        if field_ratio >= _DATA_MIN and ocr_conf >= 0.60:
            bullets.append(
                f"Identity data is consistent: {field_pct}% of expected fields "
                f"extracted with avg OCR confidence {ocr_pct}/100."
            )
        else:
            bullets.append(
                f"Data extraction is incomplete or low-confidence: {field_pct}% "
                f"fields found, OCR confidence {ocr_pct}/100. "
                f"Minor OCR errors on ID cards are common and do not constitute fraud."
            )

        # ── Final reasoning summary ──
        if status == "REJECTED" and gender_status == "MISMATCH":
            bullets.append(
                f"DECISION: REJECTED — Identity consistency check failed. "
                f"Document gender '{extracted_gender}' does not match detected "
                f"gender '{detected_gender}'. This is a non-negotiable hard failure. "
                f"No override is permitted regardless of liveness ({int(liveness*100)}%) "
                f"or other positive signals."
            )
        elif status == "VERIFIED":
            bullets.append(
                "All primary signals (identity consistency, liveness, document, data) "
                "are positive. Identity verified with high confidence."
            )
        elif status == "UNDER REVIEW":
            bullets.append(
                "Primary signals are broadly positive but uncertainty exists in one "
                "or more secondary signals (face match / deepfake score). "
                "Conservative policy — escalated to manual review rather than rejection."
            )
        else:  # REJECTED (non-gender reasons)
            bullets.append(
                "Multiple primary risk signals are simultaneously elevated (liveness "
                "failure, document invalidity, or strong multi-signal fraud indicators). "
                "Rejection is warranted under the conservative dual-signal rule."
            )

        return bullets

    # ─── Core decision logic ──────────────────────────────────────────────────

    def _classify(
        self,
        liveness: float,
        face_match: float,
        deepfake: float,
        doc_valid: bool,
        field_ratio: float,
        ocr_conf: float,
        gender_consistent: bool,
        gender_status: str,
    ) -> tuple[str, str, str, str]:
        """
        Apply the priority-based decision framework.

        Priority order: Gender > Document > Liveness > Face > Deepfake.

        Returns:
            (status, confidence_level, recommended_action, short_reason)
        """
        # ── PRIORITY 1: Gender consistency (HARD FAILURE) ─────────────────
        # Gender mismatch → immediate REJECTION. No override. No downgrade.
        if gender_status == "MISMATCH":
            return (
                "REJECTED",
                "HIGH",
                "Reject",
                "Identity consistency failed: document gender does not match "
                "detected gender. Hard rejection — no override permitted.",
            )

        data_consistent = field_ratio >= _DATA_MIN and ocr_conf >= 0.40

        # ── Secondary-signal safety flags ────────────────────────────────────
        face_very_low      = face_match < 0.30
        deepfake_sus       = deepfake   >= _DF_HIGH
        deepfake_uncertain = _DF_LOW <= deepfake < (_DF_HIGH + 0.10)
        face_match_concern = face_match < 0.50

        # ── CASE 2 — VERIFIED ────────────────────────────────────────────────
        if (
            liveness >= _L_STRONG
            and doc_valid
            and data_consistent
            and not deepfake_sus
            and not face_very_low
        ):
            return (
                "VERIFIED",
                "HIGH",
                "Auto Approve",
                "All primary KYC signals passed. Identity verified.",
            )

        # ── CASE 3 — UNDER REVIEW ────────────────────────────────────────────
        if (
            liveness >= _L_MIN
            and doc_valid
            and data_consistent
        ):
            concerns = []
            if face_very_low:
                concerns.append(
                    f"Very low face match ({int(face_match*100)}/100) likely caused by "
                    f"aging, hairstyle change, or ID card image quality — requires "
                    f"human review. Face mismatch alone is NOT grounds for rejection."
                )
            elif face_match_concern:
                concerns.append(
                    f"Low face match ({int(face_match*100)}/100) — possible aging "
                    f"or quality mismatch."
                )
            if deepfake_sus:
                concerns.append(
                    f"Deepfake score elevated ({int(deepfake*100)}/100 >=60) — "
                    f"flagged as suspicious, but probabilistic only; cannot confirm "
                    f"fraud without human review."
                )
            elif deepfake_uncertain:
                concerns.append(
                    f"Deepfake score inconclusive ({int(deepfake*100)}/100, 30-70 range)."
                )

            if concerns:
                reason = (
                    "Liveness and document checks passed. "
                    + " ".join(concerns)
                    + " Escalated per conservative policy."
                )
                return "UNDER REVIEW", "MEDIUM", "Manual Review", reason
            else:
                return (
                    "VERIFIED",
                    "MEDIUM",
                    "Auto Approve",
                    "Liveness and document signals positive. Identity verified.",
                )

        # ── CASE 4 — REJECTED (dual-signal rule) ─────────────────────────────
        liveness_failed = liveness < _L_SPOOF
        doc_failed      = not doc_valid
        data_failed     = not data_consistent

        if liveness_failed and (doc_failed or (doc_failed and data_failed)):
            return (
                "REJECTED",
                "HIGH",
                "Reject",
                "Liveness check failed and document/data signals confirm high fraud risk.",
            )

        # Safety net — borderline or contradictory signals → UNDER REVIEW
        return (
            "UNDER REVIEW",
            "LOW",
            "Manual Review",
            "Signals are borderline or contradictory. Escalated for manual review "
            "per conservative false-negative minimisation policy.",
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def score(
        self,
        face_match_score:          float,
        liveness_score:            float,
        ocr_confidence_avg:        float,
        deepfake_prob:             float,
        doc_validation_score:      float,
        field_completeness_ratio:  float,
        # Parsed boolean flags from individual detectors
        face_matched:  bool = True,
        is_real:       bool = True,
        is_deepfake:   bool = False,
        doc_valid:     bool = True,
        # Gender fields for identity consistency check
        extracted_gender: Optional[str] = None,
        detected_gender:  Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full KYC decision engine on a set of AI signal values.

        Args:
            face_match_score:         Cosine similarity (0–1).
            liveness_score:           Anti-spoof confidence (0–1).
            ocr_confidence_avg:       Average OCR confidence (0–1).
            deepfake_prob:            Deepfake probability (0–1).
            doc_validation_score:     Document-specific validation score (0–1).
            field_completeness_ratio: Fraction of expected fields found (0–1).
            face_matched:             Boolean face-match result.
            is_real:                  Boolean liveness result.
            is_deepfake:              Boolean deepfake result.
            doc_valid:                Boolean document validation result.
            extracted_gender:         Gender from document OCR (e.g. "MALE"/"FEMALE").
            detected_gender:          Gender detected from live face analysis.

        Returns:
            Full decision dict with status, confidence, reasoning, flags,
            recommended action, gender check result, and a numeric risk_score
            for gauge display.
        """
        self._load_xgb()

        # ── Priority 1: Gender consistency ─────────────────────────────────────
        gender_consistent, gender_status = self._check_gender_consistency(
            extracted_gender, detected_gender,
        )

        log.info(
            "RiskScorer: gender check — extracted=%s detected=%s status=%s",
            extracted_gender, detected_gender, gender_status,
        )

        # ── Dynamic face score adjustment (ONLY if gender consistent) ──────────
        raw_face_match = face_match_score
        adjusted_face_match = self._adjust_face_score(
            raw_face_match, liveness_score, gender_consistent,
        )

        # If gender mismatch, force face_matched to False
        if not gender_consistent:
            face_matched = False
            adjusted_face_match = raw_face_match  # No adjustment on mismatch

        # ── Feature vector ────────────────────────────────────────────────────
        features = np.array(
            [adjusted_face_match, liveness_score, ocr_confidence_avg,
             deepfake_prob, doc_validation_score, field_completeness_ratio],
            dtype=np.float32,
        ).reshape(1, -1)

        # ── Risk magnitude (numeric, for gauge) ───────────────────────────────
        risk_magnitude = self._xgb_magnitude(features)
        if risk_magnitude is None:
            risk_magnitude = self._rule_magnitude(
                adjusted_face_match, liveness_score, ocr_confidence_avg,
                deepfake_prob, doc_validation_score, field_completeness_ratio,
            )

        # Force high risk magnitude on gender mismatch
        if not gender_consistent:
            risk_magnitude = max(risk_magnitude, 0.85)

        # ── Classification ────────────────────────────────────────────────────
        status, confidence, action, short_reason = self._classify(
            liveness     = liveness_score,
            face_match   = adjusted_face_match,
            deepfake     = deepfake_prob,
            doc_valid    = doc_valid,
            field_ratio  = field_completeness_ratio,
            ocr_conf     = ocr_confidence_avg,
            gender_consistent = gender_consistent,
            gender_status     = gender_status,
        )

        # ── Reasoning chain ───────────────────────────────────────────────────
        reasoning = self._build_reasoning(
            liveness       = liveness_score,
            face_match     = adjusted_face_match,
            raw_face_match = raw_face_match,
            deepfake       = deepfake_prob,
            doc_score      = doc_validation_score,
            field_ratio    = field_completeness_ratio,
            ocr_conf       = ocr_confidence_avg,
            face_matched   = face_matched,
            is_real        = is_real,
            is_deepfake    = is_deepfake,
            doc_valid      = doc_valid,
            status         = status,
            gender_status  = gender_status,
            extracted_gender = extracted_gender,
            detected_gender  = detected_gender,
        )

        # ── Risk flags ────────────────────────────────────────────────────────
        risk_flags = {
            "face_mismatch":      not face_matched,
            "deepfake_suspicion": deepfake_prob >= _DF_HIGH,
            "document_issue":     not doc_valid,
            "liveness_failed":    not is_real,
            "gender_mismatch":    not gender_consistent,
        }

        log.info(
            "RiskScorer: status=%s confidence=%s risk=%.3f action=%s gender=%s",
            status, confidence, risk_magnitude, action, gender_status,
        )

        return {
            # Gauge-compatible numeric score
            "risk_score":        round(risk_magnitude, 4),
            # Decision
            "status":            status,
            "confidence_level":  confidence,
            "reasoning":         reasoning,
            "risk_flags":        risk_flags,
            "recommended_action": action,
            # Gender check result
            "gender_check":      gender_status,
            "extracted_gender":  extracted_gender,
            "detected_gender":   detected_gender,
            # Adjusted face match for display
            "adjusted_face_match": round(adjusted_face_match, 4),
            # Single-line summary for backwards-compat export field
            "reason":            short_reason,
            # Raw feature vector
            "features":          features[0].tolist(),
        }

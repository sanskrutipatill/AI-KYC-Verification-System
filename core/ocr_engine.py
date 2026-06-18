# FILE: kyc_system/core/ocr_engine.py
"""
OCR Engine for the KYC Verification System.

Uses PaddleOCR to extract text from document images and applies regex-based
post-processing to identify structured fields for each supported Indian
document type (Aadhaar, PAN, Voter ID, Driving Licence).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import OCR_CONFIDENCE_THRESHOLD  # noqa
from utils.logger import get_logger

log = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when a required ML model cannot be loaded."""


# ─── Field extraction helpers ─────────────────────────────────────────────────

# Date patterns accepted across all documents
_DATE_RE = re.compile(
    r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b"
)

# ─── Per-document regexes ─────────────────────────────────────────────────────

_AADHAAR_NUM_RE = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
_PAN_NUM_RE     = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
_VOTER_ID_RE    = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")
_DL_NUM_RE      = re.compile(r"\b([A-Z]{2}[0-9]{2}\s?[0-9]{4}[0-9]{7})\b")
_GENDER_RE      = re.compile(r"\b(MALE|FEMALE|TRANSGENDER|M|F)\b", re.IGNORECASE)
_NAME_HEADER_RE = re.compile(
    r"(?:Name|Father|Son|Daughter|Husband|Wife|Guardian)[:\s]+([A-Za-z\s\.]+)",
    re.IGNORECASE,
)


def _normalise_date(raw: str) -> str:
    """Convert a raw date string to DD/MM/YYYY format."""
    return raw.replace("-", "/").replace(".", "/")


def _find_first(pattern: re.Pattern, text: str) -> Optional[str]:
    """Return first match of *pattern* in *text*, or None."""
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _confidence_for_field(field_value: Optional[str], line_scores: List[float]) -> float:
    """
    Estimate per-field confidence as the average OCR score of lines that
    contributed to finding the field.  Falls back to 0.0 when the field
    was not found.
    """
    if field_value is None:
        return 0.0
    # Use the median OCR line score as a proxy for field confidence
    if not line_scores:
        return 0.5
    return float(np.median(line_scores))


# ─── Main OCR Engine ──────────────────────────────────────────────────────────

class OCREngine:
    """
    Wraps PaddleOCR to provide field-level extraction for KYC documents.

    The engine is lazily initialised on first use to avoid blocking the GUI
    at startup.

    Args:
        use_gpu:       Whether to run PaddleOCR on GPU (default False).
        use_angle_cls: Whether to enable text-angle classification.
        lang:          OCR language code (default ``'en'``).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        lang: str = "en",
    ) -> None:
        self._use_gpu = use_gpu
        self._use_angle_cls = use_angle_cls
        self._lang = lang
        self._ocr = None  # lazy-loaded

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load PaddleOCR model if not already loaded."""
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                use_angle_cls=self._use_angle_cls,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            log.info("OCREngine: PaddleOCR model loaded successfully.")
        except ImportError as exc:
            raise ModelLoadError(
                "PaddleOCR is not installed. Run:\n"
                "  pip install paddlepaddle==2.5.2 paddleocr==2.7.3"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load PaddleOCR model: {exc}") from exc

    # ─── Public API ───────────────────────────────────────────────────────────

    def extract_text(
        self, image: np.ndarray
    ) -> Tuple[List[str], List[float]]:
        """
        Run PaddleOCR on *image* and return raw text lines with their scores.

        Args:
            image: BGR numpy array of the document image.

        Returns:
            Tuple of (list-of-text-lines, list-of-confidence-scores).
        """
        self._ensure_loaded()
        result = self._ocr.ocr(image, cls=self._use_angle_cls)
        lines: List[str] = []
        scores: List[float] = []
        if result and result[0]:
            for item in result[0]:
                if item and len(item) >= 2:
                    text_info = item[1]
                    if text_info:
                        lines.append(str(text_info[0]))
                        scores.append(float(text_info[1]))
        return lines, scores

    def extract_fields(
        self, image: np.ndarray, doc_type: str
    ) -> Dict[str, Any]:
        """
        Extract structured fields from *image* according to *doc_type*.

        Args:
            image:    BGR numpy array of the document image.
            doc_type: One of ``"AADHAAR"``, ``"PAN"``, ``"VOTER_ID"``,
                      ``"DL"``.

        Returns:
            Dictionary with field names as keys.  Each value is a sub-dict::

                {
                    "value": str | None,
                    "confidence": float          # 0.0 – 1.0
                }

            Plus a top-level ``"raw_text"`` and ``"avg_confidence"`` key.

        Raises:
            ValueError: If *doc_type* is not recognised.
        """
        self._ensure_loaded()
        lines, scores = self.extract_text(image)
        full_text = " ".join(lines)
        avg_score = float(np.mean(scores)) if scores else 0.0

        extractors = {
            "AADHAAR":  self._extract_aadhaar,
            "PAN":      self._extract_pan,
            "VOTER_ID": self._extract_voter_id,
            "DL":       self._extract_dl,
        }
        if doc_type not in extractors:
            raise ValueError(f"Unknown document type: {doc_type!r}")

        fields = extractors[doc_type](full_text, lines, scores)
        fields["raw_text"] = full_text
        fields["avg_confidence"] = avg_score
        log.info("OCREngine: extracted %d fields for %s (avg_conf=%.2f)",
                 len(fields) - 2, doc_type, avg_score)
        return fields

    # ─── Document-specific extractors ─────────────────────────────────────────

    def _field(
        self,
        value: Optional[str],
        line_scores: List[float],
    ) -> Dict[str, Any]:
        """Helper: wrap a field value with its confidence estimate."""
        return {
            "value": value,
            "confidence": _confidence_for_field(value, line_scores),
        }

    def _extract_name_from_lines(
        self, lines: List[str], scores: List[float]
    ) -> Tuple[Optional[str], List[float]]:
        """
        Attempt to extract a person's name from OCR lines using header keywords.

        Returns the name string and contributing scores, or (None, []).
        """
        for i, line in enumerate(lines):
            if re.search(r"\bname\b", line, re.IGNORECASE):
                # Name is usually on the same line after a colon,
                # or on the very next line
                same_line = re.split(r"[:\-]", line, maxsplit=1)
                if len(same_line) > 1:
                    candidate = same_line[1].strip()
                    if len(candidate) > 2:
                        return candidate, [scores[i]] if i < len(scores) else []
                if i + 1 < len(lines):
                    candidate = lines[i + 1].strip()
                    if len(candidate) > 2 and not re.search(r"\d{4}", candidate):
                        sc = [scores[i + 1]] if i + 1 < len(scores) else []
                        return candidate, sc
        return None, []

    def _extract_aadhaar(
        self, full_text: str, lines: List[str], scores: List[float]
    ) -> Dict[str, Any]:
        """Extract Aadhaar-specific fields."""
        name, name_sc = self._extract_name_from_lines(lines, scores)
        dob_raw = _find_first(_DATE_RE, full_text)
        dob = _normalise_date(dob_raw) if dob_raw else None
        gender_m = _GENDER_RE.search(full_text)
        gender = gender_m.group(1).upper() if gender_m else None
        if gender == "M":
            gender = "MALE"
        elif gender == "F":
            gender = "FEMALE"
        aadhaar_num = _find_first(_AADHAAR_NUM_RE, full_text)

        return {
            "name":           self._field(name, name_sc),
            "dob":            self._field(dob, scores),
            "gender":         self._field(gender, scores),
            "aadhaar_number": self._field(aadhaar_num, scores),
        }

    def _extract_pan(
        self, full_text: str, lines: List[str], scores: List[float]
    ) -> Dict[str, Any]:
        """Extract PAN-specific fields."""
        name, name_sc = self._extract_name_from_lines(lines, scores)
        dob_raw = _find_first(_DATE_RE, full_text)
        dob = _normalise_date(dob_raw) if dob_raw else None
        pan_num = _find_first(_PAN_NUM_RE, full_text)

        # Father name: typically the line after "FATHER'S NAME" header
        father_name: Optional[str] = None
        father_sc: List[float] = []
        for i, line in enumerate(lines):
            if re.search(r"father", line, re.IGNORECASE):
                same = re.split(r"[:\-]", line, maxsplit=1)
                if len(same) > 1 and len(same[1].strip()) > 2:
                    father_name = same[1].strip()
                    father_sc = [scores[i]] if i < len(scores) else []
                elif i + 1 < len(lines):
                    father_name = lines[i + 1].strip()
                    father_sc = [scores[i + 1]] if i + 1 < len(scores) else []
                break

        return {
            "name":        self._field(name, name_sc),
            "father_name": self._field(father_name, father_sc),
            "dob":         self._field(dob, scores),
            "pan_number":  self._field(pan_num, scores),
        }

    def _extract_voter_id(
        self, full_text: str, lines: List[str], scores: List[float]
    ) -> Dict[str, Any]:
        """Extract Voter ID-specific fields."""
        name, name_sc = self._extract_name_from_lines(lines, scores)
        dob_raw = _find_first(_DATE_RE, full_text)
        dob = _normalise_date(dob_raw) if dob_raw else None
        voter_id = _find_first(_VOTER_ID_RE, full_text)

        # Address: collect lines after "Address" keyword
        address: Optional[str] = None
        addr_lines: List[str] = []
        addr_sc: List[float] = []
        in_addr = False
        for i, line in enumerate(lines):
            if re.search(r"\baddress\b", line, re.IGNORECASE):
                in_addr = True
                remain = re.split(r"[:\-]", line, maxsplit=1)
                if len(remain) > 1:
                    addr_lines.append(remain[1].strip())
                    if i < len(scores):
                        addr_sc.append(scores[i])
                continue
            if in_addr:
                if re.search(r"\b(name|dob|id|epic)\b", line, re.IGNORECASE):
                    break
                addr_lines.append(line.strip())
                if i < len(scores):
                    addr_sc.append(scores[i])
        if addr_lines:
            address = ", ".join(a for a in addr_lines if a)

        return {
            "name":     self._field(name, name_sc),
            "dob":      self._field(dob, scores),
            "voter_id": self._field(voter_id, scores),
            "address":  self._field(address, addr_sc),
        }

    def _extract_dl(
        self, full_text: str, lines: List[str], scores: List[float]
    ) -> Dict[str, Any]:
        """Extract Driving Licence-specific fields."""
        name, name_sc = self._extract_name_from_lines(lines, scores)
        dl_num = _find_first(_DL_NUM_RE, full_text)

        # Two dates possible: DOB and validity
        dates = _DATE_RE.findall(full_text)
        dob: Optional[str] = None
        validity: Optional[str] = None
        if dates:
            dob = _normalise_date(dates[0])
        if len(dates) >= 2:
            validity = _normalise_date(dates[-1])

        # Vehicle class: look for patterns like "LMV", "MCWG", "HMV"
        vc_re = re.compile(r"\b(LMV|MCWG|HMV|HGMV|MGV|LDRXCV|PSV|TRANS)\b", re.IGNORECASE)
        vc_m = vc_re.search(full_text)
        vehicle_class = vc_m.group(1).upper() if vc_m else None

        return {
            "name":          self._field(name, name_sc),
            "dob":           self._field(dob, scores),
            "dl_number":     self._field(dl_num, scores),
            "validity_date": self._field(validity, scores),
            "vehicle_class": self._field(vehicle_class, scores),
        }

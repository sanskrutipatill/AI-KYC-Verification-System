# FILE: kyc_system/core/document_validator.py
"""
Document Validator for the KYC Verification System.

Validates extracted fields from Indian identity documents using regex patterns,
algorithmic checks (Verhoeff for Aadhaar), and scoring rubrics.

Supported document types:
- AADHAAR  : 12-digit Unique Identification Number
- PAN      : 10-character Permanent Account Number
- VOTER_ID : Election Commission ID (EPIC)
- DL       : Motor Vehicle Driving Licence
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any, Dict, Optional

from config import VOTER_ID_STATE_CODES, RTO_STATE_CODES
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Verhoeff Algorithm (Aadhaar checksum) ───────────────────────────────────

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_validate(number: str) -> bool:
    """
    Validate a numeric string using the Verhoeff check-digit algorithm.

    Args:
        number: Digit-only string (spaces removed).

    Returns:
        ``True`` if the checksum is valid.
    """
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


# ─── Date helpers ─────────────────────────────────────────────────────────────

def _parse_date(raw: Optional[str]) -> Optional[date]:
    """
    Parse a date string in DD/MM/YYYY or DD-MM-YYYY format.

    Returns a :class:`datetime.date` object or ``None`` on failure.
    """
    if not raw:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    return None


def _is_future_date(raw: Optional[str]) -> bool:
    """Return True if *raw* represents a date in the future."""
    d = _parse_date(raw)
    if d is None:
        return False
    return d > date.today()


def _is_valid_date(raw: Optional[str]) -> bool:
    """Return True if *raw* can be parsed as a valid date."""
    return _parse_date(raw) is not None


# ─── Validator ────────────────────────────────────────────────────────────────

class DocumentValidator:
    """
    Validates extracted OCR fields for each Indian document type using regex
    patterns and algorithmic checks.  Produces a validation score in [0, 1].
    """

    # ── Regex patterns ────────────────────────────────────────────────────────
    _AADHAAR_RE = re.compile(r"^\d{4}\s?\d{4}\s?\d{4}$")
    _PAN_RE     = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    _VOTER_RE   = re.compile(r"^[A-Z]{3}[0-9]{7}$")
    _DL_RE      = re.compile(r"^[A-Z]{2}[0-9]{2}\s?[0-9]{4}[0-9]{7}$")

    # PAN 4th-character entity codes
    _PAN_ENTITY = {"P", "C", "H", "A", "B", "G", "J", "L", "F", "T"}

    def validate(self, doc_type: str, fields: Dict[str, Any]) -> Dict[str, object]:
        """
        Validate extracted *fields* for the given *doc_type*.

        Args:
            doc_type: One of ``"AADHAAR"``, ``"PAN"``, ``"VOTER_ID"``,
                      ``"DL"``.
            fields:   OCR extraction result from :class:`~core.ocr_engine.OCREngine`.

        Returns:
            Dictionary::

                {
                    "score":       float,   # 0.0 – 1.0
                    "checks":      dict,    # individual check results
                    "is_valid":    bool,
                    "fail_reasons": list[str],
                }

        Raises:
            ValueError: If *doc_type* is unknown.
        """
        validators = {
            "AADHAAR":  self._validate_aadhaar,
            "PAN":      self._validate_pan,
            "VOTER_ID": self._validate_voter_id,
            "DL":       self._validate_dl,
        }
        if doc_type not in validators:
            raise ValueError(f"Unknown document type: {doc_type!r}")

        result = validators[doc_type](fields)
        result["is_valid"] = result["score"] >= 0.5
        log.info(
            "DocumentValidator [%s]: score=%.2f is_valid=%s",
            doc_type, result["score"], result["is_valid"],
        )
        return result

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _get(self, fields: Dict[str, Any], key: str) -> Optional[str]:
        """Safely retrieve a field value from OCR extraction output."""
        entry = fields.get(key)
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry.get("value")
        return str(entry) if entry else None

    # ─── AADHAAR ──────────────────────────────────────────────────────────────

    def _validate_aadhaar(self, fields: Dict[str, Any]) -> Dict[str, object]:
        """
        Validate Aadhaar fields.

        Scoring:
        - +0.4  Aadhaar number format valid + Verhoeff check
        - +0.3  Name present
        - +0.2  DOB present and valid
        - +0.1  Gender present
        """
        num  = self._get(fields, "aadhaar_number")
        name = self._get(fields, "name")
        dob  = self._get(fields, "dob")
        gend = self._get(fields, "gender")

        score = 0.0
        checks: Dict[str, Any] = {}
        fails: list[str] = []

        # Number validation
        if num:
            digits = re.sub(r"\s", "", num)
            fmt_ok = bool(self._AADHAAR_RE.match(num)) and len(digits) == 12
            # Reject if first digit is 0 or 1
            starts_ok = len(digits) >= 1 and digits[0] not in ("0", "1")
            verhoeff_ok = _verhoeff_validate(digits) if fmt_ok and starts_ok else False
            checks["number_format"] = fmt_ok
            checks["number_starts_ok"] = starts_ok
            checks["verhoeff_ok"] = verhoeff_ok
            if fmt_ok and starts_ok:
                score += 0.4
                if not verhoeff_ok:
                    score -= 0.1
                    fails.append("Aadhaar checksum (Verhoeff) failed.")
            else:
                fails.append("Aadhaar number format invalid.")
        else:
            checks["number_format"] = False
            fails.append("Aadhaar number not found.")

        # Name
        if name and len(name.strip()) >= 2:
            score += 0.3
            checks["name_found"] = True
        else:
            checks["name_found"] = False
            fails.append("Name not found on document.")

        # DOB
        if _is_valid_date(dob):
            score += 0.2
            checks["dob_valid"] = True
        else:
            checks["dob_valid"] = False
            fails.append("Date of birth not found or invalid.")

        # Gender
        if gend:
            score += 0.1
            checks["gender_found"] = True
        else:
            checks["gender_found"] = False

        return {"score": round(min(score, 1.0), 3), "checks": checks, "fail_reasons": fails}

    # ─── PAN ──────────────────────────────────────────────────────────────────

    def _validate_pan(self, fields: Dict[str, Any]) -> Dict[str, object]:
        """
        Validate PAN Card fields.

        Scoring:
        - +0.5  PAN format strictly matches + entity type char valid
        - +0.3  Name present
        - +0.2  DOB present and valid
        """
        pan  = self._get(fields, "pan_number")
        name = self._get(fields, "name")
        dob  = self._get(fields, "dob")

        score = 0.0
        checks: Dict[str, Any] = {}
        fails: list[str] = []

        if pan:
            fmt_ok = bool(self._PAN_RE.match(pan))
            entity_ok = len(pan) >= 4 and pan[3].upper() in self._PAN_ENTITY
            checks["format_ok"]  = fmt_ok
            checks["entity_ok"]  = entity_ok
            if fmt_ok:
                score += 0.5
                if not entity_ok:
                    score -= 0.05
                    fails.append("PAN entity-type character not recognised.")
            else:
                fails.append("PAN number format invalid.")
        else:
            checks["format_ok"] = False
            fails.append("PAN number not found.")

        if name and len(name.strip()) >= 2:
            score += 0.3
            checks["name_found"] = True
        else:
            checks["name_found"] = False
            fails.append("Name not found on document.")

        if _is_valid_date(dob):
            score += 0.2
            checks["dob_valid"] = True
        else:
            checks["dob_valid"] = False
            fails.append("Date of birth not found or invalid.")

        return {"score": round(min(score, 1.0), 3), "checks": checks, "fail_reasons": fails}

    # ─── VOTER ID ─────────────────────────────────────────────────────────────

    def _validate_voter_id(self, fields: Dict[str, Any]) -> Dict[str, object]:
        """
        Validate Voter ID fields.

        Scoring:
        - +0.4  EPIC number format valid + known state prefix
        - +0.3  Name present
        - +0.2  DOB present and valid
        - +0.1  Address present
        """
        vid  = self._get(fields, "voter_id")
        name = self._get(fields, "name")
        dob  = self._get(fields, "dob")
        addr = self._get(fields, "address")

        score = 0.0
        checks: Dict[str, Any] = {}
        fails: list[str] = []

        if vid:
            fmt_ok = bool(self._VOTER_RE.match(vid))
            state_prefix = vid[:3].upper() if len(vid) >= 3 else ""
            state_ok = state_prefix in VOTER_ID_STATE_CODES
            checks["format_ok"]  = fmt_ok
            checks["state_code_ok"] = state_ok
            if fmt_ok:
                score += 0.3
                if state_ok:
                    score += 0.1
                else:
                    fails.append(f"Unknown Voter ID state prefix: '{state_prefix}'.")
            else:
                fails.append("Voter ID number format invalid.")
        else:
            checks["format_ok"] = False
            fails.append("Voter ID number not found.")

        if name and len(name.strip()) >= 2:
            score += 0.3
            checks["name_found"] = True
        else:
            checks["name_found"] = False
            fails.append("Name not found on document.")

        if _is_valid_date(dob):
            score += 0.2
            checks["dob_valid"] = True
        else:
            checks["dob_valid"] = False
            fails.append("Date of birth not found or invalid.")

        if addr and len(addr.strip()) >= 5:
            score += 0.1
            checks["address_found"] = True
        else:
            checks["address_found"] = False

        return {"score": round(min(score, 1.0), 3), "checks": checks, "fail_reasons": fails}

    # ─── DRIVING LICENCE ──────────────────────────────────────────────────────

    def _validate_dl(self, fields: Dict[str, Any]) -> Dict[str, object]:
        """
        Validate Driving Licence fields.

        Scoring:
        - +0.4  DL number format valid + known state code
        - +0.3  Name present
        - +0.2  Validity date present and in future
        - +0.1  Vehicle class present
        """
        dl   = self._get(fields, "dl_number")
        name = self._get(fields, "name")
        exp  = self._get(fields, "validity_date")
        vc   = self._get(fields, "vehicle_class")

        score = 0.0
        checks: Dict[str, Any] = {}
        fails: list[str] = []

        if dl:
            clean_dl = re.sub(r"\s", "", dl)
            fmt_ok = bool(self._DL_RE.match(clean_dl))
            state_code = clean_dl[:2].upper() if len(clean_dl) >= 2 else ""
            state_ok = state_code in RTO_STATE_CODES
            checks["format_ok"]   = fmt_ok
            checks["state_code_ok"] = state_ok
            if fmt_ok:
                score += 0.3
                if state_ok:
                    score += 0.1
                else:
                    fails.append(f"Unknown RTO state code: '{state_code}'.")
            else:
                fails.append("Driving Licence number format invalid.")
        else:
            checks["format_ok"] = False
            fails.append("DL number not found.")

        if name and len(name.strip()) >= 2:
            score += 0.3
            checks["name_found"] = True
        else:
            checks["name_found"] = False
            fails.append("Name not found on document.")

        if _is_future_date(exp):
            score += 0.2
            checks["expiry_valid"] = True
        else:
            checks["expiry_valid"] = False
            fails.append("Validity date not found or licence has expired.")

        if vc:
            score += 0.1
            checks["vehicle_class_found"] = True
        else:
            checks["vehicle_class_found"] = False

        return {"score": round(min(score, 1.0), 3), "checks": checks, "fail_reasons": fails}

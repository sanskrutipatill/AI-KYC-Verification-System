# FILE: kyc_system/gui/screens/result_screen.py
"""
Screen 5 — Result Dashboard.

Displays the final KYC verdict with:
  • Large status badge (VERIFIED / UNDER REVIEW / REJECTED)
  • Confidence level indicator (HIGH / MEDIUM / LOW)
  • Four score gauges (Face Match, Liveness, Deepfake Risk, Overall Risk)
  • Structured reasoning chain (explainable AI bullets)
  • Extracted identity details panel
  • Risk flags row
  • Recommended action chip
  • Action buttons: New Verification | Export Report (JSON) | Exit
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from config import EXPORT_DIR, CAPTURED_FACE_PATH, TEMP_DIR
from gui.components.score_gauge import ScoreGauge
from gui.components.status_badge import StatusBadge
from utils.logger import get_logger

log = get_logger(__name__)

# ── Confidence level colours ──────────────────────────────────────────────────
_CONF_COLORS = {
    "HIGH":   ("#166534", "#22c55e"),   # (bg, fg)
    "MEDIUM": ("#78350f", "#f59e0b"),
    "LOW":    ("#1e3a5f", "#60a5fa"),
}

# ── Recommended action colours ────────────────────────────────────────────────
_ACTION_COLORS = {
    "Auto Approve":  "#22c55e",
    "Manual Review": "#f59e0b",
    "Reject":        "#ef4444",
}


class ResultScreen(ctk.CTkFrame):
    """
    Final verification result dashboard (Step 5 of the KYC flow).

    Args:
        master:   Parent CTk widget.
        on_new:   Callback to start a new verification.
        on_exit:  Callback to close the application.
        **kwargs: Forwarded to CTkFrame.
    """

    def __init__(
        self,
        master,
        on_new:  Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_new  = on_new
        self._on_exit = on_exit
        self._results: Optional[Dict[str, Any]] = None
        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the complete dashboard layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)   # reasoning panel expands

        # ── Header: Status badge + confidence + action chip ───────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, pady=(20, 6), padx=30, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Left: status + reason
        left_hdr = ctk.CTkFrame(header, fg_color="transparent")
        left_hdr.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            left_hdr,
            text="KYC Verification Result",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w")

        self._badge = StatusBadge(left_hdr, status="")
        self._badge.pack(anchor="w", pady=(6, 0))

        # Right: confidence chip + action chip
        right_hdr = ctk.CTkFrame(header, fg_color="transparent")
        right_hdr.grid(row=0, column=2, sticky="e")

        ctk.CTkLabel(
            right_hdr,
            text="Confidence",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray55"),
        ).pack(anchor="e")
        self._conf_badge = ctk.CTkLabel(
            right_hdr,
            text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            padx=12, pady=4,
            fg_color=("gray70", "gray30"),
        )
        self._conf_badge.pack(anchor="e", pady=(2, 8))

        ctk.CTkLabel(
            right_hdr,
            text="Recommended Action",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray55"),
        ).pack(anchor="e")
        self._action_badge = ctk.CTkLabel(
            right_hdr,
            text="—",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            padx=12, pady=4,
            fg_color=("gray70", "gray30"),
        )
        self._action_badge.pack(anchor="e", pady=(2, 0))

        # ── Score gauges row ──────────────────────────────────────────────────
        gauges_frame = ctk.CTkFrame(self, fg_color="transparent")
        gauges_frame.grid(row=1, column=0, pady=(4, 4))

        self._gauge_face = ScoreGauge(gauges_frame, title="Face Match",    score=0)
        self._gauge_live = ScoreGauge(gauges_frame, title="Liveness",      score=0)
        self._gauge_deep = ScoreGauge(gauges_frame, title="Deepfake Risk", score=0)
        self._gauge_risk = ScoreGauge(gauges_frame, title="Overall Risk",  score=0)

        for i, g in enumerate([self._gauge_face, self._gauge_live,
                                self._gauge_deep, self._gauge_risk]):
            g.grid(row=0, column=i, padx=16)

        # ── Face comparison panel (doc photo vs live capture) ─────────────────
        face_cmp_frame = ctk.CTkFrame(self, corner_radius=10,
                                       fg_color=("gray91", "gray15"))
        face_cmp_frame.grid(row=2, column=0, padx=24, pady=(0, 4), sticky="ew")
        face_cmp_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            face_cmp_frame,
            text="🔍  Face Comparison",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(8, 2))

        # Doc photo
        doc_col = ctk.CTkFrame(face_cmp_frame, fg_color="transparent")
        doc_col.grid(row=1, column=0, padx=10, pady=6, sticky="n")
        ctk.CTkLabel(doc_col, text="Document Photo",
                     font=ctk.CTkFont(size=10), text_color=("gray50","gray55")
                     ).pack()
        self._doc_face_lbl = ctk.CTkLabel(doc_col, text="—",
                                           width=90, height=90,
                                           fg_color=("gray80","gray20"),
                                           corner_radius=8)
        self._doc_face_lbl.pack(pady=2)

        # Score badge between images
        mid_col = ctk.CTkFrame(face_cmp_frame, fg_color="transparent")
        mid_col.grid(row=1, column=1, padx=8, pady=6, sticky="n")
        ctk.CTkLabel(mid_col, text="Similarity",
                     font=ctk.CTkFont(size=10), text_color=("gray50","gray55")
                     ).pack(pady=(18, 2))
        self._face_score_badge = ctk.CTkLabel(
            mid_col, text="—%",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#6b7280",
        )
        self._face_score_badge.pack()
        self._face_verdict_lbl = ctk.CTkLabel(
            mid_col, text="",
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self._face_verdict_lbl.pack(pady=(2, 0))

        # Live photo
        live_col = ctk.CTkFrame(face_cmp_frame, fg_color="transparent")
        live_col.grid(row=1, column=2, padx=10, pady=6, sticky="n")
        ctk.CTkLabel(live_col, text="Live Capture",
                     font=ctk.CTkFont(size=10), text_color=("gray50","gray55")
                     ).pack()
        self._live_face_lbl = ctk.CTkLabel(live_col, text="—",
                                            width=90, height=90,
                                            fg_color=("gray80","gray20"),
                                            corner_radius=8)
        self._live_face_lbl.pack(pady=2)

        # ── Middle row: Reasoning + Identity details ──────────────────────────
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.grid(row=3, column=0, padx=20, pady=(0, 4), sticky="nsew")
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=2)
        mid.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Left: AI reasoning panel
        reason_frame = ctk.CTkFrame(mid, corner_radius=12, fg_color=("gray92", "gray13"))
        reason_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        reason_frame.grid_columnconfigure(0, weight=1)
        reason_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            reason_frame,
            text="🧠  AI Reasoning Chain",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self._reasoning_scroll = ctk.CTkScrollableFrame(
            reason_frame, fg_color="transparent", corner_radius=0
        )
        self._reasoning_scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 8))
        self._reasoning_scroll.grid_columnconfigure(0, weight=1)

        # Right panel: identity + flags
        right_panel = ctk.CTkFrame(mid, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure((0, 1), weight=1)

        # Identity details
        id_frame = ctk.CTkFrame(right_panel, corner_radius=12, fg_color=("gray92", "gray13"))
        id_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        id_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            id_frame,
            text="🪪  Identity Details",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 6))

        self._detail_rows: Dict[str, ctk.CTkLabel] = {}
        detail_keys = [
            ("name",      "Name"),
            ("dob",       "Date of Birth"),
            ("id_number", "ID Number"),
            ("gender",    "Gender"),
        ]
        for ri, (key, lbl_text) in enumerate(detail_keys):
            ctk.CTkLabel(
                id_frame,
                text=f"{lbl_text}:",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w",
                width=90,
            ).grid(row=ri + 1, column=0, sticky="w", padx=(14, 4), pady=3)
            val = ctk.CTkLabel(
                id_frame,
                text="—",
                font=ctk.CTkFont(size=11),
                anchor="w",
            )
            val.grid(row=ri + 1, column=1, sticky="w", pady=3)
            self._detail_rows[key] = val

        # Risk flags
        flags_frame = ctk.CTkFrame(right_panel, corner_radius=12, fg_color=("gray92", "gray13"))
        flags_frame.grid(row=1, column=0, sticky="nsew")
        flags_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            flags_frame,
            text="⚑  Risk Flags",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        flag_defs = [
            ("liveness_failed",    "Liveness Failed"),
            ("deepfake_suspicion", "Deepfake Suspicion"),
            ("face_mismatch",      "Face Mismatch"),
            ("document_issue",     "Document Issue"),
        ]
        self._flag_labels: Dict[str, ctk.CTkLabel] = {}
        for ri, (key, text) in enumerate(flag_defs):
            lbl = ctk.CTkLabel(
                flags_frame,
                text=f"● {text}: —",
                font=ctk.CTkFont(size=11),
                anchor="w",
            )
            lbl.grid(row=ri + 1, column=0, sticky="w", padx=14, pady=3)
            self._flag_labels[key] = lbl

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=4, column=0, pady=(6, 18))

        ctk.CTkButton(
            btn_frame,
            text="🔄  New Verification",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40, width=190,
            command=self._on_new_clicked,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame,
            text="📤  Export Report (JSON)",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40, width=210,
            fg_color=("gray65", "gray30"),
            hover_color=("gray55", "gray40"),
            command=self._export_json,
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame,
            text="✕  Exit",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40, width=120,
            fg_color=("#b91c1c", "#7f1d1d"),
            hover_color=("#991b1b", "#6b1414"),
            command=self._on_exit_clicked,
        ).pack(side="left", padx=10)

    # ─── Data binding ─────────────────────────────────────────────────────────

    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Populate every dashboard widget with pipeline *results*.

        Args:
            results: Full dict returned by the processing screen.
        """
        self._results = results

        risk   = results.get("risk",           {})
        face   = results.get("face_match",     {})
        live   = results.get("liveness",       {})
        deep   = results.get("deepfake",       {})
        doc_v  = results.get("doc_validation", {})
        fields = results.get("ocr_fields",     {})
        dtype  = results.get("doc_type",       "")

        status     = risk.get("status",              "UNDER REVIEW")
        confidence = risk.get("confidence_level",    "LOW")
        action     = risk.get("recommended_action",  "Manual Review")
        reasoning  = risk.get("reasoning",           [])
        risk_flags = risk.get("risk_flags",          {})

        # ── Status badge ──
        self._badge.set_status(status)

        # ── Confidence chip ──
        bg, fg = _CONF_COLORS.get(confidence, ("gray70", "white"))
        self._conf_badge.configure(
            text=f"  {confidence}  ",
            fg_color=bg,
            text_color=fg,
        )

        # ── Action chip ──
        action_color = _ACTION_COLORS.get(action, "#6b7280")
        self._action_badge.configure(
            text=f"  {action}  ",
            fg_color=action_color,
            text_color="#ffffff",
        )

        # ── Score gauges — always use the REAL raw score, never inflated ──
        raw_face_score = face.get("score", 0.0)
        liveness_score = live.get("liveness_score", 0.0)
        doc_valid      = doc_v.get("is_valid", True)
        df_prob        = deep.get("deepfake_prob",  0.0)
        face_error     = face.get("error")           # set if face detect failed

        # Show real score on gauge — adjusted_face_match only used for risk calc
        display_face_score = raw_face_score

        self._gauge_face.set_score(display_face_score)
        self._gauge_live.set_score(liveness_score)
        self._gauge_deep.set_score(df_prob)
        self._gauge_risk.set_score(risk.get("risk_score", 0.5))

        # ── Face comparison panel ──
        self._load_face_comparison(results, display_face_score, face_error)

        # ── Reasoning chain ──
        for child in self._reasoning_scroll.winfo_children():
            child.destroy()
        for i, bullet in enumerate(reasoning):
            row_frame = ctk.CTkFrame(self._reasoning_scroll, fg_color="transparent")
            row_frame.pack(fill="x", padx=4, pady=3)
            row_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                row_frame,
                text=f"{i+1}.",
                font=ctk.CTkFont(size=11, weight="bold"),
                width=22,
                anchor="ne",
                text_color=("gray50", "gray55"),
            ).grid(row=0, column=0, sticky="ne", padx=(0, 6), pady=2)

            ctk.CTkLabel(
                row_frame,
                text=bullet,
                font=ctk.CTkFont(size=11),
                anchor="w",
                justify="left",
                wraplength=340,
                text_color=("gray20", "gray85"),
            ).grid(row=0, column=1, sticky="w")

        # ── Identity details ──
        id_key_map = {
            "AADHAAR":  "aadhaar_number",
            "PAN":      "pan_number",
            "VOTER_ID": "voter_id",
            "DL":       "dl_number",
        }

        def _fval(key: str) -> str:
            entry = fields.get(key)
            if isinstance(entry, dict):
                return str(entry.get("value") or "—")
            return "—"

        self._detail_rows["name"].configure(text=_fval("name"))
        self._detail_rows["dob"].configure(text=_fval("dob"))
        self._detail_rows["id_number"].configure(text=_fval(id_key_map.get(dtype, "")))
        self._detail_rows["gender"].configure(text=_fval("gender"))

        # ── Risk flags ──
        gender_status = risk.get("gender_check", "UNAVAILABLE")

        # 1. Liveness / gender (Priority 1)
        if gender_status == "MISMATCH":
            self._flag_labels["liveness_failed"].configure(
                text=f"● GENDER MISMATCH: {risk.get('extracted_gender','?')} (doc) ≠ {risk.get('detected_gender','?')} (face) ⚠",
                text_color="#ef4444",
            )
        else:
            liv_fail = risk_flags.get("liveness_failed", False)
            self._flag_labels["liveness_failed"].configure(
                text=f"● Liveness Failed: {'Yes ⚠' if liv_fail else 'No ✓'}",
                text_color="#ef4444" if liv_fail else "#22c55e",
            )

        # 2. Document
        doc_fail = risk_flags.get("document_issue", False)
        self._flag_labels["document_issue"].configure(
            text=f"● Document Issue: {'Yes ⚠' if doc_fail else 'No ✓'}",
            text_color="#ef4444" if doc_fail else "#22c55e",
        )

        # 3. Face Match — always show REAL score, never inflated value
        face_err = face.get("error")
        if gender_status == "MISMATCH":
            self._flag_labels["face_mismatch"].configure(
                text="● Face Status: INVALID (Gender Mismatch)",
                text_color="#ef4444",
            )
        elif face_err:
            self._flag_labels["face_mismatch"].configure(
                text=f"● Face Status: No face detected ⚠",
                text_color="#f59e0b",
            )
        elif display_face_score >= 0.60:
            self._flag_labels["face_mismatch"].configure(
                text=f"● Face Match: {int(display_face_score*100)}% — Strong Match ✓",
                text_color="#22c55e",
            )
        elif display_face_score >= 0.30:
            self._flag_labels["face_mismatch"].configure(
                text=f"● Face Match: {int(display_face_score*100)}% — Partial Match",
                text_color="#f59e0b",
            )
        else:
            self._flag_labels["face_mismatch"].configure(
                text=f"● Face Match: {int(display_face_score*100)}% — Low / No Match ⚠",
                text_color="#ef4444",
            )

        # 4. Deepfake
        df_fail  = risk_flags.get("deepfake_suspicion", False)
        df_score = deep.get("deepfake_prob", 0.0)
        if df_fail or df_score >= 0.30:
            self._flag_labels["deepfake_suspicion"].configure(
                text="● Deepfake Risk: Moderate (Not conclusive)",
                text_color="#f59e0b",
            )
        else:
            self._flag_labels["deepfake_suspicion"].configure(
                text="● Deepfake Suspicion: No ✓",
                text_color="#22c55e",
            )

    # ─── Face comparison panel ────────────────────────────────────────────────

    def _load_face_comparison(
        self,
        results: Dict[str, Any],
        face_score: float,
        face_error: Optional[str],
    ) -> None:
        """Load document & live face images into the comparison panel."""
        SIZE = (90, 90)

        def _load_ctk_img(path: Path) -> Optional[ctk.CTkImage]:
            """Load an image file and return a CTkImage, or None on failure."""
            try:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    return None
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                pil.thumbnail(SIZE, Image.LANCZOS)
                # Pad to square
                canvas = Image.new("RGB", SIZE, (40, 40, 40))
                offset = ((SIZE[0] - pil.width) // 2, (SIZE[1] - pil.height) // 2)
                canvas.paste(pil, offset)
                return ctk.CTkImage(light_image=canvas, dark_image=canvas, size=SIZE)
            except Exception as exc:
                log.warning("Face comparison image load failed: %s", exc)
                return None

        # ── Document face ──
        doc_face_np = results.get("doc_face")  # numpy array saved by pipeline
        doc_ctk = None
        if doc_face_np is not None and isinstance(doc_face_np, np.ndarray) and doc_face_np.size > 0:
            try:
                rgb = cv2.cvtColor(doc_face_np, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pil.thumbnail(SIZE, Image.LANCZOS)
                canvas = Image.new("RGB", SIZE, (40, 40, 40))
                canvas.paste(pil, ((SIZE[0]-pil.width)//2, (SIZE[1]-pil.height)//2))
                doc_ctk = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=SIZE)
            except Exception as exc:
                log.warning("Doc face render failed: %s", exc)

        if doc_ctk:
            self._doc_face_lbl.configure(image=doc_ctk, text="")
            self._doc_face_lbl.image = doc_ctk
        else:
            self._doc_face_lbl.configure(image=None, text="No photo")

        # ── Live capture ──
        live_ctk = _load_ctk_img(CAPTURED_FACE_PATH)
        if live_ctk:
            self._live_face_lbl.configure(image=live_ctk, text="")
            self._live_face_lbl.image = live_ctk
        else:
            self._live_face_lbl.configure(image=None, text="No capture")

        # ── Score badge ──
        pct = int(round(face_score * 100))
        if face_error:
            score_txt   = "N/A"
            verdict_txt = "No face detected"
            score_color = "#6b7280"
        elif face_score >= 0.70:
            score_txt   = f"{pct}%"
            verdict_txt = "✓ Strong Match"
            score_color = "#22c55e"
        elif face_score >= 0.45:
            score_txt   = f"{pct}%"
            verdict_txt = "~ Partial Match"
            score_color = "#f59e0b"
        elif face_score >= 0.20:
            score_txt   = f"{pct}%"
            verdict_txt = "⚠ Low Match"
            score_color = "#ef4444"
        else:
            score_txt   = f"{pct}%"
            verdict_txt = "✗ No Match"
            score_color = "#ef4444"

        self._face_score_badge.configure(text=score_txt, text_color=score_color)
        self._face_verdict_lbl.configure(text=verdict_txt, text_color=score_color)

    # ─── Export ───────────────────────────────────────────────────────────────

    def _export_json(self) -> None:
        """Build and save the standardised JSON report to ~/Downloads."""
        if not self._results:
            return
        payload = self._build_export_payload()
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = EXPORT_DIR / f"kyc_report_{ts}.json"
        try:
            out_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("ResultScreen: report exported → %s", out_path)
            try:
                from CTkMessagebox import CTkMessagebox  # type: ignore
                CTkMessagebox(
                    title="Export Successful",
                    message=f"Report saved to:\n{out_path}",
                    icon="check",
                )
            except ImportError:
                import tkinter.messagebox as mb
                mb.showinfo("Export Successful", f"Report saved to:\n{out_path}")
        except Exception as exc:
            log.error("Export failed: %s", exc)
            try:
                from CTkMessagebox import CTkMessagebox  # type: ignore
                CTkMessagebox(title="Export Failed", message=str(exc), icon="cancel")
            except ImportError:
                import tkinter.messagebox as mb
                mb.showerror("Export Failed", str(exc))

    def _build_export_payload(self) -> Dict[str, Any]:
        """Construct the full JSON export payload."""
        r = self._results or {}
        face   = r.get("face_match",     {})
        live   = r.get("liveness",       {})
        deep   = r.get("deepfake",       {})
        doc_v  = r.get("doc_validation", {})
        risk   = r.get("risk",           {})
        fields = r.get("ocr_fields",     {})
        dtype  = r.get("doc_type",       "")

        id_key_map = {
            "AADHAAR": "aadhaar_number", "PAN": "pan_number",
            "VOTER_ID": "voter_id", "DL": "dl_number",
        }

        def _fval(key):
            entry = fields.get(key)
            return entry.get("value") if isinstance(entry, dict) else None

        # Always use the REAL ArcFace score — no inflation
        raw_score      = face.get("score", 0.0)
        face_match_pct = int(round(raw_score * 100))
        face_matched   = face.get("matched", False)
        face_error     = face.get("error")
        gender_status  = risk.get("gender_check", "UNAVAILABLE")
        deepfake_score = deep.get("deepfake_prob", 0.0)

        # Determine face status from real score
        if gender_status == "MISMATCH":
            face_status      = "INVALID (Gender Mismatch)"
            ui_face_mismatch = "Yes (Identity Conflict)"
        elif face_error:
            face_status      = "NO FACE DETECTED"
            ui_face_mismatch = "Yes (Detection Error)"
        elif raw_score >= 0.70:
            face_status      = "STRONG MATCH"
            ui_face_mismatch = "No"
        elif raw_score >= 0.40:
            face_status      = "PARTIAL MATCH"
            ui_face_mismatch = "No"
        elif raw_score >= 0.20:
            face_status      = "LOW MATCH"
            ui_face_mismatch = "Yes"
        else:
            face_status      = "NO MATCH"
            ui_face_mismatch = "Yes"

        # Deepfake status
        df_status = "Safe"
        ui_deepfake_detected = "No"
        if deepfake_score >= 0.60:
            df_status = "Suspicious"
            ui_deepfake_detected = "Yes"
        elif deepfake_score >= 0.30:
            df_status = "Moderate Risk (Not Conclusive)"
            ui_deepfake_detected = "Uncertain"

        reasoning = list(risk.get("reasoning", []))

        decision = {
            "face_match": face_match_pct,
            "face_match_status": face_status,
            "gender_check": gender_status,
            "extracted_gender": risk.get("extracted_gender"),
            "detected_gender": risk.get("detected_gender"),
            "confidence": risk.get("confidence_level", "LOW"),
            "final_status": risk.get("status", "UNDER REVIEW"),
            "deepfake_status": df_status,
            "reasoning": reasoning,
            "ui_flags": {
                "face_mismatch": ui_face_mismatch,
                "deepfake_detected": ui_deepfake_detected,
                "gender_mismatch": "Yes" if gender_status == "MISMATCH" else "No",
                "document_invalid": "Yes" if not doc_v.get("is_valid", True) else "No"
            }
        }

        return {
            "kyc_result": {
                "timestamp":     datetime.now().isoformat(),
                "document_type": dtype,
                "extracted_fields": {
                    "name":      _fval("name"),
                    "dob":       _fval("dob"),
                    "id_number": _fval(id_key_map.get(dtype, "")),
                    "gender":    _fval("gender"),
                },
                "scores": {
                    "face_match_pct":            face_match_pct,
                    "liveness_score":            round(live.get("liveness_score", 0.0), 4),
                    "deepfake_probability":      round(deep.get("deepfake_prob",  0.0), 4),
                    "ocr_confidence_avg":        round(fields.get("avg_confidence", 0.0), 4),
                    "document_validation_score": round(doc_v.get("score", 0.0), 4),
                    "overall_risk_score":        round(risk.get("risk_score", 0.5), 4),
                },
                "flags": {
                    "liveness_failed":   not live.get("is_real",   True),
                    "deepfake_detected": deep.get("is_deepfake",   False),
                    "face_mismatch":     ui_face_mismatch != "No",
                    "gender_mismatch":   gender_status == "MISMATCH",
                    "document_invalid":  not doc_v.get("is_valid", True),
                },
                "decision": decision
            }
        }

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def _on_new_clicked(self) -> None:
        if self._on_new:
            self._on_new()

    def _on_exit_clicked(self) -> None:
        if self._on_exit:
            self._on_exit()

# FILE: kyc_system/gui/components/score_gauge.py
"""
Score Gauge Widget for the KYC Verification System.

A custom CTk canvas widget that renders a semi-circular gauge showing a
score value from 0–100 with colour-coded arcs and a central label.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Tuple

import customtkinter as ctk


class ScoreGauge(ctk.CTkFrame):
    """
    Semi-circular gauge canvas widget for displaying a numeric score.

    Renders an arc from 0° to 180° divided into three colour zones
    (red → amber → green) with an overlaid needle and central percentage label.

    Args:
        master:    Parent CTk widget.
        title:     Label displayed below the gauge.
        score:     Initial score value (0.0 – 1.0).
        size:      Pixel diameter of the gauge (default 160).
        **kwargs:  Additional keyword arguments forwarded to CTkFrame.
    """

    def __init__(
        self,
        master: ctk.CTk | ctk.CTkFrame,
        title: str = "Score",
        score: float = 0.0,
        size: int = 160,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._title = title
        self._score = max(0.0, min(1.0, score))
        self._size = size

        # Canvas for the arc drawing
        self._canvas = tk.Canvas(
            self,
            width=size,
            height=size // 2 + 30,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(pady=(4, 0))

        # Title label
        self._title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray70"),
        )
        self._title_label.pack(pady=(0, 4))

        self._configure_canvas_bg()
        self._render_gauge(self._score)

    # ─── Private drawing helpers ───────────────────────────────────────────────

    def _configure_canvas_bg(self) -> None:
        """Match the canvas background to the CTk frame colour."""
        mode = ctk.get_appearance_mode()
        bg = "#2b2b2b" if mode == "Dark" else "#ebebeb"
        self._canvas.configure(bg=bg)

    @staticmethod
    def _score_to_color(score: float) -> str:
        """Map a score in [0,1] to a hex colour from red→amber→green."""
        if score < 0.5:
            # Red → Amber
            t = score / 0.5
            r = 239
            g = int(68 + (159 - 68) * t)
            b = 68
        else:
            # Amber → Green
            t = (score - 0.5) / 0.5
            r = int(239 + (34 - 239) * t)
            g = int(159 + (197 - 159) * t)
            b = int(68 + (94 - 68) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _render_gauge(self, score: float) -> None:
        """Redraw the gauge for the given *score*."""
        self._canvas.delete("all")
        s = self._size
        cx, cy = s // 2, s // 2 + 10
        r_outer = s // 2 - 10
        r_inner = r_outer - 22

        # ── Background arc (grey) ──
        x0, y0 = cx - r_outer, cy - r_outer
        x1, y1 = cx + r_outer, cy + r_outer
        self._canvas.create_arc(
            x0, y0, x1, y1,
            start=0, extent=180,
            style=tk.ARC,
            outline="#555555",
            width=22,
        )

        # ── Score arc (colour) ──
        extent = score * 180.0
        color = self._score_to_color(score)
        if extent > 0:
            self._canvas.create_arc(
                x0, y0, x1, y1,
                start=0, extent=extent,
                style=tk.ARC,
                outline=color,
                width=22,
            )

        # ── Needle ──
        angle_deg = score * 180.0  # 0° = right, 180° = left
        angle_rad = math.radians(180 - angle_deg)
        needle_len = r_inner - 4
        nx = cx + needle_len * math.cos(angle_rad)
        ny = cy - needle_len * math.sin(angle_rad)
        self._canvas.create_line(
            cx, cy, nx, ny,
            fill=color, width=3, capstyle=tk.ROUND,
        )
        # Pivot dot
        self._canvas.create_oval(
            cx - 5, cy - 5, cx + 5, cy + 5,
            fill=color, outline="",
        )

        # ── Central percentage label ──
        pct_text = f"{int(round(score * 100))}%"
        self._canvas.create_text(
            cx, cy + 18,
            text=pct_text,
            font=("Helvetica", 16, "bold"),
            fill=color,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def set_score(self, score: float) -> None:
        """
        Update the displayed score.

        Args:
            score: New score in [0.0, 1.0].
        """
        self._score = max(0.0, min(1.0, score))
        self._configure_canvas_bg()
        self._render_gauge(self._score)

    def set_title(self, title: str) -> None:
        """Update the gauge's text label."""
        self._title = title
        self._title_label.configure(text=title)

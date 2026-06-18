# FILE: kyc_system/gui/components/status_badge.py
"""
Status Badge Widget for the KYC Verification System.

A CTkLabel-based badge that renders VERIFIED / REVIEW / REJECTED with
appropriate background colours and icons.
"""

from __future__ import annotations

import customtkinter as ctk
from config import COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER

# Status → (background, foreground, icon)
_STATUS_STYLES = {
    "VERIFIED":     (COLOR_SUCCESS, "#ffffff", "✅  VERIFIED"),
    "UNDER REVIEW": (COLOR_WARNING, "#ffffff", "⚠️  UNDER REVIEW"),
    "REVIEW":       (COLOR_WARNING, "#ffffff", "⚠️  UNDER REVIEW"),
    "REJECTED":     (COLOR_DANGER,  "#ffffff", "❌  REJECTED"),
}


class StatusBadge(ctk.CTkLabel):
    """
    A large, colour-coded status label widget.

    Args:
        master:  Parent widget.
        status:  One of ``"VERIFIED"``, ``"REVIEW"``, ``"REJECTED"``,
                 or ``""`` (empty / loading state).
        **kwargs: Forwarded to CTkLabel.
    """

    def __init__(
        self,
        master: ctk.CTk | ctk.CTkFrame,
        status: str = "",
        **kwargs,
    ) -> None:
        # Resolve style before calling super().__init__
        text, fg, bg = self._resolve(status)
        super().__init__(
            master,
            text=text,
            font=ctk.CTkFont(size=22, weight="bold"),
            fg_color=bg,
            text_color=fg,
            corner_radius=12,
            padx=24,
            pady=10,
            **kwargs,
        )
        self._status = status

    # ─── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve(status: str):
        """Return (display_text, fg, bg) for the given *status*."""
        if status.upper() in _STATUS_STYLES:
            bg, fg, label = _STATUS_STYLES[status.upper()]
            return label, fg, bg
        return "— PENDING —", "#ffffff", "#374151"

    # ─── Public API ───────────────────────────────────────────────────────────

    def set_status(self, status: str) -> None:
        """
        Update the badge to reflect a new *status*.

        Args:
            status: ``"VERIFIED"``, ``"REVIEW"``, or ``"REJECTED"``.
        """
        self._status = status
        text, fg, bg = self._resolve(status)
        self.configure(text=text, text_color=fg, fg_color=bg)

    @property
    def status(self) -> str:
        """Current status string."""
        return self._status

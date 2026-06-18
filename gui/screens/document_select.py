# FILE: kyc_system/gui/screens/document_select.py
"""
Screen 1 — Document Type Selection.

Presents four large card-style buttons for choosing the document type.
The "Continue" button is disabled until the user makes a selection.
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from config import DOC_TYPES, COLOR_PRIMARY


class DocumentSelectScreen(ctk.CTkFrame):
    """
    Document type picker (Step 1 of the KYC flow).

    Displays four 200×120 interactive cards in a 2×2 grid.  The selected
    card gets a highlighted border and the Continue button activates.

    Args:
        master:           Parent CTk widget.
        on_continue:      Callback invoked with the selected doc-type key when
                          the user clicks "Continue →".
        **kwargs:         Forwarded to CTkFrame.
    """

    def __init__(
        self,
        master: ctk.CTk | ctk.CTkFrame,
        on_continue: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_continue = on_continue
        self._selected: Optional[str] = None

        # Map doc-type key → card widget (populated in _build_ui)
        self._cards: dict[str, ctk.CTkFrame] = {}

        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construct all child widgets."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Title ──
        title = ctk.CTkLabel(
            self,
            text="Select Document Type",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        title.grid(row=0, column=0, pady=(40, 8))

        subtitle = ctk.CTkLabel(
            self,
            text="Choose the Indian identity document you wish to verify",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray65"),
        )
        subtitle.grid(row=1, column=0, pady=(0, 30))

        # ── Card grid ──
        grid_frame = ctk.CTkFrame(self, fg_color="transparent")
        grid_frame.grid(row=2, column=0, sticky="n")

        doc_keys = list(DOC_TYPES.keys())
        for idx, key in enumerate(doc_keys):
            row, col = divmod(idx, 2)
            card = self._make_card(grid_frame, key)
            card.grid(row=row, column=col, padx=20, pady=16, sticky="nsew")
            self._cards[key] = card

        # ── Continue button ──
        self._continue_btn = ctk.CTkButton(
            self,
            text="Continue  →",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=44,
            width=220,
            state="disabled",
            command=self._on_continue_clicked,
        )
        self._continue_btn.grid(row=3, column=0, pady=(30, 40))

    def _make_card(self, parent: ctk.CTkFrame, key: str) -> ctk.CTkFrame:
        """
        Create a single 200×120 document type card.

        Args:
            parent: Parent frame (the 2×2 grid container).
            key:    Document type key from ``DOC_TYPES``.

        Returns:
            Configured CTkFrame card widget.
        """
        info = DOC_TYPES[key]
        card = ctk.CTkFrame(
            parent,
            width=200,
            height=120,
            corner_radius=14,
            border_width=2,
            border_color=("gray70", "gray30"),
            cursor="hand2",
        )
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure((0, 1, 2), weight=1)

        emoji_lbl = ctk.CTkLabel(
            card,
            text=info["emoji"],
            font=ctk.CTkFont(size=30),
        )
        emoji_lbl.grid(row=0, column=0, pady=(14, 0))

        name_lbl = ctk.CTkLabel(
            card,
            text=info["label"],
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        name_lbl.grid(row=1, column=0)

        desc_map = {
            "AADHAAR":  "12-digit UID",
            "PAN":      "10-char tax ID",
            "VOTER_ID": "EPIC / EC card",
            "DL":       "Motor vehicle licence",
        }
        desc_lbl = ctk.CTkLabel(
            card,
            text=desc_map.get(key, ""),
            font=ctk.CTkFont(size=10),
            text_color=("gray45", "gray60"),
        )
        desc_lbl.grid(row=2, column=0, pady=(0, 10))

        # Bind clicks on card and all children
        for widget in (card, emoji_lbl, name_lbl, desc_lbl):
            widget.bind("<Button-1>", lambda e, k=key: self._select_card(k))

        return card

    # ─── Interaction handlers ─────────────────────────────────────────────────

    def _select_card(self, key: str) -> None:
        """Highlight the clicked card and deselect others."""
        self._selected = key

        for k, card in self._cards.items():
            if k == key:
                card.configure(border_color=COLOR_PRIMARY, border_width=3)
            else:
                card.configure(border_color=("gray70", "gray30"), border_width=2)

        self._continue_btn.configure(state="normal")

    def _on_continue_clicked(self) -> None:
        """Invoke the parent callback with the currently-selected doc type."""
        if self._selected:
            self._on_continue(self._selected)

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all cards to the unselected state (for re-use)."""
        self._selected = None
        for card in self._cards.values():
            card.configure(border_color=("gray70", "gray30"), border_width=2)
        self._continue_btn.configure(state="disabled")

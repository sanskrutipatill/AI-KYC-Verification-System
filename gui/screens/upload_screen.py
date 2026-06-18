# FILE: kyc_system/gui/screens/upload_screen.py
"""
Screen 2 — Document Upload & OCR Extraction.

Left panel: drag-drop zone / file picker with image preview.
Right panel: scrollable extracted fields list with confidence bars.
"""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk
from PIL import Image

from config import COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER, DOCUMENT_IMAGE_PATH
from utils.logger import get_logger

log = get_logger(__name__)


class UploadScreen(ctk.CTkFrame):
    """
    Document upload screen (Step 2 of the KYC flow).

    Provides a file picker / drag-drop zone on the left and a scrollable
    field-extraction panel on the right.

    Args:
        master:          Parent CTk widget.
        doc_type:        Selected document type key (e.g. ``"AADHAAR"``).
        on_continue:     Callback invoked with (image_path, fields_dict) when
                         the user clicks "Continue →".
        **kwargs:        Forwarded to CTkFrame.
    """

    def __init__(
        self,
        master,
        doc_type: str = "AADHAAR",
        on_continue: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._doc_type = doc_type
        self._on_continue = on_continue
        self._image_path: Optional[Path] = None
        self._fields: Optional[Dict[str, Any]] = None
        self._ocr_engine = None

        self._build_ui()

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the two-panel layout."""
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──
        header = ctk.CTkLabel(
            self,
            text=f"Upload Document  ·  {self._doc_type.replace('_', ' ')}",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, columnspan=2, pady=(30, 20), padx=30, sticky="w")

        # ── Left panel (upload) ──
        self._left_panel = ctk.CTkFrame(self, corner_radius=12)
        self._left_panel.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        self._left_panel.grid_columnconfigure(0, weight=1)
        self._left_panel.grid_rowconfigure(1, weight=1)

        self._build_upload_zone()

        # ── Right panel (fields) ──
        self._right_panel = ctk.CTkFrame(self, corner_radius=12)
        self._right_panel.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        self._right_panel.grid_columnconfigure(0, weight=1)
        self._right_panel.grid_rowconfigure(1, weight=1)

        self._build_fields_panel()

        # ── Footer buttons ──
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, columnspan=2, pady=(10, 24), padx=20)

        self._extract_btn = ctk.CTkButton(
            footer,
            text="  Extract Details",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=180,
            height=40,
            state="disabled",
            command=self._run_ocr_threaded,
        )
        self._extract_btn.pack(side="left", padx=10)

        self._continue_btn = ctk.CTkButton(
            footer,
            text="Continue  →",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=180,
            height=40,
            state="disabled",
            command=self._on_continue_clicked,
        )
        self._continue_btn.pack(side="left", padx=10)

    def _build_upload_zone(self) -> None:
        """Build the file-drop / preview area inside the left panel."""
        zone_lbl = ctk.CTkLabel(
            self._left_panel,
            text="Document Image",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        zone_lbl.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 0))

        # Dashed drop zone
        self._drop_zone = ctk.CTkFrame(
            self._left_panel,
            corner_radius=10,
            border_width=2,
            border_color=("gray55", "gray40"),
            cursor="hand2",
            fg_color=("gray90", "gray20"),
        )
        self._drop_zone.grid(row=1, column=0, sticky="nsew", padx=16, pady=12)
        self._drop_zone.grid_columnconfigure(0, weight=1)
        self._drop_zone.grid_rowconfigure(0, weight=1)

        self._drop_label = ctk.CTkLabel(
            self._drop_zone,
            text="📂\n\nClick to select document image\n(JPG · PNG · PDF)",
            font=ctk.CTkFont(size=13),
            text_color=("gray50", "gray55"),
            justify="center",
        )
        self._drop_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=40)

        self._preview_label = ctk.CTkLabel(self._drop_zone, text="")
        self._preview_label.grid(row=0, column=0, sticky="nsew")
        self._preview_label.grid_remove()  # hidden until image loaded

        # Click to browse
        for w in (self._drop_zone, self._drop_label):
            w.bind("<Button-1>", lambda e: self._browse_file())

        self._progress_bar = ctk.CTkProgressBar(
            self._left_panel, mode="indeterminate", height=6
        )
        self._progress_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        self._progress_bar.grid_remove()

    def _build_fields_panel(self) -> None:
        """Build the extracted fields display inside the right panel."""
        fields_hdr = ctk.CTkLabel(
            self._right_panel,
            text="Extracted Fields",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        fields_hdr.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 0))

        self._fields_scroll = ctk.CTkScrollableFrame(
            self._right_panel, label_text="", corner_radius=8
        )
        self._fields_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        self._fields_scroll.grid_columnconfigure(0, weight=1)

        self._placeholder_lbl = ctk.CTkLabel(
            self._fields_scroll,
            text="Upload a document and click\n'Extract Details' to begin.",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray55"),
            justify="center",
        )
        self._placeholder_lbl.pack(pady=60)

    # ─── File selection ───────────────────────────────────────────────────────

    def _browse_file(self) -> None:
        """Open a file dialog and load the selected image."""
        path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("PDF", "*.pdf"), ("All", "*.*")],
        )
        if not path:
            return
        self._image_path = Path(path)
        self._load_preview(self._image_path)
        self._extract_btn.configure(state="normal")

    def _load_preview(self, path: Path) -> None:
        """Show a thumbnail preview of the selected document image."""
        try:
            img = Image.open(path)
            img.thumbnail((380, 280), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

            self._drop_label.grid_remove()
            self._preview_label.configure(image=ctk_img, text="")
            self._preview_label.image = ctk_img  # keep reference
            self._preview_label.grid()
        except Exception as exc:
            log.error("UploadScreen: failed to load preview — %s", exc)

    # ─── OCR ──────────────────────────────────────────────────────────────────

    def _run_ocr_threaded(self) -> None:
        """Kick off OCR in a background thread to keep the GUI responsive."""
        if self._image_path is None:
            return
        self._extract_btn.configure(state="disabled", text="Processing…")
        self._continue_btn.configure(state="disabled")
        self._progress_bar.grid()
        self._progress_bar.start()

        t = threading.Thread(target=self._run_ocr, daemon=True)
        t.start()

    def _run_ocr(self) -> None:
        """Background OCR worker."""
        try:
            import cv2
            from core.ocr_engine import OCREngine

            if self._ocr_engine is None:
                self._ocr_engine = OCREngine()

            img = cv2.imread(str(self._image_path))
            if img is None:
                self.after(0, self._show_error, "Could not read image file.")
                return

            # Save a copy to our temp directory for downstream modules
            import shutil
            shutil.copy2(str(self._image_path), str(DOCUMENT_IMAGE_PATH))

            fields = self._ocr_engine.extract_fields(img, self._doc_type)
            self._fields = fields
            self.after(0, self._display_fields, fields)
        except Exception as exc:
            log.error("OCR error: %s", exc)
            self.after(0, self._show_error, str(exc))
        finally:
            self.after(0, self._ocr_done)

    def _ocr_done(self) -> None:
        """Re-enable UI after OCR completes."""
        self._progress_bar.stop()
        self._progress_bar.grid_remove()
        self._extract_btn.configure(state="normal", text="  Extract Details")

    # ─── Fields display ───────────────────────────────────────────────────────

    def _display_fields(self, fields: Dict[str, Any]) -> None:
        """Render extracted field rows in the scrollable panel."""
        # Clear existing rows
        for child in self._fields_scroll.winfo_children():
            child.destroy()

        display_keys = [k for k in fields if k not in ("raw_text", "avg_confidence")]

        if not display_keys:
            ctk.CTkLabel(
                self._fields_scroll,
                text="No fields extracted. Try a clearer image.",
                text_color=COLOR_DANGER,
            ).pack(pady=20)
            return

        for key in display_keys:
            entry = fields[key]
            value = entry.get("value") if isinstance(entry, dict) else str(entry)
            conf  = entry.get("confidence", 0.0) if isinstance(entry, dict) else 0.0

            row = ctk.CTkFrame(self._fields_scroll, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=4)
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(
                row,
                text=key.replace("_", " ").title(),
                font=ctk.CTkFont(size=11, weight="bold"),
                width=120,
                anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=(4, 8))

            val_text = str(value) if value else "—"
            ctk.CTkLabel(
                row,
                text=val_text,
                font=ctk.CTkFont(size=11),
                anchor="w",
                text_color=("gray20", "gray80"),
            ).grid(row=0, column=1, sticky="w")

            bar_color = (
                COLOR_SUCCESS if conf >= 0.8
                else COLOR_WARNING if conf >= 0.5
                else COLOR_DANGER
            )
            conf_bar = ctk.CTkProgressBar(row, height=4, progress_color=bar_color)
            conf_bar.set(conf)
            conf_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 4))

        self._continue_btn.configure(state="normal")

    # ─── Error ────────────────────────────────────────────────────────────────

    def _show_error(self, message: str) -> None:
        """Show an error message in the fields panel."""
        for child in self._fields_scroll.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self._fields_scroll,
            text=f"⚠  {message}",
            text_color=COLOR_DANGER,
            wraplength=260,
        ).pack(pady=30)

    # ─── Continue ─────────────────────────────────────────────────────────────

    def _on_continue_clicked(self) -> None:
        """Invoke the parent callback."""
        if self._on_continue and self._image_path:
            self._on_continue(self._image_path, self._fields or {})

    # ─── Public API ───────────────────────────────────────────────────────────

    def reset(self, doc_type: str) -> None:
        """
        Reset the screen for a new upload session.

        Args:
            doc_type: New document type key.
        """
        self._doc_type = doc_type
        self._image_path = None
        self._fields = None
        self._preview_label.configure(image=None, text="")
        self._preview_label.grid_remove()
        self._drop_label.grid()
        self._extract_btn.configure(state="disabled", text="  Extract Details")
        self._continue_btn.configure(state="disabled")
        for child in self._fields_scroll.winfo_children():
            child.destroy()
        self._placeholder_lbl = ctk.CTkLabel(
            self._fields_scroll,
            text="Upload a document and click\n'Extract Details' to begin.",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray55"),
            justify="center",
        )
        self._placeholder_lbl.pack(pady=60)

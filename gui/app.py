# FILE: kyc_system/gui/app.py
"""
Main Application Class for the KYC Verification System.

Manages the top-level CTk window, navigation between screens, and the shared
state that flows from step to step (selected doc type, image paths, OCR fields,
pipeline results).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import customtkinter as ctk

from config import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    CAPTURED_FACE_PATH,
    DOCUMENT_IMAGE_PATH,
)
from gui.screens.document_select import DocumentSelectScreen
from gui.screens.upload_screen import UploadScreen
from gui.screens.webcam_screen import WebcamScreen
from gui.screens.processing_screen import ProcessingScreen
from gui.screens.result_screen import ResultScreen
from utils.logger import get_logger

log = get_logger(__name__)


class KYCApp(ctk.CTk):
    """
    Root application window for the KYC Verification System.

    Uses frame-switching (not multiple windows or CTkTabview) to navigate
    between the five screens.  All inter-screen state is stored as instance
    attributes and passed to each screen's callbacks.

    Attributes:
        doc_type    (str):  Currently selected document type key.
        ocr_fields  (dict): Fields extracted by the OCR engine.
        doc_image_path (Path): Saved document image path.
        live_face_path (Path): Saved webcam capture path.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Window configuration ──────────────────────────────────────────────
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Shared flow state ─────────────────────────────────────────────────
        self.doc_type:        str            = ""
        self.ocr_fields:      Dict[str, Any] = {}
        self.doc_image_path:  Path           = DOCUMENT_IMAGE_PATH
        self.live_face_path:  Path           = CAPTURED_FACE_PATH

        # ── Header ────────────────────────────────────────────────────────────
        self._build_header()

        # ── Screen container ──────────────────────────────────────────────────
        self._container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._container.grid(row=1, column=0, sticky="nsew")
        self._container.grid_columnconfigure(0, weight=1)
        self._container.grid_rowconfigure(0, weight=1)

        # ── Screens ───────────────────────────────────────────────────────────
        self._screens: Dict[str, ctk.CTkFrame] = {}
        self._current: Optional[ctk.CTkFrame]  = None

        self._init_screens()
        self._show_screen("select")

        # ── Protocol ──────────────────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        """Build the top navigation/brand bar."""
        header = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color=("gray88", "gray12"))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header,
            text="🪪  KYC Verification System",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=14, sticky="w")

        self._step_label = ctk.CTkLabel(
            header,
            text="Step 1 of 5 — Document Selection",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray55"),
        )
        self._step_label.grid(row=0, column=1, sticky="e", padx=20)

        # Appearance toggle
        self._mode_btn = ctk.CTkButton(
            header,
            text="🌙",
            width=36,
            height=30,
            fg_color="transparent",
            hover_color=("gray80", "gray25"),
            command=self._toggle_mode,
        )
        self._mode_btn.grid(row=0, column=2, padx=(0, 12))

    def _toggle_mode(self) -> None:
        """Switch between dark and light appearance mode."""
        current = ctk.get_appearance_mode()
        new_mode = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self._mode_btn.configure(text="☀️" if new_mode == "Dark" else "🌙")

    # ─── Screen management ────────────────────────────────────────────────────

    def _init_screens(self) -> None:
        """Instantiate all five screens and place them in the container grid."""

        self._screens["select"] = DocumentSelectScreen(
            master=self._container,
            on_continue=self._on_doc_selected,
            fg_color="transparent",
        )

        self._screens["upload"] = UploadScreen(
            master=self._container,
            doc_type="AADHAAR",
            on_continue=self._on_upload_done,
            fg_color="transparent",
        )

        self._screens["webcam"] = WebcamScreen(
            master=self._container,
            on_continue=self._on_capture_done,
            fg_color="transparent",
        )

        self._screens["processing"] = ProcessingScreen(
            master=self._container,
            on_done=self._on_processing_done,
            fg_color="transparent",
        )

        self._screens["result"] = ResultScreen(
            master=self._container,
            on_new=self._reset_flow,
            on_exit=self._on_close,
            fg_color="transparent",
        )

        for screen in self._screens.values():
            screen.grid(row=0, column=0, sticky="nsew")
            screen.grid_remove()

    def _show_screen(self, name: str) -> None:
        """
        Transition to the named screen.

        Args:
            name: Key in ``self._screens`` — one of
                  ``"select"``, ``"upload"``, ``"webcam"``,
                  ``"processing"``, ``"result"``.
        """
        if self._current is not None:
            self._current.grid_remove()

        screen = self._screens[name]
        screen.grid()
        self._current = screen

        steps = {
            "select":     "Step 1 of 5 — Document Selection",
            "upload":     "Step 2 of 5 — Document Upload",
            "webcam":     "Step 3 of 5 — Webcam Capture",
            "processing": "Step 4 of 5 — Processing",
            "result":     "Step 5 of 5 — Results",
        }
        self._step_label.configure(text=steps.get(name, ""))
        log.info("KYCApp: navigated to screen '%s'.", name)

    # ─── Navigation callbacks ─────────────────────────────────────────────────

    def _on_doc_selected(self, doc_type: str) -> None:
        """
        Step 1 → 2: user selected a document type.

        Args:
            doc_type: Selected document key (e.g. ``"AADHAAR"``).
        """
        self.doc_type = doc_type
        log.info("KYCApp: doc_type=%s", doc_type)

        # Rebuild upload screen for the new doc type
        upload: UploadScreen = self._screens["upload"]  # type: ignore
        upload.reset(doc_type)

        self._show_screen("upload")

    def _on_upload_done(self, image_path: Path, fields: Dict[str, Any]) -> None:
        """
        Step 2 → 3: document image uploaded and OCR complete.

        Args:
            image_path: Saved document image path.
            fields:     OCR extraction results dict.
        """
        self.doc_image_path = image_path
        self.ocr_fields = fields
        log.info("KYCApp: document image=%s fields_count=%d",
                 image_path.name, len(fields))

        webcam: WebcamScreen = self._screens["webcam"]  # type: ignore
        self._show_screen("webcam")
        webcam.start_camera()

    def _on_capture_done(self, live_face_path: Path) -> None:
        """
        Step 3 → 4: webcam frame captured.

        Args:
            live_face_path: Path to the saved face image.
        """
        self.live_face_path = live_face_path
        log.info("KYCApp: live face captured → %s", live_face_path)

        processing: ProcessingScreen = self._screens["processing"]  # type: ignore
        self._show_screen("processing")
        processing.start_processing(
            doc_image_path=self.doc_image_path,
            live_face_path=self.live_face_path,
            doc_type=self.doc_type,
            ocr_fields=self.ocr_fields,
        )

    def _on_processing_done(self, results: Dict[str, Any]) -> None:
        """
        Step 4 → 5: AI pipeline complete.

        Args:
            results: Full pipeline results dict.
        """
        result_screen: ResultScreen = self._screens["result"]  # type: ignore
        result_screen.display_results(results)
        self._show_screen("result")

    def _reset_flow(self) -> None:
        """Return to Step 1 and reset all screens for a new verification."""
        select: DocumentSelectScreen = self._screens["select"]  # type: ignore
        select.reset()
        self.doc_type = ""
        self.ocr_fields = {}
        self._show_screen("select")

    # ─── Window close ─────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Gracefully stop camera threads before destroying the window."""
        try:
            webcam: WebcamScreen = self._screens.get("webcam")  # type: ignore
            if webcam:
                webcam.stop_camera()
        except Exception:
            pass
        self.destroy()
        sys.exit(0)

# FILE: kyc_system/gui/screens/webcam_screen.py
"""
Screen 3 — Webcam Capture.

Presents a live webcam feed with real-time face detection overlay.
The user can capture a frame and retake as many times as needed.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from config import (
    WEBCAM_INDEX, WEBCAM_FRAME_RATE, WEBCAM_WIDTH, WEBCAM_HEIGHT,
    CAPTURED_FACE_PATH, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
)
from utils.webcam import WebcamThread
from utils.image_utils import draw_face_box, save_image
from utils.logger import get_logger

log = get_logger(__name__)


class WebcamScreen(ctk.CTkFrame):
    """
    Live webcam feed with face detection overlay (Step 3 of the KYC flow).

    Provides real-time face detection using InsightFace (or Haar cascade as
    fallback).  When a face is detected, the user can capture the frame.

    Args:
        master:      Parent CTk widget.
        on_continue: Callback invoked with the captured image path.
        **kwargs:    Forwarded to CTkFrame.
    """

    def __init__(
        self,
        master,
        on_continue: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_continue = on_continue
        self._cam = WebcamThread(
            index=WEBCAM_INDEX,
            width=WEBCAM_WIDTH,
            height=WEBCAM_HEIGHT,
        )
        self._face_cascade = self._load_haar_cascade()
        self._captured: Optional[np.ndarray] = None
        self._live = False
        self._after_id: Optional[str] = None
        self._face_detector = None  # InsightFace loaded lazily

        self._build_ui()

    # ─── Face detection helpers ───────────────────────────────────────────────

    def _load_haar_cascade(self):
        """Load OpenCV Haar cascade as a lightweight face detector fallback."""
        cc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return cc

    def _detect_faces_haar(self, frame: np.ndarray):
        """Return list of (x1,y1,x2,y2) bounding boxes using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        boxes = []
        for (x, y, w, h) in rects:
            boxes.append((x, y, x + w, y + h))
        return boxes

    # ─── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the webcam screen layout."""
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──
        ctk.CTkLabel(
            self,
            text="Live Face Capture",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(24, 4), sticky="w", padx=30)

        ctk.CTkLabel(
            self,
            text="Position your face inside the frame, then click Capture",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray65"),
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=30, pady=(0, 8))

        # ── Feed (left) ──
        feed_frame = ctk.CTkFrame(self, corner_radius=12)
        feed_frame.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")
        feed_frame.grid_columnconfigure(0, weight=1)
        feed_frame.grid_rowconfigure(0, weight=1)

        self._feed_label = ctk.CTkLabel(
            feed_frame,
            text="Camera not started",
            width=WEBCAM_WIDTH,
            height=WEBCAM_HEIGHT,
            fg_color=("gray85", "gray15"),
            corner_radius=8,
        )
        self._feed_label.grid(row=0, column=0, padx=10, pady=10)

        self._status_lbl = ctk.CTkLabel(
            feed_frame,
            text="⏳  Initialising camera…",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_WARNING,
        )
        self._status_lbl.grid(row=1, column=0, pady=(0, 8))

        # ── Right side (preview + controls) ──
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=2, column=1, padx=(10, 20), pady=10, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right,
            text="Captured Preview",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, pady=(16, 8))

        self._preview_lbl = ctk.CTkLabel(
            right,
            text="No capture yet",
            width=200,
            height=150,
            fg_color=("gray88", "gray18"),
            corner_radius=8,
            text_color=("gray55", "gray50"),
        )
        self._preview_lbl.grid(row=1, column=0, padx=20, pady=8)

        self._capture_btn = ctk.CTkButton(
            right,
            text="📸  Capture",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=160,
            state="disabled",
            command=self._capture_frame,
        )
        self._capture_btn.grid(row=2, column=0, pady=8)

        self._retake_btn = ctk.CTkButton(
            right,
            text="🔄  Retake",
            font=ctk.CTkFont(size=13),
            height=36,
            width=160,
            fg_color=("gray70", "gray35"),
            hover_color=("gray60", "gray40"),
            state="disabled",
            command=self._retake,
        )
        self._retake_btn.grid(row=3, column=0, pady=4)

        self._continue_btn = ctk.CTkButton(
            right,
            text="Continue  →",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=160,
            state="disabled",
            command=self._on_continue_clicked,
        )
        self._continue_btn.grid(row=4, column=0, pady=(16, 20))

    # ─── Camera lifecycle ─────────────────────────────────────────────────────

    def start_camera(self) -> None:
        """Open the webcam and begin the frame-update loop."""
        if self._cam.start():
            self._live = True
            self._update_frame()
            log.info("WebcamScreen: camera started.")
        else:
            self._status_lbl.configure(
                text="❌  Could not open camera. Check device and restart.",
                text_color=COLOR_DANGER,
            )

    def stop_camera(self) -> None:
        """Stop the frame-update loop and release the camera."""
        self._live = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._cam.stop()
        log.info("WebcamScreen: camera stopped.")

    # ─── Frame update loop ────────────────────────────────────────────────────

    def _update_frame(self) -> None:
        """Fetch the latest frame, annotate it, and schedule the next update."""
        if not self._live:
            return

        frame = self._cam.read()
        if frame is not None:
            boxes = self._detect_faces_haar(frame)
            face_found = len(boxes) > 0

            for box in boxes:
                frame = draw_face_box(frame, box, color=(0, 220, 100), thickness=2)

            if face_found:
                self._status_lbl.configure(
                    text="✅  Face detected — ready to capture",
                    text_color=COLOR_SUCCESS,
                )
                self._capture_btn.configure(state="normal")
            else:
                self._status_lbl.configure(
                    text="👤  No face detected",
                    text_color=COLOR_WARNING,
                )
                self._capture_btn.configure(state="disabled")

            self._show_frame_on_label(frame, self._feed_label, max_w=480, max_h=360)

        self._after_id = self.after(WEBCAM_FRAME_RATE, self._update_frame)

    # ─── Frame display ────────────────────────────────────────────────────────

    @staticmethod
    def _show_frame_on_label(
        frame: np.ndarray,
        label: ctk.CTkLabel,
        max_w: int = 480,
        max_h: int = 360,
    ) -> None:
        """Convert a BGR OpenCV frame to CTkImage and display on *label*."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        label.configure(image=ctk_img, text="")
        label.image = ctk_img  # prevent garbage collection

    # ─── Capture / Retake ─────────────────────────────────────────────────────

    def _capture_frame(self) -> None:
        """Freeze the current frame as the captured image."""
        frame = self._cam.read()
        if frame is None:
            return
        self._captured = frame.copy()
        self._live = False  # pause the feed

        # Show thumbnail preview
        self._show_frame_on_label(frame, self._preview_lbl, max_w=200, max_h=150)

        # Save to temp path
        try:
            save_image(self._captured, CAPTURED_FACE_PATH)
            log.info("WebcamScreen: frame captured → %s", CAPTURED_FACE_PATH)
        except Exception as exc:
            log.error("WebcamScreen: could not save capture — %s", exc)

        self._capture_btn.configure(state="disabled")
        self._retake_btn.configure(state="normal")
        self._continue_btn.configure(state="normal")

    def _retake(self) -> None:
        """Resume the live feed for a new capture attempt."""
        self._captured = None
        self._preview_lbl.configure(image=None, text="No capture yet")
        self._preview_lbl.image = None
        self._continue_btn.configure(state="disabled")
        self._retake_btn.configure(state="disabled")
        self._live = True
        self._update_frame()

    # ─── Continue ─────────────────────────────────────────────────────────────

    def _on_continue_clicked(self) -> None:
        """Invoke the parent callback with the captured image path."""
        self.stop_camera()
        if self._on_continue:
            self._on_continue(CAPTURED_FACE_PATH)

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Ensure camera is released when the screen is destroyed."""
        self.stop_camera()
        super().destroy()

# FILE: kyc_system/utils/webcam.py
"""
OpenCV webcam capture thread for the KYC Verification System.

The :class:`WebcamThread` runs in a background daemon thread, continuously
reading frames from the default camera and making the latest frame available
to the GUI via a :class:`threading.Lock`-guarded buffer.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class WebcamThread:
    """
    Background thread that continuously captures frames from a webcam.

    Usage::

        cam = WebcamThread(index=0)
        cam.start()
        frame = cam.read()   # numpy BGR array or None
        cam.stop()

    Attributes:
        index   (int):  Camera device index passed to ``cv2.VideoCapture``.
        running (bool): ``True`` while the capture loop is active.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        """
        Initialise the webcam thread.

        Args:
            index:  OpenCV camera index (0 for the first/default camera).
            width:  Requested capture width in pixels.
            height: Requested capture height in pixels.
        """
        self.index = index
        self.width = width
        self.height = height
        self.running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open the camera and start the background capture loop.

        Returns:
            ``True`` if the camera was opened successfully, ``False`` otherwise.
        """
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            log.warning("WebcamThread: could not open camera index=%d", self.index)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("WebcamThread: started (index=%d, %dx%d)", self.index, self.width, self.height)
        return True

    def stop(self) -> None:
        """Signal the capture loop to stop and release the camera resource."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("WebcamThread: stopped.")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most recently captured frame.

        Returns:
            BGR numpy array, or ``None`` if no frame is available yet.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_open(self) -> bool:
        """Return ``True`` if the camera is open and the thread is running."""
        return self.running and self._cap is not None and self._cap.isOpened()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Continuously read frames from the camera until ``self.running`` is False.
        Frames are stored in the internal buffer under a lock so that :meth:`read`
        is always thread-safe.
        """
        while self.running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                log.debug("WebcamThread: failed to read frame — skipping.")

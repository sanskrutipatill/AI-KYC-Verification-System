# FILE: kyc_system/main.py
"""
Entry point for the KYC Verification System.

Sets the global CustomTkinter appearance and launches the main application
window.  All heavy ML work is done on background threads — this module only
bootstraps the GUI.
"""

import sys
from pathlib import Path

# ── Ensure the project root is on PYTHONPATH so all modules resolve correctly ──
sys.path.insert(0, str(Path(__file__).parent))

import customtkinter as ctk

from config import APPEARANCE_MODE, COLOR_THEME, WINDOW_TITLE
from gui.app import KYCApp
from utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    """
    Initialise CustomTkinter and start the KYC application event loop.

    This function should be the sole entry point; do not import from it.
    """
    log.info("KYC Verification System starting up…")

    # ── Global CTk settings ───────────────────────────────────────────────────
    ctk.set_appearance_mode(APPEARANCE_MODE)
    ctk.set_default_color_theme(COLOR_THEME)

    # ── Launch application ────────────────────────────────────────────────────
    app = KYCApp()
    app.mainloop()

    log.info("KYC Verification System exited.")


if __name__ == "__main__":
    main()

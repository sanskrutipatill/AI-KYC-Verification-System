# FILE: kyc_system/config.py
"""
Global configuration constants, thresholds, and paths for the KYC Verification System.
All tunable parameters should be modified here rather than in individual modules.
"""

from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ─── GUI Settings ─────────────────────────────────────────────────────────────
WINDOW_TITLE = "KYC Verification System"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600
APPEARANCE_MODE = "System"   # "Dark", "Light", or "System"
COLOR_THEME = "blue"

# ─── Color Palette ────────────────────────────────────────────────────────────
COLOR_SUCCESS = "#22c55e"      # green-500
COLOR_WARNING = "#f59e0b"      # amber-500
COLOR_DANGER  = "#ef4444"      # red-500
COLOR_PRIMARY = "#1f6aa5"      # CTk blue
COLOR_MUTED   = "#6b7280"      # gray-500
COLOR_VERIFIED_BG  = "#166534"
COLOR_REJECTED_BG  = "#7f1d1d"
COLOR_REVIEW_BG    = "#78350f"

# ─── Face Matching ────────────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD = 0.40    # ArcFace cosine: >=0.40 partial match; same person ~0.60-0.90
FACE_MATCH_MODEL = "buffalo_l" # insightface model pack (ArcFace + detection)

# ─── Liveness Detection ───────────────────────────────────────────────────────
LIVENESS_THRESHOLD = 0.60      # min score to be considered "Real"
LIVENESS_MODEL_DIR = MODELS_DIR / "silent_face"
LIVENESS_MODEL_V1 = "4_0_0_80x80_MiniFASNetV1SE.pth"
LIVENESS_MODEL_V2 = "2.7_80x80_MiniFASNetV2.pth"
LIVENESS_INPUT_SIZE = (80, 80)

# ─── Deepfake Detection ───────────────────────────────────────────────────────
DEEPFAKE_THRESHOLD = 0.50      # prob >= threshold → deepfake
DEEPFAKE_INPUT_SIZE = (299, 299)
DEEPFAKE_MODEL_PATH = MODELS_DIR / "deepfake_detector.pth"

# ─── OCR ──────────────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.50  # below this → field flagged low confidence
OCR_USE_GPU = False
OCR_USE_ANGLE_CLS = True
OCR_LANG = "en"

# ─── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "face_match": 0.35,
    "liveness":   0.25,
    "ocr":        0.15,
    "deepfake":   0.10,   # weight for (1 - deepfake_prob)
    "doc_valid":  0.10,
    "fields":     0.05,
}
RISK_THRESHOLD_VERIFIED = 0.30  # risk <= this → VERIFIED
RISK_THRESHOLD_REVIEW   = 0.55  # risk <= this → REVIEW  else REJECTED
XGBOOST_MODEL_PATH = MODELS_DIR / "risk_model.json"

# ─── Document Types ───────────────────────────────────────────────────────────
DOC_TYPES = {
    "AADHAAR":  {"label": "Aadhaar Card",       "emoji": "🪪"},
    "PAN":      {"label": "PAN Card",            "emoji": "💳"},
    "VOTER_ID": {"label": "Voter ID",            "emoji": "🗳️"},
    "DL":       {"label": "Driving License",     "emoji": "🚗"},
}

# ─── Voter ID State Prefixes (all 36 states/UTs) ──────────────────────────────
VOTER_ID_STATE_CODES = {
    "AND", "ARP", "ASM", "BIH", "CHH", "GOA", "GUJ", "HAR",
    "HPR", "JHK", "KAR", "KER", "LAD", "MPR", "MAH", "MNP",
    "MEG", "MIZ", "NAG", "ODI", "PDU", "PNJ", "RAJ", "SKM",
    "TNA", "TEL", "TRP", "UPR", "UTK", "WBG", "DLH", "CHD",
    "ANI", "DDA", "LKD", "JNK",
}

# ─── RTO State Codes (Driving Licence) ────────────────────────────────────────
RTO_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# ─── Webcam ───────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0
WEBCAM_FRAME_RATE = 30         # ms delay between frame refreshes (33ms ≈ 30fps)
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
CAPTURED_FACE_PATH = TEMP_DIR / "live_face.jpg"
DOCUMENT_IMAGE_PATH = TEMP_DIR / "document_image.jpg"

# ─── Export ───────────────────────────────────────────────────────────────────
EXPORT_DIR = Path.home() / "Downloads"

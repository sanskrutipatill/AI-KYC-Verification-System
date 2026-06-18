# KYC Verification System — Project Description

## Overview

This is a **desktop-based Know Your Customer (KYC) Verification System** built in Python, designed specifically for **Indian identity document verification**. It combines multiple AI/ML technologies — computer vision, OCR, deep learning, and anti-spoofing — into a unified, step-by-step pipeline with a clean GUI. The goal is to automatically verify whether the person appearing on a webcam is the same person shown on a submitted government identity document, while also detecting possible fraud signals like spoofed faces or deepfakes.

---

## Purpose & Domain

KYC (Know Your Customer) is a mandatory compliance process used by banks, fintech apps, telecom providers, and other regulated industries to verify the identity of their customers. This system automates that process locally on a Windows desktop using AI models — no cloud APIs or external services required.

It supports **four Indian government-issued identity documents**:
- 🪪 **Aadhaar Card** — UIDAI's 12-digit biometric ID
- 💳 **PAN Card** — Income Tax Department's Permanent Account Number
- 🗳️ **Voter ID (EPIC)** — Election Commission of India card
- 🚗 **Driving Licence** — Regional Transport Office issued licence

---

## Architecture

The project is structured into four main layers:

```
kyc_system/
├── core/           ← AI/ML pipeline modules
├── gui/            ← Desktop UI (screens + components)
├── utils/          ← Shared utilities (logging, webcam, image tools)
├── models/         ← Pre-trained model weights storage
├── config.py       ← All thresholds, paths, constants
└── main.py         ← Entry point
```

---

## AI/ML Pipeline — Core Modules

The system runs a **5-stage AI pipeline**, each handled by a dedicated module:

### 1. OCR Engine (`core/ocr_engine.py`)
- Uses **PaddleOCR** to extract raw text from document images
- Applies **regex-based post-processing** to identify structured fields per document type:
  - Aadhaar: number, name, DOB, gender
  - PAN: number, name, father's name, DOB
  - Voter ID: number, name, DOB, address
  - Driving Licence: number, name, DOB, validity date, vehicle class
- Provides per-field confidence scores estimated from OCR line scores
- Lazy-loaded at first use to avoid GUI startup delay

### 2. Document Validator (`core/document_validator.py`)
- Validates the OCR-extracted fields using format checks and algorithmic rules:
  - **Aadhaar**: Regex format + **Verhoeff checksum algorithm** (cryptographic check-digit verification unique to Aadhaar)
  - **PAN**: Strict 10-character regex + entity-type character validation (`P` = individual, `C` = company, etc.)
  - **Voter ID**: EPIC format regex + validation of the 3-letter state prefix against all 36 Indian states/UTs
  - **DL**: Format regex + RTO 2-letter state/UT code validation
- Returns a numeric score (0–1) indicating document integrity

### 3. Face Matcher (`core/face_matcher.py`)
- Uses **InsightFace** (`buffalo_l` model) implementing **ArcFace** — a state-of-the-art deep learning face recognition algorithm
- Generates 512-dimensional face embeddings from both:
  - The face detected in the uploaded document image
  - The live face captured by the webcam
- Computes **cosine similarity** between the two embeddings to determine if they are the same person
- Automatically detects and crops the best face from any input frame (handles full webcam frames gracefully)
- Scores are **never artificially inflated** — the raw ArcFace similarity is always used honestly

### 4. Liveness Detector (`core/liveness_detector.py`)
- Implements **anti-spoofing** using a **MiniFASNet ensemble** (two models: V1SE + V2) from the Silent-Face-Anti-Spoofing project
- The neural network classifies a face as "Real", "Spoof", or "Unknown" (3 classes), and the "Real" class probability is used as the liveness score
- Both models run at 80×80 resolution and their scores are averaged for robustness
- **Fallback**: If model weights are not installed, a heuristic based on **Laplacian variance** (image texture complexity) is used — real faces have higher high-frequency content than printed photos or screens
- This confirms that a **real person is present** in front of the camera (not a photo, video replay, or mask), but does NOT confirm identity

### 5. Deepfake Detector (`core/deepfake_detector.py`)
- Uses an **EfficientNet-B4** backbone (via `timm`) with a sigmoid binary classification head
- Estimates the probability that the live face is **AI-generated or digitally manipulated**
- When fine-tuned KYC-specific weights aren't present, falls back to ImageNet-pretrained weights as a structural signal
- Input: 299×299 normalised RGB face tensor

---

## Decision Engine (`core/risk_scorer.py`)

The `RiskScorer` is the brain of the system — a **strict, fair, explainable decision framework** that takes all 5 AI signals and produces a final KYC verdict. It implements a **priority-based hierarchy**:

| Priority | Signal | Type |
|----------|--------|------|
| 1 (Highest) | Gender Consistency | Hard binary check |
| 2 | Document Validity | Scored check |
| 3 | Liveness | Scored check |
| 4 | Face Match | Supporting signal |
| 5 (Lowest) | Deepfake Probability | Probabilistic advisory |

### Decision Outcomes

| Verdict | Conditions |
|---------|-----------|
| ✅ **VERIFIED** | Gender match + Liveness ≥ 90% + Document valid + Data consistent + Deepfake < 60% + Face match not very low |
| 🔁 **UNDER REVIEW** | Gender match + Liveness ≥ 80% + Document valid + Data consistent — but face match low OR deepfake uncertain |
| ❌ **REJECTED** | Gender MISMATCH (immediate, no override) OR Liveness < 50% + Document/data also failed |

### Key Design Principles
- **Gender mismatch = non-negotiable hard failure**: if the document says "MALE" but the face analysis detects "FEMALE", the case is immediately rejected with no override allowed
- **Liveness is about presence, not identity**: it confirms a real person is in front of the camera, but doesn't confirm who they are
- **Face match alone cannot reject**: aging, poor ID card image quality, and hairstyle changes are common — low face score alone (< 30%) sends a case to UNDER REVIEW, not rejection
- **Deepfake is probabilistic**: it is an advisory signal only and can never by itself be the sole cause of rejection
- **Prefer UNDER REVIEW over false rejection**: when signals are contradictory or borderline, the conservative policy escalates for manual review
- **Optional XGBoost model**: if a trained `risk_model.json` is present, it is used for risk magnitude estimation; otherwise, a weighted rule-based formula is used

### Risk Score Formula (Rule-based)
```
quality = 0.35 × face_match
        + 0.25 × liveness
        + 0.15 × ocr_confidence
        + 0.10 × (1 − deepfake_prob)
        + 0.10 × doc_valid_score
        + 0.05 × field_completeness

risk_score = 1 − quality   (0 = safe, 1 = high risk)
```

### Explainability
Every decision comes with a **structured reasoning chain** — a list of human-readable bullets explaining each signal's contribution, actual numeric values, and why the final verdict was reached. This is crucial for auditability in regulated KYC contexts.

---

## GUI — User Interface (`gui/`)

Built with **CustomTkinter** (a modern-looking wrapper over Python's Tkinter), the UI follows a **5-step linear wizard flow**:

| Step | Screen | Purpose |
|------|--------|---------|
| 1 | Document Selection | Choose document type (Aadhaar/PAN/Voter ID/DL) |
| 2 | Document Upload | Upload/drag-drop document image; OCR runs in background |
| 3 | Webcam Capture | Live camera feed; captures face on button press |
| 4 | Processing | AI pipeline runs (all 5 stages) with animated progress |
| 5 | Results | Displays verdict, risk score gauge, detailed reasoning |

### GUI Components
- **`score_gauge.py`**: Animated circular gauge showing the risk score (0–100)
- **`status_badge.py`**: Color-coded status badge (green = VERIFIED, amber = REVIEW, red = REJECTED)
- **`processing_screen.py`**: Runs all AI stages in a background thread with live status updates
- **`result_screen.py`**: Full detailed report — verdict, scores per signal, reasoning bullets, export option
- Has a **dark/light mode toggle** in the header bar
- Window is resizable with a minimum size of 900×600

---

## Utilities (`utils/`)

- **`logger.py`**: Structured logging to both console and rotating log files in `/logs/`
- **`webcam.py`**: Background thread-based webcam frame capture using OpenCV, with robust error handling for `CvCapture_MSMF` errors (Windows Media Foundation)
- **`image_utils.py`**: Image preprocessing helpers (resizing, normalisation, format conversions)

---

## Technology Stack

|------------------------|--------------------------|---------------|
| Category               | Library                  | Version       |
|------------------------|--------------------------|---------------|
| GUI                    | CustomTkinter            | 5.2.2         |
| Computer Vision        | OpenCV                   | 4.8.1         |
| OCR                    | PaddleOCR + PaddlePaddle | 2.8.1 / 2.6.2 |
| Face Recognition       | InsightFace (ArcFace)    | 0.7.3         |
| Inference Runtime      | ONNX Runtime             | 1.16.3        |
| Anti-Spoofing/Deepfake | PyTorch + timm           | 2.1.2 / 0.9.7 |
| Optional Risk Model    | XGBoost                  | 2.0.3         |
| Image Processing       | Pillow                   | 10.1.0        |

> **Note**: InsightFace has no official Python 3.11 wheel on PyPI, so a pre-built community wheel for Windows x64 is bundled in the `models/` directory.

---

## Configuration (`config.py`)

All thresholds are centralised and tunable without touching any module:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `FACE_MATCH_THRESHOLD` | 0.40 | Minimum cosine similarity for a "match" |
| `LIVENESS_THRESHOLD` | 0.60 | Minimum score to call a face "Real" |
| `DEEPFAKE_THRESHOLD` | 0.50 | Probability above which face is flagged as deepfake |
| `OCR_CONFIDENCE_THRESHOLD` | 0.50 | Below this, a field is flagged as low-confidence |
| `RISK_THRESHOLD_VERIFIED` | 0.30 | Risk score ≤ this → VERIFIED |
| `RISK_THRESHOLD_REVIEW` | 0.55 | Risk score ≤ this → UNDER REVIEW, else REJECTED |

---

## Data Flow Summary

```
[User] → Select Doc Type
       → Upload Document Image → [OCR Engine] → Extract Fields
       → [Document Validator] → Validate Fields + Score
       → Webcam Capture → [Liveness Detector] → Is Real Person?
                        → [Deepfake Detector] → Is AI Fake?
                        → [Face Matcher] → Does face match document?
       → [Risk Scorer] → Gender Check + Priority Decision
       → [Result Screen] → Verdict + Reasoning + Risk Gauge
```

---

## Summary

This project is a **production-quality, privacy-preserving, local KYC verification system** that:
- Runs entirely **on-device** — no internet or cloud required
- Supports India's 4 major government IDs with **document-specific validation logic**
- Uses **state-of-the-art AI** (ArcFace, MiniFASNet, EfficientNet-B4, PaddleOCR)
- Has a **fair, explainable decision engine** that prefers caution over false rejection
- Provides a **professional desktop GUI** with step-by-step guidance
- Produces **audit-ready reasoning chains** for every decision

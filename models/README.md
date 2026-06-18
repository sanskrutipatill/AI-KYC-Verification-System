# Model Download Instructions

This directory holds pre-trained and fine-tuned model weights required by
the KYC Verification System.  **None of these files are included in the
repository** — follow the steps below to download each one before running
the application.

---

## 1. InsightFace — `buffalo_l` Model Pack

Used by: `core/face_extractor.py`, `core/face_matcher.py`

The `buffalo_l` pack is downloaded **automatically** by InsightFace the first
time `FaceAnalysis` is initialised.  The files are stored in:

- **Windows**: `C:\Users\<you>\.insightface\models\buffalo_l\`
- **Linux/macOS**: `~/.insightface/models/buffalo_l/`

No manual action is required unless you are in an air-gapped environment.
For offline installation, download manually:

```bash
# Download the zip and extract to the models directory
curl -L -o buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
mkdir -p ~/.insightface/models/buffalo_l
unzip buffalo_l.zip -d ~/.insightface/models/buffalo_l/
```

---

## 2. Silent-Face-Anti-Spoofing — MiniFASNet Weights

Used by: `core/liveness_detector.py`

Place the two weight files inside `models/silent_face/`:

```
models/
└── silent_face/
    ├── 2.7_80x80_MiniFASNetV2.pth
    └── 4_0_0_80x80_MiniFASNetV1SE.pth
```

### Download steps

```bash
# Clone the reference repo to get the weights
git clone --depth 1 \
  https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git /tmp/sfa

# Create target directory
mkdir -p kyc_system/models/silent_face

# Copy the two model files
cp "/tmp/sfa/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth" \
   kyc_system/models/silent_face/

cp "/tmp/sfa/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth" \
   kyc_system/models/silent_face/
```

> **Note**: If the weights are unavailable, the liveness detector automatically
> falls back to a Laplacian-variance texture heuristic.  The application will
> not crash, but accuracy will be lower.

---

## 3. Deepfake Detector — EfficientNet-B4 Weights *(optional)*

Used by: `core/deepfake_detector.py`

Expected path: `models/deepfake_detector.pth`

If this file is **absent**, the module loads ImageNet-pretrained weights and
applies a sigmoid head.  The probabilities will not be calibrated, but the
pipeline will function normally.

To use fine-tuned weights (e.g. from FaceForensics++):

```bash
# Example using a model fine-tuned on FaceForensics++ with EfficientNet-B4
# Replace the URL with your own fine-tuned checkpoint
curl -L -o kyc_system/models/deepfake_detector.pth \
  https://example.com/your-finetuned-efficientnet-b4.pth
```

---

## 4. XGBoost Risk Model *(optional)*

Used by: `core/risk_scorer.py`

Expected path: `models/risk_model.json`

If absent, a rule-based weighted formula is used automatically.

To train your own model from historical KYC verification data:

```python
import xgboost as xgb
import numpy as np

# Feature columns (6 features):
# [face_match_score, liveness_score, ocr_confidence_avg,
#  deepfake_prob, doc_validation_score, field_completeness_ratio]

X_train = np.load("kyc_features.npy")
y_train = np.load("kyc_labels.npy")   # 0=clean, 1=fraudulent

dtrain = xgb.DMatrix(X_train, label=y_train)
params  = {"max_depth": 5, "eta": 0.1, "objective": "binary:logistic"}
model  = xgb.train(params, dtrain, num_boost_round=100)
model.save_model("kyc_system/models/risk_model.json")
```

---

## Summary Table

| File | Required | Auto-download | Fallback |
|------|----------|---------------|---------|
| `~/.insightface/models/buffalo_l/` | ✅ Yes | ✅ Yes (InsightFace) | None |
| `models/silent_face/*.pth` | Recommended | ❌ Manual | Texture heuristic |
| `models/deepfake_detector.pth` | ❌ Optional | ❌ Manual | ImageNet pretrained |
| `models/risk_model.json` | ❌ Optional | ❌ Manual | Rule-based scoring |

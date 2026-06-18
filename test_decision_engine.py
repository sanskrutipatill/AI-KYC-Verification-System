import sys
sys.path.insert(0, 'kyc_system')
from core.risk_scorer import RiskScorer

scorer = RiskScorer()

cases = [
    ("CASE 1 - Clearly Genuine", dict(
        face_match_score=0.82, liveness_score=0.93, ocr_confidence_avg=0.88,
        deepfake_prob=0.12, doc_validation_score=0.90, field_completeness_ratio=0.90,
        face_matched=True, is_real=True, is_deepfake=False, doc_valid=True),
     "VERIFIED"),
    ("CASE 2a - Likely Genuine / low face match (aging)", dict(
        face_match_score=0.38, liveness_score=0.85, ocr_confidence_avg=0.80,
        deepfake_prob=0.15, doc_validation_score=0.85, field_completeness_ratio=0.85,
        face_matched=False, is_real=True, is_deepfake=False, doc_valid=True),
     "UNDER REVIEW"),
    ("CASE 2b - Likely Genuine / deepfake uncertain zone", dict(
        face_match_score=0.65, liveness_score=0.88, ocr_confidence_avg=0.75,
        deepfake_prob=0.45, doc_validation_score=0.80, field_completeness_ratio=0.80,
        face_matched=True, is_real=True, is_deepfake=False, doc_valid=True),
     "UNDER REVIEW"),
    ("CASE 3 - High Risk (liveness fail + doc invalid)", dict(
        face_match_score=0.30, liveness_score=0.35, ocr_confidence_avg=0.40,
        deepfake_prob=0.72, doc_validation_score=0.20, field_completeness_ratio=0.30,
        face_matched=False, is_real=False, is_deepfake=True, doc_valid=False),
     "REJECTED"),
    ("SAFETY - deepfake alone must NOT reject", dict(
        face_match_score=0.70, liveness_score=0.91, ocr_confidence_avg=0.85,
        deepfake_prob=0.68, doc_validation_score=0.88, field_completeness_ratio=0.88,
        face_matched=True, is_real=True, is_deepfake=True, doc_valid=True),
     # deepfake>=60 correctly escalates to UNDER REVIEW (manual review),
     # NOT REJECTED — deepfake alone can NEVER reject per spec
     "UNDER REVIEW"),
    ("SAFETY - face mismatch alone must NOT reject", dict(
        face_match_score=0.20, liveness_score=0.92, ocr_confidence_avg=0.82,
        deepfake_prob=0.10, doc_validation_score=0.85, field_completeness_ratio=0.85,
        face_matched=False, is_real=True, is_deepfake=False, doc_valid=True),
     "UNDER REVIEW"),
    ("EDGE - borderline liveness 75, valid doc", dict(
        face_match_score=0.55, liveness_score=0.75, ocr_confidence_avg=0.70,
        deepfake_prob=0.20, doc_validation_score=0.75, field_completeness_ratio=0.70,
        face_matched=True, is_real=True, is_deepfake=False, doc_valid=True),
     "UNDER REVIEW"),
    ("EDGE - liveness fail only, doc valid (must NOT reject alone)", dict(
        face_match_score=0.65, liveness_score=0.40, ocr_confidence_avg=0.80,
        deepfake_prob=0.15, doc_validation_score=0.85, field_completeness_ratio=0.85,
        face_matched=True, is_real=False, is_deepfake=False, doc_valid=True),
     "UNDER REVIEW"),
]

print()
passed = 0
failed = 0
for name, kwargs, expected in cases:
    result = scorer.score(**kwargs)
    status = result["status"]
    conf   = result["confidence_level"]
    action = result["recommended_action"]
    risk   = result["risk_score"]
    ok = status == expected
    passed += ok
    failed += not ok
    tag = "PASS" if ok else f"FAIL (expected {expected})"
    print(f"  {'[OK]' if ok else '[!!]'}  {name}")
    print(f"       status={status}  confidence={conf}  action={action}  risk={risk:.3f}  -> {tag}")
    for bullet in result["reasoning"][:2]:
        short = bullet[:95] + "..." if len(bullet) > 95 else bullet
        print(f"       > {short}")
    print()

print("=" * 65)
print(f"  Decision Engine: {passed} passed / {failed} failed out of {len(cases)} cases")
print("=" * 65)

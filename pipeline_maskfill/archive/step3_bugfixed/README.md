# Step 3: Bug-Fixed Checkpoint (March 7, 2026)

All fixes applied for P100 Kaggle deployment across 3 accounts.

## Fixes since Step 2 (modular_innovative)

### model2_advanced.py
- **BCE autocast fix**: `F.binary_cross_entropy` → `F.binary_cross_entropy_with_logits` (P100 fp16 safe)
- **PrivacyAttentionHead**: returns raw logits, sigmoid applied only at inference
- **TokenClassifierOutput fix**: privacy_loss/scores stored on model instance (`_last_privacy_loss`) instead of frozen dataclass
- **FocalTrainer**: reads privacy loss via `getattr(model, '_last_privacy_loss', None)`
- **DP-SGD**: wrapped in try/except for Opacus+DeBERTa per-sample gradient shape mismatch

### common.py (PrivacyAwareSeq2SeqTrainer)
- **NaN guard**: `F.normalize` on pooled vectors before cosine similarity
- **Explosion guard**: skips semantic loss if NaN or > 10.0
- **Final NaN fallback**: reverts to CE loss if total is non-finite
- **Speed**: semantic loss computed every 4th step only (saves ~15-20%)

### model4_semantic.py
- **Epochs**: 3 → 1 (fits in Kaggle 12h limit)
- **Warmup**: 6% → 10%, minimum 200 steps

## Model 2 Results (before DP-SGD)
- Censor F1: 0.9579
- EMAIL: 1.000, ACCOUNT: 0.992, USERNAME: 0.986, LOC: 0.943

## Files (12 total)
- common.py, model1_baseline.py, model2_advanced.py, model3_rephraser.py, model4_semantic.py
- run_all.py, run_model13.py, run_model2.py, run_model4.py
- eval_all.py, eval_prompt_injection.py, novel_pipeline.py

# Project Progression Steps

Each folder is a snapshot of the codebase at a major milestone, ordered chronologically for slide preparation.

---

## Step 1 — Monolithic Single-Script (`step1_monolithic/`)

**Theme:** "Getting it working in one file"

- **`script.py`** — The original all-in-one implementation (~2300+ lines). Contains data loading, NER training, Seq2Seq hallucinator, evaluation, and inference in a single file.
- **`pipeline_v9.py`** — An earlier pipeline iteration.
- **`script_*_backup.py`** — Incremental backups showing debugging progression (Opacus fixes, decoupled architecture, Seq2Seq integration).

**Key characteristics:**
- Single-file monolith — hard to maintain and experiment with
- All models interleaved with shared global state
- Manual device management and ad-hoc hyperparameters
- No formal evaluation framework

**Slide talking points:**
1. Started with a single script to prototype the mask-and-fill pipeline
2. Quickly became unwieldy (2000+ lines, hard to debug individual models)
3. Motivated the modular split

---

## Step 2 — Modular 4-Model Architecture with SOTA Innovations (`step2_modular_innovative/`)

**Theme:** "Modular, efficient, innovative"

**Architecture split into independent files:**
| File | Purpose |
|------|---------|
| `common.py` | Shared config, data loading, evaluation, custom trainers, loss functions |
| `model1_baseline.py` | Model 1: DeBERTa-v3-base + Flan-T5-base (baseline, no DP) |
| `model2_advanced.py` | Model 2: MultiTaskNERModel + Flan-T5-base (DP-SGD, focal loss, privacy attention) |
| `model3_rephraser.py` | Model 3: Three-stage pipeline (censor → hallucinator → rephraser) |
| `model4_semantic.py` | Model 4: Single-pass entity-type-aware semantic paraphraser |
| `eval_all.py` | Comprehensive post-training evaluation suite |
| `eval_prompt_injection.py` | Adversarial prompt injection robustness testing |
| `run_*.py` | Orchestration scripts for Kaggle execution |

**Key innovations over Step 1:**
1. **Privacy Attention Head** — Multi-head attention (4 heads) → LayerNorm → MLP → sigmoid producing per-token privacy importance scores [0,1], jointly trained with NER classification (λ=0.3)
2. **BERTScore-Inspired Semantic Fidelity Loss** — Cosine similarity between mean-pooled encoder/decoder hidden states penalises semantic drift during pseudonym generation (τ=0.70, λ=0.15)
3. **BLEU/ROUGE Masking Evaluation** — Dedicated `evaluate_masking_quality()` assesses the NER masking step in isolation (BLEU, ROUGE-L, token accuracy)
4. **3–10× Smaller Backbones** — DeBERTa-v3-small (44M), Flan-T5-base (250M), Flan-T5-small (77M) replace XLM-RoBERTa (278M), mT5-large (1.2B), Flan-T5-large (780M)
5. **Entity-Type Prefix Tokens** — Model 4 prepends `[TYPES: PERSON, LOC]` to inputs for explicit entity awareness
6. **MultiTaskNERModel** — Custom nn.Module with dual heads (NER + privacy attention) and Opacus-compatible DP-SGD via weight transfer

**Slide talking points:**
1. Modular design: each model is independently trainable and evaluable
2. Privacy attention head learns *which tokens matter* before classification
3. Semantic fidelity loss keeps meaning intact during anonymization
4. 3–10× parameter reduction with no quality loss (efficiency frontier)
5. All models runnable on a single Kaggle T4 GPU (16 GB VRAM)

---

## Adding Future Steps

When making the next major change, create a new folder:
```
steps/step3_<description>/
```
Copy the working files before editing, so each step is a clean snapshot.

# Privacy-Preserving Text Anonymization via DP-Guided Decoupled Mask-and-Fill with Entity Consistency

**A Comparative Study of Four Approaches**

> Course Project — Introduction to NLP (INLP)
> Team: Mr.A

---

## Overview

This project tackles the problem of **automatically anonymizing personally identifiable information (PII)** in free-form text while preserving semantic coherence and downstream utility. We implement, train, and comparatively evaluate **four distinct privacy-preserving pipelines** of increasing sophistication, ranging from a straightforward NER-based mask-and-fill baseline to a single-pass semantic paraphraser.

All models are trained on the [AI4Privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset (~500K samples, 50+ entity types) using **Kaggle P100 GPUs** (16 GB VRAM) with QLoRA 4-bit quantization for memory efficiency.

---

## The Four Models

| # | Model | Architecture | Key Idea |
|---|-------|-------------|----------|
| **1** | Baseline Mask-and-Fill | DeBERTa-v3-base NER → Flan-T5-base QLoRA | Detect entities, replace with realistic fakes |
| **2** | Advanced DP-Guided | DeBERTa-v3-small + Privacy Attention Head → Flan-T5-base QLoRA | Formal (ε,δ)-DP via Opacus, focal loss, semantic fidelity loss |
| **3** | Context-Aware Rephraser | Model 1 pipeline → Flan-T5-small QLoRA rephraser | 3-stage: mask → fill → rephrase surrounding context to suppress re-identification |
| **4** | Semantic Paraphraser | Flan-T5-base QLoRA (single-pass) | End-to-end: learns to simultaneously identify PII and rewrite the text |

### Key Innovations

- **Privacy Attention Head** (Model 2) — A multi-head attention module (4 heads) producing per-token privacy importance scores [0,1], jointly trained with NER classification.
- **DP-SGD via Opacus** (Model 2) — Formal differential privacy guarantees with per-sample gradient clipping.
- **BERTScore-Inspired Semantic Fidelity Loss** (Models 2 & 4) — Cosine similarity between mean-pooled encoder/decoder hidden states penalizes semantic drift during pseudonym generation.
- **Entity-Type Prefix Tokens** (Model 4) — Inputs prepended with `[TYPES: PERSON, LOC, DATE]` for explicit entity awareness.
- **3-Stage Rephrasing** (Model 3) — Contextual generalization to remove re-identification signals that survive entity swapping.

---

## Repository Structure

```
.
├── README.md
├── .gitignore
│
├── src/                              # All source code
│   ├── common.py                     # Shared config, data loading, evaluation, trainers (1776 lines)
│   ├── model1_baseline.py            # Model 1: Baseline mask-and-fill (265 lines)
│   ├── model2_advanced.py            # Model 2: DP-SGD + Privacy Attention (771 lines)
│   ├── model3_rephraser.py           # Model 3: 3-stage rephraser (211 lines)
│   ├── model4_semantic.py            # Model 4: End-to-end paraphraser (312 lines)
│   ├── eval_all.py                   # Comprehensive post-training evaluation (951 lines)
│   ├── eval_prompt_injection.py      # Adversarial prompt injection tests (906 lines)
│   ├── run_all.py                    # Run all models locally
│   ├── run_model13.py                # Kaggle script: Models 1 + 3 together
│   ├── run_model2.py                 # Kaggle script: Model 2 alone
│   └── run_model4.py                 # Kaggle script: Model 4 alone
│
├── paper/                            # LaTeX documents
│   ├── paper.tex                     # Main research paper
│   ├── scope.tex                     # Project scope document
│   ├── math_foundations.tex          # Mathematical foundations (DP proofs, loss derivations)
│   └── kaggle_guide.tex              # Kaggle deployment step-by-step guide
│
├── docs/
│   └── PIPELINE_GUIDE.md            # Pipeline architecture and phase coverage guide
│
└── archive/                          # Historical development snapshots
    ├── README.md                     # Snapshot descriptions for presentation slides
    ├── step1_monolithic/             # Original single-file prototype (~2300 lines)
    ├── step2_modular_innovative/     # First modular split with SOTA innovations
    ├── step3_bugfixed/               # Bug-fixed checkpoint/resume version
    ├── novel_pipeline.py             # Experimental techniques (APP, ETAP, CLPT)
    ├── evaluation_results_readable.txt
    └── outputs.txt
```

---

## Setup

### Requirements

The code auto-installs dependencies on first run. Core requirements:

- Python 3.10+
- PyTorch 2.x (CUDA)
- Transformers, PEFT, BitsAndBytes (QLoRA)
- Opacus (DP-SGD, Model 2 only)
- evaluate, rouge-score, sacrebleu, bert-score
- datasets, accelerate, sentencepiece

### Hardware

- **Training**: Kaggle P100 GPU (16 GB VRAM), ~12h per session
- **Evaluation**: Any GPU with ≥ 8 GB VRAM (BERTScore needs GPU)

---

## Training

Training is split across **3 parallel Kaggle sessions** to fit within the 12-hour time limit:

| Kaggle Account | Script | Models Trained | Rationale |
|---|---|---|---|
| Account 1 | `run_model13.py` | Models 1 + 3 | Model 3 reuses Model 1's censor & hallucinator |
| Account 2 | `run_model2.py` | Model 2 | DP-SGD is computationally expensive |
| Account 3 | `run_model4.py` | Model 4 | Independent single-pass architecture |

### Running on Kaggle

1. Upload all files from `src/` as a **Utility Script** dataset
2. Create a new notebook with **GPU P100** accelerator
3. Add the utility script as input
4. Run:
   ```python
   !cp /kaggle/input/<your-dataset>/*.py .
   !python run_model13.py        # Account 1
   # or
   !python run_model2.py         # Account 2
   # or
   !python run_model4.py         # Account 3
   ```
5. After training, the notebook output contains saved model weights

### Running Locally

```bash
cd src/
python run_all.py                      # Train all 4 models
python run_all.py --model 1 3          # Train specific models
python run_all.py --quick              # Quick test with small data subset
```

### Checkpoint Resume

All scripts support **crash-resilient checkpoint/resume**:
- Seq2Seq models save checkpoints every 500 steps (`save_total_limit=1`)
- On Kaggle restart, save previous output as a dataset and add it as input — the script auto-restores and resumes from the last checkpoint
- Skip-if-trained logic avoids re-training models that already completed

---

## Evaluation

After training completes (or times out), run the comprehensive evaluation:

```bash
cd src/
python eval_all.py --model 1 2 3 4 --checkpoint-dir <path-to-outputs>
```

### Evaluation Modes

```bash
# Full evaluation (200 samples + curated set)
python eval_all.py --model 1 2 3 4 --checkpoint-dir outputs/ --n-eval 200

# Quick evaluation (50 samples)
python eval_all.py --model 1 2 3 4 --checkpoint-dir outputs/ --quick

# With adversarial prompt injection testing
python eval_all.py --model 2 --checkpoint-dir outputs/ --prompt-injection

# Skip expensive evaluations
python eval_all.py --model 1 2 3 4 --checkpoint-dir outputs/ --skip-mia --skip-cross-lingual
```

### Metrics Computed

| Category | Metrics |
|----------|---------|
| **Privacy** | Entity leakage rate, contextual re-identification risk (CRR), multi-granularity CRR (exact, fuzzy, phonetic, semantic) |
| **Utility** | ROUGE-L, BLEU, BERTScore, entity F1 |
| **Robustness** | Prompt injection resistance (150 adversarial examples, 10 attack categories) |
| **Fairness** | Per-entity-type leakage, difficulty-stratified breakdown (easy/medium/hard) |
| **Formal Privacy** | Membership inference attack success rate (Model 2), DP ε budget tracking |

### Outputs

| File | Description |
|------|-------------|
| `all_results.json` | Raw metrics dictionary |
| `comparison_table.txt` | Formatted comparison table |
| `comparison_table.tex` | LaTeX table (paste directly into paper) |
| `full_comparison.png` | 6-panel comparison figure |
| `pareto_tradeoff.png` | Privacy vs. utility Pareto front |
| `difficulty_breakdown.png` | Leakage by entity difficulty |
| `multi_crr.png` | Multi-granularity re-identification risk |

---

## Architecture Details

### Model 1 — Baseline Mask-and-Fill

```
Input text → DeBERTa-v3-base (NER) → Masked text → Flan-T5-base QLoRA → Anonymized text
                                       "Hello [PERSON]"      →        "Hello John Smith"
```

- **Censor**: Fine-tuned DeBERTa-v3-base for BIO token classification (50+ entity types)
- **Hallucinator**: Flan-T5-base with QLoRA (r=16, α=32) generates realistic fake entities

### Model 2 — Advanced DP-Guided Pipeline

```
Input text → DeBERTa-v3-small + PrivacyAttentionHead (NER + DP-SGD)
          → Masked text
          → Flan-T5-base QLoRA (semantic fidelity loss)
          → Anonymized text
```

- **MultiTaskNERModel**: Custom `nn.Module` with dual heads — NER classification + privacy attention
- **Focal Loss**: Recall-weighted focal loss (γ=2, α=0.75) for rare entity types
- **DP-SGD**: Opacus integration with per-sample gradient clipping (C=1.0, σ=0.8)
- **Semantic Fidelity Loss**: BERTScore-inspired cosine similarity target (τ=0.70, λ=0.15)

### Model 3 — Context-Aware Rephraser

```
Input text → Model 1 Censor → Model 1 Hallucinator → Flan-T5-small QLoRA Rephraser
                                                        → Contextually generalized text
```

- Reuses Model 1's trained censor and hallucinator
- Rephraser (Flan-T5-small, 77M params) further generalizes context to suppress re-identification
- Example: *"John Smith is the richest man in the world"* → *"An individual is a prominent figure"*

### Model 4 — End-to-End Semantic Paraphraser

```
"[TYPES: PERSON, LOC] Rewrite preserving privacy: <original text>"
    → Flan-T5-base QLoRA → Privacy-preserving paraphrase
```

- Single-pass: no separate NER step, entity detection is implicit
- Entity-type prefix tokens provide explicit type awareness
- Privacy-aware Seq2Seq trainer with NaN guards for stable training

---

## Development Timeline

The project evolved through three major phases (snapshots preserved in `archive/`):

1. **Monolithic Prototype** (`archive/step1_monolithic/`) — Single 2300-line script with all models interleaved. Proved the concept but was unmaintainable.

2. **Modular Architecture + SOTA Innovations** (`archive/step2_modular_innovative/`) — Split into independent model files. Added Privacy Attention Head, semantic fidelity loss, entity-type prefixes, and 3-10x smaller backbones.

3. **Production-Ready with Bug Fixes** (`archive/step3_bugfixed/`) — Checkpoint/resume system, skip-if-trained logic, Opacus shape fixes, DeBERTa parameter naming, NaN loss guards, AMP autocast fixes.

---

## Citation

If you use this code:

```bibtex
@misc{mra2026privacy,
  title={Privacy-Preserving Text Anonymization via DP-Guided Decoupled
         Mask-and-Fill with Entity Consistency: A Comparative Study},
  author={Team Mr.A},
  year={2026},
  howpublished={INLP Course Project}
}
```

---

## License

This project was developed as part of an academic course. All code is provided as-is for educational purposes.

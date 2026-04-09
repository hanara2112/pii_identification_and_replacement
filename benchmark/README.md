---
license: mit
task_categories:
  - text-generation
  - token-classification
language:
  - en
tags:
  - privacy
  - anonymization
  - pii
size_categories:
  - 100K<n<1M
pretty_name: SAHA-AL
---

# SAHA-AL: PII Anonymization Benchmark

**SAHA-AL** is a benchmark suite for training and evaluating text anonymization systems. It evaluates anonymization as a **system under attack** — incorporating adversarial, contextual, and generative privacy risks alongside a formalized privacy-utility tradeoff framework.

## 1. Tasks

| Task | Input | Output |
|------|-------|--------|
| **Task 1: PII Detection** | `original_text` | `detected_entities: [{start, end, type}]` |
| **Task 2: Text Anonymization** | `original_text` + `entities` | `anonymized_text` |
| **Task 3: Privacy Risk** | Evaluator-computed over Task 2 | CRR-3, ERA, LRR, UAC |

## 2. Metrics

### Detection (Task 1)
- Span P/R/F1 (exact, partial, type-aware)
- Per-type recall

### Anonymization Quality (Task 2)
| Metric | Definition | Direction |
|--------|-----------|-----------|
| **ELR** | Entity string found in prediction | ↓ lower |
| **Token Recall** | Entity-span tokens absent from prediction | ↑ higher |
| **OMR** | Non-entity tokens altered | ↓ lower |
| **FPR** | Structured replacements match format regex | ↑ higher |
| **BERTScore F1** | Semantic similarity (distilbert-base-uncased) | ↑ higher |

### Privacy Risk (Task 3)
| Metric | Definition | Direction |
|--------|-----------|-----------|
| **CRR-3** | Capitalized 3-gram survival rate | ↓ lower |
| **ERA** | Entity Recovery Attack (retrieval adversary) | ↓ lower |
| **LRR** | LLM Re-identification Rate (generative adversary) | ↓ lower |
| **UAC** | Unique Attribute Combination rate (k-anonymity proxy) | ↓ lower |

### Privacy-Utility Frontier
- **PUS(λ) = λ · Privacy + (1-λ) · Utility** — parameterized tradeoff score

## 3. Dataset

- **Train:** ~113k records (28.8k gold + ~84k augmented)
- **Validation:** ~3.6k records (gold only)
- **Test:** 3,600 records, 9,271 gold entities (frozen)

Entity types: 20 fine-grained PII types (FULLNAME, FIRST_NAME, LAST_NAME, EMAIL, PHONE, SSN, etc.)

## 4. Repository Structure

```
benchmark/
├── README.md
├── revised_plan.md              # Full benchmark design document
│
├── eval/                        # Evaluation modules
│   ├── utils.py                 # Shared: span matching, format regexes, I/O
│   ├── eval_detection.py        # Task 1: span P/R/F1
│   ├── eval_anonymization.py    # Task 2: ELR, TokRecall, OMR, FPR, BERTScore, NLI
│   ├── eval_privacy.py          # Task 3: CRR-3, ERA, LRR, UAC
│   └── bootstrap.py             # 95% bootstrap confidence intervals
│
├── baselines/                   # Baseline systems
│   ├── regex_faker_baseline.py  # Regex/spaCy/Presidio (+ --save-spans)
│   ├── seq2seq_inference.py     # 6 fine-tuned seq2seq models
│   ├── bert_ner_baseline.py     # BERT-base token classifier
│   ├── llm_baseline.py          # GPT-4o-mini zero-shot
│   ├── hybrid_baseline.py       # spaCy detect + GPT-4o-mini replace
│   └── maskfill_inference.py    # pipeline_maskfill cross-dataset baselines
│
├── analysis/                    # Analysis scripts
│   ├── dataset_stats.py         # Dataset statistics
│   ├── pareto_frontier.py       # Privacy-utility frontier + PUS sweep
│   ├── failure_taxonomy.py      # 5-category error classification
│   └── tab_transfer.py          # Cross-dataset evaluation on TAB
│
├── scripts/                     # Utility scripts
│   ├── prepare_dataset.py       # Build train/val/test splits
│   ├── push_to_hf.py            # Push dataset to HuggingFace
│   └── run_all.sh               # Run full evaluation pipeline
│
├── data/                        # Dataset splits (JSONL)
├── predictions/                 # Model prediction outputs
└── results/                     # Evaluation result JSONs
```

## 5. Quick Start

```bash
# Install dependencies
pip install faker spacy bert_score sentence-transformers transformers openai
python -m spacy download en_core_web_lg

# Prepare dataset (from raw data)
python -m scripts.prepare_dataset

# Run a baseline
python -m baselines.regex_faker_baseline --gold data/test.jsonl --mode spacy --save-spans

# Evaluate Task 2
python -m eval.eval_anonymization --gold data/test.jsonl --pred predictions/spacy_predictions.jsonl --print-types

# Evaluate Task 1
python -m eval.eval_detection --gold data/test.jsonl --pred predictions/spacy_spans.jsonl

# Evaluate Task 3 (privacy)
python -m eval.eval_privacy --gold data/test.jsonl --pred predictions/spacy_predictions.jsonl --train data/train.jsonl --skip-lrr

# Run everything
bash scripts/run_all.sh
```

## 6. Leaderboard

*Measured on the frozen test split (3,600 records, 9,271 gold entities).*

### Task 2: Text Anonymization

| Model | ELR ↓ | BERTScore F1 ↑ | CRR-3 ↓ |
|-------|-------|----------------|---------|
| Regex+Faker | 83.49% | 98.13 | 92.53 |
| spaCy+Faker | 26.70% | 91.84 | 40.62 |
| Presidio | 33.86% | 90.02 | 50.33 |
| BART-base + PII | **0.93%** | **92.74** | 34.62 |
| Flan-T5-small + PII | 0.99% | 92.47 | 34.94 |
| DistilBART + PII | 1.23% | 86.34 | 34.69 |
| T5-small + PII | 1.54% | 92.59 | 35.27 |
| T5-efficient-tiny + PII | 4.14% | 92.57 | 37.06 |

## 7. Submission Format

```json
{"id": "sample_00000", "anonymized_text": "David lives in Chicago."}
```

For Task 1 submissions:
```json
{"id": "sample_00000", "detected_entities": [{"start": 0, "end": 5, "type": "FULLNAME"}]}
```

## 8. Benchmark Protocol

1. **No test set contamination** — do not train or tune on test.jsonl
2. **Report compute** — GPU type, training time, inference time
3. **Standard evaluation** — use the provided eval scripts
4. **Disclose LLM pre-training data** for LLM-based systems

## Limitations

- Source texts use synthetic/LLM-generated prompts
- English-only
- IAA was entity-level F1=0.83 (not Cohen's kappa as previously reported)
- Augmentation is ~3.3× actual expansion (not 12× as documented in pipeline README)

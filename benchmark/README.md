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
  - ner
  - text-rewriting
  - benchmark
size_categories:
  - 100K<n<1M
pretty_name: "SAHA-AL: PII Anonymization Benchmark"
dataset_info:
  - config_name: default
    features:
      - name: id
        dtype: string
      - name: original_text
        dtype: string
      - name: anonymized_text
        dtype: string
      - name: entities
        list:
          - name: text
            dtype: string
          - name: type
            dtype: string
          - name: start
            dtype: int32
          - name: end
            dtype: int32
    splits:
      - name: train
        num_examples: 113133
      - name: validation
        num_examples: 3600
      - name: test
        num_examples: 3600
---

# SAHA-AL: PII Anonymization Benchmark

**SAHA-AL** is a benchmark for training and evaluating text anonymization systems. It goes beyond detection accuracy by evaluating anonymization as a **system under attack** — measuring adversarial re-identification risk, contextual privacy leakage, and a formalized privacy-utility tradeoff.

## Key Features

- **3 evaluation tasks:** PII detection, text anonymization quality, and adversarial privacy risk
- **11 metrics** spanning leakage, utility, format preservation, and multi-vector attack resistance
- **8 baseline systems** evaluated: 3 rule-based (Regex, spaCy, Presidio) and 5 fine-tuned seq2seq models
- **Privacy-Utility Score (PUS):** A parameterized framework for navigating the privacy-utility tradeoff
- **Failure taxonomy:** Diagnostic error classification (full leak, boundary error, format break, context retention)

## Dataset


| Split      | Records | Entities | Avg Entities/Record | PII Density |
| ---------- | ------- | -------- | ------------------- | ----------- |
| Train      | 113,133 | 286,542  | 2.53                | 14.9%       |
| Validation | 3,600   | 9,211    | 2.56                | 15.0%       |
| Test       | 3,600   | 9,271    | 2.58                | 15.2%       |


**Entity types (12):** FULLNAME, UNKNOWN, PHONE, EMAIL, ADDRESS, DATE, FIRST_NAME, ID_NUMBER, ZIPCODE, CREDIT_CARD, SSN, ORGANIZATION

Test set is dominated by FULLNAME (57.5%) and UNKNOWN (19.6%).

### Data Format

Each record is a JSON object:

```json
{
  "id": "sample_00123",
  "original_text": "Please contact John Smith at john.smith@email.com for details.",
  "anonymized_text": "Please contact Michael Jones at m.jones@mail.org for details.",
  "entities": [
    {"text": "John Smith", "type": "FULLNAME", "start": 15, "end": 25},
    {"text": "john.smith@email.com", "type": "EMAIL", "start": 29, "end": 49}
  ]
}
```

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("huggingbahl21/saha-al")

# Access splits
train = dataset["train"]       # 113,133 records
val = dataset["validation"]    # 3,600 records
test = dataset["test"]         # 3,600 records (frozen)

# Example record
record = test[0]
print(record["original_text"])
print(record["entities"])
```

## Tasks

### Task 1: PII Detection

**Input:** `original_text` | **Output:** `detected_entities: [{start, end, type}]`

Evaluate span-level precision, recall, and F1 in three modes: exact match, partial (IoU > 0.5), and type-aware.

### Task 2: Text Anonymization

**Input:** `original_text` | **Output:** `anonymized_text`

Replace all PII with realistic synthetic alternatives while preserving non-PII text and document structure.

### Task 3: Privacy Risk Assessment

Evaluator-computed over Task 2 outputs. Simulates adversarial attacks against anonymized text to measure residual re-identification risk.

## Metrics

### Anonymization Quality (Task 2)


| Metric           | What it measures                                                         | Direction          |
| ---------------- | ------------------------------------------------------------------------ | ------------------ |
| **ELR**          | Entity Leakage Rate — fraction of entities found verbatim in output      | ↓ lower is better  |
| **Token Recall** | Fraction of entity tokens absent from output                             | ↑ higher is better |
| **OMR**          | Over-Masking Rate — non-entity tokens unnecessarily altered              | ↓ lower is better  |
| **FPR**          | Format Preservation Rate — structured replacements match expected format | ↑ higher is better |
| **BERTScore F1** | Semantic similarity between original and anonymized text                 | ↑ higher is better |


### Privacy Risk (Task 3)


| Metric    | Attack model                                                                | Direction         |
| --------- | --------------------------------------------------------------------------- | ----------------- |
| **CRR-3** | Capitalized 3-gram survival rate (statistical)                              | ↓ lower is better |
| **ERA@k** | Entity Recovery Attack — retrieval adversary with candidate pool (SBERT)    | ↓ lower is better |
| **LRR**   | LLM Re-identification Rate — generative adversary guesses original entities | ↓ lower is better |
| **UAC**   | Unique Attribute Combination rate — k-anonymity proxy                       | ↓ lower is better |


### Privacy-Utility Tradeoff

**PUS(lambda) = lambda * (1 - ELR/100) + (1 - lambda) * (BERTScore/100)**

A single parameterized score where lambda controls the privacy-utility balance (lambda=0.5 gives equal weight).

## Leaderboard

*Evaluated on the frozen test split (3,600 records, 9,271 entities).*

### Task 2: Anonymization Quality


| System              | ELR ↓     | Token Recall ↑ | BERTScore F1 ↑ | PUS (lambda=0.5) |
| ------------------- | --------- | -------------- | -------------- | ---------------- |
| **BART-base + PII** | **0.93%** | **95.50%**     | 92.74          | **0.959**        |
| Flan-T5-small + PII | 0.99%     | 95.33%         | 92.47          | 0.957            |
| T5-small + PII      | 1.54%     | 94.15%         | 92.59          | 0.955            |
| T5-eff-tiny + PII   | 4.14%     | 90.42%         | 92.57          | 0.942            |
| DistilBART + PII    | 1.23%     | 94.41%         | 86.34          | 0.926            |
| spaCy+Faker         | 26.44%    | 74.59%         | 91.86          | 0.827            |
| Presidio            | 33.77%    | 68.18%         | 90.04          | 0.781            |
| Regex+Faker         | 83.39%    | 23.05%         | 98.15          | 0.574            |


### Task 3: Privacy Risk


| System              | CRR-3 ↓    | ERA@1 ↓   | ERA@5 ↓   | LRR Exact ↓ | UAC ↓     |
| ------------------- | ---------- | --------- | --------- | ----------- | --------- |
| **BART-base + PII** | **34.62%** | **1.90%** | **4.84%** | **0.13%**   | **0.33%** |
| Flan-T5-small + PII | 34.94%     | 1.67%     | 5.08%     | —           | 0.22%     |
| spaCy+Faker         | 40.62%     | 9.36%     | 14.27%    | —           | 1.58%     |
| Presidio            | 50.33%     | 20.46%    | 26.09%    | 2.12%       | 1.78%     |


### Task 1: PII Detection


| System      | Exact F1 | Partial F1 | Type-aware F1 |
| ----------- | -------- | ---------- | ------------- |
| spaCy+Faker | 56.30    | 69.38      | 18.18         |
| Presidio    | 48.57    | 60.11      | 4.97          |
| Regex+Faker | 25.86    | 27.99      | 24.39         |


## Key Findings

1. **Seq2seq models achieve <1% entity leakage** while preserving >92% semantic similarity, dramatically outperforming rule-based systems (26–83% leakage).
2. **Structured PII (email, phone, SSN, credit card) is fully solved** — 0% leakage across all seq2seq models. **Names remain the open challenge**, accounting for 83% of residual leakage.
3. **Retrieval attacks are more effective than LLM attacks.** ERA@1 recovers 1.9% of entities vs LRR's 0.13%, indicating replacement-based anonymization is more vulnerable to database-level adversaries than generative inference.
4. **BART-base dominates the Pareto frontier** with PUS=0.959, achieving near-maximum privacy (99.1%) and high utility (92.7%).

## Evaluation Quick Start

```bash
pip install faker spacy bert_score sentence-transformers transformers
python -m spacy download en_core_web_lg
```

### Run a baseline

```bash
# Rule-based (regex / spacy / presidio)
python -m baselines.regex_faker_baseline --gold data/test.jsonl --mode spacy --save-spans

# Seq2seq (downloads checkpoint from HuggingFace)
python -m baselines.seq2seq_inference --gold data/test.jsonl --model-name bart-base-pii --output predictions/predictions_bart-base-pii.jsonl
```

### Evaluate

```bash
# Task 2: Anonymization quality
python -m eval.eval_anonymization \
    --gold data/test.jsonl \
    --pred predictions/predictions_bart-base-pii.jsonl \
    --print-types

# Task 1: Detection (requires --save-spans from baseline)
python -m eval.eval_detection \
    --gold data/test.jsonl \
    --pred predictions/spacy_spans.jsonl

# Task 3: Privacy risk
python -m eval.eval_privacy \
    --gold data/test.jsonl \
    --pred predictions/predictions_bart-base-pii.jsonl \
    --train data/train.jsonl \
    --skip-lrr

# Bootstrap confidence intervals
python -m eval.bootstrap \
    --gold data/test.jsonl \
    --pred predictions/predictions_bart-base-pii.jsonl \
    --metrics elr token_recall crr3

# Failure taxonomy
python -m analysis.failure_taxonomy \
    --gold data/test.jsonl \
    --pred predictions/predictions_bart-base-pii.jsonl

# Pareto frontier analysis
python -m analysis.pareto_frontier \
    --results Results/all_eval_results.json \
    --plot figures/pareto_frontier.png

# Run the full pipeline
bash scripts/run_all.sh
```

## Submission Format

**Task 2** (one JSONL line per record):

```json
{"id": "sample_00000", "anonymized_text": "Please contact Michael Jones at m.jones@mail.org."}
```

**Task 1** (one JSONL line per record):

```json
{"id": "sample_00000", "detected_entities": [{"start": 15, "end": 29, "type": "FULLNAME"}]}
```

## Repository Structure

```
benchmark/
├── eval/                          # Evaluation modules
│   ├── utils.py                   #   Shared: span matching, format regexes, I/O
│   ├── eval_detection.py          #   Task 1: span P/R/F1
│   ├── eval_anonymization.py      #   Task 2: ELR, Token Recall, OMR, FPR, BERTScore
│   ├── eval_privacy.py            #   Task 3: CRR-3, ERA, LRR, UAC
│   └── bootstrap.py               #   95% bootstrap confidence intervals
│
├── baselines/                     # Baseline systems
│   ├── regex_faker_baseline.py    #   Regex / spaCy / Presidio pipelines
│   ├── seq2seq_inference.py       #   BART / T5 / Flan-T5 / DistilBART inference
│   ├── bert_ner_baseline.py       #   BERT-base token classifier (NER)
│   ├── llm_baseline.py            #   GPT-4o-mini / local LLM zero-shot
│   └── hybrid_baseline.py         #   spaCy detect + LLM replace
│
├── analysis/                      # Analysis & visualization
│   ├── dataset_stats.py           #   Dataset statistics
│   ├── pareto_frontier.py         #   Privacy-utility Pareto + PUS sweep
│   ├── failure_taxonomy.py        #   5-category error classification
│   └── plot_results.py            #   Publication figures
│
├── scripts/                       # Utilities
│   ├── prepare_dataset.py         #   Build train/val/test from raw data
│   ├── push_to_hf.py             #   Push to HuggingFace Hub
│   └── run_all.sh                 #   Full evaluation pipeline
│
├── data/                          # Dataset splits (JSONL)
├── predictions/                   # Model outputs
├── Results/                       # Evaluation JSONs
├── figures/                       # Plots
│
├── RESULTS.md                     # Key results for reporting
├── RESULTS_FULL.md                # Exhaustive results reference
├── METHODOLOGY.md                 # Full methodology & theory guide
└── RESULTS_INTERPRETATION.md      # In-depth results discussion
```

## Trained Model Checkpoints

All seq2seq checkpoints are hosted on HuggingFace:

**Repository:** `[JALAPENO11/pii_identification_and_anonymisations](https://huggingface.co/JALAPENO11/pii_identification_and_anonymisations)`


| Model               | Base Architecture              | Parameters | Checkpoint Path                                              |
| ------------------- | ------------------------------ | ---------- | ------------------------------------------------------------ |
| BART-base + PII     | `facebook/bart-base`           | 139M       | `checkpoints_pii_aware_loss/bart-base/best_model.pt`         |
| Flan-T5-small + PII | `google/flan-t5-small`         | 77M        | `checkpoints_pii_aware_loss/flan-t5-small/best_model.pt`     |
| T5-small + PII      | `t5-small`                     | 60M        | `checkpoints_pii_aware_loss/t5-small/best_model.pt`          |
| DistilBART + PII    | `sshleifer/distilbart-cnn-6-6` | 230M       | `checkpoints_pii_aware_loss/distilbart/best_model.pt`        |
| T5-eff-tiny + PII   | `google/t5-efficient-tiny`     | 16M        | `checkpoints_pii_aware_loss/t5-efficient-tiny/best_model.pt` |


## Benchmark Protocol

1. **No test set contamination** — do not train or tune on `test.jsonl`
2. **Use the provided evaluation scripts** for fair comparison
3. **Report compute** — GPU type, training time, inference time
4. **Disclose pre-training data** for LLM-based systems

## Citation

If you use SAHA-AL in your research, please cite:

```bibtex
@misc{saha-al-2026,
  title={SAHA-AL: A Multi-Task Benchmark for PII Anonymization with Adversarial Privacy Evaluation},
  author={Mr. A},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/huggingbahl21/saha-al}},
}
```

## Limitations

- **Synthetic data:** Source texts are generated via Faker-based templates. Results may not fully generalize to real-world PII distributions.
- **English only:** All data, models, and evaluation metrics are English-specific.
- **Entity type ambiguity:** ~19.6% of test entities are typed as UNKNOWN due to heuristic type inference limitations.
- **Train/test entity overlap:** 43.9% of test entity strings appear in training data, which may inflate seq2seq model performance.
- **IAA:** Inter-annotator agreement was entity-level F1=0.83.

## License

This benchmark is released under the [MIT License](https://opensource.org/licenses/MIT).
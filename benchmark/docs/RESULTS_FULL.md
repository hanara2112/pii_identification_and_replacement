# SAHA-AL Benchmark — Complete Results Reference

> Exhaustive record of all evaluation results. For report-ready summaries, see `RESULTS.md`.

---

## 1. Dataset Statistics

### Split Summary

| Split | Records | Entities | Avg Words/Record | Avg Ents/Record | PII Density | Entity Types | Unique Strings |
|-------|---------|----------|-------------------|-----------------|-------------|-------------|----------------|
| Train | 113,133 | 286,542 | 17.0 | 2.53 | 0.149 | 12 | 183,720 |
| Validation | 3,600 | 9,211 | 17.0 | 2.56 | 0.150 | 11 | 8,276 |
| Test | 3,600 | 9,271 | 17.0 | 2.58 | 0.152 | 12 | 8,344 |

**Train/Test entity string overlap:** 43.9% (3,661 / 8,344 unique test strings)

### Entity Type Distribution (Test Set — Dataset Labels)

| Type | Count | % |
|------|-------|---|
| FULLNAME | 5,332 | 57.5% |
| UNKNOWN | 1,813 | 19.6% |
| PHONE | 714 | 7.7% |
| EMAIL | 373 | 4.0% |
| ADDRESS | 366 | 3.9% |
| DATE | 270 | 2.9% |
| FIRST_NAME | 228 | 2.5% |
| ID_NUMBER | 81 | 0.9% |
| ZIPCODE | 47 | 0.5% |
| CREDIT_CARD | 33 | 0.4% |
| SSN | 13 | 0.1% |
| ORGANIZATION | 1 | 0.0% |

### Entity Type Distribution (Evaluation Mapping)

The evaluation scripts infer entity types from text heuristics, producing a different mapping than the dataset labels above. The aggregate ELR is unaffected (text-based matching), but per-type breakdowns use this mapping:

| Type | Count |
|------|-------|
| FULLNAME | 4,177 |
| UNKNOWN | 1,081 |
| ORGANIZATION | 1,062 |
| LOCATION | 753 |
| PHONE | 714 |
| DATE | 599 |
| EMAIL | 373 |
| TIME | 337 |
| ID_NUMBER | 81 |
| ZIPCODE | 47 |
| CREDIT_CARD | 33 |
| SSN | 13 |
| ADDRESS | 1 |

### Train Split Type Distribution

| Type | Count |
|------|-------|
| FULLNAME | 164,338 |
| UNKNOWN | 55,110 |
| PHONE | 21,687 |
| EMAIL | 12,615 |
| ADDRESS | 11,396 |
| FIRST_NAME | 7,980 |
| DATE | 7,908 |
| ID_NUMBER | 2,080 |
| ZIPCODE | 1,493 |
| CREDIT_CARD | 1,446 |
| SSN | 439 |
| ORGANIZATION | 50 |

---

## 2. Task 1: PII Detection (Span-Level P/R/F1)

### Regex+Faker

| Mode | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| Exact | 92.32 | 15.04 | 25.86 | 1,394 | 116 | 7,877 |
| Partial | 99.93 | 16.28 | 27.99 | 1,509 | 1 | 7,762 |
| Type-aware | 87.09 | 14.18 | 24.39 | 1,315 | 195 | 7,956 |

### spaCy+Faker

| Mode | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| Exact | 69.02 | 47.55 | 56.30 | 4,408 | 1,979 | 4,863 |
| Partial | 85.05 | 58.59 | 69.38 | 5,432 | 955 | 3,839 |
| Type-aware | 22.28 | 15.35 | 18.18 | 1,423 | 4,964 | 7,848 |

### Presidio

| Mode | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| Exact | 56.34 | 42.68 | 48.57 | 3,957 | 3,066 | 5,314 |
| Partial | 69.73 | 52.82 | 60.11 | 4,897 | 2,126 | 4,374 |
| Type-aware | 5.77 | 4.37 | 4.97 | 405 | 6,618 | 8,866 |

### Per-Type Recall (Exact Match)

| Type | Regex | spaCy | Presidio |
|------|-------|-------|----------|
| CREDIT_CARD | 100.00% | 100.00% | 100.00% |
| EMAIL | 100.00% | 99.73% | 100.00% |
| SSN | 100.00% | 100.00% | 92.31% |
| ID_NUMBER | 100.00% | 98.77% | 100.00% |
| PHONE | 95.10% | 93.98% | 58.68% |
| DATE | 31.72% | 81.64% | 83.47% |
| TIME | 0.30% | 0.30% | 54.30% |
| LOCATION | 0.00% | 64.28% | 61.49% |
| ORGANIZATION | 0.00% | 50.56% | 13.75% |
| FULLNAME | 0.00% | 38.54% | 37.56% |
| ZIPCODE | 0.00% | 44.68% | 44.68% |
| UNKNOWN | 2.22% | 8.97% | 14.52% |
| ADDRESS | 0.00% | 0.00% | 0.00% |

### Per-Type Detail — spaCy

| Type | TP | Total | Recall |
|------|-----|-------|--------|
| ADDRESS | 0 | 1 | 0.00% |
| CREDIT_CARD | 33 | 33 | 100.00% |
| DATE | 489 | 599 | 81.64% |
| EMAIL | 372 | 373 | 99.73% |
| FULLNAME | 1,610 | 4,177 | 38.54% |
| ID_NUMBER | 80 | 81 | 98.77% |
| LOCATION | 484 | 753 | 64.28% |
| ORGANIZATION | 537 | 1,062 | 50.56% |
| PHONE | 671 | 714 | 93.98% |
| SSN | 13 | 13 | 100.00% |
| TIME | 1 | 337 | 0.30% |
| UNKNOWN | 97 | 1,081 | 8.97% |
| ZIPCODE | 21 | 47 | 44.68% |

### Per-Type Detail — Presidio

| Type | TP | Total | Recall |
|------|-----|-------|--------|
| ADDRESS | 0 | 1 | 0.00% |
| CREDIT_CARD | 33 | 33 | 100.00% |
| DATE | 500 | 599 | 83.47% |
| EMAIL | 373 | 373 | 100.00% |
| FULLNAME | 1,569 | 4,177 | 37.56% |
| ID_NUMBER | 81 | 81 | 100.00% |
| LOCATION | 463 | 753 | 61.49% |
| ORGANIZATION | 146 | 1,062 | 13.75% |
| PHONE | 419 | 714 | 58.68% |
| SSN | 12 | 13 | 92.31% |
| TIME | 183 | 337 | 54.30% |
| UNKNOWN | 157 | 1,081 | 14.52% |
| ZIPCODE | 21 | 47 | 44.68% |

### Per-Type Detail — Regex

| Type | TP | Total | Recall |
|------|-----|-------|--------|
| ADDRESS | 0 | 1 | 0.00% |
| CREDIT_CARD | 33 | 33 | 100.00% |
| DATE | 190 | 599 | 31.72% |
| EMAIL | 373 | 373 | 100.00% |
| FULLNAME | 0 | 4,177 | 0.00% |
| ID_NUMBER | 81 | 81 | 100.00% |
| LOCATION | 0 | 753 | 0.00% |
| ORGANIZATION | 0 | 1,062 | 0.00% |
| PHONE | 679 | 714 | 95.10% |
| SSN | 13 | 13 | 100.00% |
| TIME | 1 | 337 | 0.30% |
| UNKNOWN | 24 | 1,081 | 2.22% |
| ZIPCODE | 0 | 47 | 0.00% |

---

## 3. Task 2: Text Anonymization Quality

### Main Comparison

| System | ELR ↓ | Leaked | Token Recall ↑ | OMR ↓ | FPR ↑ | BERTScore F1 ↑ |
|--------|-------|--------|----------------|-------|-------|----------------|
| **BART-base** | **0.93%** | 86 | **95.50%** | **0.13%** | 0.84% | 92.74 |
| Flan-T5-small | 0.99% | 92 | 95.33% | 0.22% | 1.12% | 92.47 |
| DistilBART | 1.23% | 114 | 94.41% | 0.15% | 1.85% | 86.34 |
| T5-small | 1.54% | 143 | 94.15% | 0.19% | 0.28% | 92.59 |
| T5-eff-tiny | 4.14% | 384 | 90.42% | 0.49% | 0.84% | 92.57 |
| spaCy+Faker | 26.44% | 2,451 | 74.59% | 2.47% | 3.20% | 91.86 |
| Presidio | 33.77% | 3,131 | 68.18% | 1.18% | **12.37%** | 90.04 |
| Regex+Faker | 83.39% | 7,731 | 23.05% | 0.00% | 24.17% | **98.15** |

### CRR-3 by System

| System | CRR-3 |
|--------|-------|
| BART-base | 34.62% |
| DistilBART | 34.69% |
| Flan-T5-small | 34.94% |
| T5-small | 35.27% |
| T5-eff-tiny | 37.06% |
| spaCy+Faker | 40.62% |
| Presidio | 50.33% |

### Per-Type ELR — All Seq2Seq Models

#### BART-base (ELR = 0.93%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 45 | 4,177 | 1.08% |
| UNKNOWN | 26 | 1,081 | 2.41% |
| LOCATION | 6 | 753 | 0.80% |
| ORGANIZATION | 6 | 1,062 | 0.57% |
| TIME | 2 | 337 | 0.59% |
| DATE | 1 | 599 | 0.17% |
| EMAIL | 0 | 373 | 0.00% |
| PHONE | 0 | 714 | 0.00% |
| SSN | 0 | 13 | 0.00% |
| CREDIT_CARD | 0 | 33 | 0.00% |
| ID_NUMBER | 0 | 81 | 0.00% |
| ZIPCODE | 0 | 47 | 0.00% |
| ADDRESS | 0 | 1 | 0.00% |

#### Flan-T5-small (ELR = 0.99%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 54 | 4,177 | 1.29% |
| UNKNOWN | 26 | 1,081 | 2.41% |
| LOCATION | 5 | 753 | 0.66% |
| ORGANIZATION | 4 | 1,062 | 0.38% |
| TIME | 2 | 337 | 0.59% |
| DATE | 1 | 599 | 0.17% |
| All others | 0 | 1,262 | 0.00% |

#### T5-small (ELR = 1.54%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 73 | 4,177 | 1.75% |
| UNKNOWN | 52 | 1,081 | 4.81% |
| LOCATION | 9 | 753 | 1.20% |
| ORGANIZATION | 5 | 1,062 | 0.47% |
| TIME | 3 | 337 | 0.89% |
| DATE | 1 | 599 | 0.17% |
| All others | 0 | 1,262 | 0.00% |

#### DistilBART (ELR = 1.23%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 64 | 4,177 | 1.53% |
| UNKNOWN | 30 | 1,081 | 2.78% |
| LOCATION | 13 | 753 | 1.73% |
| ORGANIZATION | 5 | 1,062 | 0.47% |
| TIME | 1 | 337 | 0.30% |
| DATE | 1 | 599 | 0.17% |
| All others | 0 | 1,262 | 0.00% |

#### T5-efficient-tiny (ELR = 4.14%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 250 | 4,177 | 5.98% |
| UNKNOWN | 71 | 1,081 | 6.57% |
| ORGANIZATION | 29 | 1,062 | 2.73% |
| LOCATION | 25 | 753 | 3.32% |
| TIME | 6 | 337 | 1.78% |
| DATE | 3 | 599 | 0.50% |
| All others | 0 | 1,262 | 0.00% |

### Per-Type ELR — Rule-Based Systems

#### spaCy+Faker (ELR = 26.44%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| TIME | 317 | 337 | 94.07% |
| FULLNAME | 1,179 | 4,177 | 28.23% |
| UNKNOWN | 745 | 1,081 | 68.92% |
| ORGANIZATION | 106 | 1,062 | 9.98% |
| LOCATION | 52 | 753 | 6.91% |
| DATE | 33 | 599 | 5.51% |
| ZIPCODE | 18 | 47 | 38.30% |
| ADDRESS | 1 | 1 | 100.00% |
| EMAIL | 0 | 373 | 0.00% |
| PHONE | 0 | 714 | 0.00% |
| SSN | 0 | 13 | 0.00% |
| CREDIT_CARD | 0 | 33 | 0.00% |
| ID_NUMBER | 0 | 81 | 0.00% |

#### Presidio (ELR = 33.77%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 1,286 | 4,177 | 30.79% |
| ORGANIZATION | 727 | 1,062 | 68.46% |
| UNKNOWN | 712 | 1,081 | 65.86% |
| PHONE | 164 | 714 | 22.97% |
| LOCATION | 99 | 753 | 13.15% |
| TIME | 86 | 337 | 25.52% |
| DATE | 36 | 599 | 6.01% |
| ZIPCODE | 19 | 47 | 40.43% |
| SSN | 1 | 13 | 7.69% |
| ADDRESS | 1 | 1 | 100.00% |
| EMAIL | 0 | 373 | 0.00% |
| CREDIT_CARD | 0 | 33 | 0.00% |
| ID_NUMBER | 0 | 81 | 0.00% |

#### Regex+Faker (ELR = 83.39%)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 4,167 | 4,177 | 99.76% |
| ORGANIZATION | 1,060 | 1,062 | 99.81% |
| UNKNOWN | 993 | 1,081 | 91.86% |
| LOCATION | 753 | 753 | 100.00% |
| DATE | 385 | 599 | 64.27% |
| TIME | 325 | 337 | 96.44% |
| ZIPCODE | 47 | 47 | 100.00% |
| ADDRESS | 1 | 1 | 100.00% |
| EMAIL | 0 | 373 | 0.00% |
| PHONE | 0 | 714 | 0.00% |
| SSN | 0 | 13 | 0.00% |
| CREDIT_CARD | 0 | 33 | 0.00% |
| ID_NUMBER | 0 | 81 | 0.00% |

---

## 4. Task 3: Privacy Risk Assessment

### CRR-3 / ERA / UAC

| System | CRR-3 ↓ | ERA@1 ↓ | ERA@5 ↓ | UAC ↓ | ERA Evaluated |
|--------|---------|---------|---------|-------|---------------|
| **BART-base** | 34.62% | 1.90% | 4.84% | 0.33% | 1,261 |
| Flan-T5-small | 34.94% | 1.67% | 5.08% | 0.22% | 1,261 |
| spaCy+Faker | 40.62% | 9.36% | 14.27% | 1.58% | 1,261 |
| Presidio | 50.33% | 20.46% | 26.09% | 1.78% | 1,261 |

### LRR — LLM Re-identification Rate

Two separate LRR evaluations were conducted with different attacker models:

#### Run 1: Qwen 2.5 14B (newer, higher coverage, zero failures)

| System | LRR Exact ↓ | LRR Fuzzy ↓ | Entities Evaluated | Sample N | API Failures |
|--------|-------------|-------------|-------------------|----------|--------------|
| BART-base | 0.13% | 0.53% | 753 | 300 | 0 |
| Presidio | 2.12% | 3.45% | 753 | 300 | 0 |

#### Run 2: Llama 3.3 70B (older, stronger attacker)

| System | LRR Exact ↓ | LRR Fuzzy ↓ | Entities Evaluated | API Failures |
|--------|-------------|-------------|-------------------|--------------|
| BART-base | 0.20% | 0.40% | 495 | 0 |
| spaCy+Faker | 0.94% | 2.83% | 106 | 152 |

### Attack Comparison — BART-base (All Methods)

| Attack Type | Recovery Rate | Description |
|-------------|--------------|-------------|
| ELR (verbatim) | 0.93% | Entity text appears unchanged in output |
| ERA@1 (retrieval) | 1.90% | Embedding-based candidate ranking, top-1 |
| ERA@5 (retrieval) | 4.84% | Embedding-based candidate ranking, top-5 |
| CRR-3 (character) | 34.62% | 3-gram character overlap between original and anonymized |
| UAC (unique) | 0.33% | Entities uniquely identifiable from anonymized text |
| LRR exact (Qwen 14B) | 0.13% | LLM guesses original from anonymized text |
| LRR fuzzy (Qwen 14B) | 0.53% | LLM guess has >0.8 character similarity |
| LRR exact (Llama 70B) | 0.20% | LLM guesses original from anonymized text |
| LRR fuzzy (Llama 70B) | 0.40% | LLM guess has >0.8 character similarity |

---

## 5. Failure Taxonomy

### Updated Counts (from latest evaluation JSON files)

| Category | BART-base | spaCy | Presidio | Regex |
|----------|-----------|-------|----------|-------|
| **Clean** | 5,700 (61.5%) | 3,892 (42.0%) | 2,615 (28.2%) | 327 (3.5%) |
| **Full Leak** | 86 (0.9%) | 2,451 (26.4%) | 3,131 (33.8%) | 7,731 (83.4%) |
| **Context Retention** | 3,374 (36.4%) | 2,728 (29.4%) | 2,340 (25.2%) | 1,182 (12.7%) |
| **Boundary Error** | 101 (1.1%) | 165 (1.8%) | 191 (2.1%) | 28 (0.3%) |
| **Format Break** | 10 (0.1%) | 35 (0.4%) | 994 (10.7%) | 3 (0.0%) |
| **Type Confusion** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Over-Masking** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Total** | 9,271 | 9,271 | 9,271 | 9,271 |

### Failure Examples — BART-base

#### Boundary Errors (101 total)

| Sample ID | Entity | Type | Leaked Tokens |
|-----------|--------|------|---------------|
| sample_30726 | 15th November 1948 | DATE | "15th" |
| sample_45225 | Hockett Country Lane | ORGANIZATION | "Lane" |
| sample_28340 | Jennett Tree Lane | FULLNAME | "Lane" |
| sample_76273 | 7 o'clock | TIME | "clock" |
| sample_47756 | 26/01/2005 | DATE | "2005" |
| sample_14691 | Bou Chakra | FULLNAME | "Chakra" |
| sample_07968 | 17/10/2024 | DATE | "2024" |

#### Format Breaks (10 total)

| Sample ID | Entity | Type | Replacement |
|-----------|--------|------|-------------|
| sample_72480 | 089 8921.2729 | PHONE | "er discussion." |
| sample_93074 | 002 213.352.2254 | PHONE | "67" |
| sample_131240 | 5199 | DATE | "Heigh" |
| sample_83823 | +84.62 399 7586 | PHONE | "'" |
| sample_107291 | 32719 | ZIPCODE | "5" |
| sample_100009 | 42FH@tutanota.com | EMAIL | "mail.com " |

### Failure Examples — Presidio

#### Format Breaks (994 total) — predominantly `<TAG>` replacements

| Sample ID | Entity | Type | Replacement |
|-----------|--------|------|-------------|
| sample_17035 | 96362-93741 | DATE | `<DATE_TIME>` |
| sample_138062 | November 20th, 1969 | DATE | `<DATE_TIME>` |
| sample_83956 | 24/03/2001 | DATE | `<DATE_TIME>` |
| sample_67948 | tal-a@hotmail.com | EMAIL | `<EMAIL_ADDRESS>` |
| sample_87293 | +585 01.139-1554 | PHONE | `<PHONE_NUMBER>` |

### Failure Examples — spaCy

#### Boundary Errors (165 total)

| Sample ID | Entity | Type | Leaked Tokens |
|-----------|--------|------|---------------|
| sample_86516 | Azaria Machi | LOCATION | "Azaria" |
| sample_109293 | Burton upon Trent | FULLNAME | "upon" |
| sample_48195 | Diss Wortham | FULLNAME | "Diss" |
| sample_38307 | East Camp Wisdom Road | ORGANIZATION | "East" |
| sample_07031 | North Baltimore | LOCATION | "North" |

#### Format Breaks (35 total)

| Sample ID | Entity | Type | Replacement |
|-----------|--------|------|-------------|
| sample_36095 | 17090 | ZIPCODE | "[FAC]" |
| sample_133033 | 2334403519 | PHONE | "ese" |
| sample_48114 | +7-96.925-4307 | PHONE | "[EVENT]" |
| sample_01191 | P1962@outlook.com | EMAIL | "Brady LLC" |
| sample_83519 | +962.53-077.7119 | PHONE | "Barron" |

---

## 6. Pareto Frontier Analysis

### Pareto-Optimal Systems

`regex`, `bart-base`

### System Coordinates

| System | Privacy (1−ELR) | Utility (BERT/100) | PUS (λ=0.5) |
|--------|----------------|--------------------|-------------|
| bart-base | 0.991 | 0.927 | 0.959 |
| flan-t5-small | 0.990 | 0.925 | 0.957 |
| t5-small | 0.985 | 0.926 | 0.955 |
| t5-eff-tiny | 0.959 | 0.926 | 0.942 |
| distilbart | 0.988 | 0.863 | 0.926 |
| spacy | 0.736 | 0.919 | 0.827 |
| presidio | 0.662 | 0.900 | 0.781 |
| regex | 0.166 | 0.982 | 0.574 |

### PUS Sweep (λ = 0.0 to 1.0)

| System | λ=0.0 | λ=0.1 | λ=0.2 | λ=0.3 | λ=0.4 | λ=0.5 | λ=0.6 | λ=0.7 | λ=0.8 | λ=0.9 | λ=1.0 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| regex | 0.982 | 0.900 | 0.818 | 0.737 | 0.655 | 0.574 | 0.492 | 0.411 | 0.329 | 0.248 | 0.166 |
| spacy | 0.919 | 0.900 | 0.882 | 0.864 | 0.845 | 0.827 | 0.809 | 0.791 | 0.772 | 0.754 | 0.736 |
| presidio | 0.900 | 0.877 | 0.853 | 0.829 | 0.805 | 0.781 | 0.758 | 0.734 | 0.710 | 0.686 | 0.662 |
| bart-base | 0.927 | 0.934 | 0.940 | 0.946 | 0.953 | 0.959 | 0.965 | 0.972 | 0.978 | 0.984 | 0.991 |
| flan-t5 | 0.925 | 0.931 | 0.938 | 0.944 | 0.951 | 0.957 | 0.964 | 0.971 | 0.977 | 0.984 | 0.990 |
| t5-small | 0.926 | 0.932 | 0.938 | 0.944 | 0.949 | 0.955 | 0.961 | 0.967 | 0.973 | 0.979 | 0.985 |
| distilbart | 0.863 | 0.876 | 0.888 | 0.901 | 0.913 | 0.926 | 0.938 | 0.950 | 0.963 | 0.975 | 0.988 |
| t5-eff-tiny | 0.926 | 0.929 | 0.932 | 0.936 | 0.939 | 0.942 | 0.945 | 0.949 | 0.952 | 0.955 | 0.959 |

BART-base is the only system whose PUS **increases** with higher λ (privacy weight), confirming it dominates across all privacy-utility tradeoff preferences.

---

## 7. Prediction Files Inventory

| File | System | Records |
|------|--------|---------|
| predictions_bart-base-pii.jsonl | BART-base seq2seq | 3,600 |
| predictions_flan-t5-small-pii.jsonl | Flan-T5-small seq2seq | 3,600 |
| predictions_t5-small-pii.jsonl | T5-small seq2seq | 3,600 |
| predictions_distilbart-pii.jsonl | DistilBART seq2seq | 3,600 |
| predictions_t5-efficient-tiny-pii.jsonl | T5-efficient-tiny seq2seq | 3,600 |
| regex_predictions.jsonl | Regex+Faker pipeline | 3,600 |
| spacy_predictions.jsonl | spaCy+Faker pipeline | 3,600 |
| presidio_predictions.jsonl | Presidio pipeline | 3,600 |
| predictions_llm.jsonl | LLM baseline | 3,600 |
| regex_spans.jsonl | Regex detected spans | 3,600 |
| spacy_spans.jsonl | spaCy detected spans | 3,600 |
| presidio_spans.jsonl | Presidio detected spans | 3,600 |

---

## 8. Known Issues & Caveats

1. **Entity type mapping inconsistency:** The dataset labels use {FULLNAME, FIRST_NAME, ADDRESS, ...} while the evaluation scripts infer types via text heuristics, producing {FULLNAME, LOCATION, TIME, ORGANIZATION, ...}. Aggregate ELR is unaffected (text-based string matching), but per-type breakdowns use the evaluation mapping and are not directly comparable to dataset-label distributions.

2. **Two separate LRR evaluations exist.** Run 1 used Qwen 2.5-14B (753 entities, 0 failures). Run 2 used Llama 3.3 70B (495 entities for BART, 106 for spaCy with 152 failures). The Qwen run is more reliable (higher coverage, zero failures) but uses a weaker attacker. The Llama run uses a stronger attacker but has lower coverage.

3. **ERA values not in latest JSON files.** The ERA@1/ERA@5 values (1.90%/4.84% for BART-base) come from an earlier evaluation run. The latest `eval_privacy_bart_lrr.json` has `"era": null`. ERA results are retained from the earlier run in this document.

4. **Context retention inflation.** The context overlap metric (>80% shared context words) flags expected non-PII text preservation as "context retention." Values of ~36% for seq2seq models reflect faithful text reproduction of surrounding words, not re-identification risk.

5. **Boundary error false positives.** Common substrings (e.g., "Lane", "2024", "clock") trigger boundary errors even when the original entity was correctly replaced. The latest evaluation run reduced boundary errors significantly (e.g., BART-base: 439 → 101) by tightening the detection criteria.

6. **19.6% UNKNOWN entities.** Entity type inference from text heuristics leaves ~20% of test entities untyped. Per-type analysis is limited for these.

7. **Synthetic data.** All test data is generated via Faker. Results may not generalize to real-world PII distributions.

8. **43.9% train/test entity string overlap.** Nearly half of test entity strings appear in training data. Seq2seq model performance may partly reflect memorization.

9. **Presidio type-aware F1 is 4.97%.** This is largely a taxonomy mismatch — Presidio uses its own entity type names (e.g., `PERSON` vs `FULLNAME`, `PHONE_NUMBER` vs `PHONE`) rather than reflecting actual detection failure.

10. **Flan-T5/T5-small/DistilBART/T5-eff-tiny failure taxonomies not re-evaluated.** Only BART-base, spaCy, Presidio, and Regex have updated failure taxonomy counts from the latest evaluation. Earlier counts for the other seq2seq models (from previous run) are not included in this document to avoid mixing evaluation runs.

---

## 9. Figures

| Figure | File | Description |
|--------|------|-------------|
| Task 2 Comparison | `figures/task2_comparison.png` | Dual-panel: ELR (top) and BERTScore (bottom) for all systems |
| Pareto Frontier | `figures/pareto_frontier.png` | Privacy vs Utility scatter with Pareto frontier line |
| Failure Taxonomy | `figures/failure_taxonomy.png` | Stacked bar chart of failure categories by system |
| Detection Recall | `figures/detection_recall.png` | Per-type recall heatmap for rule-based detectors |
| Attack Heatmap | `figures/attack_heatmap.png` | Recovery rates by system and attack type |

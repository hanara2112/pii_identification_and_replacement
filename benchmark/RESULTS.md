# SAHA-AL Benchmark — Complete Results

## Dataset Statistics

| Split | Records | Entities | Avg Words/Record | Avg Ents/Record | PII Density | Entity Types |
|-------|---------|----------|-------------------|-----------------|-------------|-------------|
| Train | 113,133 | 286,542 | 17.0 | 2.53 | 0.149 | 12 |
| Validation | 3,600 | 9,211 | 17.0 | 2.56 | 0.150 | 11 |
| Test | 3,600 | 9,271 | 17.0 | 2.58 | 0.152 | 12 |

**Train/Test entity string overlap:** 43.9% (3,661 / 8,344 unique test strings)

### Test Set Entity Type Distribution

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

---

## Task 1: PII Detection (Span-Level P/R/F1)

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

---

## Task 2: Text Anonymization Quality

| System | ELR ↓ | Token Recall ↑ | OMR ↓ | FPR ↑ | BERTScore F1 ↑ |
|--------|-------|----------------|-------|-------|----------------|
| **BART-base** | **0.93%** | **95.50%** | **0.13%** | 0.28% | 92.74 |
| Flan-T5-small | 0.99% | 95.33% | 0.22% | 0.62% | 92.47 |
| DistilBART | 1.23% | 94.41% | 0.15% | 1.85% | 86.34 |
| T5-small | 1.54% | 94.15% | 0.19% | 0.14% | 92.59 |
| T5-eff-tiny | 4.14% | 90.42% | 0.49% | 0.84% | 92.57 |
| spaCy+Faker | 26.44% | 74.59% | 2.47% | 3.20% | 91.86 |
| Presidio | 33.77% | 68.18% | 1.18% | **12.37%** | 90.04 |
| Regex+Faker | 83.39% | 23.05% | 0.00% | 24.17% | **98.15** |

### Per-Type ELR — BART-base (best model)

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 53 | 5,332 | 0.99% |
| UNKNOWN | 29 | 1,813 | 1.60% |
| FIRST_NAME | 3 | 228 | 1.32% |
| ADDRESS | 1 | 366 | 0.27% |
| EMAIL | 0 | 373 | 0.00% |
| PHONE | 0 | 714 | 0.00% |
| DATE | 0 | 270 | 0.00% |
| SSN | 0 | 13 | 0.00% |
| CREDIT_CARD | 0 | 33 | 0.00% |
| ID_NUMBER | 0 | 81 | 0.00% |
| ZIPCODE | 0 | 47 | 0.00% |
| ORGANIZATION | 0 | 1 | 0.00% |

---

## Task 3: Privacy Risk Assessment

### CRR-3 / ERA / UAC

| System | CRR-3 ↓ | ERA@1 ↓ | ERA@5 ↓ | UAC ↓ | ERA Evaluated |
|--------|---------|---------|---------|-------|---------------|
| **BART-base** | 34.62% | 1.90% | 4.84% | 0.33% | 1,261 |
| Flan-T5-small | 34.94% | 1.67% | 5.08% | 0.22% | 1,261 |
| spaCy+Faker | 40.62% | 9.36% | 14.27% | 1.58% | 1,261 |
| Presidio | 50.33% | 20.46% | 26.09% | 1.78% | 1,261 |

### LRR — LLM Re-identification Rate (Llama 3.3 70B)

| System | LRR Exact ↓ | LRR Fuzzy ↓ | Entities Evaluated | API Failures |
|--------|-------------|-------------|-------------------|--------------|
| BART-base | 0.20% | 0.40% | 495 | 0 |
| spaCy+Faker | 0.94% | 2.83% | 106 | 152 |

### Attack Comparison — BART-base

| Attack Type | Recovery Rate | Description |
|-------------|--------------|-------------|
| ELR (verbatim) | 0.93% | Entity text appears unchanged in output |
| ERA@1 (retrieval) | 1.90% | Embedding-based candidate ranking, top-1 |
| ERA@5 (retrieval) | 4.84% | Embedding-based candidate ranking, top-5 |
| LRR exact (Llama 70B) | 0.20% | LLM guesses original from anonymized text |
| LRR fuzzy (Llama 70B) | 0.40% | LLM guess has >0.8 character similarity |

**Key finding:** Retrieval attacks (ERA@1=1.90%) are more effective than generative LLM attacks (LRR=0.40%) against replacement-based anonymization.

---

## Failure Taxonomy

| Category | BART-base | Flan-T5 | T5-small | spaCy | Presidio | Regex |
|----------|-----------|---------|----------|-------|----------|-------|
| **Clean** | 5,496 (59.3%) | 5,477 (59.1%) | 5,309 (57.3%) | 3,795 (40.9%) | 2,587 (27.9%) | 280 (3.0%) |
| **Full Leak** | 86 (0.9%) | 92 (1.0%) | 143 (1.5%) | 2,451 (26.4%) | 3,131 (33.8%) | 7,731 (83.4%) |
| **Ghost Leak** | 3,241 (35.0%) | 3,275 (35.3%) | 3,324 (35.9%) | 2,667 (28.8%) | 2,306 (24.9%) | 1,080 (11.6%) |
| **Boundary** | 439 (4.7%) | 418 (4.5%) | 486 (5.2%) | 323 (3.5%) | 253 (2.7%) | 177 (1.9%) |
| **Format Break** | 9 (0.1%) | 9 (0.1%) | 9 (0.1%) | 35 (0.4%) | 994 (10.7%) | 3 (0.0%) |
| **Type Confusion** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Over-Masking** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Total** | 9,271 | 9,271 | 9,271 | 9,271 | 9,271 | 9,271 |

**Note:** Ghost leak measures context word overlap (>80%) around removed entities. High values for seq2seq models indicate faithful non-PII text preservation, not necessarily re-identification risk. Boundary errors for BART-base are mostly false positives from common substrings like "com" in email replacements.

---

## Pareto Frontier Analysis

**Pareto-optimal systems:** `regex`, `bart-base`

| System | Privacy (1−ELR) | Utility (BERT/100) | PUS (λ=0.5) |
|--------|----------------|--------------------|-------------|
| regex | 0.166 | 0.982 | 0.574 |
| spacy | 0.736 | 0.919 | 0.827 |
| presidio | 0.662 | 0.900 | 0.781 |
| bart-base | 0.991 | 0.927 | 0.959 |
| flan-t5-small | 0.990 | 0.925 | 0.957 |
| t5-small | 0.985 | 0.926 | 0.955 |
| distilbart | 0.988 | 0.863 | 0.926 |
| t5-eff-tiny | 0.959 | 0.926 | 0.942 |

BART-base achieves the highest balanced PUS score (0.959 at λ=0.5).

---

## Known Issues & Caveats

1. **Entity type inconsistency:** DistilBART and T5-eff-tiny per-type breakdowns use a different entity type mapping (LOCATION, TIME, ORGANIZATION) than BART/Flan-T5/T5-small (FIRST_NAME, ADDRESS). Overall ELR is unaffected (text-based). Per-type comparisons require re-evaluation against the same test split.
2. **spaCy LRR unreliable:** 152/200 API calls failed due to rate limiting. Only 106 entities evaluated.
3. **Ghost leak inflation:** The context overlap metric flags expected non-PII preservation as "ghost leaks." Values of ~35% for seq2seq models reflect faithful text reproduction, not re-identification risk.
4. **Boundary false positives:** Common substrings (e.g., "com" from email domains) trigger boundary errors even when the replacement correctly uses a different email.
5. **19.6% UNKNOWN entities:** Entity type inference from text heuristics leaves ~20% untyped. Per-type analysis is limited for these.
6. **Synthetic data:** All test data is generated via Faker. Results may not generalize to real-world PII distributions.

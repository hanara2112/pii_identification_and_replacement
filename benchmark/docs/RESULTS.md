# PII Anonymization Benchmark — Key Results

## Dataset Overview

| Split | Records | Entities | Avg Ents/Record | PII Density |
|-------|---------|----------|-----------------|-------------|
| Train | 113,133 | 286,542 | 2.53 | 14.9% |
| Test | 3,600 | 9,271 | 2.58 | 15.2% |

Dominant entity types in test: FULLNAME (57.5%), UNKNOWN (19.6%), PHONE (7.7%), EMAIL (4.0%).
Train/test entity string overlap: 43.9%.

---

## Systems Evaluated

| Category | Systems |
|----------|---------|
| Rule-based | Regex+Faker, spaCy+Faker, Presidio |
| Seq2Seq | BART-base, Flan-T5-small, T5-small, DistilBART, T5-efficient-tiny |

---

## Task 1: PII Detection (Span-Level)

| System | Exact F1 | Partial F1 | Type-aware F1 |
|--------|----------|------------|---------------|
| Regex+Faker | 25.86 | 27.99 | 24.39 |
| spaCy+Faker | 56.30 | 69.38 | 18.18 |
| Presidio | 48.57 | 60.11 | 4.97 |

spaCy achieves the best overall detection (F1=56.3) but all rule-based systems struggle with FULLNAME (0–39% recall) and ADDRESS (0% recall). Structured types (EMAIL, CREDIT_CARD, SSN) are detected near-perfectly by all systems.

---

## Task 2: Text Anonymization Quality

| System | ELR ↓ | Token Recall ↑ | OMR ↓ | BERTScore F1 ↑ |
|--------|-------|----------------|-------|----------------|
| **BART-base** | **0.93%** | **95.50%** | **0.13%** | 92.74 |
| Flan-T5-small | 0.99% | 95.33% | 0.22% | 92.47 |
| DistilBART | 1.23% | 94.41% | 0.15% | 86.34 |
| T5-small | 1.54% | 94.15% | 0.19% | 92.59 |
| T5-eff-tiny | 4.14% | 90.42% | 0.49% | 92.57 |
| spaCy+Faker | 26.44% | 74.59% | 2.47% | 91.86 |
| Presidio | 33.77% | 68.18% | 1.18% | 90.04 |
| Regex+Faker | 83.39% | 23.05% | 0.00% | 98.15 |

All seq2seq models achieve ELR < 5%. BART-base leads with only 86/9,271 entities leaked (0.93%) while preserving 95.5% of non-PII tokens.

### Per-Type Leakage — BART-base

| Type | Leaked | Total | ELR |
|------|--------|-------|-----|
| FULLNAME | 45 | 4,177 | 1.08% |
| UNKNOWN | 26 | 1,081 | 2.41% |
| LOCATION | 6 | 753 | 0.80% |
| ORGANIZATION | 6 | 1,062 | 0.57% |
| TIME | 2 | 337 | 0.59% |
| DATE | 1 | 599 | 0.17% |
| EMAIL / PHONE / SSN / CC / ID / ZIP | 0 | 1,262 | 0.00% |

Residual leakage concentrates on names (FULLNAME, UNKNOWN) — structured types are fully protected.

---

## Task 3: Privacy Risk Assessment

### Re-identification Attacks

| System | CRR-3 ↓ | ERA@1 ↓ | ERA@5 ↓ | UAC ↓ |
|--------|---------|---------|---------|-------|
| **BART-base** | 34.62% | 1.90% | 4.84% | 0.33% |
| Flan-T5-small | 34.94% | 1.67% | 5.08% | 0.22% |
| spaCy+Faker | 40.62% | 9.36% | 14.27% | 1.58% |
| Presidio | 50.33% | 20.46% | 26.09% | 1.78% |

### LLM Re-identification (LRR)

| System | Model | LRR Exact ↓ | LRR Fuzzy ↓ | Entities | Failures |
|--------|-------|-------------|-------------|----------|----------|
| BART-base | Qwen 2.5 14B | 0.13% | 0.53% | 753 | 0 |
| Presidio | Qwen 2.5 14B | 2.12% | 3.45% | 753 | 0 |

### Attack Comparison — BART-base

| Attack | Recovery Rate | Method |
|--------|--------------|--------|
| ELR (verbatim leak) | 0.93% | Direct string match |
| ERA@1 (retrieval) | 1.90% | Embedding-based candidate ranking |
| ERA@5 (retrieval) | 4.84% | Top-5 candidate ranking |
| LRR exact (LLM) | 0.13% | LLM guesses original entity |
| LRR fuzzy (LLM) | 0.53% | LLM guess >0.8 char similarity |

Retrieval attacks (ERA) pose higher risk than generative LLM attacks (LRR) against replacement-based anonymization.

---

## Failure Taxonomy

| Category | BART-base | spaCy | Presidio | Regex |
|----------|-----------|-------|----------|-------|
| **Clean** | 5,700 (61.5%) | 3,892 (42.0%) | 2,615 (28.2%) | 327 (3.5%) |
| **Full Leak** | 86 (0.9%) | 2,451 (26.4%) | 3,131 (33.8%) | 7,731 (83.4%) |
| **Context Retention** | 3,374 (36.4%) | 2,728 (29.4%) | 2,340 (25.2%) | 1,182 (12.7%) |
| **Boundary Error** | 101 (1.1%) | 165 (1.8%) | 191 (2.1%) | 28 (0.3%) |
| **Format Break** | 10 (0.1%) | 35 (0.4%) | 994 (10.7%) | 3 (0.0%) |

Context retention for seq2seq models reflects faithful non-PII text preservation, not re-identification risk.

---

## Pareto Frontier

| System | Privacy (1−ELR) | Utility (BERT/100) | PUS (λ=0.5) |
|--------|----------------|--------------------|-------------|
| **BART-base** | **0.991** | 0.927 | **0.959** |
| Flan-T5-small | 0.990 | 0.925 | 0.957 |
| T5-small | 0.985 | 0.926 | 0.955 |
| T5-eff-tiny | 0.959 | 0.926 | 0.942 |
| DistilBART | 0.988 | 0.863 | 0.926 |
| spaCy+Faker | 0.736 | 0.919 | 0.827 |
| Presidio | 0.662 | 0.900 | 0.781 |
| Regex+Faker | 0.166 | 0.982 | 0.574 |

BART-base dominates the Pareto frontier with PUS = 0.959, achieving near-perfect privacy (99.1%) with high utility (92.7%).

---

## Key Findings

1. **Seq2seq models dramatically outperform rule-based systems.** BART-base leaks 0.93% of entities vs 26–83% for rule-based approaches, while maintaining >92% BERTScore.
2. **Structured PII is solved.** All seq2seq models achieve 0% leakage on EMAIL, PHONE, SSN, CREDIT_CARD, ID_NUMBER, and ZIPCODE.
3. **Names remain the hardest category.** FULLNAME and UNKNOWN account for 83% of all residual leakage across seq2seq models.
4. **Retrieval attacks > LLM attacks.** Embedding-based ERA@1 (1.90%) recovers more entities than Llama/Qwen-based LRR (0.13–0.53%), suggesting replacement-based anonymization is more vulnerable to database-level attacks than generative guessing.
5. **BART-base is the best overall system.** Highest PUS score (0.959), lowest ELR (0.93%), lowest CRR-3 (34.62%), and lowest LRR (0.13% exact).
6. **Presidio suffers from format breaks.** 10.7% of entities trigger format-breaking replacements (e.g., `<DATE_TIME>` tags instead of natural text), hurting downstream utility.

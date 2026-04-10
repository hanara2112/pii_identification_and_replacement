# SAHA-AL Benchmark — Results Interpretation & Discussion

> This document provides a deep interpretation of the benchmark results: what the numbers mean, the patterns they reveal, and their implications for PII anonymization system design.

---

## 1. The Core Finding: End-to-End Wins Over Detect-Then-Replace

The single most important result from this benchmark is the order-of-magnitude gap between seq2seq and rule-based systems on entity leakage:

| Approach | Best ELR | Worst ELR |
|----------|----------|-----------|
| Seq2Seq models | 0.93% | 4.14% |
| Rule-based systems | 26.44% | 83.39% |

This is not a marginal improvement — it is a **qualitative shift**. The best rule-based system (spaCy+Faker) leaks 28x more entities than the best seq2seq model (BART-base). The gap exists because rule-based systems rely on an explicit detect-then-replace pipeline, meaning any entity the detector misses is leaked entirely. Seq2seq models, by contrast, learn to rewrite text holistically, implicitly handling entities that don't match any predefined pattern.

**Why this matters for practitioners:** If privacy is the primary concern, a fine-tuned seq2seq model is categorically superior to any rule-based pipeline tested here. The choice is not about marginal gains — it is about whether a system leaks <1% or >25% of PII.

---

## 2. Understanding Detection Failures (Task 1)

### 2.1 The Precision-Recall Tradeoff

The three rule-based detectors occupy different points on the precision-recall curve:

- **Regex:** Very high precision (92.3%) but catastrophically low recall (15.0%). It only finds entities that match strict regex patterns.
- **spaCy:** Balanced but imperfect (P=69.0%, R=47.6%). Its NER catches names and locations but misses many.
- **Presidio:** Moderate on both axes (P=56.3%, R=42.7%) but generates many false positives (3,066 FP vs 1,979 for spaCy).

The F1 scores (25.9, 56.3, 48.6) reveal that **none of these systems are adequate as standalone PII detectors** on this dataset. All three miss over half of all entities.

### 2.2 The Type-Aware Collapse

The most striking detection result is the type-aware F1 collapse for Presidio (4.97%) and spaCy (18.18%). This is primarily a **taxonomy mismatch problem**, not a detection failure. Presidio labels entities as `PERSON` while the gold standard uses `FULLNAME`; Presidio uses `PHONE_NUMBER` while the gold uses `PHONE`. The detection itself is reasonable, but type normalization was not applied.

**Implication:** When reporting detection metrics, type-aware evaluation requires careful type mapping. The exact and partial F1 scores are more meaningful for comparing these systems.

### 2.3 The Entity Type Spectrum

Detection recall varies dramatically by entity type, forming a clear spectrum:

**Fully solved (>92% recall, all systems):** EMAIL, CREDIT_CARD, SSN, ID_NUMBER. These have rigid, distinctive formats that regex patterns capture near-perfectly.

**Partially solved (50–95% recall, NER-based only):** PHONE, DATE, LOCATION, ORGANIZATION. These require either flexible patterns (PHONE, DATE) or NER-based recognition (LOCATION, ORGANIZATION).

**Unsolved (0–39% recall):** FULLNAME (0–39%), UNKNOWN (2–15%), ADDRESS (0%), TIME (0–54%). Names are the dominant challenge — they comprise 57.5% of all entities but are detected at <40% recall even by the best system.

**Why names are hard:** Personal names have no structural pattern. They are arbitrary sequences of capitalized words, indistinguishable from other capitalized phrases (place names, titles, organizations) without semantic understanding. With 4,177 names in the test set and the best detector finding only 1,610 (38.5%), the remaining 2,567 flow through to become leaks in any detect-then-replace pipeline.

---

## 3. Interpreting Anonymization Quality (Task 2)

### 3.1 ELR: What Does <1% Leakage Mean?

BART-base leaks 86 out of 9,271 entities (0.93%). In absolute terms, this means roughly 1 in 108 entities survives anonymization. At the record level, since each record has ~2.58 entities on average, approximately 2.3% of records contain at least one leaked entity.

This is excellent but not zero. In a deployment scenario with 100,000 records, approximately 2,300 records would contain at least one verbatim PII leak. Whether this is acceptable depends on the sensitivity of the data and the regulatory context (e.g., GDPR, HIPAA).

### 3.2 The Regex Paradox: High BERTScore, Terrible Privacy

Regex+Faker achieves the highest BERTScore (98.15) — seemingly the best utility — while having the worst ELR (83.39%). This is not contradictory:

- **Why high BERTScore:** Regex only modifies the few entities it detects (15% recall). The vast majority of text is unchanged, so the original and anonymized texts are nearly identical, yielding near-perfect semantic similarity.
- **Why terrible ELR:** The 85% of entities it *doesn't* detect remain verbatim.

This reveals a fundamental limitation of BERTScore as a standalone metric: a system that changes nothing has BERTScore ~100% and ELR ~100%. **BERTScore measures utility but is blind to privacy.** The PUS score (which combines both) correctly ranks Regex last (PUS=0.574).

### 3.3 The DistilBART Anomaly

DistilBART has better ELR than T5-small (1.23% vs 1.54%) but much worse BERTScore (86.34 vs 92.59). This 6-point BERTScore gap is the largest among seq2seq models and suggests DistilBART generates text that diverges more from the original in ways beyond entity replacement. Possible explanations:

- **Summarization bias:** DistilBART is distilled from a summarization model (`distilbart-cnn-6-6`), which may shorten or rephrase text more aggressively than models trained from language-model pretraining.
- **Truncation artifacts:** The model may truncate longer inputs, losing non-PII content.
- **Lower generation quality:** Distillation may have reduced the model's ability to faithfully preserve context around entities.

**Recommendation:** DistilBART is not recommended despite its good privacy score, because the utility loss is disproportionate. Flan-T5-small achieves better ELR (0.99%) *and* better BERTScore (92.47) with fewer parameters (77M vs 230M).

### 3.4 The T5-Efficient-Tiny Scaling Effect

T5-efficient-tiny (16M params) has 4.5x higher ELR than BART-base (4.14% vs 0.93%), confirming that **model capacity matters for anonymization quality**. With only 16M parameters, the model lacks the capacity to consistently generate plausible replacements for diverse entity types. However, its BERTScore (92.57) is nearly identical to BART-base (92.74), meaning it preserves non-PII text well — it just fails to replace PII as effectively.

The ELR scaling across model sizes:

| Model | Parameters | ELR |
|-------|-----------|-----|
| T5-eff-tiny | 16M | 4.14% |
| T5-small | 60M | 1.54% |
| Flan-T5-small | 77M | 0.99% |
| BART-base | 139M | 0.93% |
| DistilBART | 230M | 1.23% |

The relationship is roughly log-linear between 16M and 139M, but DistilBART breaks the trend, suggesting that architecture and pretraining matter as much as raw parameter count.

### 3.5 Per-Type Leakage: Where Do Seq2Seq Models Fail?

Across all seq2seq models, the leakage profile is remarkably consistent:

- **Zero leakage:** EMAIL, PHONE, SSN, CREDIT_CARD, ID_NUMBER, ZIPCODE — all structured types
- **Low leakage (0.2–0.8%):** LOCATION, ORGANIZATION, TIME, DATE — semi-structured types
- **Residual leakage (1–6%):** FULLNAME, UNKNOWN — unstructured names

FULLNAME and UNKNOWN together account for 71/86 (82.6%) of BART-base's leaked entities. The model has effectively "solved" structured PII replacement but still struggles with names — particularly unusual or multi-word names that may not appear frequently in training data.

The UNKNOWN category (2.41% ELR for BART-base) has the highest per-type leakage rate. These are entities that the type-inference heuristic couldn't classify — likely unusual formats, edge cases, or genuinely ambiguous strings. Their higher leakage rate suggests that entity ambiguity at the dataset level correlates with model difficulty.

---

## 4. Interpreting Privacy Attacks (Task 3)

### 4.1 The Attack Hierarchy

For BART-base, the five attack types form a clear hierarchy of effectiveness:

```
LRR exact (0.13%) < LRR fuzzy (0.53%) < ELR (0.93%) < ERA@1 (1.90%) < ERA@5 (4.84%)
```

**Interpretation:**

- **LRR is the weakest attack.** An LLM trying to *guess* the original entity from context almost never succeeds. Replacement-based anonymization (substituting one plausible name for another) defeats generative inference because the context is semantically consistent with the replacement — there are no "holes" or redaction markers that would signal what was changed.

- **ELR captures verbatim failures.** These are entities the model simply passed through unchanged. They represent implementation bugs or edge cases, not fundamental vulnerability.

- **ERA is the most effective attack.** A retrieval adversary with access to a candidate pool recovers 1.9% of entities at rank-1 and 4.8% at rank-5. This is because the anonymized text's embedding still carries some signal about the entity types present, and candidate entities of the same type have correlated embeddings.

### 4.2 Why ERA > LRR: The Inference vs Retrieval Gap

The fact that retrieval attacks outperform LLM attacks has an important theoretical implication: **replacement-based anonymization is more vulnerable to database adversaries than to generative adversaries.**

When an entity is replaced with a plausible fake (e.g., "John Smith" → "Michael Jones"), the surrounding text remains coherent. An LLM cannot distinguish "Michael Jones" from "John Smith" based on context alone because both are valid names. But a retrieval adversary doesn't need to reason about plausibility — it exploits the statistical correlation between the anonymized text's global embedding and the candidate entities' embeddings.

This suggests that **future anonymization systems should consider embedding-level privacy**, not just surface-level text replacement. Techniques like adding noise to the semantic space or using structurally different replacement entities (different length, different ethnicity, different gender) could reduce ERA vulnerability.

### 4.3 CRR-3: Interpreting 34.6%

A CRR-3 of 34.62% means that roughly one-third of all capitalized 3-grams in the original text survive in the anonymized output. This might seem high, but it includes non-PII capitalized phrases (e.g., "Please note that", "We are writing") which *should* survive because they are not sensitive.

Comparing across systems:

| System | CRR-3 |
|--------|-------|
| BART-base | 34.62% |
| Presidio | 50.33% |

The 16-point gap reflects Presidio's higher entity leakage — more PII-containing 3-grams survive. The fact that even BART-base retains 34.6% confirms that CRR-3 is a coarse measure that includes both expected non-PII retention and actual leakage. It is useful as a relative comparison metric but not as an absolute privacy guarantee.

### 4.4 UAC: Compositional Privacy

BART-base's UAC of 0.33% means that only 0.33% of records have a unique quasi-identifier combination in the anonymized output. This is very low — it means that almost all records are "blended" into groups with shared attribute patterns, providing k-anonymity with k >> 1 for the vast majority.

Presidio's UAC of 1.78% (5.4x higher) reflects its higher leakage rate: more surviving entities create more unique attribute combinations.

---

## 5. Interpreting the Failure Taxonomy

### 5.1 BART-base: A 61/36/1/1/0 Profile

BART-base's failure distribution is:
- 61.5% Clean (entity properly replaced)
- 36.4% Context Retention (entity removed, context preserved)
- 0.9% Full Leak (entity passed through unchanged)
- 1.1% Boundary Error (entity partially replaced)
- 0.1% Format Break (replacement is malformed)

The **dominant "failure" is context retention (36.4%)**, but this is expected and benign for seq2seq models. The model faithfully reproduces non-PII text around entities — this is exactly what it should do. The context retention rate is essentially measuring how much surrounding text the model preserves, which correlates with BERTScore.

The actual privacy-relevant failures (Full Leak + Boundary Error) total only 2.0% of entities. This is closely aligned with the ELR of 0.93% (Full Leak alone), with boundary errors adding another 1.1% of partial exposure.

### 5.2 Boundary Errors: Mostly False Positives

The 101 boundary errors in BART-base are dominated by common words that happen to appear in entity names:

- **"Lane"** appears in "Jennett Tree Lane", "Leeming Lane", "Moss Side Lane" — but "Lane" is a common English word that naturally appears in other contexts too.
- **"2024", "2005"** appear as partial matches of dates — but year numbers frequently appear in non-PII text.
- **"clock"** from "7 o'clock" — part of a time expression, not identifying.

Many of these are **false positive boundary errors**: the leaked token appears in the output coincidentally, not because the model failed to replace the entity. The boundary error metric is useful for flagging *potential* partial leaks, but manual inspection reveals that most are benign.

### 5.3 Presidio's Format Break Problem

Presidio's 10.7% format break rate (994 entities) is its most distinctive failure mode. The root cause is Presidio's default anonymization strategy: it uses placeholder tags like `<DATE_TIME>`, `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>` instead of generating realistic replacements. While this effectively removes PII (low ELR for those specific entities), it:

1. Destroys the natural text format (reducing BERTScore)
2. Makes the anonymization obvious to a reader
3. Reduces downstream utility for any NLP task consuming the text

This is a design choice, not a bug — Presidio prioritizes explicit redaction over plausible replacement. But it explains why Presidio's BERTScore (90.04) and PUS (0.781) are lower than spaCy+Faker despite having a similar detection capability.

### 5.4 Regex: An Instructive Failure Case

Regex+Faker's profile (83.4% Full Leak, 3.5% Clean) is essentially the inverse of BART-base. It leaks nearly everything because it can only detect structured types (EMAIL, PHONE, SSN, etc.) — which together constitute only ~13% of test entities. The remaining 87% (names, locations, organizations, unknown types) pass through unchanged.

The 12.7% context retention for Regex is also instructive: even for entities it successfully detects and replaces, the surrounding context is preserved (because Regex only modifies the specific span). This is similar to BART-base's behavior but for a much smaller fraction of entities.

---

## 6. Understanding the Pareto Frontier

### 6.1 Two Pareto-Optimal Systems, Two Philosophies

The Pareto-optimal set is `{Regex, BART-base}`, representing two extreme strategies:

- **Regex:** Maximum utility (BERTScore 98.2%) at the cost of almost no privacy (16.6%). It barely changes the text.
- **BART-base:** Near-maximum privacy (99.1%) with high utility (92.7%). It rewrites entities effectively.

All other systems are **dominated** — there exists at least one system with both better privacy and better utility. This is particularly notable for spaCy+Faker and Presidio, which are dominated by BART-base on both axes.

### 6.2 The PUS Crossover

The PUS sweep reveals that Regex is preferred only when λ < 0.07 (almost pure utility focus). For any λ ≥ 0.07, at least one seq2seq model dominates. At λ = 0.5 (equal privacy-utility weight), BART-base leads with PUS = 0.959.

BART-base is unique among all systems: its PUS **increases** with λ. This happens because its privacy score (0.991) exceeds its utility score (0.927), so increasing the privacy weight improves the combined score. For all rule-based systems, PUS decreases with λ because their privacy scores are lower than their utility scores.

### 6.3 The Seq2Seq Cluster

The five seq2seq models cluster tightly in the top-right of the Pareto plot (Privacy ∈ [0.96, 0.99], Utility ∈ [0.86, 0.93]). Within this cluster:

- **BART-base, Flan-T5-small, T5-small** form a tight trio (PUS 0.955–0.959), differentiated mainly by small ELR differences.
- **T5-eff-tiny** trades ~3% privacy for slightly lower utility, positioning it as a viable option when compute is constrained (16M params vs 139M).
- **DistilBART** is an outlier — good privacy but poor utility (BERTScore 86.3), making it the worst seq2seq choice despite being the largest model.

---

## 7. Cross-Cutting Themes

### 7.1 Structured vs Unstructured PII: A Solved/Unsolved Divide

The benchmark reveals a clear divide in PII difficulty:

| Difficulty | Entity Types | Best Detection Recall | Best ELR |
|------------|-------------|----------------------|----------|
| **Solved** | EMAIL, SSN, CREDIT_CARD, ID_NUMBER | 100% | 0.00% |
| **Mostly solved** | PHONE, DATE | 83–95% | 0.00–0.17% |
| **Hard** | FULLNAME, LOCATION, ORG, TIME | 0–64% | 0.57–2.41% |
| **Unsolved** | UNKNOWN, ADDRESS | 0–15% | 2.41% (UNKNOWN) |

Structured PII with distinctive formats is fully solved by all approaches. The remaining challenge is entirely about names and free-form identifiers — entities that look like ordinary text.

### 7.2 Detection is the Bottleneck for Rule-Based Systems

For rule-based systems, the anonymization quality ceiling is set by detection recall. spaCy detects 47.6% of entities and leaks 26.4%; the correlation is almost perfect because undetected entities are leaked verbatim. Improving rule-based anonymization therefore requires better detection — which is fundamentally limited by the ambiguity of names in text.

Seq2seq models bypass this bottleneck entirely by not requiring explicit detection. They learn an implicit entity model during training that generalizes beyond pattern matching.

### 7.3 The 43.9% Overlap Question

The train/test entity string overlap of 43.9% raises the question: how much of the seq2seq models' success is memorization vs generalization?

Several observations suggest **genuine generalization dominates:**
- Even for the 56.1% of test entities *not* seen in training, seq2seq models achieve very low leakage.
- The models correctly handle novel names, unusual phone formats, and edge-case dates that are unlikely to be memorized.
- T5-eff-tiny (16M params) achieves 4.14% ELR despite having far too few parameters to memorize 183,720 unique training entity strings.

However, a definitive answer would require a zero-overlap evaluation split, which is not currently available.

### 7.4 Adversarial Robustness Rankings

When ranked by adversarial robustness (Task 3), the system ordering is consistent across all attack types:

```
BART-base > Flan-T5 > spaCy+Faker > Presidio
```

BART-base leads on every metric: lowest CRR-3, lowest ERA@1/ERA@5, lowest LRR, lowest UAC. This consistency suggests that the privacy advantages of seq2seq models are not metric-specific artifacts — they reflect a genuine, robust improvement in anonymization quality.

---

## 8. Practical Implications

### 8.1 System Selection Guide

| Scenario | Recommended System | Rationale |
|----------|-------------------|-----------|
| Maximum privacy, moderate compute | BART-base | Best ELR, best PUS, 139M params |
| Privacy with minimal compute | T5-eff-tiny | ELR 4.14% with only 16M params |
| No GPU available | spaCy+Faker | Best rule-based option, CPU-only |
| Structured PII only | Regex+Faker | Perfect for emails, phones, SSNs |
| Regulatory audit trail needed | Presidio | Explicit detection + redaction tags |

### 8.2 What Would Make BART-base Even Better?

The 86 remaining leaks in BART-base are concentrated in FULLNAME (45) and UNKNOWN (26). Targeted improvements could include:

1. **Name-focused data augmentation:** Generate more training examples with unusual, multi-word, or non-Western names.
2. **Entity-aware loss functions:** Increase the loss weight for tokens within entity spans during training.
3. **Post-processing filter:** Apply a lightweight NER check on the output and re-replace any detected entities.
4. **Ensemble with rule-based:** Use BART-base for general anonymization, then overlay regex checks for structured types as a safety net (though this is already at 0% leakage for structured types).

### 8.3 Caveats for Deployment

1. **Synthetic ≠ real-world.** All results are on Faker-generated data. Real-world PII has different distributions, formats, and context patterns. Performance on real clinical notes, legal documents, or social media will differ.
2. **ELR < 1% ≠ GDPR compliance.** Regulatory compliance depends on the specific legal framework, the sensitivity of the data, and the definition of "adequate" anonymization. ELR provides a technical measure but not a legal guarantee.
3. **BERTScore ≠ downstream utility.** High BERTScore indicates semantic preservation at the embedding level, but specific NLP tasks (NER, relation extraction, classification) may be affected differently by anonymization artifacts.

---

## 9. Summary

The SAHA-AL benchmark demonstrates that:

1. **Fine-tuned seq2seq models represent a paradigm shift** in text anonymization, achieving <1% entity leakage while preserving >92% semantic similarity.
2. **Structured PII is a solved problem** — regex patterns suffice for emails, phone numbers, SSNs, and credit cards. **Names remain the open challenge.**
3. **Retrieval-based attacks (ERA) are the most potent threat** against replacement-based anonymization, outperforming both verbatim detection and LLM-based inference.
4. **Privacy and utility are not fundamentally at odds** — BART-base achieves near-maximum scores on both axes, dominating the Pareto frontier with PUS = 0.959.
5. **The failure taxonomy reveals qualitatively different error profiles**: seq2seq models produce rare, specific failures (boundary leaks of common words), while rule-based systems produce systematic, categorical failures (all names leaked).

The benchmark's multi-task, multi-attack, privacy-utility framework provides a comprehensive lens for evaluating anonymization systems — moving beyond simple accuracy metrics to capture the full complexity of the privacy protection problem.

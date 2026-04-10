# SAHA-AL Benchmark: Methodology & Theoretical Framework

> A complete methodological and theoretical reference for the SAHA-AL benchmark for PII identification and replacement in text.

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Problem Formulation](#2-problem-formulation)
3. [Dataset Construction](#3-dataset-construction)
4. [Task Definitions](#4-task-definitions)
5. [Evaluation Framework — Task 1: PII Detection](#5-evaluation-framework--task-1-pii-detection)
6. [Evaluation Framework — Task 2: Anonymization Quality](#6-evaluation-framework--task-2-anonymization-quality)
7. [Evaluation Framework — Task 3: Privacy Risk Assessment](#7-evaluation-framework--task-3-privacy-risk-assessment)
8. [Privacy-Utility Tradeoff Framework](#8-privacy-utility-tradeoff-framework)
9. [Failure Taxonomy](#9-failure-taxonomy)
10. [Baseline Systems](#10-baseline-systems)
11. [Statistical Methodology](#11-statistical-methodology)
12. [Theoretical Foundations & Related Work](#12-theoretical-foundations--related-work)
13. [Known Limitations](#13-known-limitations)
14. [References](#14-references)

---

## 1. Introduction & Motivation

Text anonymization — the process of removing or replacing personally identifiable information (PII) in free-form text — is a prerequisite for privacy-preserving data sharing in healthcare, legal, financial, and research domains. Unlike tabular data where k-anonymity and differential privacy offer well-studied guarantees, unstructured text poses unique challenges:

- **Context leakage:** Even after entity removal, surrounding words can reveal identity.
- **Semantic fidelity:** Aggressive redaction destroys downstream utility.
- **Diverse entity types:** Names, addresses, dates, IDs, and free-form identifiers require different handling strategies.
- **Adversarial recovery:** Sophisticated attackers can re-identify individuals through embedding retrieval or LLM-based inference.

Existing benchmarks (TAB, PII-Masking datasets) typically evaluate only detection accuracy or simple token-level recall. SAHA-AL extends this by evaluating anonymization as a **system under attack**, incorporating adversarial privacy risk metrics alongside a formalized privacy-utility tradeoff framework.

### Design Principles

1. **Multi-task evaluation:** Separate detection, anonymization quality, and privacy risk into distinct measurable tasks.
2. **Adversarial threat modeling:** Evaluate systems against retrieval-based (ERA), generative (LRR), and statistical (CRR-3, UAC) attacks.
3. **Privacy-utility formalization:** Provide a parameterized score (PUS) that allows practitioners to navigate the privacy-utility tradeoff according to their risk tolerance.
4. **Error taxonomy:** Move beyond aggregate metrics to categorize *how* systems fail, enabling targeted improvement.

---

## 2. Problem Formulation

### Definitions

Let \( \mathcal{D} = \{(x_i, E_i)\}_{i=1}^{N} \) be a dataset where:
- \( x_i \) is a text document (the original text)
- \( E_i = \{e_{i,1}, e_{i,2}, \ldots, e_{i,k_i}\} \) is the set of PII entities in \( x_i \)
- Each entity \( e = (\text{text}, \text{type}, \text{start}, \text{end}) \) has surface text, a type label, and character-level span offsets

An **anonymization system** \( f \) produces:
\[ \hat{x}_i = f(x_i) \]
where \( \hat{x}_i \) is the anonymized text with PII entities replaced by synthetic alternatives.

A **detection system** \( g \) produces:
\[ \hat{E}_i = g(x_i) = \{(\text{start}_j, \text{end}_j, \text{type}_j)\}_{j=1}^{m_i} \]

### Threat Model

We consider three classes of adversaries:
1. **Passive observer:** Has access only to \( \hat{x}_i \). Can the original entities be read directly? (ELR)
2. **Knowledge-enriched adversary:** Has access to a database of candidate entities (e.g., from training data). Can they recover the original entity by similarity matching? (ERA)
3. **Generative adversary:** Has access to a powerful language model. Can they infer the original entity from contextual clues in \( \hat{x}_i \)? (LRR)

---

## 3. Dataset Construction

### Source Data

The raw dataset is derived from `anonymized_dataset_final.jsonl`, a collection of synthetic text records generated using Faker-based text templates. Each record contains an `original_text` with embedded PII entities and a reference `anonymized_text` with those entities replaced.

### Entity Type Inference

The raw data labels all entities as `type=UNKNOWN`. The preparation script (`scripts/prepare_dataset.py`) infers entity types using a cascaded heuristic:

1. **Regex matching** (highest priority): Structured types are identified by pattern:
   - EMAIL: `user@domain.tld`
   - SSN: `###-##-####`
   - CREDIT_CARD: 13–19 digit sequences
   - PHONE: International/domestic phone patterns
   - DATE: Multiple date formats (MM/DD/YYYY, Month DD YYYY, etc.)
   - ID_NUMBER: 1–3 uppercase letters followed by 6–10 digits
   - ZIPCODE: 5-digit or ZIP+4 format

2. **Heuristic rules** (medium priority):
   - ADDRESS: Starts with number + street suffix (St, Ave, Rd, etc.)
   - ORGANIZATION: Contains corporate suffixes (Inc, Corp, LLC, etc.)
   - FULLNAME: 2–4 capitalized words
   - FIRST_NAME: Single capitalized word preceded by title keywords (Mr, Mrs, Dr, etc.)

3. **Fallback:** Remaining entities retain `UNKNOWN` type (~19.6% of test entities)

### Span Alignment

Entity character offsets are verified against `original_text`. If the stored `(start, end)` span doesn't match the entity text at those positions, a greedy non-overlapping string search locates the correct span. Entities that cannot be aligned receive `start=-1, end=-1` and are flagged as `invalid`.

### Data Splits

| Split | Source | Records | Entities |
|-------|--------|---------|----------|
| Train | First 36,000 records (80%) + augmented remainder | 113,133 | 286,542 |
| Validation | First 36,000 records (10%) | 3,600 | 9,211 |
| Test | First 36,000 records (10%) | 3,600 | 9,271 |

The split is performed with `random.seed(42)` after shuffling the first 36,000 "gold" records. Augmented records (beyond index 36,000) are appended only to the training set. A leakage assertion verifies zero ID overlap between train and test.

**Train/test entity string overlap:** 43.9% of unique test entity strings also appear in the training data. This is a consequence of the synthetic generation process (Faker reuses name/email pools).

---

## 4. Task Definitions

### Task 1: PII Detection

**Input:** Original text \( x_i \)
**Output:** Set of detected entity spans \( \hat{E}_i = \{(\text{start}, \text{end}, \text{type})\} \)
**Goal:** Maximize span-level precision, recall, and F1.

This task evaluates the ability to *locate* PII in text, independent of how it is anonymized. It is relevant for detect-then-replace pipelines (regex, spaCy, Presidio, BERT-NER).

### Task 2: Text Anonymization

**Input:** Original text \( x_i \) with entity annotations \( E_i \)
**Output:** Anonymized text \( \hat{x}_i \) where all PII has been replaced
**Goal:** Minimize entity leakage (ELR) while maximizing text utility (BERTScore, token recall).

This task evaluates end-to-end anonymization quality. Seq2seq models receive only the original text and produce anonymized output directly, without explicit entity annotations at inference time.

### Task 3: Privacy Risk Assessment

**Input:** (Original text, anonymized text) pairs from Task 2
**Output:** Privacy risk scores under multiple attack models
**Goal:** Quantify residual re-identification risk after anonymization.

This task simulates adversarial attacks against anonymized outputs to measure how much private information can be recovered.

---

## 5. Evaluation Framework — Task 1: PII Detection

### 5.1 Span Matching Modes

Three matching modes are defined in `eval/utils.py:span_match()`:

**Exact Match:**
\[ \text{match}(p, g) = \mathbb{1}[p.\text{start} = g.\text{start} \wedge p.\text{end} = g.\text{end}] \]

**Partial Match (IoU > 0.5):**
\[ \text{overlap} = \max(0, \min(p.\text{end}, g.\text{end}) - \max(p.\text{start}, g.\text{start})) \]
\[ \text{union} = \max(p.\text{end}, g.\text{end}) - \min(p.\text{start}, g.\text{start}) \]
\[ \text{match}(p, g) = \mathbb{1}\left[\frac{\text{overlap}}{\text{union}} > 0.5\right] \]

**Type-Aware Match:**
\[ \text{match}(p, g) = \mathbb{1}[p.\text{start} = g.\text{start} \wedge p.\text{end} = g.\text{end} \wedge \text{upper}(p.\text{type}) = \text{upper}(g.\text{type})] \]

### 5.2 Matching Algorithm

Predicted spans are matched to gold spans via **greedy bipartite matching**: for each predicted span (in order), find the first unmatched gold span that satisfies the match criterion. This yields:

- **TP** (true positives): matched gold spans
- **FP** (false positives): unmatched predicted spans
- **FN** (false negatives): unmatched gold spans

### 5.3 Corpus-Level Metrics

\[ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R} \]

All metrics are computed at the **corpus level** (micro-averaged across all records).

### 5.4 Per-Type Recall

For each entity type \( t \):
\[ \text{Recall}_t = \frac{\text{TP}_t}{\text{Total}_t} \]

where \( \text{Total}_t \) is the number of gold entities of type \( t \) and \( \text{TP}_t \) is the number correctly detected (exact match).

---

## 6. Evaluation Framework — Task 2: Anonymization Quality

### 6.1 Entity Leakage Rate (ELR)

The primary privacy metric. Measures the fraction of gold entities whose text appears verbatim in the anonymized output.

\[ \text{ELR} = \frac{\sum_{i=1}^{N} \sum_{j=1}^{k_i} \mathbb{1}[\text{match}(e_{i,j}, \hat{x}_i)]}{|\mathcal{E}|} \times 100\% \]

where \( |\mathcal{E}| = \sum_i k_i \) is the total entity count, and the match function uses **word-boundary regex**:

```
pattern = r"(?<!\w)" + re.escape(entity_text) + r"(?!\w)"
```

This ensures that "com" inside "communication" is not flagged as a leak of entity "com", but "John Smith" appearing unchanged is flagged. The match is case-insensitive.

**Per-type ELR** is computed by grouping entities by their inferred type and computing ELR within each group.

### 6.2 Token Recall

Measures the fraction of entity-span tokens that are successfully *absent* from the anonymized text. Higher is better.

\[ \text{Token Recall} = \frac{\sum_{\text{tok} \in \mathcal{T}} \mathbb{1}[\text{tok} \notin \text{tokens}(\hat{x}_i)]}{|\mathcal{T}|} \times 100\% \]

where \( \mathcal{T} \) is the set of all entity tokens across all records (tokens with length < 2 are excluded to avoid noise from single-character tokens).

Token extraction uses the regex `\w+` (alphanumeric word characters). The check is against the full set of tokens in the anonymized text (lowercased).

### 6.3 Over-Masking Rate (OMR)

Measures the fraction of *non-entity tokens* that were altered during anonymization. Lower is better — high OMR indicates the system is unnecessarily modifying non-PII content.

\[ \text{OMR} = \frac{\sum_{\text{tok} \in \mathcal{T}_{\text{non-ent}}} \mathbb{1}[\text{tok} \notin \hat{x}_i]}{|\mathcal{T}_{\text{non-ent}}|} \times 100\% \]

Non-entity tokens are identified using character-level span exclusion: a token is "non-entity" if none of its character positions overlap with any gold entity span.

### 6.4 Format Preservation Rate (FPR)

For structured entity types (EMAIL, PHONE, SSN, CREDIT_CARD, ZIPCODE, DATE), FPR checks whether the replacement string matches an expected format regex.

\[ \text{FPR} = \frac{\sum_{\text{ent} \in \mathcal{S}} \mathbb{1}[\text{FORMAT}_t.\text{match}(\text{replacement})]}{|\mathcal{S}|} \times 100\% \]

where \( \mathcal{S} \) is the set of entities with a defined format pattern, and the replacement string is extracted via `difflib.SequenceMatcher` alignment between original and anonymized text.

Format patterns used:

| Type | Regex Pattern |
|------|--------------|
| EMAIL | `^[^@\s]+@[^@\s]+\.[^@\s]+$` |
| PHONE | `^[\d\s\+\-\(\)\.]{5,20}$` |
| SSN | `^\d{3}[- ]?\d{2}[- ]?\d{4}$` |
| CREDIT_CARD | `^[\d\s\-]{12,25}$` |
| ZIPCODE | `^[A-Z0-9\- ]{3,12}$` |
| DATE | `\d` (contains at least one digit) |

### 6.5 BERTScore F1

Semantic similarity between original and anonymized text, measuring how well the anonymization preserves meaning.

\[ \text{BERTScore}(x, \hat{x}) = F_1(\text{BERT-emb}(x), \text{BERT-emb}(\hat{x})) \]

Implementation uses the `bert_score` library with `distilbert-base-uncased` as the embedding model. The score is computed per-record and then averaged (mean F1 across all records), reported as a percentage (0–100 scale).

BERTScore captures semantic similarity at the token-embedding level using greedy matching between contextual embeddings of the reference and candidate texts (Zhang et al., 2020).

### 6.6 NLI Consistency (Optional)

Fraction of (original, anonymized) text pairs classified as "entailment" by an NLI model (`roberta-large-mnli`). This measures whether the anonymized text preserves the logical content of the original.

\[ \text{NLI} = \frac{\sum_{i} \mathbb{1}[\text{NLI}(x_i, \hat{x}_i) = \text{ENTAILMENT}]}{N} \times 100\% \]

---

## 7. Evaluation Framework — Task 3: Privacy Risk Assessment

### 7.1 Capitalized 3-gram Survival Rate (CRR-3)

A statistical measure of how much identifiable n-gram structure survives anonymization. Capitalized 3-grams are proxies for named entities and proper nouns.

**Extraction:** From the original text, extract all 3-grams where at least one token starts with an uppercase letter.

**Survival check:** A 3-gram "survives" if its lowercase form appears as a substring in the (lowercased) anonymized text.

\[ \text{CRR-3} = \frac{\sum_{i} |\{g \in \text{Cap3}(x_i) : \text{lower}(g) \in \text{lower}(\hat{x}_i)\}|}{\sum_{i} |\text{Cap3}(x_i)|} \times 100\% \]

CRR-3 captures entity preservation that ELR might miss (e.g., partial name matches, multi-word entities that span n-gram boundaries).

### 7.2 Entity Recovery Attack (ERA)

A **retrieval-based adversary** that attempts to recover original entities by embedding similarity. This simulates an attacker with access to a database of candidate entities.

**Threat model:** The adversary knows:
- The anonymized text \( \hat{x}_i \)
- A pool of candidate entities from the training data, partitioned by type

**Algorithm:**
1. Build per-type candidate pools from training data (capped at 500 candidates per type).
2. Encode all candidate entities and anonymized texts using Sentence-BERT (`all-MiniLM-L6-v2`).
3. For each gold entity \( e \) in each test record:
   a. Compute cosine similarity between the anonymized text embedding and all candidate embeddings for the entity's type.
   b. Rank candidates by similarity.
   c. Check if the original entity appears in the top-1 (ERA@1) or top-5 (ERA@5) ranked candidates.

\[ \text{ERA@}k = \frac{|\{e : e.\text{text} \in \text{Top-}k(\text{cossim}(\hat{x}, \text{pool}_{e.\text{type}}))\}|}{|\mathcal{E}_{\text{eval}}|} \times 100\% \]

**Fair evaluation:** If the original entity is not in the candidate pool, it is appended with its embedding to ensure the attack has the opportunity to succeed.

ERA measures vulnerability to database-level attacks, which are relevant when an adversary has auxiliary knowledge (e.g., a census, social media scrape, or leaked dataset).

### 7.3 LLM Re-identification Rate (LRR)

A **generative adversary** that uses a large language model to infer original entities from contextual clues in the anonymized text.

**Prompt:**
```
You are given an anonymized text where personal information has been replaced with fake alternatives.

Anonymized text: {anonymized_text}

For each piece of replaced information, try to guess the ORIGINAL value based on contextual clues.
Output one guess per line in the format:
REPLACEMENT -> YOUR_GUESS

Only output guesses, nothing else.
```

**Matching:**
- **LRR Exact:** The LLM's guess exactly matches the original entity text.
- **LRR Fuzzy:** The LLM's guess has >0.8 character-level similarity (SequenceMatcher ratio) to the original.

\[ \text{LRR}_{\text{exact}} = \frac{|\{e : e.\text{text} \in \text{LLM-guesses}(\hat{x}_i)\}|}{|\mathcal{E}_{\text{eval}}|} \times 100\% \]

\[ \text{LRR}_{\text{fuzzy}} = \frac{|\{e : \exists g \in \text{guesses}, \text{sim}(e.\text{text}, g) > 0.8\}|}{|\mathcal{E}_{\text{eval}}|} \times 100\% \]

LRR is evaluated on a sample of records (default \( n = 300 \)) due to the cost of LLM inference. The benchmark supports both API-based models (GPT-4o-mini, Llama 3.3 70B via Groq) and local models (Qwen 2.5, Phi-3.5, Gemma-2) with optional 4-bit quantization.

LRR measures vulnerability to inference attacks where an adversary reasons about the original content based on contextual coherence, writing style, and world knowledge.

### 7.4 Unique Attribute Combination Rate (UAC)

A **compositional privacy proxy** grounded in k-anonymity (Sweeney, 2002). Measures the fraction of records that have a unique combination of surviving quasi-identifiers, making them potentially uniquely identifiable.

**Algorithm:**
1. For each record, identify "surviving" entity types — entities whose text appears in the anonymized output (leaked) or whose type can be inferred from context hints within a ±30-character window.
2. Form a type-combination tuple for each record (e.g., `(EMAIL, FULLNAME, PHONE)`).
3. Count how many records share each combination.
4. Records with a unique combination (count = 1) violate k=1 anonymity.

Context hints use keyword matching:

| Type | Hint Keywords |
|------|--------------|
| EMAIL | "email", "@", "mail" |
| PHONE | "phone", "call", "tel" |
| SSN | "ssn", "social security" |
| DATE | "born", "date", "birthday" |
| ADDRESS | "lives", "address", "street", "road" |

\[ \text{UAC} = \frac{|\{r : \text{count}(\text{combo}(r)) = 1\}|}{N} \times 100\% \]

---

## 8. Privacy-Utility Tradeoff Framework

### 8.1 Privacy-Utility Score (PUS)

A parameterized scalar that unifies privacy protection and text utility into a single score:

\[ \text{PUS}(\lambda) = \lambda \cdot \text{Privacy} + (1 - \lambda) \cdot \text{Utility} \]

where:
\[ \text{Privacy} = 1 - \frac{\text{ELR}}{100}, \quad \text{Utility} = \frac{\text{BERTScore}_{F_1}}{100} \]

The parameter \( \lambda \in [0, 1] \) controls the tradeoff:
- \( \lambda = 0 \): Pure utility focus (PUS = Utility)
- \( \lambda = 0.5 \): Equal weight (default comparison point)
- \( \lambda = 1 \): Pure privacy focus (PUS = Privacy)

### 8.2 Pareto Frontier

A system \( s \) is **Pareto-optimal** if no other system achieves both strictly higher privacy *and* strictly higher utility:

\[ s \in \text{Pareto} \iff \nexists s' : \text{Privacy}(s') \geq \text{Privacy}(s) \wedge \text{Utility}(s') \geq \text{Utility}(s) \]

with at least one strict inequality. The Pareto frontier represents the set of non-dominated systems — the best achievable tradeoff at each operating point.

### 8.3 PUS Sweep

Computing PUS across \( \lambda = \{0.0, 0.1, \ldots, 1.0\} \) reveals each system's behavior across the tradeoff spectrum. A system is **unconditionally dominant** if its PUS is highest for all \( \lambda \) values — this is only possible if it lies on the Pareto frontier and no other system offers a better tradeoff at any operating point.

---

## 9. Failure Taxonomy

Beyond aggregate metrics, the benchmark classifies each entity replacement into one of six categories to provide diagnostic insight into *how* systems fail.

### 9.1 Categories

**1. Full Leak:** The entity text appears unchanged in the anonymized output (word-boundary regex match). This is the most severe failure — the PII is completely exposed.

**2. Boundary Error:** The entity is partially masked — some tokens (length > 3) from the entity appear in the output, but not the full entity string. Examples: "Lane" surviving from "Jennett Tree Lane", or "2024" surviving from "17/10/2024".

**3. Format Break:** For structured types with defined format patterns (EMAIL, PHONE, SSN, etc.), the replacement string does not match the expected format regex. Examples: a phone number replaced with "er discussion." or a date replaced with "Heigh". This indicates the model failed to generate a type-appropriate substitute.

**4. Context Retention (Ghost Leak):** The entity was successfully removed, but >80% of the surrounding context words (±50 characters) are preserved in the anonymized text. While this is *expected* for seq2seq models that faithfully reproduce non-PII text, it could theoretically aid re-identification in combination with auxiliary information.

**5. Type Confusion:** The replacement doesn't match the semantic type of the original (e.g., a name replaced with a phone number). Not observed in practice in the current evaluation.

**6. Over-Masking:** Non-entity content was unnecessarily altered. Not observed as a distinct failure category in the current evaluation (captured instead by the aggregate OMR metric).

**7. Clean:** The entity was properly replaced with no detectable issues.

### 9.2 Classification Logic

The classification is **priority-ordered**: each entity is assigned to the *first* matching category in this order:

1. Full leak → word-boundary regex match of full entity text
2. Boundary error → any token (length > 3) matches individually, but not the full entity
3. Format break → structured type with a replacement that fails the format regex
4. Context retention → >80% context word overlap in a ±50-character window around the entity position
5. Clean → none of the above conditions triggered

### 9.3 Interpretation

- **Full Leak + Boundary Error** = direct privacy violations
- **Format Break** = utility/plausibility failures (not necessarily privacy-violating, but reduces the quality of the anonymized text)
- **Context Retention** = an indirect risk factor, but high values for seq2seq models primarily reflect faithful non-PII text preservation, not actual re-identification vulnerability

---

## 10. Baseline Systems

### 10.1 Rule-Based Systems

**Regex+Faker:**
1. Apply ordered regex patterns (EMAIL → SSN → CREDIT_CARD → ... → ID_NUMBER) to detect spans.
2. Resolve overlapping spans (greedy, left-to-right, longest-first).
3. Replace each span with a Faker-generated value of the corresponding type.

*Strength:* Near-perfect precision for structured types.
*Weakness:* Cannot detect unstructured entities (names, organizations, locations).

**spaCy+Faker:**
1. Run `en_core_web_lg` NER to detect PERSON, ORG, GPE, LOC, DATE, FAC, NORP, EVENT entities.
2. Overlay regex patterns from the Regex baseline for structured types.
3. Resolve overlaps and replace with Faker values.

*Strength:* Detects names and organizations via NER.
*Weakness:* spaCy NER has moderate recall on non-standard names; type taxonomy mismatch.

**Presidio (Microsoft):**
1. Use `AnalyzerEngine` with all default recognizers (PII, NER, regex).
2. Use `AnonymizerEngine` to apply default anonymization operators.

*Strength:* Comprehensive recognizer suite, production-grade.
*Weakness:* Uses tag-style replacements (e.g., `<DATE_TIME>`) for some types, causing format breaks. High false-positive rate.

### 10.2 Seq2Seq Models

Five encoder-decoder models fine-tuned on the SAHA-AL training data to perform text-to-text anonymization:

| Model | Base | Parameters | Prefix |
|-------|------|-----------|--------|
| BART-base | `facebook/bart-base` | 139M | (none) |
| Flan-T5-small | `google/flan-t5-small` | 77M | "Replace all personal identifiable information..." |
| T5-small | `t5-small` | 60M | "anonymize: " |
| DistilBART | `sshleifer/distilbart-cnn-6-6` | 230M | (none) |
| T5-efficient-tiny | `google/t5-efficient-tiny` | 16M | "anonymize: " |

**Training:** All models are fine-tuned with a PII-aware loss function. Checkpoints are hosted on HuggingFace (`JALAPENO11/pii_identification_and_anonymisations`).

**Inference:** Beam search with `num_beams=4`, `max_input_length=128`, `max_output_length=128`, `do_sample=False`.

**Key difference from rule-based systems:** Seq2seq models perform *end-to-end* anonymization — they take original text as input and directly generate anonymized text without an explicit detect-then-replace pipeline. This allows them to leverage contextual understanding to produce more natural and consistent replacements.

### 10.3 LLM Baselines

**Zero-shot LLM:** Direct prompting of GPT-4o-mini or local models (Qwen 2.5, Phi-3.5, Gemma-2) to anonymize text.

**Hybrid (spaCy + LLM):** spaCy detects entities, then an LLM generates contextually appropriate replacements for each detected entity.

### 10.4 BERT-NER Baseline

Fine-tuned `bert-base-cased` as a BIO token classifier for entity detection (Task 1), followed by Faker-based replacement for detected spans (Task 2).

---

## 11. Statistical Methodology

### 11.1 Bootstrap Confidence Intervals

The benchmark uses **non-parametric bootstrap** (Efron & Tibshirani, 1993) to compute 95% confidence intervals for key metrics.

**Procedure:**
1. Draw \( B = 1000 \) bootstrap samples of size \( N \) (with replacement) from the test set.
2. Compute the metric on each bootstrap sample.
3. Report the 2.5th and 97.5th percentiles as the 95% CI.

\[ \text{CI}_{95\%} = \left[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}\right] \]

Supported metrics: ELR, Token Recall, OMR, FPR, CRR-3.

### 11.2 Record Alignment

All evaluations require aligned (gold, prediction) pairs. The `align_records()` function performs **ID-based join**: each gold record is matched to its prediction by the `id` field. Missing predictions cause a `ValueError`.

### 11.3 Text Normalization

All text comparisons use `normalize_text()` which strips whitespace and replaces `None`/empty strings with `[EMPTY]`. Entity matching uses `re.IGNORECASE` for case-insensitive comparison.

---

## 12. Theoretical Foundations & Related Work

### 12.1 Privacy Definitions

**k-Anonymity (Sweeney, 2002):** A dataset satisfies k-anonymity if every combination of quasi-identifiers appears in at least k records. UAC approximates the violation rate (k=1 records) for text-based quasi-identifiers.

**Differential Privacy (Dwork, 2006):** While not directly measured by this benchmark, the ELR and ERA metrics provide an empirical estimate of privacy leakage that complements formal DP guarantees. Future work could integrate DP-based text generation with SAHA-AL evaluation.

### 12.2 Text Anonymization Literature

**Entity Recognition:** The benchmark's detection evaluation (Task 1) follows the CoNLL-style span evaluation tradition but extends it with partial (IoU) and type-aware matching modes.

**Text Rewriting:** The seq2seq anonymization approach draws from paraphrase generation and controllable text generation. The key insight is that end-to-end models can learn entity replacement as a latent subtask, avoiding the error propagation of detect-then-replace pipelines.

### 12.3 Adversarial Privacy Evaluation

**Embedding-based Attacks (ERA):** Inspired by membership inference attacks (Shokri et al., 2017) and attribute inference. ERA uses Sentence-BERT (Reimers & Gurevych, EMNLP 2019) to embed both the anonymized text and candidate entities, then ranks candidates by cosine similarity.

**LLM-based Attacks (LRR):** Inspired by Staab et al. (2023) on LLM privacy inference capabilities. The LRR metric measures whether modern LLMs can "undo" anonymization by reasoning about contextual clues.

### 12.4 Evaluation Metrics

**BERTScore (Zhang et al., 2020):** Token-level matching using contextual embeddings, providing a more semantically meaningful similarity measure than BLEU or ROUGE.

**NLI Consistency:** Using textual entailment as a proxy for semantic preservation follows the tradition of faithfulness evaluation in summarization (Maynez et al., 2020).

---

## 13. Known Limitations

1. **Synthetic data:** All text is generated via Faker-based templates. Entity distributions, naming patterns, and contextual structures may not reflect real-world PII. Results should be interpreted as a controlled comparison rather than absolute performance estimates.

2. **English-only:** The benchmark, NER models, and evaluation metrics are English-specific. Multilingual extension would require adapted Faker locales, NER models, and format patterns.

3. **Entity type inference artifacts:** The heuristic type inference assigns ~19.6% of entities to UNKNOWN, and the evaluation-time type mapping (which re-infers types from text) produces a different distribution than the dataset labels. Aggregate metrics (ELR, BERTScore) are unaffected, but per-type comparisons require careful interpretation.

4. **Train/test entity overlap (43.9%):** The high overlap means seq2seq models may benefit from memorization. This inflates their absolute performance but does not affect the relative ranking of systems.

5. **Fixed threat model:** The adversary in ERA has access to training-set entities (closed-world assumption). Real-world adversaries may have different or broader auxiliary knowledge. LRR uses a fixed prompt; adversarial prompt engineering could yield higher recovery rates.

6. **BERTScore limitations:** BERTScore using `distilbert-base-uncased` may not capture all aspects of semantic preservation, particularly for domain-specific text.

7. **No latency/efficiency evaluation:** The benchmark does not measure inference time, memory usage, or computational cost, which are critical for deployment decisions.

8. **IAA:** Inter-annotator agreement for the source data was entity-level F1=0.83, not Cohen's kappa.

---

## 14. References

- Dwork, C. (2006). Differential privacy. *ICALP*.
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Maynez, J., et al. (2020). On faithfulness and factuality in abstractive summarization. *ACL*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.
- Shokri, R., et al. (2017). Membership inference attacks against machine learning models. *IEEE S&P*.
- Staab, R., et al. (2023). Beyond memorization: Violating privacy via inference with large language models. *arXiv:2310.07298*.
- Sweeney, L. (2002). k-Anonymity: A model for protecting privacy. *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*.
- Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. *ICLR*.

# SAHA-AL Benchmark Suite — Final Design (2-Day Depth Build)

**Core thesis:** Existing benchmarks evaluate anonymization as a static text transformation. SAHA-AL evaluates it as a **system under attack**, incorporating adversarial, contextual, and generative privacy risks alongside a formalized privacy-utility tradeoff framework.

---

## 0. Blockers (First 4 Hours)

### 0a. Fix test set entity types

`prepare_dataset.py` publishes test entities with `type = "UNKNOWN"`. Everything downstream dies without real types.

**Fix:** Rebuild splits from `anonymized_dataset_final.jsonl` with types traced from `gold_standard.jsonl`. Validate:

```python
unknown_count = sum(1 for e in entities if e["type"] == "UNKNOWN")
assert unknown_count == 0, f"Record {entry_id}: {unknown_count} UNKNOWN types"
```

**Time: 2h.** Non-negotiable.

### 0b. Dataset statistics

Add `analysis/dataset_stats.py`. Compute per-split: record counts, entity counts, type distribution, avg text length, PII density, entity string overlap between train/test (memorization check).

**Time: 1h.** Templated script, no judgment calls.

### 0c. Verify seq2seq inference format alignment

**Problem:** `Seq2Seq_model/dataset.py` trains models on `prefix + original_text` (plain text, no entity markers). But `benchmark/kaggle_inference.py` injects `<TYPE> entity_text </TYPE>` XML markers at inference time. If the HuggingFace checkpoints (`JALAPENO11/pii_identification_and_anonymisations`) were trained without markers, feeding them markers at inference will degrade output quality and invalidate existing leaderboard numbers.

**Fix:** Check which training script produced the uploaded checkpoints:

- If trained with `Seq2Seq_model/train.py` or `train2.py` (no markers): strip markers from `kaggle_inference.py`, re-run inference, re-evaluate.
- If trained with a separate script that used markers: document this and verify consistency.

**Validation:** Run 50 test samples with and without markers, compare ELR. If they diverge by >2 percentage points, the mismatch is confirmed.

**Time: 1h.** Critical for trusting any seq2seq number.

---

## 1. Benchmark Structure — 3 Tasks

### Task 1: PII Detection

- Input: `original_text`
- Output: `{"id": "...", "detected_entities": [{"start": 0, "end": 4, "type": "FULLNAME"}, ...]}`

### Task 2: Text Anonymization

- Input: `original_text` + `entities`
- Output: `{"id": "...", "anonymized_text": "..."}`

### Task 3: Privacy Risk Assessment

- Evaluator-computed over Task 2 submissions. No separate submission.

### 1a. Entity Taxonomy Standardization

Three different entity type sets exist across directories:


| Source                            | Count | Notable differences                                                                                                                |
| --------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `saha_al/config.py`               | 21    | Includes TITLE, GENDER, NUMBER, OTHER_PII, UNKNOWN                                                                                 |
| `pipeline_maskfill/src/common.py` | 19    | Uses PERSON (not FULLNAME/FIRST_NAME/LAST_NAME), adds IP_ADDRESS, IBAN, DRIVER_LICENSE, USERNAME, URL, MEDICAL, BUILDING, POSTCODE |
| `benchmark/README.md`             | ~18   | Mix of both                                                                                                                        |


**Benchmark canonical set (20 types):** Use `saha_al/config.py` ENTITY_TYPES minus UNKNOWN (which is a data bug, not a type). For the BERT NER baseline, further collapse noisy labels: drop TITLE, GENDER, NUMBER, OTHER_PII and map them to O (non-entity). This gives **16 clean NER types** for token classification.

For cross-dataset baselines from `pipeline_maskfill`, apply an inverse entity map at evaluation time (PERSON → {FULLNAME, FIRST_NAME, LAST_NAME}; LOC → LOCATION; etc.).

---

## 2. Metrics — Depth Over Breadth

### A. Detection Metrics (Task 1)


| Metric                | Definition                 | Direction |
| --------------------- | -------------------------- | --------- |
| Span P/R/F1 (exact)   | `(start, end)` exact match | higher    |
| Span P/R/F1 (partial) | Character IoU > 0.5        | higher    |
| Type-aware F1         | Exact span + type match    | higher    |
| Per-type recall       | Recall per entity type     | higher    |


### B. Anonymization Quality Metrics (Task 2)


| Metric           | Definition                                                 | Direction |
| ---------------- | ---------------------------------------------------------- | --------- |
| **ELR**          | Entity string found in prediction (word-boundary regex)    | lower     |
| **Token Recall** | Entity-span tokens absent from prediction                  | higher    |
| **OMR**          | Non-entity tokens altered in prediction                    | lower     |
| **FPR**          | Structured-type replacements matching format regex         | higher    |
| **BERTScore F1** | Semantic similarity (pinned `distilbert-base-uncased`) [2] | higher    |


**FPR format patterns** — aligned with `saha_al/utils/quality_checks.py`:

```python
FORMAT_PATTERNS = {
    "EMAIL":       re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "PHONE":       re.compile(r"^[\d\s\+\-\(\)\.]{5,20}$"),
    "SSN":         re.compile(r"^\d{3}[- ]?\d{2}[- ]?\d{4}$"),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{12,25}$"),
    "ZIPCODE":     re.compile(r"^[A-Z0-9\- ]{3,12}$", re.IGNORECASE),
    "DATE":        re.compile(r"\d"),
}
```

Note: IBAN and IP_ADDRESS patterns are not implemented in the existing codebase. Add them only if time permits; do not claim they exist.

### C. Privacy Metrics (Task 3) — THE DEPTH LAYER


| Metric                  | Definition                                                              | Direction |
| ----------------------- | ----------------------------------------------------------------------- | --------- |
| **CRR-3**               | Capitalized 3-gram survival rate                                        | lower     |
| **ERA**                 | Entity Recovery Attack: retrieval adversary top-k accuracy [3, 14]      | lower     |
| **LRR**                 | LLM Re-identification Rate: generative adversary exact/fuzzy match [13] | lower     |
| **UAC**                 | Unique Attribute Combination rate: compositional privacy proxy [10, 11] | lower     |
| **MIA** (supplementary) | Membership Inference Attack on seq2seq models [12]                      | lower     |


MIA is already implemented in `pipeline_maskfill/src/model2_advanced.py::membership_inference_attack()`. Port it to `eval/eval_privacy.py` — it tests whether the model memorized training examples, complementing ERA/LRR which test output-level privacy. Low cost to add since the code exists.

**ERA** (retrieval adversary — embedding-based re-identification [3, 14]):

```python
def entity_recovery_attack(gold_records, predictions, train_records, top_k=5):
    from sentence_transformers import SentenceTransformer  # [3]
    model = SentenceTransformer('all-MiniLM-L6-v2')

    top1_hits, top5_hits, total = 0, 0, 0
    for g, p in zip(gold_records, predictions):
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0: continue
            ent_type = ent.get("type", "UNKNOWN")
            original_val = ent["text"]

            pool = list(set(
                e["text"] for r in train_records for e in r.get("entities", [])
                if e.get("type") == ent_type and e.get("text")
            ))
            if original_val not in pool:
                pool.append(original_val)
            if len(pool) < 2: continue

            anon_emb = model.encode(p["anonymized_text"], convert_to_tensor=True)
            pool_embs = model.encode(pool, convert_to_tensor=True)

            from sentence_transformers.util import cos_sim
            scores = cos_sim(anon_emb, pool_embs)[0]
            ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)

            total += 1
            if pool[ranked[0]] == original_val: top1_hits += 1
            if original_val in [pool[ranked[i]] for i in range(min(top_k, len(ranked)))]:
                top5_hits += 1

    return {
        "era_top1": top1_hits / total if total else 0,
        "era_top5": top5_hits / total if total else 0,
    }
```

**LRR** (generative adversary — the novelty injection [13]):

```python
LRR_PROMPT = """You are given an anonymized text where personal information has been replaced with fake alternatives.

Anonymized text: {anonymized_text}

For each piece of replaced information, try to guess the ORIGINAL value based on contextual clues. Output one guess per line in the format:
REPLACEMENT -> YOUR_GUESS

Only output guesses, nothing else."""

def llm_reidentification_rate(gold_records, predictions, sample_n=300):
    """
    Prompt an LLM to recover original entities from anonymized text.
    Measures: exact match rate, fuzzy match rate (>0.8 char similarity).
    Inspired by Staab et al. [13] on LLM-based privacy inference.
    """
    from openai import OpenAI
    from difflib import SequenceMatcher
    client = OpenAI()

    exact, fuzzy, total = 0, 0, 0
    for g, p in zip(gold_records[:sample_n], predictions[:sample_n]):
        original_entities = {e["text"] for e in g.get("entities", []) if e.get("text")}
        if not original_entities: continue

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": LRR_PROMPT.format(anonymized_text=p["anonymized_text"])}],
            temperature=0, max_tokens=512
        )
        guesses = response.choices[0].message.content.strip().split("\n")
        parsed_guesses = set()
        for line in guesses:
            if "->" in line:
                parsed_guesses.add(line.split("->")[-1].strip())

        for orig_ent in original_entities:
            total += 1
            if orig_ent in parsed_guesses:
                exact += 1
            elif any(SequenceMatcher(None, orig_ent.lower(), guess.lower()).ratio() > 0.8
                     for guess in parsed_guesses):
                fuzzy += 1

    return {
        "lrr_exact": exact / total if total else 0,
        "lrr_fuzzy": (exact + fuzzy) / total if total else 0,
    }
```

Cost: ~$3-5 for 300 test records on GPT-4o-mini. Run on a sample, not all 3.6k.

**UAC** (compositional privacy — lightweight k-anonymity proxy [10, 11]):

```python
def unique_attribute_combination_rate(gold_records, predictions):
    """
    For each record, extract the non-masked quasi-identifier tuple
    (entity types still inferable from context). Measure how many
    records have a unique combination — higher = more re-identifiable.

    Grounded in Sweeney's observation that 87% of US residents can be
    uniquely identified by {ZIP, gender, birth date} alone [11].
    """
    from collections import Counter
    combos = Counter()
    record_combos = []

    for g, p in zip(gold_records, predictions):
        pred_text = p["anonymized_text"]
        surviving_types = []
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0: continue
            if ent["text"].lower() in pred_text.lower():
                surviving_types.append(ent["type"])
            else:
                start = max(0, ent["start"] - 30)
                end = min(len(g["original_text"]), ent["end"] + 30)
                context = g["original_text"][start:end].lower()
                type_hints = {
                    "EMAIL": ["email", "@", "mail"],
                    "PHONE": ["phone", "call", "tel"],
                    "SSN": ["ssn", "social security"],
                    "DATE": ["born", "date", "birthday"],
                    "ADDRESS": ["lives", "address", "street", "road"],
                }
                if any(h in context for h in type_hints.get(ent["type"], [])):
                    surviving_types.append(ent["type"])

        combo = tuple(sorted(surviving_types))
        combos[combo] += 1
        record_combos.append(combo)

    unique = sum(1 for c in record_combos if combos[c] == 1)
    return unique / len(record_combos) if record_combos else 0
```

### D. Utility Metrics (Task 2, supplementary)


| Metric              | Definition                                                             | Direction |
| ------------------- | ---------------------------------------------------------------------- | --------- |
| **NLI Consistency** | Fraction of (original, anonymized) pairs classified as entailment [19] | higher    |


One metric, no training, off-the-shelf model (`roberta-large-mnli`). Downstream Δ dropped — not worth the implementation time for a 2-day build.

---

## 3. THE NOVELTY: Privacy-Utility Frontier

This is what elevates the paper from "more metrics on a bigger dataset" to "a new evaluation paradigm." Extends the privacy-utility tradeoff concept from differential privacy theory [23, 24] to text anonymization benchmarking.

### 3a. Formalization

Define the **Privacy-Utility Score (PUS)**:

```
PUS(λ) = λ · Privacy + (1 - λ) · Utility
```

Where:

- `Privacy = (1 - ELR/100)` — fraction of entities successfully masked
- `Utility = BERTScore_F1 / 100` — semantic preservation [2]

Sweep `λ ∈ {0.0, 0.1, 0.2, ..., 1.0}`.

### 3b. Analysis (implement in `analysis/pareto_frontier.py`)

```python
def compute_pareto_frontier(results_dict):
    """
    results_dict: {model_name: {"elr": float, "bertscore": float}}
    Returns: Pareto-optimal models and frontier plot data.
    """
    import numpy as np
    models = list(results_dict.keys())
    points = np.array([(1 - r["elr"]/100, r["bertscore"]/100) for r in results_dict.values()])

    pareto = []
    for i, (px, py) in enumerate(points):
        dominated = any(points[j][0] >= px and points[j][1] >= py and
                        (points[j][0] > px or points[j][1] > py)
                        for j in range(len(points)) if j != i)
        if not dominated:
            pareto.append(models[i])

    return pareto, points

def sweep_pus(results_dict, lambdas=None):
    """Sweep λ and report PUS for each model at each operating point."""
    if lambdas is None:
        lambdas = [i/10 for i in range(11)]
    table = {}
    for name, r in results_dict.items():
        privacy = 1 - r["elr"] / 100
        utility = r["bertscore"] / 100
        table[name] = {f"λ={l:.1f}": round(l * privacy + (1-l) * utility, 4)
                       for l in lambdas}
    return table
```

### 3c. What this produces (Figure 1 in the paper)

A scatter plot with:

- X-axis: Privacy (1 - ELR)
- Y-axis: Utility (BERTScore)
- Each model is a point
- Pareto frontier drawn
- λ iso-curves overlaid

This single figure tells the story: Regex+Faker is high utility / zero privacy. BART-base is high privacy / moderate utility. LLMs fall somewhere. The Pareto frontier shows what's achievable.

**Paper claim:** "We introduce the first formalized privacy-utility tradeoff framework for text anonymization benchmarking, enabling principled comparison of systems at any operating point."

---

## 4. THE NOVELTY: Failure Taxonomy

Not just "where models fail" — a structured categorization that future papers can reuse.

### 4a. Taxonomy (5 categories)


| Category           | Definition                                                                                                                                                   | Detection Method                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Boundary Error** | Entity partially masked: "ohn Smith" instead of "John Smith"                                                                                                 | Gold entity substring partially matches prediction     |
| **Type Confusion** | Entity replaced with wrong type: email replaced with phone number                                                                                            | FPR check: replacement doesn't match type format regex |
| **Ghost Leak**     | Entity string removed but semantically equivalent context remains: "the surgeon from Gothenburg" still present. Related to quasi-identifier linking [10, 11] | CRR-3 on per-entity context window; LRR guess matches  |
| **Over-Masking**   | Non-entity content unnecessarily altered                                                                                                                     | OMR token-level detection                              |
| **Format Break**   | Replacement destroys document structure: "Dear [FULLNAME]" → "Dear 555-0123"                                                                                 | FPR failure on structured types                        |


### 4b. Implementation (`analysis/failure_taxonomy.py`)

```python
def classify_failures(gold_records, predictions):
    counts = {"boundary": 0, "type_confusion": 0, "ghost_leak": 0,
              "over_mask": 0, "format_break": 0, "clean": 0}
    examples = {k: [] for k in counts}

    for g, p in zip(gold_records, predictions):
        orig = g["original_text"]
        pred = p["anonymized_text"]
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0: continue
            ent_text = ent["text"]
            ent_type = ent.get("type", "UNKNOWN")

            leaked = re.search(rf'(?<!\w){re.escape(ent_text)}(?!\w)',
                               pred, re.IGNORECASE)

            if leaked:
                continue

            tokens = re.findall(r'\w+', ent_text)
            partial_leaked = [t for t in tokens if len(t) > 2 and
                              re.search(rf'(?<!\w){re.escape(t)}(?!\w)',
                                        pred, re.IGNORECASE)]
            if partial_leaked and len(partial_leaked) < len(tokens):
                counts["boundary"] += 1
                examples["boundary"].append({
                    "id": g["id"], "entity": ent_text, "type": ent_type,
                    "leaked_tokens": partial_leaked
                })
                continue

            pattern = FORMAT_PATTERNS.get(ent_type)
            if pattern:
                replacement = extract_replacement(orig, pred, ent["start"], ent["end"])
                if replacement and not pattern.search(replacement):
                    counts["format_break"] += 1
                    examples["format_break"].append({
                        "id": g["id"], "entity": ent_text, "type": ent_type,
                        "replacement": replacement
                    })
                    continue

            ctx_start = max(0, ent["start"] - 50)
            ctx_end = min(len(orig), ent["end"] + 50)
            orig_ctx = orig[ctx_start:ctx_end].lower()
            pred_ctx_start = max(0, ent["start"] - 50)
            pred_ctx_end = min(len(pred), ent["end"] + 50)
            if pred_ctx_end <= len(pred):
                pred_ctx = pred[pred_ctx_start:pred_ctx_end].lower()
                ctx_tokens = set(re.findall(r'\w+', orig_ctx)) - set(re.findall(r'\w+', ent_text.lower()))
                pred_tokens = set(re.findall(r'\w+', pred_ctx))
                if ctx_tokens and len(ctx_tokens & pred_tokens) / len(ctx_tokens) > 0.8:
                    counts["ghost_leak"] += 1
                    examples["ghost_leak"].append({
                        "id": g["id"], "entity": ent_text, "type": ent_type,
                        "shared_context": list(ctx_tokens & pred_tokens)[:5]
                    })
                    continue

            counts["clean"] += 1

    return counts, examples
```

### 4c. Paper output (Table + Figure)

**Table: Failure Distribution (BART-base on test set)**


| Category          | Count | % of Entities | Example                                          |
| ----------------- | ----- | ------------- | ------------------------------------------------ |
| Clean replacement |       |               |                                                  |
| Boundary error    |       |               | "ohn Smith" leaked from "John Smith"             |
| Type confusion    |       |               | SSN replaced with "John Doe"                     |
| Ghost leak        |       |               | Entity removed but "47-year-old surgeon" remains |
| Over-masking      |       |               | "the court ruled" → "the building decided"       |
| Format break      |       |               | Email replaced with "555-0123"                   |


**Figure 2:** Stacked bar chart across all baselines showing failure distribution. This is the visual that reviewers remember.

---

## 5. Baselines — 13 Systems (Depth > Breadth)

### A. Rule-based (3 systems, existing)


| System             | Code                                      | Effort   |
| ------------------ | ----------------------------------------- | -------- |
| Regex + Faker      | `regex_faker_baseline.py --mode regex`    | 0 (done) |
| spaCy + Faker [20] | `regex_faker_baseline.py --mode spacy`    | 0 (done) |
| Presidio [21]      | `regex_faker_baseline.py --mode presidio` | 0 (done) |


Modify `regex_faker_baseline.py`: add `--save-spans` to emit detected spans before replacement.

### B. Transformer NER (1 system, new)


| System             | Code                             | Effort   |
| ------------------ | -------------------------------- | -------- |
| BERT-base NER [24] | `baselines/bert_ner_baseline.py` | 4h + GPU |


Fine-tune `bert-base-cased` with `AutoModelForTokenClassification`, BIO scheme. Use 16 clean entity types (drop TITLE, GENDER, NUMBER, OTHER_PII, UNKNOWN from saha_al's 21 — these are noisy catch-all labels that degrade NER training).

### C. Seq2seq Fine-tuned (6 systems, existing)


| System                              | Base Model                                | Source                    |
| ----------------------------------- | ----------------------------------------- | ------------------------- |
| BART-base + PII [4]                 | `facebook/bart-base`                      | `Seq2Seq_model/train3.py` |
| T5-small + PII [5]                  | `google-t5/t5-small`                      | `Seq2Seq_model/train3.py` |
| Flan-T5-small + PII [6]             | `google/flan-t5-small`                    | `Seq2Seq_model/train3.py` |
| DistilBART + PII [7]                | `sshleifer/distilbart-cnn-6-6`            | `Seq2Seq_model/train3.py` |
| T5-efficient-tiny + PII [5]         | `google/t5-efficient-tiny`                | `Seq2Seq_model/train3.py` |
| **Flan-T5-base QLoRA + PII** [6, 9] | `google/flan-t5-base` (4-bit + LoRA r=16) | `Seq2Seq_model/train3.py` |


The 6th model (`flan-t5-base-qlora`) exists in `Seq2Seq_model/config.py` but was previously omitted from the leaderboard. Include it — it's free and demonstrates parameter-efficient fine-tuning [9, 26].

**Important caveat (see Blocker 0c):** Verify that the checkpoints on HuggingFace were trained with the same input format as `kaggle_inference.py` uses. If there's a mismatch, re-run inference without entity markers.

Although i have checkpoints saved for this !!

### D. LLM + Hybrid (2 systems, new)


| System                                         | Code                           | Effort |
| ---------------------------------------------- | ------------------------------ | ------ |
| GPT-4o-mini (zero-shot)                        | `baselines/llm_baseline.py`    | 2h     |
| **Hybrid: spaCy detect + GPT-4o-mini replace** | `baselines/hybrid_baseline.py` | 2h     |


**Hybrid baseline — why it matters:**

The hybrid isolates **detection quality vs generation quality**. If it outperforms both spaCy+Faker and GPT-4o-mini zero-shot, it proves that combining best-in-class detection with best-in-class generation is the optimal strategy. If not, it tells you about error propagation.

```python
def hybrid_anonymize(text, nlp, openai_client):
    doc = nlp(text)
    entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents]
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append((match.start(), match.end(), ent_type, match.group()))
    entities = resolve_overlapping_spans(entities)

    if not entities:
        return text

    entity_list = "\n".join(f"- \"{e[3]}\" (type: {e[2]})" for e in entities)
    prompt = f"""The following entities were detected as PII in this text:
{entity_list}

Original text: {text}

Replace ONLY the listed entities with realistic fake alternatives of the same type. Keep everything else exactly the same. Return only the result."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=512
    )
    return response.choices[0].message.content.strip()
```

### E. Mask-Fill Pipeline (2 systems, cross-dataset — from `pipeline_maskfill/`)


| System                   | Architecture                                                                                                   | Training Data               | Effort              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- | --------------------------- | ------------------- |
| MaskFill-Baseline [8, 6] | DeBERTa-v3-base NER → Flan-T5-base fill (QLoRA [9])                                                            | ai4privacy/pii-masking-400k | 2h (inference only) |
| MaskFill-DP [8, 22, 23]  | Multi-task NER (privacy attention + focal loss [25]) → Flan-T5-base fill + DP fine-tuning (Opacus, ε=8.0 [22]) | ai4privacy/pii-masking-400k | 2h (inference only) |


**Why these matter (high novelty, low cost):**

These models already exist in `pipeline_maskfill/src/` and were trained on a **different dataset** (ai4privacy [pii-masking-400k]). Running them zero-shot on our test set provides:

1. **Cross-dataset transfer evidence** — how well do models trained on external data generalize to SAHA-AL?
2. **The only system with differential privacy guarantees** (MaskFill-DP, ε=8.0 via Opacus [22, 23]) — directly testable on the privacy-utility frontier. If DP training reduces ELR at the cost of BERTScore, we have a concrete data point on the Pareto curve.
3. **Architectural diversity** — mask-fill (detect → mask → generate) vs end-to-end seq2seq.

**Implementation:** Load checkpoints from `pipeline_maskfill/outputs/`, write a thin adapter in `baselines/maskfill_inference.py` that maps their entity types to the benchmark taxonomy (PERSON → {FULLNAME, FIRST_NAME, LAST_NAME}, LOC → LOCATION, etc.). No retraining needed.

**Caveat:** These use 19 entity types collapsed from ai4privacy's taxonomy. Evaluation must use the type mapping from `pipeline_maskfill/src/common.py::ENTITY_MAP` in reverse. Per-type metrics will be approximate for types that collapse (e.g., PERSON covers names at all granularities).

**Existing evaluations to port:** `pipeline_maskfill` already implements several adversarial evaluations:

- `model2_advanced.py::membership_inference_attack()` — MIA [12]
- `model4_semantic.py::evaluate_reidentification_risk()` — re-identification risk assessment
- `eval_prompt_injection.py` — 150 prompt injection examples across 10 attack categories
- `eval_all.py` — multi-granularity CRR, difficulty-stratified leakage

Port MIA to `eval/eval_privacy.py` as a supplementary metric. Acknowledge the prompt injection evaluation in the paper's Related Work as an existing component, but do not include it in the core benchmark metrics (it tests model robustness, not anonymization quality).

---

## 6. Ablations — Trimmed to Essentials

### 6a. Scale + Augmentation (single table)

Train BART-base [4] on 4 configurations. Reports the story without 5 sub-ablations.


| Config                             | ELR | TokRecall | FPR | BERTScore | LRR |
| ---------------------------------- | --- | --------- | --- | --------- | --- |
| 25% gold                           |     |           |     |           |     |
| 100% gold                          |     |           |     |           |     |
| 100% gold + aug                    |     |           |     |           |     |
| 100% gold + aug (entity-swap only) |     |           |     |           |     |


Row 1→2: scale effect. Row 2→3: augmentation effect. Row 4: best single augmentation strategy (entity swap [16]).

**Actual augmentation numbers:** The dataset contains ~~120k total entries (36k gold + ~84k augmented). This is a **~~3.3× expansion** of the gold set, not "12×" as previously documented. The discrepancy arises because `saha_al/config.py` defines multipliers (entity swap ×4 [16], template fill 5000 [17], EDA ×3 [15]) that would yield ~257k augmented entries at full capacity, but only ~84k were retained in the final dataset. The plan and paper should report the actual 3.3× number.

### 6b. Cross-Dataset Transfer to TAB

Run BART-base [4] on TAB test (119 ECHR documents) [1]. Extract spans via diff. Report TAB metrics.


| Metric                | BART-base (ours) | Longformer (TAB paper [1]) |
| --------------------- | ---------------- | -------------------------- |
| Token recall (all)    |                  |                            |
| Token recall (DIRECT) |                  |                            |
| Token recall (QUASI)  |                  |                            |
| Token precision       |                  |                            |
| Token F1              |                  |                            |


This is the single strongest generalization experiment. One table, maximum impact.

**Dropped from previous plan:** Pipeline layer ablation (requires re-running full annotation pipeline — not feasible in 2 days), per-strategy augmentation ablation (5 rows → compressed to 1 row), robustness tests (typo injection, case perturbation — breadth, not depth).

---

## 7. Contributions — Revised (4 Claims)

### Contribution 1: Anonymization-Under-Attack Evaluation Paradigm

Existing benchmarks (TAB [1], i2b2) evaluate anonymization as static text transformation — mask recall and precision. SAHA-AL introduces an **adversarial evaluation paradigm**: every anonymized output is tested against a retrieval adversary (ERA, using sentence embeddings [3]), a generative adversary (LRR, extending LLM-based privacy inference [13]), and a compositional privacy probe (UAC, grounded in k-anonymity theory [10, 11]). This shifts evaluation from "did you mask the right tokens" to "can an attacker still re-identify from your output."

### Contribution 2: Formalized Privacy-Utility Frontier

We introduce PUS(λ), a parameterized privacy-utility score that extends the differential privacy tradeoff concept [23, 24] to text anonymization benchmarking. The Pareto frontier analysis reveals that no existing system dominates across the full tradeoff space, and identifies the optimal operating region for practical deployment.

### Contribution 3: Scalable Semi-Automatic Annotation Pipeline at 3× Throughput

SAHA-AL's 4-layer pipeline achieves entity-level F1=0.83 inter-annotator agreement at ~26 sec/entry (vs TAB's [1] fully manual annotation at minutes/document), producing 36k gold entries with 20 fine-grained PII types. Augmentation via entity swap [16], template fill [17], and EDA [15] extends training data to ~120k entries (3.3× expansion). This represents a 28× scale increase over TAB's 1,355 documents.

**Note on IAA metric:** The `saha_al/utils/iaa.py` implementation computes entity-level set-overlap F1 on (start, end, type) tuples, not Cohen's kappa. The benchmark README previously reported this as "κ=0.83" — this should be corrected to "entity-level agreement F1=0.83" in the paper.

### Contribution 4: Failure Taxonomy for Anonymization Systems

We define a 5-category failure taxonomy (boundary error, type confusion, ghost leak, over-masking, format break) and show that failure distributions vary dramatically across system architectures — seq2seq models produce ghost leaks while NER+replace systems produce boundary errors. This taxonomy provides a reusable diagnostic framework for future anonymization research, analogous to error categorization work in NER [16] and machine translation.

---

## 8. Benchmark Protocol

### Splits

Based on `prepare_dataset.py` with `GOLD_CUTOFF=36000`:

- **train:** ~113k (28.8k gold + ~84k augmented)
- **validation:** ~3.6k (gold only)
- **test:** 3,600 records, 9,271 gold entities (gold only, frozen)

### Submission Formats

- Task 1: `{"id": "...", "detected_entities": [...]}`
- Task 2: `{"id": "...", "anonymized_text": "..."}`
- Task 3: evaluator-computed

### Fairness Rules

1. No test set contamination
2. Report compute (GPU type, training time, inference time)
3. Provide code or HuggingFace link
4. Disclose LLM pre-training data
5. Cross-dataset baselines (pipeline_maskfill) must be clearly labeled as such — they are not trained on SAHA-AL data

---

## 9. Directory Structure

```
benchmark/
├── README.md
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── eval/
│   ├── eval_detection.py          # Task 1: span P/R/F1
│   ├── eval_anonymization.py      # Task 2: ELR, TokRecall, OMR, FPR, BERTScore, NLI
│   ├── eval_privacy.py            # Task 3: CRR-3, ERA, LRR, UAC, MIA
│   ├── bootstrap.py               # 95% CIs [18]
│   └── utils.py                   # Shared: normalization, span matching, format regexes
│
├── baselines/
│   ├── regex_faker_baseline.py    # Regex/spaCy/Presidio (add --save-spans)
│   ├── bert_ner_baseline.py       # BERT-base token classifier [24]
│   ├── seq2seq_inference.py       # 6 fine-tuned seq2seq [4, 5, 6, 7]
│   ├── llm_baseline.py            # GPT-4o-mini zero-shot
│   ├── hybrid_baseline.py         # spaCy detect + GPT-4o-mini replace
│   └── maskfill_inference.py      # pipeline_maskfill Models 1+2 [8, 22]
│
├── analysis/
│   ├── dataset_stats.py           # Dataset statistics
│   ├── pareto_frontier.py         # Privacy-utility frontier + PUS sweep
│   ├── failure_taxonomy.py        # 5-category error classification
│   └── tab_transfer.py            # Cross-dataset eval on TAB [1]
│
├── results/
│   └── eval_*.json
│
├── predictions/
│   └── predictions_*.jsonl
│
└── scripts/
    ├── prepare_dataset.py
    ├── push_to_hf.py
    └── run_all.sh
```

---

## 10. Results Tables (Paper)

### Table 1: Task 2 — Text Anonymization


| Model                     | ELR ↓    | TokRecall ↑ | OMR ↓ | FPR ↑ | BERTScore ↑ |
| ------------------------- | -------- | ----------- | ----- | ----- | ----------- |
| Regex+Faker               | 83.49    |             |       |       | 98.13       |
| spaCy+Faker [20]          | 26.70    |             |       |       | 91.84       |
| Presidio [21]             | 33.86    |             |       |       | 90.02       |
| BERT-base NER [24]        |          |             |       |       |             |
| T5-efficient-tiny [5]     | 4.14     |             |       |       | 92.57       |
| T5-small [5]              | 1.54     |             |       |       | 92.59       |
| Flan-T5-small [6]         | 0.99     |             |       |       | 92.47       |
| DistilBART [7]            | 1.23     |             |       |       | 86.34       |
| BART-base [4]             | **0.93** |             |       |       | **92.74**   |
| Flan-T5-base QLoRA [6, 9] |          |             |       |       |             |
| GPT-4o-mini (0-shot)      |          |             |       |       |             |
| Hybrid (spaCy+GPT)        |          |             |       |       |             |
| MaskFill-Baseline† [8]    |          |             |       |       |             |
| MaskFill-DP† [8, 22]      |          |             |       |       |             |


† Cross-dataset: trained on ai4privacy/pii-masking-400k, evaluated zero-shot on SAHA-AL test.

### Table 2: Task 1 — PII Detection


| Model                                | Exact F1 | Partial F1 | Type-aware F1 |
| ------------------------------------ | -------- | ---------- | ------------- |
| Regex                                |          |            |               |
| spaCy [20]                           |          |            |               |
| Presidio [21]                        |          |            |               |
| BERT-base NER [24]                   |          |            |               |
| MaskFill-Baseline† (DeBERTa NER) [8] |          |            |               |
| BART-base (diff) [4]                 |          |            |               |


### Table 3: Task 3 — Privacy Under Attack


| Model                | CRR-3 ↓ | ERA@1 ↓ | ERA@5 ↓ | LRR exact ↓ | LRR fuzzy ↓ | UAC ↓ |
| -------------------- | ------- | ------- | ------- | ----------- | ----------- | ----- |
| spaCy+Faker [20]     | 40.62   |         |         |             |             |       |
| BART-base [4]        | 34.62   |         |         |             |             |       |
| GPT-4o-mini          |         |         |         |             |             |       |
| Hybrid               |         |         |         |             |             |       |
| MaskFill-DP† [8, 22] |         |         |         |             |             |       |


### Table 4: Per-Entity-Type ELR (top 4 systems)

### Table 5: Scale + Augmentation Ablation

### Table 6: Cross-Dataset Transfer to TAB [1]

### Table 7: Failure Taxonomy Distribution


| Category       | Regex | spaCy | BERT | BART | QLoRA | GPT-4o | Hybrid | MaskFill† |
| -------------- | ----- | ----- | ---- | ---- | ----- | ------ | ------ | --------- |
| Clean          |       |       |      |      |       |        |        |           |
| Boundary error |       |       |      |      |       |        |        |           |
| Type confusion |       |       |      |      |       |        |        |           |
| Ghost leak     |       |       |      |      |       |        |        |           |
| Over-masking   |       |       |      |      |       |        |        |           |
| Format break   |       |       |      |      |       |        |        |           |


### Figure 1: Privacy-Utility Pareto Frontier (all 13 systems)

### Figure 2: Failure Distribution Stacked Bar Chart

### Figure 3: PUS(λ) Sweep Curves

---

## 11. Discussion — Focused on Novelty

1. **String masking ≠ privacy.** If ERA/LRR rankings diverge from ELR rankings, we prove that removing entity strings is insufficient — context enables re-identification. This validates the adversarial evaluation paradigm (Contribution 1) and aligns with recent findings on LLM inference attacks [13].
2. **Ghost leaks dominate seq2seq failures.** Seq2seq models have near-zero ELR but context leakage remains. The failure taxonomy will show this concretely.
3. **DP training on the Pareto frontier.** MaskFill-DP (ε=8.0 [22, 23]) provides the only data point with formal privacy guarantees. If it achieves lower ERA/LRR at the cost of BERTScore, this validates that DP constraints improve anonymization quality — a finding connecting text anonymization to differential privacy theory [24].
4. **Hybrid isolation.** If hybrid (spaCy+GPT) outperforms both components individually, the field should move toward modular architectures. If not, end-to-end generation is the right paradigm.
5. **Pareto frontier reveals no dominant system.** No model is best at all λ. High-privacy applications (medical) need different systems than high-utility applications (analytics). The benchmark enables this choice.
6. **Cross-dataset gap.** TAB [1] transfer reveals the synthetic-to-real domain shift. QUASI recall will be low — honest signal for future work. pipeline_maskfill baselines (trained on ai4privacy) provide a second data point on cross-dataset generalization.

---

## 12. Limitations

1. **Synthetic source data.** TAB cross-eval and pipeline_maskfill cross-dataset baselines are the primary controls.
2. **No quasi-identifier modeling.** UAC is a lightweight proxy grounded in k-anonymity theory [10, 11], not a full model.
3. **No co-reference consistency checking.**
4. **English only.**
5. **LRR depends on GPT-4o-mini capability** — adversary strength may change with model updates [13].
6. **ERA assumes closed candidate pool** [14].
7. **IAA was computed as entity-level F1, not Cohen's kappa** — the original benchmark README mislabeled this.
8. **Augmentation is 3.3× actual (not 12× as documented in saha_al README)** — the config-defined multipliers were not fully applied.
9. **pipeline_maskfill baselines use different training data** — results are cross-dataset and not directly comparable to in-domain baselines without caveats.

---

## 13. Implementation Schedule — 2 Days

### Day 1 (Hours 0–15)


| Hour  | Task                                                                             | Output                                |
| ----- | -------------------------------------------------------------------------------- | ------------------------------------- |
| 0–2   | Fix entity types in `prepare_dataset.py`, rebuild splits                         | Clean test.jsonl with real types      |
| 2–3   | Verify seq2seq format (Blocker 0c): run 50 samples ± markers                     | Confirmed or fixed inference pipeline |
| 3–4   | `analysis/dataset_stats.py`                                                      | Stats JSON + paper table              |
| 4–8   | `eval/eval_anonymization.py` (TokRecall, OMR, FPR, BERTScore, NLI)               | Complete Task 2 evaluator             |
| 8–9   | Run Task 2 eval on all 8 existing baselines + QLoRA                              | Fill Table 1 cells                    |
| 9–10  | `eval/bootstrap.py` [18]                                                         | CIs for all metrics                   |
| 10–12 | `baselines/llm_baseline.py` (GPT-4o-mini)                                        | GPT-4o-mini predictions               |
| 12–14 | `baselines/hybrid_baseline.py` (spaCy + GPT-4o-mini)                             | Hybrid predictions                    |
| 14–15 | `baselines/bert_ner_baseline.py` [24] — setup + launch training (runs overnight) | Training started                      |


### Day 2 (Hours 15–32)


| Hour  | Task                                                                                | Output                             |
| ----- | ----------------------------------------------------------------------------------- | ---------------------------------- |
| 15–16 | Collect BERT NER results, run Task 1 + Task 2 eval                                  | BERT row in Tables 1+2             |
| 16–17 | `eval/eval_detection.py` (span P/R/F1) + modify regex_faker `--save-spans`          | Task 1 evaluator + detection preds |
| 17–18 | Run Task 1 eval on all span-producing baselines                                     | Fill Table 2                       |
| 18–21 | `eval/eval_privacy.py` — ERA + LRR + UAC + MIA                                      | Task 3 evaluator                   |
| 21–22 | Run Task 3 eval on top 5 systems                                                    | Fill Table 3                       |
| 22–24 | `analysis/pareto_frontier.py` + `analysis/failure_taxonomy.py`                      | Figure 1 + Table 7                 |
| 24–26 | `analysis/tab_transfer.py` — cross-dataset eval on TAB [1]                          | Table 6                            |
| 26–28 | Scale ablation: train BART on {25%, 100%, 100%+aug}                                 | Table 5                            |
| 28–30 | `baselines/maskfill_inference.py` — load pipeline_maskfill checkpoints, run on test | MaskFill rows in Tables 1-3        |
| 30–32 | Final: fill all table cells, generate figures, write README leaderboard             | Done                               |


**Total: 32 hours across 2 days.**

pipeline_maskfill baselines (Hours 28–30) are **stretch goals** — they depend on having saved checkpoints available. If checkpoints are not accessible, cut these and report 11 baselines instead of 13.

The three novelty items (LRR [13], Pareto frontier, failure taxonomy) are scheduled on Day 2 Hours 18–24. Everything before that is infrastructure that enables them.

---

## 14. What Was Cut (and Why)


| Cut                                   | Reason                                                                                                                            |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| RoBERTa NER baseline                  | One transformer NER is enough. Breadth.                                                                                           |
| Llama-3-8B baseline                   | One LLM is enough. Use the better one (GPT-4o-mini).                                                                              |
| CIA metric                            | Replaced by stronger LRR (generative adversary > classifier probe) [13].                                                          |
| RC metric                             | Overlaps with FPR. Removed redundancy.                                                                                            |
| Downstream Δ utility                  | NLI consistency [19] is sufficient and requires no training.                                                                      |
| Typo injection robustness             | Breadth item. Not in the depth story.                                                                                             |
| Case perturbation robustness          | Same.                                                                                                                             |
| Pipeline layer ablation               | Requires re-running annotation pipeline. Day+ effort.                                                                             |
| Per-strategy augmentation ablation    | Compressed to single swap-only row in scale table [16].                                                                           |
| pipeline_maskfill Model 3 (rephraser) | Model 1+2 provide sufficient architectural diversity. Model 3 adds a third stage (rephrase) that would need separate explanation. |
| pipeline_maskfill Model 4 (semantic)  | End-to-end paraphraser overlaps with seq2seq baselines.                                                                           |
| Prompt injection evaluation           | Already in pipeline_maskfill; tests model robustness, not anonymization quality. Acknowledge in Related Work only.                |


---

## 15. References

[1] Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M. (2022). The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization. *Computational Linguistics*, 48(4), 1053–1101.

[2] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.

[3] Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*, 3982–3992.

[4] Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *ACL*, 7871–7880.

[5] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*, 21(140), 1–67.

[6] Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. (2024). Scaling Instruction-Finetuned Language Models. *JMLR*, 25(70), 1–53.

[7] Shleifer, S. & Rush, A. M. (2020). Pre-trained Summarization Distillation. *arXiv:2010.13002*.

[8] He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *ICLR*.

[9] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS*.

[10] Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy. *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*, 10(5), 557–570.

[11] Sweeney, L. (2000). Simple Demographics Often Identify People Uniquely. *Carnegie Mellon University, Data Privacy Working Paper 3*.

[12] Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE Symposium on Security and Privacy*, 3–18.

[13] Staab, R., Vero, M., Balunović, M., & Vechev, M. (2023). Beyond Memorization: Violating Privacy Via Inference with Large Language Models. *arXiv:2310.07298*.

[14] Narayanan, A. & Shmatikov, V. (2008). Robust De-anonymization of Large Sparse Datasets. *IEEE Symposium on Security and Privacy*, 111–125.

[15] Wei, J. & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. *EMNLP*, 6382–6388.

[16] Dai, X. & Adel, H. (2020). An Analysis of Simple Data Augmentation for Named Entity Recognition. *COLING*, 3861–3867.

[17] Anaby-Tavor, A., Carmeli, B., Goldbraich, E., Kanber, A., Kour, G., Shlomov, S., Tepper, N., & Zwerdling, N. (2020). Do Not Have Enough Data? Deep Learning to the Rescue! *AAAI*, 7383–7390.

[18] Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

[19] Williams, A., Nangia, N., & Bowman, S. R. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. *NAACL*, 1112–1122.

[20] Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-Strength Natural Language Processing in Python. *Zenodo*.

[21] Microsoft (2020). Presidio — Data Protection and De-identification SDK. *GitHub: microsoft/presidio*.

[22] Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Maber, M., Lipman, H., Zovich, S., & Çilingir, E. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. *arXiv:2109.12298*.

[23] Dwork, C. (2006). Differential Privacy. *ICALP*, Lecture Notes in Computer Science, 4052, 1–12.

[24] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*, 4171–4186.

[25] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*, 2980–2988.

[26] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
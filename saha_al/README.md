# 🛡️ SAHA-AL: Semi-Automatic Human-Augmented Active Learning Pipeline

> **Dataset Creation Pipeline for Privacy-Preserving Text Anonymization**
>
> Part of the project: *Privacy-Preserving Text Anonymization via Decoupled Mask-and-Fill Architecture*

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Dataset Summary](#dataset-summary)
4. [Entity Taxonomy](#entity-taxonomy)
5. [Layer-by-Layer Walkthrough](#layer-by-layer-walkthrough)
   - [Layer 1 — Pre-Annotation Engine](#layer-1--pre-annotation-engine)
   - [Layer 2 — Confidence-Based Routing](#layer-2--confidence-based-routing)
   - [Layer 3 — Active Learning](#layer-3--active-learning-bootstrap-ner)
   - [Layer 4 — Streamlit Annotation Interface](#layer-4--streamlit-annotation-interface)
6. [Phase 2 — Data Augmentation](#phase-2--data-augmentation)
   - [Strategy 1: Entity Swap](#strategy-1-entity-swap-dai--adel-coling-2020)
   - [Strategy 2: Template Fill](#strategy-2-template-fill-anaby-tavor-et-al-aaai-2020)
   - [Strategy 3: EDA](#strategy-3-eda--easy-data-augmentation-wei--zou-emnlp-2019)
7. [Quality Assurance](#quality-assurance)
8. [How to Run](#how-to-run)
9. [Directory Structure](#directory-structure)
10. [Efficiency Innovations](#efficiency-innovations)
11. [Dataset Analysis & Reporting](#dataset-analysis--reporting)
12. [Annotation Guidelines Summary](#annotation-guidelines-summary)
13. [Output Schema](#output-schema)

---

## Overview

**SAHA-AL** is a 4-layer annotation pipeline designed to efficiently create a high-quality gold-standard anonymization dataset from raw text. The pipeline combines **automated NER** (spaCy + regex), **confidence-based routing**, **active learning**, and a **custom Streamlit annotation interface** to maximize annotator throughput while maintaining data quality.

A **Phase 2 augmentation stage** then expands the gold standard using three peer-reviewed strategies — **Entity Swap** (Dai & Adel, COLING 2020), **Template Fill** (Anaby-Tavor et al., AAAI 2020), and **EDA** (Wei & Zou, EMNLP 2019) — producing a training set **~12× larger** than the manually annotated core.

### The Problem

We need a gold-standard dataset of `(original_text, anonymized_text, entities)` triples where every PII span is:
- **Detected** (character-level start/end offsets)
- **Classified** (into one of 21 entity types)
- **Replaced** (with a realistic, format-preserving Faker substitute)

Manually annotating 120K+ entries from scratch is infeasible. SAHA-AL solves this by **pre-annotating ~85% of entities automatically** and routing only uncertain cases to human annotators.

### Key Numbers

| Metric | Value |
|--------|-------|
| Source dataset | AI4Privacy (English subset) |
| Total entries | 120,333 |
| Entity types | 21 PII categories |
| Annotators | 4 team members |
| Estimated throughput | ~10,000 entries in 2.5 weeks |
| Auto-approved (GREEN) | ~55–65% of entries |
| Human-reviewed (YELLOW) | ~25–30% of entries |
| Expert-reviewed (RED) | ~10–15% of entries |
| Augmentation multiplier | ~12× gold standard (Entity Swap ×4 + Template Fill ×5 + EDA ×3) |

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          SAHA-AL PIPELINE                                │
│                                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐        │
│  │   Layer 1    │    │   Layer 2    │    │      Layer 3          │        │
│  │  Pre-Annot.  │───▶│   Routing    │───▶│   Active Learning     │        │
│  │ spaCy+Regex  │    │ GREEN/Y/RED  │    │  (Bootstrap NER)      │        │
│  └──────────────┘    └──────────────┘    └───────────────────────┘        │
│                              │                       │                    │
│                              ▼                       ▼                    │
│                     ┌──────────────────────────────────────┐              │
│                     │          Layer 4                      │              │
│                     │   Streamlit Annotation Interface      │              │
│                     │  ┌────────┬─────────┬──────────────┐ │              │
│                     │  │Annotate│  Stats  │Flagged Review│ │              │
│                     │  └────────┴─────────┴──────────────┘ │              │
│                     └──────────────────────────────────────┘              │
│                              │                                            │
│                              ▼                                            │
│                     ┌──────────────────┐                                  │
│                     │  Gold Standard   │                                  │
│                     │ (gold_standard   │                                  │
│                     │    .jsonl)       │                                  │
│                     └────────┬─────────┘                                  │
│                              │                                            │
│              ┌───────────────┼────────────────┐                           │
│              ▼               ▼                ▼                           │
│  ┌───────────────┐ ┌─────────────────┐ ┌──────────────┐                  │
│  │ Entity Swap   │ │ Template Fill   │ │    EDA       │                  │
│  │ Dai & Adel    │ │ Anaby-Tavor     │ │ Wei & Zou   │                  │
│  │ COLING 2020   │ │ AAAI 2020       │ │ EMNLP 2019  │                  │
│  │    ×4         │ │    ×5           │ │    ×3        │                  │
│  └───────┬───────┘ └────────┬────────┘ └──────┬───────┘                  │
│          └──────────────────┼─────────────────┘                           │
│                             ▼                                             │
│                    ┌──────────────────┐                                   │
│                    │  Training Data   │                                   │
│                    │ (training_data   │                                   │
│                    │    .jsonl)       │                                   │
│                    │  gold + ~12×aug  │                                   │
│                    └──────────────────┘                                   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Summary

### Source Data

- **File:** `data/original_data.jsonl`
- **Size:** 120,333 entries
- **Schema:** `{ "entry_id": int, "original_text": str }`
- **Origin:** AI4Privacy dataset (English subset) — crowd-sourced text samples containing synthetic PII

### Output Data (Gold Standard)

- **File:** `data/gold_standard.jsonl`
- **Schema:** See [Output Schema](#output-schema)
- **Contains:** Fully annotated entries with detected entities, character offsets, entity types, Faker replacements, and anonymized text

---

## Entity Taxonomy

We define **21 PII entity types** organized by sensitivity:

| # | Entity Type | Description | Example | Detection |
|---|-------------|-------------|---------|-----------|
| 1 | `FULLNAME` | Full person name | John Smith | spaCy PERSON |
| 2 | `FIRST_NAME` | Given name only | John | Manual / context |
| 3 | `LAST_NAME` | Family name only | Smith | Manual / context |
| 4 | `ID_NUMBER` | Generic ID (CURP, DNI, etc.) | MARGA7101160M9183 | Regex (alphanumeric) |
| 5 | `PASSPORT` | Passport number | AB1234567 | Regex |
| 6 | `SSN` | Social Security Number | 123-45-6789 | Regex (high priority) |
| 7 | `PHONE` | Phone number (any format) | +1 (555) 123-4567 | Regex (3 patterns) |
| 8 | `EMAIL` | Email address | user@example.com | Regex (highest priority) |
| 9 | `ADDRESS` | Street / postal address | 123 Main St, Apt 4 | spaCy FAC + context |
| 10 | `DATE` | Calendar date | March 15, 2024 | spaCy DATE + Regex (6 patterns) |
| 11 | `TIME` | Time of day | 5:52:48 AM | Regex (4 patterns) |
| 12 | `LOCATION` | City / state / country | San Francisco, CA | spaCy GPE/LOC |
| 13 | `ORGANIZATION` | Company / institution | Google Inc. | spaCy ORG |
| 14 | `ACCOUNT_NUMBER` | Bank / financial account | 3847261590142 | Regex (10–18 digits) |
| 15 | `CREDIT_CARD` | Credit card number | 4532-1234-5678-9012 | Regex (16 digits) |
| 16 | `ZIPCODE` | Postal / ZIP code | 43431-9599 | Regex |
| 17 | `TITLE` | Honorific | Mr, Dr, Prof | Lookup set |
| 18 | `GENDER` | Gender reference | Male, Female | Lookup set |
| 19 | `NUMBER` | Numeric PII (amounts, etc.) | $50,000 | Context |
| 20 | `OTHER_PII` | Any PII not in above categories | Ethnicity, religion | Manual |
| 21 | `UNKNOWN` | Unclassified (needs review) | — | Fallback |

---

## Layer-by-Layer Walkthrough

### Layer 1 — Pre-Annotation Engine

**File:** `layer1_preannotation.py`

**Purpose:** Automatically detect and tag PII entities in every entry using two complementary systems.

#### Step 1: spaCy NER (`en_core_web_lg`)

- Runs the spaCy `en_core_web_lg` model on each text
- Maps spaCy labels to our taxonomy via `SPACY_TO_SAHA`:
  - `PERSON` → `FULLNAME`
  - `GPE` → `LOCATION`
  - `ORG` → `ORGANIZATION`
  - `DATE` → `DATE`
  - `TIME` → `TIME`
  - `FAC` → `ADDRESS`
  - Non-PII labels (`CARDINAL`, `MONEY`, etc.) → skipped
- **Strength:** Good at names, organizations, locations, dates in natural language

#### Step 2: Regex Pattern Matching

- **23 regex patterns** with priority scores (1–10) defined in `utils/regex_patterns.py`
- Patterns are ordered by specificity:
  - **Priority 10:** Email, SSN, ISO dates, placeholder tokens
  - **Priority 8–9:** Credit cards, phones, verbal dates, times
  - **Priority 5–7:** Generic IDs, ZIP codes, account numbers
- **Strength:** Catches structured PII that spaCy misses (emails, SSNs, phone numbers, IDs)

#### Step 3: Merge & Deduplicate

- `utils/merger.py` combines both detection lists
- **Overlap resolution:**
  - **Identical span + type** from both sources → `agreement: "full"` (highest confidence)
  - **Overlapping but different** → `agreement: "partial"` (winner = higher priority)
  - **No overlap** → `agreement: "single_source"`
- Each entity gets: `{text, start, end, entity_type, source, confidence, agreement}`

#### Step 4: Faker Replacement Generation

- `utils/faker_replacements.py` generates 3 replacement candidates per entity
- **Multi-locale support:** en_US, en_GB, fr_FR, de_DE, es_MX, ja_JP, ru_RU
- **Format-preserving:** Phone separators, date formats, ID character classes are preserved
- First candidate is auto-selected; annotators can override in Layer 4

#### Step 5: Build Anonymized Text

- Entities processed right-to-left (preserves character offsets)
- Original PII spans replaced with Faker substitutes
- Output: complete `anonymized_text` string

**Output:** `data/pre_annotated.jsonl` — every entry enriched with entities, replacements, anonymized text, and metadata.

**Run:**
```bash
python scripts/run_layer1.py --limit 1000  # test on first 1000
python scripts/run_layer1.py               # full dataset (resume-safe)
```

---

### Layer 2 — Confidence-Based Routing

**File:** `layer2_routing.py`

**Purpose:** Triage every pre-annotated entry into one of three queues based on detection confidence and agreement levels.

#### Routing Rules

| Queue | Criteria | Action |
|-------|----------|--------|
| 🟢 **GREEN** | All entities: confidence ≥ 0.85, no `partial` agreement, no `UNKNOWN` types | Auto-approved (spot-checked) |
| 🟡 **YELLOW** | Medium confidence, some ambiguity, or minor disagreements | Human review required |
| 🔴 **RED** | Any entity: confidence < 0.50, OR >30% `UNKNOWN` types, OR ≥2 `partial` agreements | Expert annotator review |

#### Routing Logic (pseudocode)

```
if min_confidence < 0.50 → RED
if unknown_ratio > 0.30 → RED
if partial_disagreements ≥ 2 → RED
if min_confidence ≥ 0.85 AND no_partial AND no_unknown → GREEN
else → YELLOW
```

**Output:** Three queue files:
- `data/queues/green_queue.jsonl`
- `data/queues/yellow_queue.jsonl`
- `data/queues/red_queue.jsonl`

**Run:**
```bash
python scripts/run_layer2.py
```

---

### Layer 3 — Active Learning (Bootstrap NER)

**File:** `layer3_active_learning.py`

**Purpose:** Prioritize annotation effort by showing the most uncertain entries first.

#### Bootstrap Training

Once ≥200 gold-standard entries are annotated:
1. Gold entries are converted to spaCy training format
2. A lightweight blank `en` NER model is trained (10 epochs)
3. Model is saved to `models/bootstrap/bootstrap_YYYYMMDD_HHMMSS/`
4. Retraining triggered every 500 new gold entries

#### Uncertainty Scoring

For each entry in YELLOW/RED queues:
1. Run the bootstrap model on the original text
2. Compare predicted entity count vs. pre-annotation count
3. **Uncertainty = |pre_count - boot_count| / max(pre_count, boot_count)**
4. Higher uncertainty → shown to annotator first

This ensures annotators spend time on the **hardest, most valuable** examples.

**Run:**
```bash
python scripts/run_bootstrap_train.py train     # train model
python scripts/run_bootstrap_train.py score     # re-sort queues
```

---

### Layer 4 — Streamlit Annotation Interface

**File:** `layer4_app.py`

**Purpose:** A full-featured web-based annotation tool for human review.

#### Features

| Feature | Description |
|---------|-------------|
| **Highlighted Text** | Original text with color-coded PII spans (each entity type has a unique color) |
| **Entity Review Table** | Per-entity row with: text, type dropdown, confidence badge, replacement editor, delete button |
| **Type Correction** | Dropdown to change entity type from all 21 options |
| **Replacement Editing** | Inline text input to modify Faker replacements |
| **Regenerate Replacements** | One-click to generate fresh Faker candidates |
| **Three Actions** | ✅ Accept (save to gold) / 🚩 Flag (expert review) / ⏭️ Skip |
| **Queue Navigation** | Switch between GREEN/YELLOW/RED queues |
| **Statistics Tab** | Entity distribution charts, annotator contributions, annotation log |
| **Flagged Review Tab** | View all flagged entries with reasons |
| **Auto-backup** | Gold standard backed up every 50 annotations |
| **Session Tracking** | Annotator name, per-session count, timestamps |

#### Screenshot Layout

```
┌─ Sidebar ──────────┬─ Main Panel ─────────────────────────────────┐
│ 🛡️ SAHA-AL         │  📝 Annotation │ 📊 Statistics │ 🚩 Flagged  │
│                     ├─────────────────────────────────────────────┤
│ 👤 Annotator: ___   │  Entry 42 of 3801         [progress bar]    │
│                     │  ID: 58923  Queue: 🟡 YELLOW                │
│ 📂 Queue: YELLOW    │                                             │
│ 🔄 Reload           │  📄 Original Text (highlighted PII):        │
│                     │  ┌─────────────────────────────────────┐    │
│ 📊 Queue Sizes      │  │ My name is [John Smith] and I live  │    │
│ 🟢 GREEN:  67,412   │  │ at [123 Main St], [NYC]. Call me    │    │
│ 🟡 YELLOW: 38,201   │  │ at [+1-555-0123]. SSN: [123-45-678]│    │
│ 🔴 RED:    14,720   │  └─────────────────────────────────────┘    │
│                     │                                             │
│ ✅ Gold: 1,247      │  🔍 Entity Review:                          │
│ 📝 Session: 83      │  Text        │ Type    │Conf│ Replacement   │
│                     │  John Smith  │FULLNAME │🟢.92│ Jane Doe     │
│ 💾 Backup           │  123 Main St │ADDRESS  │🟡.71│ 456 Oak Ave  │
│                     │  NYC         │LOCATION │🟢.88│ Boston       │
│                     │  +1-555-0123 │PHONE    │🟢.90│ +1-555-9876  │
│                     │  123-45-6789 │SSN      │🟢.94│ 987-65-4321  │
│                     │                                             │
│                     │  ⚡ [✅ Accept] [🚩 Flag ___] [⏭️ Skip]     │
└─────────────────────┴─────────────────────────────────────────────┘
```

**Run:**
```bash
streamlit run saha_al/layer4_app.py
```

---

## Phase 2 — Data Augmentation

**File:** `augmentation.py` &nbsp;|&nbsp; **Script:** `scripts/run_augmentation.py`

After gold-standard annotation is complete (or at checkpoints), the augmentation module expands the verified dataset using three peer-reviewed strategies. Augmentation is applied **ONLY to verified gold entries** to prevent error propagation.

| Technique | Reference | Multiplier | What changes | What stays |
|-----------|-----------|------------|--------------|------------|
| Entity Swap | Dai & Adel, COLING 2020 | ×4 | All PII values | Sentence structure |
| Template Fill | Anaby-Tavor et al., AAAI 2020 | ×5 | Sentence structure + entity combos | Entity value format |
| EDA | Wei & Zou, EMNLP 2019 | ×3 | Non-PII context words | All PII spans exactly |

### Why These Three?

We evaluated five candidate strategies and selected the three that best cover **orthogonal failure modes** without requiring heavy external dependencies:

| Strategy | Selected? | Rationale |
|----------|-----------|-----------|
| **Entity Swap** | ✅ | Directly prevents entity-value memorisation — highest impact for NER |
| **Template Fill** | ✅ | Creates novel entity co-occurrences + structural diversity; zero external deps |
| **EDA** | ✅ | Lightweight, proven context-diversity boost; no model needed |
| Back-Translation | ❌ | Requires translation API or large seq2seq model — heavy dependency for marginal gain |
| Contextual MLM | ❌ | Needs a loaded LLM (BERT/RoBERTa) at augmentation time — overkill for this task |

### Strategy 1: Entity Swap (Dai & Adel, COLING 2020)

**What it does:** Creates N copies of each gold entry where every PII span is replaced with a **new** Faker value of the same entity type. The surrounding sentence is kept verbatim.

```
Gold:     "My name is Igorche Ramtin Eshekary and my SSN is 123-45-6789."
Swap #1:  "My name is François Michel Dupont and my SSN is 847-29-3156."
Swap #2:  "My name is Yuki Tanaka Sato and my SSN is 562-81-4073."
Swap #3:  "My name is Carlos Rivera López and my SSN is 391-67-8204."
Swap #4:  "My name is Aisha Patel Nair and my SSN is 204-53-7189."
```

**Why it works:** Without swap, the model could learn that *"123-45-6789"* specifically is an SSN. After swapping, the model is forced to learn that **any `XXX-XX-XXXX` pattern in this position** is an SSN — it learns the *pattern*, not the *value*.

**Implementation details:**
- Entities are processed right-to-left to preserve character offsets
- After multi-entity swaps, offsets are re-anchored by sequential string search
- Each variant gets an independent anonymised version with fresh Faker values
- Format-preserving: phone separators, date formats, ID character classes are matched

### Strategy 2: Template Fill (Anaby-Tavor et al., AAAI 2020)

**What it does:** Builds entity pools from the gold standard, defines 40 sentence templates across diverse domains, and fills them with randomly sampled entities.

```
Template:  "{FULLNAME} (SSN: {SSN}, Phone: {PHONE}) registered on {DATE}."
Pool:      FULLNAME → ["Maria Garcia", "Ahmed Hassan", ...]
           SSN      → ["123-45-6789", "987-65-4321", ...]
           PHONE    → ["805 099 8045", "+52 55 1234 5678", ...]
           DATE     → ["March 15, 2024", "4th August 1942", ...]

Generated: "Ahmed Hassan (SSN: 987-65-4321, Phone: 805 099 8045) registered on March 15, 2024."
```

**Why it works:** In the gold data, "Ahmed Hassan" only appears with *his* specific phone and SSN. After template fill, the model sees his name combined with completely different identifiers — breaking **spurious co-occurrence correlations**.

**40 templates** spanning:
- Immigration / legal (appointments, case references, passport verification)
- Financial (account flags, transfers, credit card charges, tax returns)
- Administrative (registrations, reservations, memberships)
- Simple patterns (single/dual entity for short-text robustness)
- Complex multi-entity (4–6 entities per sentence)

**Pool expansion:** Small pools are topped up with Faker-generated values (30 extra per type) to ensure diversity even when the gold standard is still small.

### Strategy 3: EDA — Easy Data Augmentation (Wei & Zou, EMNLP 2019)

**What it does:** Applies four lightweight word-level operations to the **non-PII context** surrounding entities. Entity spans are *never* touched.

| Operation | Code | Example |
|-----------|------|---------|
| **Synonym Replacement (SR)** | Replace ≈ α×n context words with synonyms | "Please **send** to user@mail.com" → "Please **forward** to user@mail.com" |
| **Random Insertion (RI)** | Insert synonym of a random word at random position | "The deadline is March 15" → "The **cutoff** deadline is March 15" |
| **Random Swap (RS)** | Swap two random context words | "**Contact** {NAME} **immediately**" → "**Immediately** {NAME} **contact**" |
| **Random Deletion (RD)** | Delete each context word with probability α | "Please kindly **also** verify the SSN" → "Please kindly verify the SSN" |

**Parameter α** (default 0.1): Controls the fraction of context words affected per operation. Lower α = subtler changes. Higher α = more aggressive augmentation.

**Why it works:** The gold data might contain "Please send to {EMAIL} for verification" 50 times. Without EDA, the model overfits to those exact context words. After EDA, it also sees "Kindly forward to {EMAIL} to be verified", "Please dispatch to {EMAIL} for confirmation", etc. The model learns that EMAIL detection depends on the **format** of the entity, not the specific surrounding words.

**Implementation details:**
- Text is decomposed into alternating (non-entity, entity) segments
- EDA operations are applied **only** to non-entity segments
- Entity offsets are recalculated deterministically during reassembly (no drift)
- Built-in synonym dictionary with 40+ common context words (no WordNet dependency)
- Each variant uses one randomly selected EDA operation

### Augmentation Pipeline Flow

```
Gold Standard (N entries)
        │
        ├──▶ Entity Swap (×4)       → N × 4 variants
        │     └ new PII values, same context
        │
        ├──▶ Template Fill (×5)     → ~5 × N entries (or custom count)
        │     └ new sentences from templates + entity pools
        │
        └──▶ EDA (×3)              → N × 3 variants
              └ modified context words, same PII spans
              │
              ▼
        augmented_data.jsonl  (~12N entries)
              │
              ▼
        gold + augmented → shuffle → training_data.jsonl  (~13N total)
```

### Expected Output Sizes

| Gold Standard | Swap ×4 | Template ×5 | EDA ×3 | Augmented Total | Training Set |
|---|---|---|---|---|---|
| 1,000 | 4,000 | 5,000 | 3,000 | ~12,000 | ~13,000 |
| 5,000 | 20,000 | 25,000 | 15,000 | ~60,000 | ~65,000 |
| 10,000 | 40,000 | 50,000 | 30,000 | ~120,000 | ~130,000 |

### Run Augmentation

```bash
# All three strategies with default multipliers
python scripts/run_augmentation.py --strategy all

# Only Entity Swap with custom multiplier
python scripts/run_augmentation.py --strategy swap --multiplier 6

# Only Template Fill with custom count
python scripts/run_augmentation.py --strategy template --count 20000

# Only EDA with custom multiplier and α
python scripts/run_augmentation.py --strategy eda --multiplier 5 --alpha 0.15

# Custom output path
python scripts/run_augmentation.py --strategy all --output data/my_augmented.jsonl
```

---

## Quality Assurance

### Automated Quality Checks (`utils/quality_checks.py`)

Run on every **Accept** action (non-blocking — saves regardless, but logs warnings):

| Check | Severity | What it catches |
|-------|----------|-----------------|
| **Leakage** | 🔴 High | Original PII text still present in anonymized output |
| **Format Preservation** | 🟡 Medium | Replacement doesn't match expected format (e.g., email → non-email) |
| **Replacement Validity** | 🔴 High | Missing replacement or replacement identical to original |

### Inter-Annotator Agreement (`utils/iaa.py`)

- Entity-level precision, recall, F1 computed between annotator pairs
- Measures agreement on `(start, end, entity_type)` tuples
- Used for periodic quality audits

### Automatic Backups

- Gold standard auto-backed up every 50 annotations
- Manual backup button in sidebar
- Timestamped backups in `data/backups/`

---

## How to Run

> 📖 **For a comprehensive step-by-step guide** with full CLI reference, troubleshooting, and FAQ, see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

### Prerequisites

```bash
# Python 3.10+
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r saha_al/requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### Full Pipeline Execution

```bash
# Step 1: Pre-annotate all entries (resume-safe)
python saha_al/scripts/run_layer1.py --limit 5000   # test first
python saha_al/scripts/run_layer1.py                  # full run

# Step 2: Route into GREEN/YELLOW/RED queues
python saha_al/scripts/run_layer2.py

# Step 3: Launch annotation interface
streamlit run saha_al/layer4_app.py

# Step 4 (after 200+ gold entries): Train bootstrap model
python saha_al/scripts/run_bootstrap_train.py train

# Step 5: Re-sort queues by uncertainty
python saha_al/scripts/run_bootstrap_train.py score

# Step 6: Augment gold standard (Phase 2)
python saha_al/scripts/run_augmentation.py --strategy all

# Step 7: Export statistics report
python saha_al/scripts/export_report_stats.py
```

---

## Directory Structure

```
saha_al/
├── __init__.py                          # Package init
├── config.py                            # Central configuration (all paths, thresholds, constants)
├── layer1_preannotation.py              # Layer 1: spaCy + regex pre-annotation engine
├── layer2_routing.py                    # Layer 2: Confidence-based routing (GREEN/YELLOW/RED)
├── layer3_active_learning.py            # Layer 3: Bootstrap NER training + uncertainty scoring
├── layer4_app.py                        # Layer 4: Streamlit annotation interface
├── augmentation.py                      # Phase 2: Data augmentation (Entity Swap, Template Fill, EDA)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── HOW_TO_RUN.md                        # Detailed setup & run guide
│
├── utils/
│   ├── __init__.py
│   ├── entity_types.py                  # Entity schema with colors, descriptions, examples
│   ├── regex_patterns.py                # 23 regex patterns for structured PII detection
│   ├── faker_replacements.py            # Multi-locale Faker replacement generator
│   ├── merger.py                        # Merge spaCy + regex detections, resolve overlaps
│   ├── quality_checks.py               # Post-annotation quality checks (leakage, format, validity)
│   ├── io_helpers.py                    # JSONL read/write/append, backup, id tracking
│   └── iaa.py                           # Inter-annotator agreement computation
│
├── scripts/
│   ├── run_layer1.py                    # CLI: Run Layer 1 pre-annotation
│   ├── run_layer2.py                    # CLI: Run Layer 2 routing
│   ├── run_bootstrap_train.py           # CLI: Layer 3 train/score commands
│   ├── run_augmentation.py              # CLI: Phase 2 data augmentation
│   ├── export_report_stats.py           # CLI: Export JSON + Markdown report
│   └── dataset_analysis.py             # CLI: Generate charts, tables, and statistics
│
├── data/
│   ├── original_data.jsonl              # Source data (120,333 entries)
│   ├── pre_annotated.jsonl              # Layer 1 output
│   ├── gold_standard.jsonl              # Final annotated dataset
│   ├── skipped.jsonl                    # Skipped entries
│   ├── flagged.jsonl                    # Flagged for expert review
│   ├── augmented_data.jsonl             # Phase 2 augmentation output
│   ├── training_data.jsonl              # Gold + augmented combined (final training set)
│   ├── queues/
│   │   ├── green_queue.jsonl            # Auto-approved entries
│   │   ├── yellow_queue.jsonl           # Needs human review
│   │   └── red_queue.jsonl              # Needs expert review
│   └── backups/                         # Timestamped gold standard backups
│
├── models/
│   └── bootstrap/                       # Saved bootstrap NER models
│
├── reports/                             # Generated by dataset_analysis.py
│   ├── figures/                         # PNG charts (entity dist, routing, heatmap, etc.)
│   ├── tables/                          # LaTeX table snippets (.tex)
│   ├── dataset_analysis_report.md       # Human-readable Markdown report
│   └── dataset_statistics.json          # Machine-readable stats
│
└── logs/
    ├── annotation_log.jsonl             # Per-action log (accept/flag/skip)
    └── model_versions.jsonl             # Bootstrap model training log
```

---

## Efficiency Innovations

### 1. Hybrid Detection (spaCy + Regex)

Instead of relying on a single NER model, we combine:
- **spaCy** for contextual entities (names, organizations, locations)
- **23 priority-ranked regex patterns** for structured entities (emails, SSNs, phones, IDs)

This gives **~85% entity recall** before any human touches the data.

### 2. Confidence-Based Routing

Not all entries need the same level of attention:
- **GREEN (~60%):** Both systems agree with high confidence → auto-approved, spot-checked
- **YELLOW (~25%):** Some ambiguity → quick human review
- **RED (~15%):** Genuine disagreement → careful expert review

This means annotators spend **zero time on easy entries** and focus entirely on the hard cases.

### 3. Active Learning with Bootstrap NER

After initial annotations, a lightweight NER model is trained on gold data. It then **scores remaining entries by prediction uncertainty**, ensuring annotators always see the most informative examples first. This maximizes the value of each human annotation.

### 4. Format-Preserving Replacements

Faker replacements maintain the **structural format** of the original PII:
- Phone `(555) 123-4567` → `(555) 987-6543` (same separator pattern)
- Date `March 15, 2024` → `July 22, 1998` (same verbal format)
- ID `MARGA-7101-T9` → `BXKPL-4829-Q7` (same character class pattern)

This ensures the anonymized text remains **realistic and usable** for downstream NLP tasks.

### 5. Multi-Locale Diversity

Faker generators use **7 locales** (US, UK, French, German, Spanish, Japanese, Russian) to produce diverse, non-repetitive replacements — critical for training robust models.

### 6. Streamlit Interface Innovation

Instead of Google Sheets or Excel annotation, a **custom web UI** provides:
- Color-coded entity highlights (21 unique colors)
- Inline type correction dropdowns
- One-click replacement regeneration
- Real-time quality warnings
- Built-in statistics dashboard
- Automatic backups

Estimated **3x throughput improvement** over spreadsheet-based annotation.

### 7. Three-Strategy Data Augmentation (Phase 2)

Rather than collecting more raw data, we **amplify verified annotations** with three complementary, peer-reviewed augmentation strategies:
- **Entity Swap (×4)** — Prevents entity-value memorisation (Dai & Adel, COLING 2020)
- **Template Fill (×5)** — Creates novel entity co-occurrences + structural diversity (Anaby-Tavor et al., AAAI 2020)
- **EDA (×3)** — Increases context-word diversity without touching PII spans (Wei & Zou, EMNLP 2019)

Combined multiplier of **×12** means 10K gold annotations produce **~130K training examples** — competitive with datasets requiring 10× the human effort. All three strategies are **zero-dependency** (no external APIs, no heavy models at augmentation time).

---

## Dataset Analysis & Reporting

The pipeline includes a comprehensive **dataset analysis script** that generates publication-ready charts, LaTeX tables, and statistical summaries.

```bash
# Generate full analysis (charts + tables + Markdown + JSON)
python saha_al/scripts/dataset_analysis.py

# Custom output directory
python saha_al/scripts/dataset_analysis.py --output-dir reports/v2/

# Text-only (skip chart generation)
python saha_al/scripts/dataset_analysis.py --no-charts
```

### Generated Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Entity type distribution | PNG (bar + pie) | Count and proportion of each entity type |
| Routing distribution | PNG | GREEN / YELLOW / RED bar chart |
| Confidence histogram | PNG | Distribution of entity confidence scores |
| Agreement breakdown | PNG (pie) | Full / partial / single-source agreement |
| Text length distribution | PNG | Histogram of character counts per entry |
| PII density | PNG | Distribution of PII-to-text ratio per entry |
| Entities per entry | PNG | Histogram of entity count per entry |
| Co-occurrence heatmap | PNG | Which entity types appear together |
| Detection source | PNG (pie) | spaCy vs regex vs both |
| Annotator contributions | PNG | Per-annotator entry counts |
| Augmentation breakdown | PNG | Entries per augmentation strategy |
| 5 LaTeX tables | `.tex` | Copy-paste into your report (`\input{...}`) |
| Markdown report | `.md` | Human-readable summary with all key numbers |
| Stats JSON | `.json` | Machine-readable statistics for downstream tools |

All files are saved to `saha_al/reports/` (configurable via `--output-dir`).

---

## Annotation Guidelines Summary

### Entity Tagging Rules

1. **Tag the minimal span** — "John Smith" not "My name is John Smith"
2. **Prefer specific types** — `SSN` over `ID_NUMBER`, `EMAIL` over `OTHER_PII`
3. **When in doubt, flag** — flagged entries go to expert review
4. **Check replacements** — ensure they're realistic and format-preserving
5. **Don't skip PII** — even partial PII (first name only) should be tagged

### Annotator Workflow

1. Read the highlighted text
2. Review each entity: correct type if wrong, edit replacement if needed
3. Delete false positives (non-PII detected as PII)
4. Click **Accept** if everything looks good
5. Click **Flag** with a reason if something is confusing
6. Click **Skip** if the text is garbage/unreadable

---

## Output Schema

### Gold Standard Entry (`gold_standard.jsonl`)

```json
{
  "entry_id": 58923,
  "original_text": "My name is John Smith and I live at 123 Main St, NYC. Call me at +1-555-0123. SSN: 123-45-6789",
  "anonymized_text": "My name is Jane Doe and I live at 456 Oak Ave, Boston. Call me at +1-555-9876. SSN: 987-65-4321",
  "entities": [
    {
      "text": "John Smith",
      "start": 11,
      "end": 21,
      "entity_type": "FULLNAME",
      "source": "spacy",
      "confidence": 0.92,
      "agreement": "single_source"
    },
    {
      "text": "123-45-6789",
      "start": 81,
      "end": 92,
      "entity_type": "SSN",
      "source": "regex",
      "confidence": 0.94,
      "agreement": "single_source"
    }
  ],
  "replacements": {
    "John Smith": "Jane Doe",
    "123 Main St": "456 Oak Ave",
    "NYC": "Boston",
    "+1-555-0123": "+1-555-9876",
    "123-45-6789": "987-65-4321"
  },
  "metadata": {
    "preannotation_timestamp": "2025-03-01T14:23:45",
    "spacy_count": 3,
    "regex_count": 2,
    "merged_count": 5,
    "agreement_counts": { "full": 0, "partial": 0, "single_source": 5 },
    "annotator": "Ayush",
    "action": "ACCEPT",
    "annotated_at": "2025-03-02T09:15:30",
    "source_queue": "YELLOW",
    "quality_warnings": 0,
    "warnings": []
  }
}
```

### Augmented Entry (`augmented_data.jsonl`)

```json
{
  "entry_id": "58923_swap_2",
  "original_text": "My name is François Dupont and I live at 789 Rue de Rivoli, Paris. Call me at +33-1-42-68-5309. SSN: 847-29-3156",
  "anonymized_text": "My name is Yuki Tanaka and I live at 12 Cherry Blossom Rd, Osaka. Call me at +81-3-1234-5678. SSN: 562-81-4073",
  "entities": [
    {
      "text": "François Dupont",
      "start": 11,
      "end": 26,
      "entity_type": "FULLNAME",
      "source": "augmentation",
      "confidence": 1.0,
      "agreement": "augmented"
    },
    {
      "text": "847-29-3156",
      "start": 100,
      "end": 111,
      "entity_type": "SSN",
      "source": "augmentation",
      "confidence": 1.0,
      "agreement": "augmented"
    }
  ],
  "metadata": {
    "augmentation": {
      "strategy": "entity_swap",
      "source_entry_id": 58923,
      "variant_index": 2,
      "timestamp": "2025-03-10T16:45:12"
    }
  }
}
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.28.0 | Annotation web interface |
| `spacy` | ≥ 3.7.0 | NER pre-annotation + bootstrap training |
| `faker` | ≥ 20.0.0 | Realistic PII replacement generation |
| `tqdm` | ≥ 4.65.0 | Progress bars for batch processing |
| `pandas` | ≥ 2.0.0 | Statistics and data manipulation |
| `nltk` | ≥ 3.8.0 | WordNet synonyms for EDA augmentation (optional — built-in fallback) |

**spaCy Model:** `en_core_web_lg` (560MB, CPU) or `en_core_web_trf` (GPU)

---

## Authors

INLP Project Team — Semester 6, 2025

---

# 📋 SAHA-AL Annotation Guidelines

> **Version 2.0** — Last updated: March 2026
>
> These guidelines govern all human annotation within the SAHA-AL pipeline.
> Every team member **must read this document** before annotating.

---

## Table of Contents

1. [Goal](#1-goal)
2. [Entity Taxonomy — Definitions & Decision Rules](#2-entity-taxonomy--definitions--decision-rules)
3. [Span Selection Rules](#3-span-selection-rules)
4. [Ambiguity Resolution — Decision Trees](#4-ambiguity-resolution--decision-trees)
5. [Replacement Quality Criteria](#5-replacement-quality-criteria)
6. [Annotator Actions — When to Accept / Flag / Skip](#6-annotator-actions--when-to-accept--flag--skip)
7. [Common Mistakes to Avoid](#7-common-mistakes-to-avoid)
8. [Edge Cases & Worked Examples](#8-edge-cases--worked-examples)
9. [Inter-Annotator Agreement Protocol](#9-inter-annotator-agreement-protocol)
10. [Queue-Specific Instructions](#10-queue-specific-instructions)
11. [Quality Checklist (Pre-Accept)](#11-quality-checklist-pre-accept)

---

## 1. Goal

Produce a gold-standard dataset of `(original_text, anonymized_text, entities)` triples where:

- **Every PII span** is detected with exact character offsets (`start`, `end`)
- **Every PII span** is classified into exactly one of **21 entity types**
- **Every PII span** has a **format-preserving replacement** that is realistic and plausible
- **No original PII** leaks into the anonymized text

The downstream task is training NER models for privacy-preserving text anonymization. Annotation quality directly determines model quality.

---

## 2. Entity Taxonomy — Definitions & Decision Rules

### High-Sensitivity PII (Identity-Critical)

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `FULLNAME` | Complete person name (≥2 tokens: first + last, possibly middle) | "Igorche Ramtin Eshekary", "Anna-Lynn Recica" | "Brandy" (→ FIRST_NAME), "the team" | ≥2 name tokens → FULLNAME |
| `FIRST_NAME` | Given / first name in isolation (1 token, clearly a name) | "Brandy", "Tami", "Cajetan" | "Brandy Haroon" (→ FULLNAME), "Sale" (context needed) | Standalone single name token |
| `LAST_NAME` | Family name / surname in isolation | "Eshekary", "Haroon" | "Brandy Haroon" (→ FULLNAME) | Standalone surname only |
| `SSN` | US Social Security Number (XXX-XX-XXXX) | "123-45-6789", "753-78-3595" | "123-456-7890" (→ PHONE or ID) | Exactly 3-2-4 digit pattern |
| `PASSPORT` | Passport or travel document number | "AB1234567", "PT1954390" | "MARGA7101160M9183" (→ ID_NUMBER) | Matches known passport formats (2 letters + 7 digits, etc.) |
| `CREDIT_CARD` | Credit card number (16 digits) | "4532-1234-5678-9012", "5323815020502804" | "3667081227" (10 digits → ACCOUNT_NUMBER) | Exactly 16 digits (with optional separators) |

### Contact PII

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `PHONE` | Any telephone/fax number | "+1-555-0123", "805 099 8045", "(32)-8048 7576" | "476506330" (9 digits, no format → ACCOUNT_NUMBER) | Has phone-like separators/prefix OR ≥10 digits with clear phone context |
| `EMAIL` | Email address | "user@gmail.com", "NNS17@tutanota.com" | "user at gmail" (not valid format) | Contains `@` + valid domain |
| `ADDRESS` | Street / postal / physical address | "123 Main St, Apt 4", "27451 Range Road 3045" | "NYC" (→ LOCATION), "43431" (→ ZIPCODE) | Contains street name/number/road/avenue |
| `ZIPCODE` | Postal / ZIP code | "43431-9599", "517214", "NE13 6HB" | "123 Main St" (→ ADDRESS) | Standalone postal code pattern |

### Temporal PII

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `DATE` | Any calendar date | "4th August 1942", "2022-06-23T00:00:00", "12/12/1974" | "March" alone (too vague) | Contains day+month or month+year or full date |
| `TIME` | Time of day | "10:17", "5:52:48 AM", "half past 1" | "1942" (→ DATE) | HH:MM format or verbal time expression |

### Location / Organization PII

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `LOCATION` | City / state / country / named geographic area | "NYC", "San Francisco", "Firebaugh" | "123 Main St" (→ ADDRESS) | Named place without street address |
| `ORGANIZATION` | Company / institution / named group | "Google Inc.", "[ORGANISATIONPLACEHOLDER_14]" | "the team" (generic) | Proper noun referring to an organization |

### Financial PII

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `ACCOUNT_NUMBER` | Bank / financial / reference account number | "3847261590142", "0798155157" | "4532-1234-5678-9012" (16 digits → CREDIT_CARD) | 10–18 digit sequence not matching CC/SSN/Phone |

### Demographic PII

| Type | Definition | Positive Examples | Negative Examples | Decision Rule |
|------|-----------|-------------------|-------------------|---------------|
| `TITLE` | Honorific / salutation | "Mr", "Mrs", "Dr", "Prof", "Madame", "Mstr" | "Manager" (job title, not PII) | From the known titles set |
| `GENDER` | Gender identifier | "Male", "Female", "Non-binary" | "M" (ambiguous, could be initial) → use context | Explicit gender word |
| `NUMBER` | Numeric PII (age, count, amount) | "28 individuals", "$50,000" | "123-45-6789" (→ SSN) | Numeric value that is PII in context |

### Catch-All

| Type | Definition | When to Use |
|------|-----------|-------------|
| `ID_NUMBER` | Generic alphanumeric ID not matching a more specific type | "MARGA7101160M9183", "3LPHEOTUUH" — only when not SSN/Passport/CC |
| `OTHER_PII` | Any PII not covered by the 20 types above | Ethnicity, religion, medical conditions, biometric data |
| `UNKNOWN` | **Never use in gold standard.** Auto-assigned by Layer 1 when type cannot be determined. Must be resolved by annotator. | Annotator MUST change this to a specific type or delete the entity |

---

## 3. Span Selection Rules

### Rule 1: Minimal Span

Tag the **shortest contiguous text** that constitutes the PII entity.

| ✅ Correct | ❌ Wrong | Why |
|-----------|---------|-----|
| `[John Smith]` | `My name is [John Smith]` | "My name is" is context, not PII |
| `[123-45-6789]` | `SSN: [123-45-6789]` | "SSN:" is a label, not the entity |
| `[user@gmail.com]` | `email [user@gmail.com] address` | Only the email itself is PII |

### Rule 2: Include Full Entities

Don't truncate entities mid-word.

| ✅ Correct | ❌ Wrong | Why |
|-----------|---------|-----|
| `[Igorche Ramtin Eshekary]` → FULLNAME | `[Igorche] [Ramtin Eshekary]` | Should be one FULLNAME span |
| `[27451 Range Road 3045, Willingdon]` → ADDRESS | `[27451 Range Road 3045]` + `[Willingdon]` separately | Full address is one span |

### Rule 3: Separate Distinct Entities

Don't merge unrelated entities into one span.

| ✅ Correct | ❌ Wrong |
|-----------|---------|
| `[John Smith]` FULLNAME + `[123-45-6789]` SSN | `[John Smith 123-45-6789]` as one entity |
| `[NYC]` LOCATION + `[+1-555-0123]` PHONE | `[NYC, +1-555-0123]` as one entity |

### Rule 4: Preserve Exact Offsets

Character offsets `(start, end)` must precisely match the highlighted text. If the auto-detection is off by even 1 character, **flag the entry** for correction.

### Rule 5: Handle Hyphenated / Compound Names

- Hyphenated names → single FULLNAME: `[Anna-Lynn Recica]`
- Name with particles → single FULLNAME: `[Rem Abdirahin van Veelen Ottrich]`
- Titles attached → separate: `[Mr]` TITLE + `[John Smith]` FULLNAME

---

## 4. Ambiguity Resolution — Decision Trees

### Is it a PHONE or ACCOUNT_NUMBER?

```
Does it start with + or have () or contain - between groups?
  ├─ YES → Does it have 7-15 digits total?
  │    ├─ YES → PHONE
  │    └─ NO  → Check context (mentioned "call", "contact"?)
  │         ├─ YES → PHONE
  │         └─ NO  → ACCOUNT_NUMBER
  └─ NO → Is it 10-18 pure digits?
       ├─ YES → Is it mentioned with "account", "payment", "tax"?
       │    ├─ YES → ACCOUNT_NUMBER
       │    └─ NO  → Check digit count: 16 digits? → CREDIT_CARD, else ACCOUNT_NUMBER
       └─ NO → NUMBER or ID_NUMBER based on context
```

### Is it a NAME or LOCATION?

```
Is it a proper noun (capitalized)?
  ├─ YES → Does it follow "Mr/Mrs/Dear" or precede a verb (said, wrote, asked)?
  │    ├─ YES → FULLNAME / FIRST_NAME
  │    └─ NO  → Does it follow "in/at/from/to" + is a known place?
  │         ├─ YES → LOCATION
  │         └─ NO  → Check context: is it a person acting in the sentence?
  │              ├─ YES → FULLNAME / FIRST_NAME
  │              └─ NO  → LOCATION (default for ambiguous proper nouns after prepositions)
  └─ NO → Not a name or location entity
```

### Is it a DATE or TIME?

```
Does it contain HH:MM pattern?
  ├─ YES → Is there also a date component (month, day, year)?
  │    ├─ YES → Separate into DATE + TIME if they're distinct spans
  │    └─ NO  → TIME
  └─ NO → Does it contain month name or DD/MM/YYYY pattern?
       ├─ YES → DATE
       └─ NO → Is it just a year like "1942"?
            ├─ YES → DATE (if clearly referring to a specific date/year)
            └─ NO  → NUMBER or not PII
```

### FIRST_NAME vs FULLNAME

```
How many name tokens?
  ├─ 1 token  → FIRST_NAME (unless clearly a surname from context)
  ├─ 2 tokens → FULLNAME
  ├─ 3+ tokens → FULLNAME
  └─ With title prefix ("Mr John Smith")?
       → [Mr] TITLE + [John Smith] FULLNAME (separate entities)
```

---

## 5. Replacement Quality Criteria

Every replacement must satisfy **all four** criteria:

### C1: Format Preservation

The replacement must match the structural format of the original.

| Original | ✅ Good Replacement | ❌ Bad Replacement |
|----------|-------------------|--------------------|
| `123-45-6789` (SSN) | `847-29-3156` | `847293156` (missing dashes) |
| `(32)-8048 7576` (phone) | `(55)-1234 5678` | `+551234567` (different format) |
| `4th August 1942` (date) | `17th March 1965` | `1965-03-17` (different format) |
| `NNS17@tutanota.com` (email) | `abc42@inbox.net` | `John Smith` (not an email) |

### C2: Type Consistency

The replacement must be a valid instance of the declared entity type.

| Entity Type | ✅ Valid | ❌ Invalid |
|-------------|---------|-----------|
| FULLNAME | "Jane Doe" | "123 Main St" |
| PHONE | "+1-555-9876" | "john@mail.com" |
| LOCATION | "Boston" | "Dr. Smith" |

### C3: Plausibility

The replacement should produce a sentence that reads naturally.

| Original | ✅ Natural | ❌ Jarring |
|----------|----------|-----------|
| "Meet [Brandy] at 10:17" | "Meet [Yuki] at 10:17" | "Meet [AAAA] at 10:17" |
| "Visit us at [NYC]" | "Visit us at [Portland]" | "Visit us at [Z]" |

### C4: Non-Leakage

The replacement must be **different from the original**. If the auto-generated replacement happens to be identical, **click Regenerate**.

---

## 6. Annotator Actions — When to Accept / Flag / Skip

### ✅ Accept — Save to Gold Standard

Use when **ALL** of the following are true:
- [ ] Every PII span is detected (no missed entities)
- [ ] Every entity has the correct type
- [ ] All offsets look correct (text matches span)
- [ ] Replacements are format-preserving and plausible
- [ ] No UNKNOWN types remain
- [ ] Anonymized text reads naturally

### 🚩 Flag — Send to Expert Review

Use when **ANY** of the following are true:
- The text contains PII you're not sure how to classify
- The auto-detection missed a span and you're not sure where the boundaries are
- The text is in a mixed language and PII spans are unclear
- The entry contains entity types not in our taxonomy (potential `OTHER_PII`)
- Character offsets appear misaligned (highlight doesn't match text)
- You and another annotator disagree on the same entry

**Always write a reason** when flagging. Examples:
- "Unsure if '1221' is a ZIPCODE or just a NUMBER"
- "Ambiguous: 'Sale' could be a FIRST_NAME or LOCATION"
- "Mixed-language text, PII boundaries unclear"
- "Offsets misaligned for entity 3"

### ⏭️ Skip — Remove from Queue

Use when:
- The text is garbage / unreadable / corrupted
- The text contains no PII at all (rare — should already be filtered)
- The text is a duplicate of a previously annotated entry

---

## 7. Common Mistakes to Avoid

### Mistake 1: Leaving UNKNOWN Types

❌ Never accept an entry with `UNKNOWN` entity types. Every entity **must** be reclassified to a specific type or deleted.

### Mistake 2: Over-Tagging Non-PII

Not everything that looks like a number is PII:

| Text | PII? | Why |
|------|------|-----|
| "28 individuals" | ❌ NO | Generic count, not personally identifiable |
| "the 19th Century" | ❌ NO | Historical reference, not PII |
| "111 meter long deck" | ❌ NO | Measurement, not PII |
| "28" in "age: 28" | ✅ YES (NUMBER) | Age is PII in context |
| "111" in "room 111" | It depends | Room number could be PII if it identifies a person's location |

**Rule of thumb:** Would knowing this number help identify or locate a specific individual? If YES → PII. If NO → skip it.

### Mistake 3: Splitting Full Names

❌ `[Igorche] [Ramtin] [Eshekary]` → three FIRST_NAMEs
✅ `[Igorche Ramtin Eshekary]` → one FULLNAME

### Mistake 4: Merging Title + Name

❌ `[Mr John Smith]` → one FULLNAME
✅ `[Mr]` TITLE + `[John Smith]` FULLNAME

### Mistake 5: Ignoring Placeholders

The source data contains pre-existing placeholders like `[ORGANISATIONPLACEHOLDER_14]`. These should be tagged as their respective types:
- `[ORGANISATIONPLACEHOLDER_14]` → ORGANIZATION
- `[AMOUNTPLACEHOLDER_9]` → NUMBER
- `[LANGUAGEPLACEHOLDER_26]` → OTHER_PII

### Mistake 6: Accepting Misaligned Offsets

If the highlighted span in the UI doesn't exactly match the entity text (e.g., includes an extra space or cuts off a letter), **flag the entry**. Misaligned offsets will corrupt the training data.

### Mistake 7: Not Checking the Anonymized Preview

Always scroll down to the anonymized text preview. Read it once. If anything looks wrong (PII still visible, replacement doesn't fit, text is garbled), investigate before accepting.

---

## 8. Edge Cases & Worked Examples

### Example 1: Multiple Entity Types in One Sentence

**Text:** `"Igorche Ramtin Eshekary will need to bring their MARGA7101160M9183 and 805 099 8045 for verification."`

| Span | Type | Reasoning |
|------|------|-----------|
| `Igorche Ramtin Eshekary` | FULLNAME | 3-token person name |
| `MARGA7101160M9183` | ID_NUMBER | Alphanumeric, not SSN/Passport format |
| `805 099 8045` | PHONE | 10 digits with spaces, phone-like grouping |

### Example 2: Ambiguous Numbers

**Text:** `"Shola Kenzi Zimeri used 0111-284596398 to schedule an interview about the effects of tech-free days on 28 individuals."`

| Span | Type | Reasoning |
|------|------|-----------|
| `Shola Kenzi Zimeri` | FULLNAME | 3-token name |
| `0111-284596398` | PHONE | Dash-separated digits, "used [X] to schedule" = phone |
| `28` | ❌ Not PII | "28 individuals" is a generic count, not identifying |

### Example 3: Title + Name Separation

**Text:** `"Dear Mr Shamaila Kifajete, We have received your payment of 644693204822782691."`

| Span | Type |
|------|------|
| `Mr` | TITLE |
| `Shamaila Kifajete` | FULLNAME |
| `644693204822782691` | CREDIT_CARD (18 digits, payment context) or ACCOUNT_NUMBER |

### Example 4: Address vs Location

**Text:** `"27451 Range Road 3045, Willingdon is the new location for our Kite Design Studio."`

| Span | Type | Reasoning |
|------|------|-----------|
| `27451 Range Road 3045, Willingdon` | ADDRESS | Full street address with number + road name + city |
| (Don't separately tag `Willingdon` as LOCATION — it's part of the ADDRESS) | | |

### Example 5: Timestamps and Chat Messages

**Text:** `"19:23 Stellie Mekka: 'Hello Gxime Roxana-Georgiana...'"`

| Span | Type | Reasoning |
|------|------|-----------|
| `19:23` | TIME | HH:MM format |
| `Stellie Mekka` | FULLNAME | Speaker name |
| `Gxime Roxana-Georgiana` | FULLNAME | Addressee (hyphenated) |

### Example 6: Non-PII That Looks Like PII

**Text:** `"The affiliate program is open to all 13 and above."`

| Span | PII? | Reasoning |
|------|------|-----------|
| `13` | Maybe | "13 and above" = age threshold? If it's a policy (not about a specific person), it's not PII. If "John is 13", it would be PII (NUMBER). Context decides. |

**Guidance:** When in doubt, **don't tag** generic thresholds/counts. Only tag numbers that refer to a **specific individual's** data.

---

## 9. Inter-Annotator Agreement Protocol

### Double Annotation

- **10% of entries** are randomly assigned to two annotators
- After annotation, IAA is computed using entity-level F1 (see `utils/iaa.py`)
- **Target:** Entity-level F1 ≥ 0.85

### Disagreement Resolution

1. Both annotators review each other's annotations for the disagreed entries
2. If they agree after review → update to agreed version
3. If still disagreeing → escalate to **expert adjudicator** (team lead)
4. Expert's decision is final and logged

### IAA Checkpoints

| After N Gold Entries | Action |
|---------------------|--------|
| 200 | First IAA check — calibrate annotators |
| 500 | Second IAA check — expect F1 ≥ 0.80 |
| 1,000 | Third IAA check — expect F1 ≥ 0.85 |
| Every 2,000 | Periodic audits |

If IAA drops below 0.80, **stop annotation** and hold a calibration session to discuss disagreements.

---

## 10. Queue-Specific Instructions

### 🟢 GREEN Queue (Auto-Approved)

- **Spot-check only** — review every 10th entry
- Focus on: missed entities (false negatives), obviously wrong types
- Expected time: ~5 seconds per entry
- Accept rate: ~95%+

### 🟡 YELLOW Queue (Primary Annotation Queue)

- **Review every entity** — this is where most annotation time is spent
- Focus on: correcting types, fixing replacements, catching false positives/negatives
- Expected time: ~15–30 seconds per entry
- Accept rate: ~75–85%

### 🔴 RED Queue (Expert Review)

- **Careful, thorough review** — these are the hardest entries
- May require domain knowledge, multi-entity reasoning
- Expected time: ~30–60 seconds per entry
- Flag rate: ~15–25% (it's OK to flag RED entries)

---

## 11. Quality Checklist (Pre-Accept)

Before clicking ✅ **Accept**, mentally run through this checklist:

- [ ] **Completeness:** Are ALL PII spans detected? (scan the text for missed names, numbers, emails, dates)
- [ ] **Types:** Is every entity type correct? (no UNKNOWN, no obviously wrong types)
- [ ] **Spans:** Do highlights match the text exactly? (no extra spaces, no truncated words)
- [ ] **Replacements:** Is each replacement format-preserving and plausible?
- [ ] **Anonymized preview:** Does the anonymized text read naturally?
- [ ] **No leakage:** Is the original PII completely absent from the anonymized text?
- [ ] **Consistency:** Would another annotator make the same decisions?

If you can answer YES to all seven, click Accept. Otherwise, fix the issues or Flag.

---

> **Remember:** Quality > Speed. One bad gold entry corrupts the model far more than one skipped entry helps it.

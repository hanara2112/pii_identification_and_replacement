"""
Shared evaluation utilities for the SAHA-AL benchmark.

Provides: JSONL I/O, text normalization, span matching, format regexes,
replacement extraction via difflib, and capitalized n-gram extraction.
"""

import json
import re
from difflib import SequenceMatcher


# ── Format regex patterns (aligned with saha_al/utils/quality_checks.py) ──

FORMAT_PATTERNS = {
    "EMAIL":       re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "PHONE":       re.compile(r"^[\d\s\+\-\(\)\.]{5,20}$"),
    "SSN":         re.compile(r"^\d{3}[- ]?\d{2}[- ]?\d{4}$"),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{12,25}$"),
    "ZIPCODE":     re.compile(r"^[A-Z0-9\- ]{3,12}$", re.IGNORECASE),
    "DATE":        re.compile(r"\d"),
}


# ── I/O ──

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def align_records(gold: list[dict], preds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align gold and prediction records by id, raising on mismatch."""
    pred_map = {p["id"]: p for p in preds}
    aligned_gold, aligned_pred = [], []
    for g in gold:
        gid = g.get("id", g.get("entry_id"))
        if gid not in pred_map:
            raise ValueError(f"Gold id {gid} not found in predictions")
        aligned_gold.append(g)
        aligned_pred.append(pred_map[gid])
    return aligned_gold, aligned_pred


# ── Text normalization ──

def normalize_text(text: str | None) -> str:
    if text is None:
        return "[EMPTY]"
    text = text.strip()
    return text if text else "[EMPTY]"


# ── Span matching ──

def span_match(pred: dict, gold: dict, mode: str = "exact") -> bool:
    """
    Compare two entity spans.

    Args:
        pred: dict with 'start', 'end' (and optionally 'type')
        gold: dict with 'start', 'end' (and optionally 'type')
        mode: 'exact' | 'partial' | 'type_aware'
    """
    if mode == "exact":
        return pred["start"] == gold["start"] and pred["end"] == gold["end"]
    elif mode == "partial":
        overlap = max(0, min(pred["end"], gold["end"]) - max(pred["start"], gold["start"]))
        union = max(pred["end"], gold["end"]) - min(pred["start"], gold["start"])
        return (overlap / union) > 0.5 if union > 0 else False
    elif mode == "type_aware":
        return (
            pred["start"] == gold["start"]
            and pred["end"] == gold["end"]
            and pred.get("type", "").upper() == gold.get("type", "").upper()
        )
    else:
        raise ValueError(f"Unknown span match mode: {mode}")


# ── Entity-text matching ──

def exact_entity_match(ent_text: str, pred_text: str) -> bool:
    """Check if entity text appears in prediction as a whole word."""
    if not ent_text or not pred_text:
        return False
    pattern = rf"(?<!\w){re.escape(ent_text)}(?!\w)"
    return re.search(pattern, pred_text, flags=re.IGNORECASE) is not None


# ── Replacement extraction ──

def extract_replacement(original: str, prediction: str, start: int, end: int) -> str | None:
    """
    Heuristically extract the replacement string for an entity at [start:end]
    in the original text, by aligning original and prediction via SequenceMatcher.
    """
    sm = SequenceMatcher(None, original, prediction)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i1 <= start and i2 >= end:
            return prediction[j1:j2]
        if tag == "equal" and i1 <= start < end <= i2:
            offset = start - i1
            return prediction[j1 + offset : j1 + offset + (end - start)]
    return None


# ── Capitalized n-gram extraction ──

def get_capitalized_ngrams(text: str, n: int = 3) -> list[tuple[str, ...]]:
    """Extract n-grams where at least one token starts with uppercase."""
    tokens = text.split()
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return [ng for ng in ngrams if any(len(t) > 0 and t[0].isupper() for t in ng)]

"""
Quality Checks
Post-annotation quality checks for accepted entries.
Checks:
  1. Leakage  – original PII text still present in anonymised text
  2. Format   – replacement preserves rough format (e.g., email → email)
  3. Validity – replacement looks like the declared entity type
"""

import re


# ── 1. Leakage check ────────────────────────────────────────────────
def check_leakage(original_text: str, anonymized_text: str, entities: list) -> list:
    """
    Return list of warnings where original PII text still appears
    verbatim in the anonymised text.
    """
    warnings = []
    for ent in entities:
        original_pii = ent.get("text", "")
        if not original_pii or len(original_pii) < 3:
            continue  # skip very short tokens to avoid false positives
        if original_pii.lower() in anonymized_text.lower():
            warnings.append({
                "check": "leakage",
                "severity": "high",
                "entity_type": ent.get("entity_type", "UNKNOWN"),
                "original_text": original_pii,
                "message": f"Original PII '{original_pii}' still found in anonymised text.",
            })
    return warnings


# ── 2. Format preservation check ────────────────────────────────────
_FORMAT_RULES = {
    "EMAIL":       re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "PHONE":       re.compile(r"^[\d\s\+\-\(\)\.]{5,20}$"),
    "SSN":         re.compile(r"^\d{3}[- ]?\d{2}[- ]?\d{4}$"),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{12,25}$"),
    "ZIPCODE":     re.compile(r"^[A-Z0-9\- ]{3,12}$", re.IGNORECASE),
    "DATE":        re.compile(r"\d"),  # must contain at least one digit
}


def check_format_preservation(entities: list, replacements: dict) -> list:
    """
    For structured types, verify that the replacement roughly matches
    the expected format.
    """
    warnings = []
    for ent in entities:
        etype = ent.get("entity_type", "")
        replacement = replacements.get(ent.get("text", ""), "")
        if not replacement:
            continue
        pattern = _FORMAT_RULES.get(etype)
        if pattern and not pattern.search(replacement):
            warnings.append({
                "check": "format",
                "severity": "medium",
                "entity_type": etype,
                "replacement": replacement,
                "message": f"Replacement '{replacement}' may not match expected format for {etype}.",
            })
    return warnings


# ── 3. Replacement validity ─────────────────────────────────────────
def check_replacement_validity(entities: list, replacements: dict) -> list:
    """
    Ensure that every entity has a non-empty replacement and that it
    actually differs from the original.
    """
    warnings = []
    for ent in entities:
        original = ent.get("text", "")
        replacement = replacements.get(original, "")
        if not replacement:
            warnings.append({
                "check": "validity",
                "severity": "high",
                "entity_type": ent.get("entity_type", "UNKNOWN"),
                "original_text": original,
                "message": f"No replacement provided for entity '{original}'.",
            })
        elif replacement.strip().lower() == original.strip().lower():
            warnings.append({
                "check": "validity",
                "severity": "medium",
                "entity_type": ent.get("entity_type", "UNKNOWN"),
                "original_text": original,
                "replacement": replacement,
                "message": f"Replacement is identical to original for '{original}'.",
            })
    return warnings


# ── Aggregate runner ─────────────────────────────────────────────────
def run_all_checks(
    original_text: str,
    anonymized_text: str,
    entities: list,
    replacements: dict,
) -> list:
    """
    Run all quality checks and return combined warnings list.
    Each warning is a dict with keys: check, severity, entity_type, message.
    """
    warnings = []
    warnings.extend(check_leakage(original_text, anonymized_text, entities))
    warnings.extend(check_format_preservation(entities, replacements))
    warnings.extend(check_replacement_validity(entities, replacements))
    return warnings

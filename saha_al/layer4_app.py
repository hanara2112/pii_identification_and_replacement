"""
Layer 4 — Streamlit Annotation Interface (SAHA-AL)
Full-featured annotation tool with:
  - Sidebar: queue selector, annotator name, statistics
  - Main panel: highlighted text, entity review table, replacement editor
  - Actions: Accept, Flag, Skip
  - Tabs: Annotation, Statistics, Flagged Review
"""

import json
import os
import sys
import time
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from saha_al.config import (
    GREEN_QUEUE_PATH,
    YELLOW_QUEUE_PATH,
    RED_QUEUE_PATH,
    GOLD_STANDARD_PATH,
    SKIPPED_PATH,
    FLAGGED_PATH,
    BACKUP_DIR,
    ANNOTATION_LOG_PATH,
    ENTITY_TYPES,
    MAX_LENGTH_RATIO,
    MIN_LENGTH_RATIO,
)
from saha_al.utils.entity_types import ENTITY_SCHEMA
from saha_al.utils.faker_replacements import generate_replacements
from saha_al.utils.quality_checks import run_all_checks
from saha_al.utils.io_helpers import (
    read_jsonl,
    append_jsonl,
    backup_gold_standard,
    get_annotated_ids,
    count_lines,
)


# ─────────────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAHA-AL Annotation Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "annotator_name": "",
        "current_queue": "YELLOW",
        "queue_data": [],
        "current_index": 0,
        "modified_entities": [],
        "modified_replacements": {},
        "annotation_count": 0,
        "session_start": datetime.now().isoformat(),
        "flag_reason": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─────────────────────────────────────────────────────────────────────
#  Queue Loading
# ─────────────────────────────────────────────────────────────────────
QUEUE_PATHS = {
    "GREEN": GREEN_QUEUE_PATH,
    "YELLOW": YELLOW_QUEUE_PATH,
    "RED": RED_QUEUE_PATH,
}


def load_queue(queue_name: str) -> list:
    """Load queue data, filtering out already-annotated ids."""
    path = QUEUE_PATHS.get(queue_name)
    if not path:
        return []
    entries = read_jsonl(path)
    done_ids = get_annotated_ids(GOLD_STANDARD_PATH)
    return [e for e in entries if e.get("entry_id") not in done_ids]


# ─────────────────────────────────────────────────────────────────────
#  Highlighted HTML
# ─────────────────────────────────────────────────────────────────────
def build_highlighted_html(text: str, entities: list) -> str:
    """Build HTML with highlighted PII spans using entity colors."""
    if not entities:
        return f"<p style='font-size:16px; line-height:1.8;'>{_html_escape(text)}</p>"

    # Sort by start position
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))

    html_parts = []
    last_end = 0

    for ent in sorted_ents:
        start = ent.get("start", 0)
        end = ent.get("end", 0)
        etype = ent.get("entity_type", "UNKNOWN")
        color = ENTITY_SCHEMA.get(etype, {}).get("color", "#CCCCCC")

        # Text before this entity
        if start > last_end:
            html_parts.append(_html_escape(text[last_end:start]))

        # Entity span with highlight
        entity_text = _html_escape(text[start:end])
        tooltip = f"{etype} | conf={ent.get('confidence', '?')} | {ent.get('agreement', '?')}"
        html_parts.append(
            f'<span style="background-color:{color}; padding:2px 4px; '
            f'border-radius:4px; font-weight:600; cursor:help;" '
            f'title="{tooltip}">'
            f'{entity_text}'
            f'<sub style="font-size:10px; color:#333;"> {etype}</sub>'
            f'</span>'
        )
        last_end = end

    # Remaining text
    if last_end < len(text):
        html_parts.append(_html_escape(text[last_end:]))

    return (
        f"<div style='font-size:16px; line-height:2.0; "
        f"background:#1E1E1E; color:#E0E0E0; padding:16px; "
        f"border-radius:8px; border:1px solid #333;'>"
        f"{''.join(html_parts)}</div>"
    )


def _html_escape(text: str) -> str:
    """Basic HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ─────────────────────────────────────────────────────────────────────
#  Annotation Actions
# ─────────────────────────────────────────────────────────────────────
def action_accept(entry: dict, entities: list, replacements: dict, annotator: str):
    """Accept the annotation: run quality checks, save to gold standard."""
    original_text = entry.get("original_text", "")
    anonymized_text = entry.get("anonymized_text", "")

    # Run quality checks (non-blocking)
    warnings = run_all_checks(original_text, anonymized_text, entities, replacements)

    # Build gold entry
    gold_entry = {
        "entry_id": entry.get("entry_id"),
        "original_text": original_text,
        "anonymized_text": anonymized_text,
        "entities": entities,
        "replacements": replacements,
        "metadata": {
            **entry.get("metadata", {}),
            "annotator": annotator,
            "action": "ACCEPT",
            "annotated_at": datetime.now().isoformat(),
            "source_queue": entry.get("routing", {}).get("queue", "UNKNOWN"),
            "quality_warnings": len(warnings),
            "warnings": warnings[:5],  # keep first 5 warnings
        },
    }
    append_jsonl(GOLD_STANDARD_PATH, gold_entry)

    # Log
    _log_action(entry, annotator, "ACCEPT", warnings)

    # Backup periodically
    st.session_state["annotation_count"] += 1
    if st.session_state["annotation_count"] % 50 == 0:
        backup_gold_standard(GOLD_STANDARD_PATH, BACKUP_DIR)

    return warnings


def action_flag(entry: dict, annotator: str, reason: str):
    """Flag entry for expert review."""
    flag_entry = {
        "entry_id": entry.get("entry_id"),
        "original_text": entry.get("original_text", ""),
        "entities": entry.get("entities", []),
        "flag_reason": reason,
        "flagged_by": annotator,
        "flagged_at": datetime.now().isoformat(),
        "source_queue": entry.get("routing", {}).get("queue", "UNKNOWN"),
    }
    append_jsonl(FLAGGED_PATH, flag_entry)
    _log_action(entry, annotator, "FLAG")


def action_skip(entry: dict, annotator: str):
    """Skip entry, save to skipped file."""
    skip_entry = {
        "entry_id": entry.get("entry_id"),
        "skipped_by": annotator,
        "skipped_at": datetime.now().isoformat(),
    }
    append_jsonl(SKIPPED_PATH, skip_entry)
    _log_action(entry, annotator, "SKIP")


def _log_action(entry: dict, annotator: str, action: str, warnings: list = None):
    """Append to annotation log."""
    log_entry = {
        "entry_id": entry.get("entry_id"),
        "annotator": annotator,
        "action": action,
        "timestamp": datetime.now().isoformat(),
        "queue": entry.get("routing", {}).get("queue", "UNKNOWN"),
        "warnings_count": len(warnings) if warnings else 0,
    }
    append_jsonl(ANNOTATION_LOG_PATH, log_entry)


# ─────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🛡️ SAHA-AL")
        st.markdown("**Semi-Automatic Human-Augmented Active Learning**")
        st.markdown("---")

        # Annotator name
        st.session_state["annotator_name"] = st.text_input(
            "👤 Annotator Name",
            value=st.session_state["annotator_name"],
            placeholder="Enter your name",
        )

        st.markdown("---")

        # Queue selector
        selected_queue = st.radio(
            "📂 Select Queue",
            ["YELLOW", "RED", "GREEN"],
            index=["YELLOW", "RED", "GREEN"].index(st.session_state["current_queue"]),
            help="YELLOW=medium confidence, RED=low confidence, GREEN=auto-approved",
        )

        if selected_queue != st.session_state["current_queue"]:
            st.session_state["current_queue"] = selected_queue
            st.session_state["queue_data"] = load_queue(selected_queue)
            st.session_state["current_index"] = 0

        # Load queue button
        if st.button("🔄 Reload Queue", use_container_width=True):
            st.session_state["queue_data"] = load_queue(st.session_state["current_queue"])
            st.session_state["current_index"] = 0
            st.rerun()

        st.markdown("---")

        # Queue stats
        st.markdown("### 📊 Queue Sizes")
        for q_name in ["GREEN", "YELLOW", "RED"]:
            count = count_lines(QUEUE_PATHS.get(q_name, ""))
            color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[q_name]
            st.metric(f"{color} {q_name}", count)

        st.markdown("---")

        # Gold standard count
        gold_count = count_lines(GOLD_STANDARD_PATH)
        st.metric("✅ Gold Standard", gold_count)
        st.metric("📝 This Session", st.session_state["annotation_count"])

        st.markdown("---")

        # Backup button
        if st.button("💾 Backup Gold Standard", use_container_width=True):
            path = backup_gold_standard(GOLD_STANDARD_PATH, BACKUP_DIR)
            if path:
                st.success(f"Backed up to {os.path.basename(path)}")
            else:
                st.info("No gold standard file to back up yet.")


# ─────────────────────────────────────────────────────────────────────
#  Main: Annotation Tab
# ─────────────────────────────────────────────────────────────────────
def render_annotation_tab():
    queue_data = st.session_state["queue_data"]
    idx = st.session_state["current_index"]

    if not queue_data:
        st.info(
            f"No entries in the **{st.session_state['current_queue']}** queue "
            f"(or all have been annotated). Click **Reload Queue** in the sidebar."
        )
        return

    if idx >= len(queue_data):
        st.success("🎉 You've processed all entries in this queue!")
        if st.button("Start Over"):
            st.session_state["current_index"] = 0
            st.rerun()
        return

    entry = queue_data[idx]

    # ── Progress bar ──
    st.progress(idx / len(queue_data), text=f"Entry {idx + 1} of {len(queue_data)}")

    # ── Entry header ──
    col_id, col_queue, col_agreement = st.columns(3)
    with col_id:
        st.markdown(f"**Entry ID:** `{entry.get('entry_id')}`")
    with col_queue:
        queue = entry.get("routing", {}).get("queue", "?")
        badge_color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(queue, "⚪")
        st.markdown(f"**Queue:** {badge_color} {queue}")
    with col_agreement:
        metadata = entry.get("metadata", {})
        agr = metadata.get("agreement_counts", {})
        st.markdown(
            f"**Agreement:** Full={agr.get('full', 0)} "
            f"Partial={agr.get('partial', 0)} "
            f"Single={agr.get('single_source', 0)}"
        )

    st.markdown("---")

    # ── Highlighted original text ──
    st.markdown("#### 📄 Original Text (with detected PII)")
    entities = entry.get("entities", [])
    html = build_highlighted_html(entry.get("original_text", ""), entities)
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("")

    # ── Anonymized text preview ──
    with st.expander("👀 Anonymized Text Preview", expanded=False):
        st.text(entry.get("anonymized_text", ""))

    st.markdown("---")

    # ── Entity Review Table ──
    st.markdown("#### 🔍 Entity Review")

    if not entities:
        st.info("No entities detected in this entry.")
    else:
        # Initialize modified entities from entry entities on first load
        if (
            not st.session_state.get("modified_entities")
            or st.session_state.get("_loaded_entry_id") != entry.get("entry_id")
        ):
            st.session_state["modified_entities"] = [dict(e) for e in entities]
            st.session_state["modified_replacements"] = dict(entry.get("replacements", {}))
            st.session_state["_loaded_entry_id"] = entry.get("entry_id")

        mod_entities = st.session_state["modified_entities"]
        mod_replacements = st.session_state["modified_replacements"]

        for i, ent in enumerate(mod_entities):
            col_text, col_type, col_conf, col_repl, col_del = st.columns([3, 2, 1, 3, 1])

            with col_text:
                st.text(ent.get("text", ""))

            with col_type:
                current_type = ent.get("entity_type", "UNKNOWN")
                type_idx = ENTITY_TYPES.index(current_type) if current_type in ENTITY_TYPES else len(ENTITY_TYPES) - 1
                new_type = st.selectbox(
                    "Type",
                    ENTITY_TYPES,
                    index=type_idx,
                    key=f"type_{i}",
                    label_visibility="collapsed",
                )
                mod_entities[i]["entity_type"] = new_type

            with col_conf:
                conf = ent.get("confidence", 0)
                color = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.5 else "🔴")
                st.markdown(f"{color} {conf:.2f}")

            with col_repl:
                pii_text = ent.get("text", "")
                current_repl = mod_replacements.get(pii_text, "")
                new_repl = st.text_input(
                    "Replacement",
                    value=current_repl,
                    key=f"repl_{i}",
                    label_visibility="collapsed",
                )
                mod_replacements[pii_text] = new_repl

            with col_del:
                if st.button("🗑️", key=f"del_{i}", help="Remove this entity"):
                    mod_entities.pop(i)
                    st.rerun()

        # Regenerate replacements button
        if st.button("🔄 Regenerate All Replacements"):
            for ent in mod_entities:
                pii_text = ent.get("text", "")
                etype = ent.get("entity_type", "UNKNOWN")
                candidates = generate_replacements(etype, pii_text, n=3)
                if candidates:
                    mod_replacements[pii_text] = candidates[0]
            st.rerun()

    st.markdown("---")

    # ── Action Buttons ──
    st.markdown("#### ⚡ Actions")
    col_accept, col_flag, col_skip, col_nav = st.columns(4)

    annotator = st.session_state.get("annotator_name", "").strip()

    with col_accept:
        if st.button("✅ Accept", type="primary", use_container_width=True):
            if not annotator:
                st.error("Please enter your annotator name in the sidebar.")
                return
            warnings = action_accept(
                entry,
                st.session_state.get("modified_entities", entities),
                st.session_state.get("modified_replacements", entry.get("replacements", {})),
                annotator,
            )
            if warnings:
                st.warning(f"⚠️ {len(warnings)} quality warning(s) — saved anyway.")
                for w in warnings[:3]:
                    st.caption(f"  • [{w.get('severity')}] {w.get('message')}")
            else:
                st.success("Saved to gold standard!")
            time.sleep(0.5)
            _advance()

    with col_flag:
        flag_reason = st.text_input("Flag reason", key="flag_input", placeholder="Why?")
        if st.button("🚩 Flag", use_container_width=True):
            if not annotator:
                st.error("Please enter your annotator name in the sidebar.")
                return
            action_flag(entry, annotator, flag_reason or "No reason given")
            st.info("Entry flagged for expert review.")
            time.sleep(0.3)
            _advance()

    with col_skip:
        if st.button("⏭️ Skip", use_container_width=True):
            action_skip(entry, annotator or "anonymous")
            _advance()

    with col_nav:
        st.markdown("**Navigation**")
        sub1, sub2 = st.columns(2)
        with sub1:
            if st.button("⬅️ Prev") and idx > 0:
                st.session_state["current_index"] -= 1
                st.rerun()
        with sub2:
            if st.button("➡️ Next") and idx < len(queue_data) - 1:
                st.session_state["current_index"] += 1
                st.rerun()


def _advance():
    """Move to next entry and rerun."""
    st.session_state["current_index"] += 1
    st.session_state["modified_entities"] = []
    st.session_state["modified_replacements"] = {}
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
#  Statistics Tab
# ─────────────────────────────────────────────────────────────────────
def render_statistics_tab():
    st.markdown("### 📊 Annotation Statistics")

    # Gold standard stats
    gold_entries = read_jsonl(GOLD_STANDARD_PATH)
    if not gold_entries:
        st.info("No gold standard entries yet. Start annotating!")
        return

    st.metric("Total Gold Standard Entries", len(gold_entries))

    # Entity type distribution
    type_counts = Counter()
    for entry in gold_entries:
        for ent in entry.get("entities", []):
            type_counts[ent.get("entity_type", "UNKNOWN")] += 1

    if type_counts:
        st.markdown("#### Entity Type Distribution")
        df = pd.DataFrame(
            sorted(type_counts.items(), key=lambda x: -x[1]),
            columns=["Entity Type", "Count"],
        )
        st.bar_chart(df.set_index("Entity Type"))
        st.dataframe(df, use_container_width=True)

    # Annotator distribution
    annotator_counts = Counter()
    for entry in gold_entries:
        annotator = entry.get("metadata", {}).get("annotator", "unknown")
        annotator_counts[annotator] += 1

    if annotator_counts:
        st.markdown("#### Annotator Contributions")
        df_ann = pd.DataFrame(
            sorted(annotator_counts.items(), key=lambda x: -x[1]),
            columns=["Annotator", "Count"],
        )
        st.dataframe(df_ann, use_container_width=True)

    # Queue source distribution
    queue_counts = Counter()
    for entry in gold_entries:
        q = entry.get("metadata", {}).get("source_queue", "UNKNOWN")
        queue_counts[q] += 1

    if queue_counts:
        st.markdown("#### Source Queue Distribution")
        df_q = pd.DataFrame(
            sorted(queue_counts.items(), key=lambda x: -x[1]),
            columns=["Queue", "Count"],
        )
        st.dataframe(df_q, use_container_width=True)

    # Warnings summary
    total_warnings = sum(
        entry.get("metadata", {}).get("quality_warnings", 0) for entry in gold_entries
    )
    st.metric("Total Quality Warnings", total_warnings)

    # Annotation log timeline
    log_entries = read_jsonl(ANNOTATION_LOG_PATH)
    if log_entries:
        st.markdown("#### Recent Annotation Log")
        recent = log_entries[-20:]  # last 20
        df_log = pd.DataFrame(recent)
        if not df_log.empty:
            st.dataframe(df_log, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
#  Flagged Review Tab
# ─────────────────────────────────────────────────────────────────────
def render_flagged_tab():
    st.markdown("### 🚩 Flagged Entries")

    flagged = read_jsonl(FLAGGED_PATH)
    if not flagged:
        st.info("No flagged entries yet.")
        return

    st.metric("Total Flagged", len(flagged))

    for i, entry in enumerate(flagged):
        with st.expander(
            f"Entry {entry.get('entry_id')} — flagged by {entry.get('flagged_by', '?')}",
            expanded=False,
        ):
            st.markdown(f"**Reason:** {entry.get('flag_reason', 'N/A')}")
            st.markdown(f"**Flagged at:** {entry.get('flagged_at', '?')}")
            st.markdown(f"**Source queue:** {entry.get('source_queue', '?')}")
            st.text(entry.get("original_text", "")[:500])

            entities = entry.get("entities", [])
            if entities:
                df = pd.DataFrame([
                    {
                        "Text": e.get("text"),
                        "Type": e.get("entity_type"),
                        "Conf": e.get("confidence"),
                    }
                    for e in entities
                ])
                st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Load queue on first run
    if not st.session_state["queue_data"]:
        st.session_state["queue_data"] = load_queue(st.session_state["current_queue"])

    # Tabs
    tab_annotate, tab_stats, tab_flagged = st.tabs([
        "📝 Annotation",
        "📊 Statistics",
        "🚩 Flagged",
    ])

    with tab_annotate:
        render_annotation_tab()

    with tab_stats:
        render_statistics_tab()

    with tab_flagged:
        render_flagged_tab()


if __name__ == "__main__":
    main()

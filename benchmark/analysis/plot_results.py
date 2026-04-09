"""
SAHA-AL Benchmark — Publication-Quality Results Visualization

Generates four figures:
  1. task2_comparison.png  — ELR + BERTScore grouped bar chart
  2. attack_heatmap.png    — Systems × Attack types recovery-rate heatmap
  3. failure_taxonomy.png  — Stacked horizontal bar of failure categories
  4. detection_recall.png  — Per-type recall heatmap for rule-based detectors

Usage:
  python -m analysis.plot_results [--results-dir results] [--eval-dir Results] [--output-dir figures]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── 1. Task 2: ELR + BERTScore comparison ──

def plot_task2_comparison(eval_dir, output_path):
    systems = [
        ("BART-base",     "eval_anon_bart-base-pii.json"),
        ("Flan-T5",       "eval_anon_flan-t5-small-pii.json"),
        ("T5-small",      "eval_anon_t5-small-pii.json"),
        ("DistilBART",    "eval_anon_distilbart-pii.json"),
        ("T5-eff-tiny",   "eval_anon_t5-efficient-tiny-pii.json"),
        ("spaCy+Faker",   "eval_anon_spacy.json"),
        ("Presidio",      "eval_anon_presidio.json"),
        ("Regex+Faker",   "eval_anon_regex.json"),
    ]

    names, elrs, berts = [], [], []
    for name, fname in systems:
        d = _load(os.path.join(eval_dir, fname))
        if d:
            names.append(name)
            elrs.append(d.get("elr", 0))
            berts.append(d.get("bertscore_f1", 0))

    if not names:
        print("[SKIP] No Task 2 eval files found")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    x = np.arange(len(names))
    colors_elr = ["#2196F3" if e < 5 else "#FF9800" if e < 30 else "#f44336" for e in elrs]
    ax1.bar(x, elrs, color=colors_elr, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Entity Leakage Rate ↓ (%)", fontsize=11)
    ax1.set_title("Task 2: Text Anonymization Quality", fontsize=13, fontweight="bold")
    for i, v in enumerate(elrs):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(0, max(elrs) * 1.15)

    ax2.bar(x, berts, color="#4CAF50", edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("BERTScore F1 ↑", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    for i, v in enumerate(berts):
        ax2.text(i, v - 1.5, f"{v:.1f}", ha="center", va="top", fontsize=8, color="white",
                 fontweight="bold")
    ax2.set_ylim(min(berts) - 5, 100)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── 2. Attack-System heatmap ──

def plot_attack_heatmap(results_dir, eval_dir, output_path):
    system_cfg = [
        ("BART-base",   "eval_anon_bart-base-pii.json",  "eval_privacy_bart.json",     "eval_privacy_bart_lrr.json"),
        ("spaCy+Faker", "eval_anon_spacy.json",           "eval_privacy_spacy.json",    None),
        ("Presidio",    "eval_anon_presidio.json",         "eval_privacy_presidio.json", "eval_privacy_presidio_lrr.json"),
    ]

    attack_names = ["ELR", "CRR-3", "ERA@1", "ERA@5", "UAC", "LRR (exact)"]
    rows, row_labels = [], []

    for name, anon_f, priv_f, lrr_f in system_cfg:
        anon = _load(os.path.join(eval_dir, anon_f))
        priv = _load(os.path.join(results_dir, priv_f))
        lrr = _load(os.path.join(results_dir, lrr_f)) if lrr_f else None

        if not anon or not priv:
            continue

        era = priv.get("era") or {}
        lrr_data = (lrr or {}).get("lrr") or {}
        row = [
            anon.get("elr", 0),
            priv.get("crr3", 0),
            era.get("era_top1", 0),
            era.get("era_top5", 0),
            priv.get("uac", 0),
            lrr_data.get("lrr_exact", float("nan")),
        ]
        rows.append(row)
        row_labels.append(name)

    if not rows:
        print("[SKIP] Insufficient data for attack heatmap")
        return

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.9 + 1)))

    mask = np.isnan(data)
    display_data = np.where(mask, 0, data)

    im = ax.imshow(display_data, cmap="RdYlGn_r", aspect="auto", vmin=0,
                   vmax=max(50, np.nanmax(data) * 1.1))

    ax.set_xticks(np.arange(len(attack_names)))
    ax.set_xticklabels(attack_names, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    for i in range(len(row_labels)):
        for j in range(len(attack_names)):
            if mask[i, j]:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
            else:
                v = data[i, j]
                color = "white" if v > 20 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=10,
                        fontweight="bold", color=color)

    ax.set_title("Privacy Under Attack: Recovery Rates by System × Attack Type",
                 fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, label="Recovery Rate (%)", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── 3. Failure Taxonomy stacked bar ──

def plot_failure_taxonomy(results_dir, output_path):
    system_files = [
        ("BART-base",  "failure_bart-base-pii.json"),
        ("spaCy+Faker","failure_spacy.json"),
        ("Presidio",   "failure_presidio.json"),
        ("Regex+Faker","failure_regex.json"),
    ]

    categories = ["clean", "full_leak", "ghost_leak", "boundary", "format_break"]
    cat_labels = ["Clean", "Full Leak", "Context Retention", "Boundary Error", "Format Break"]
    cat_colors = ["#4CAF50", "#f44336", "#FF9800", "#9C27B0", "#795548"]

    sys_names, sys_data = [], []
    for name, fname in system_files:
        d = _load(os.path.join(results_dir, fname))
        if d:
            counts = d["counts"]
            total = sum(counts.values())
            if total > 0:
                sys_names.append(name)
                sys_data.append([counts.get(c, 0) / total * 100 for c in categories])

    if not sys_names:
        print("[SKIP] No failure taxonomy files found")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(sys_names) * 0.8 + 1)))
    y = np.arange(len(sys_names))
    data = np.array(sys_data)

    left = np.zeros(len(sys_names))
    for j, (cat_label, color) in enumerate(zip(cat_labels, cat_colors)):
        bars = ax.barh(y, data[:, j], left=left, color=color, edgecolor="white",
                       linewidth=0.5, label=cat_label)
        for i, bar in enumerate(bars):
            w = bar.get_width()
            if w > 5:
                ax.text(left[i] + w / 2, i, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += data[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels(sys_names, fontsize=11)
    ax.set_xlabel("Percentage of Entities (%)", fontsize=11)
    ax.set_title("Failure Taxonomy: Where Do Systems Fail?", fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=9)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── 4. Detection per-type recall heatmap ──

def plot_detection_recall(results_dir, output_path):
    det_files = [
        ("Regex+Faker", "eval_det_regex.json"),
        ("spaCy+Faker", "eval_det_spacy.json"),
        ("Presidio",    "eval_det_presidio.json"),
    ]

    sys_names, type_recalls = [], []
    all_types = set()

    for name, fname in det_files:
        d = _load(os.path.join(results_dir, fname))
        if d and "per_type_recall" in d:
            sys_names.append(name)
            ptr = d["per_type_recall"]
            type_recalls.append(ptr)
            all_types.update(ptr.keys())

    if not sys_names:
        print("[SKIP] No detection eval files found")
        return

    important_types = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "DATE", "FULLNAME",
                       "ADDRESS", "ID_NUMBER", "ZIPCODE"]
    types = [t for t in important_types if t in all_types]

    data = np.zeros((len(sys_names), len(types)))
    for i, ptr in enumerate(type_recalls):
        for j, t in enumerate(types):
            data[i, j] = ptr.get(t, 0)

    fig, ax = plt.subplots(figsize=(10, max(3, len(sys_names) * 0.8 + 1)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(sys_names)))
    ax.set_yticklabels(sys_names, fontsize=11)

    for i in range(len(sys_names)):
        for j in range(len(types)):
            v = data[i, j]
            color = "white" if v < 50 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_title("PII Detection: Per-Type Recall (Exact Match)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Recall (%)", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Benchmark — Visualizations")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--eval-dir", default="Results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating publication figures...")
    plot_task2_comparison(args.eval_dir, os.path.join(args.output_dir, "task2_comparison.png"))
    plot_attack_heatmap(args.results_dir, args.eval_dir,
                        os.path.join(args.output_dir, "attack_heatmap.png"))
    plot_failure_taxonomy(args.results_dir, os.path.join(args.output_dir, "failure_taxonomy.png"))
    plot_detection_recall(args.results_dir, os.path.join(args.output_dir, "detection_recall.png"))
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ==============================================================================
# eval_all.py — Unified Post-Training Evaluation for All 4 Models
# ==============================================================================
# Run AFTER training on each Kaggle account. Loads saved checkpoints and
# produces comprehensive, comparable metrics that show gradations between
# models in privacy, utility, robustness, and edge-case handling.
#
# Usage:
#   # Evaluate a single model:
#   python eval_all.py --model 2 --checkpoint-dir /kaggle/input/model2-outputs
#
#   # Evaluate models 1 & 3 (trained together):
#   python eval_all.py --model 1 3 --checkpoint-dir /kaggle/input/model13-outputs
#
#   # Evaluate all 4 (after gathering all outputs into one dataset):
#   python eval_all.py --model 1 2 3 4 --checkpoint-dir /kaggle/input/all-outputs
#
#   # Quick sanity check:
#   python eval_all.py --model 1 --checkpoint-dir outputs --quick
#
#   # Include prompt-injection eval (slow but thorough):
#   python eval_all.py --model 1 2 3 4 --checkpoint-dir outputs --prompt-injection
#
# Expected directory structure under --checkpoint-dir:
#   model1_baseline/
#       censor/              (tokenizer + model weights)
#       hallucinator/        (tokenizer + PEFT adapter)
#       results.json         (training-time results, optional)
#   model2_advanced/
#       censor/
#       hallucinator/
#   model3_rephraser/
#       censor/  hallucinator/  rephraser/
#   model4_semantic/
#       paraphraser/
# ==============================================================================

import os, sys, argparse, json, time, gc, torch, re, warnings
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from common import (
    CFG, DEVICE, log, install_deps, cleanup_gpu,
    load_ai4privacy, language_stratified_split,
    get_source_text, quick_subsample,
    evaluate_anonymization, run_curated_eval,
    compute_leakage, compute_per_entity_leakage,
    compute_contextual_reidentification_risk,
    plot_comparison, plot_evaluation_summary,
    BIO_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    EntityConsistency, _FP16_OK, _BF16_OK, BASE_DIR,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model Loaders — Reconstruct anonymize_fn from saved checkpoints
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_model_dir(model_dir):
    """Find the best available weights directory.

    If the final model files exist directly in model_dir, return model_dir.
    Otherwise, look for the highest-numbered checkpoint-NNNN/ subdir.
    This handles the case where training timed out before save_model() ran.
    """
    # Check for final saved model (NER or LoRA adapter)
    for marker in ["model.safetensors", "pytorch_model.bin",
                    "adapter_config.json"]:
        if os.path.exists(os.path.join(model_dir, marker)):
            return model_dir

    # Fall back to latest checkpoint subdir
    ckpts = [d for d in os.listdir(model_dir)
             if os.path.isdir(os.path.join(model_dir, d))
             and re.match(r"checkpoint-\d+$", d)] if os.path.isdir(model_dir) else []
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split("-")[1]))
        resolved = os.path.join(model_dir, ckpts[-1])
        log.info(f"  No final model in {model_dir} — using {ckpts[-1]}")
        return resolved

    raise FileNotFoundError(
        f"No model weights or checkpoints found in {model_dir}")

def load_model1(ckpt_dir):
    """Load Model 1 (Baseline) from checkpoint directory."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from peft import PeftModel

    censor_dir = _resolve_model_dir(
        os.path.join(ckpt_dir, "model1_baseline", "censor"))
    halluc_dir = _resolve_model_dir(
        os.path.join(ckpt_dir, "model1_baseline", "hallucinator"))

    log.info(f"Loading Model 1 censor from {censor_dir}")
    censor_tok = AutoTokenizer.from_pretrained(censor_dir)
    censor_model = AutoModelForTokenClassification.from_pretrained(
        censor_dir, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE).eval()
    from common import fix_deberta_params
    censor_model = fix_deberta_params(censor_model)

    log.info(f"Loading Model 1 hallucinator from {halluc_dir}")
    halluc_tok = AutoTokenizer.from_pretrained(halluc_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.HALLUC_BASE, device_map="auto", torch_dtype=torch.float16)
    halluc_model = PeftModel.from_pretrained(base_model, halluc_dir)
    halluc_model.eval()

    from model1_baseline import anonymize_baseline
    def anonymize_fn(text):
        return anonymize_baseline(text, censor_model, censor_tok,
                                  halluc_model, halluc_tok)
    return anonymize_fn, {"censor": (censor_model, censor_tok),
                          "halluc": (halluc_model, halluc_tok)}


def load_model2(ckpt_dir):
    """Load Model 2 (Advanced) from checkpoint directory."""
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from peft import PeftModel

    censor_raw = os.path.join(ckpt_dir, "model2_advanced", "censor")
    censor_dir = _resolve_model_dir(censor_raw)
    halluc_dir = _resolve_model_dir(
        os.path.join(ckpt_dir, "model2_advanced", "hallucinator"))

    log.info(f"Loading Model 2 censor from {censor_dir}")
    # Tokenizer is always saved in the root censor dir (not inside checkpoint)
    tok_dir = censor_raw if os.path.exists(
        os.path.join(censor_raw, "tokenizer.json")) else censor_dir
    censor_tok = AutoTokenizer.from_pretrained(tok_dir)
    # Model 2 uses custom MultiTaskNERModel, not AutoModelForTokenClassification
    from model2_advanced import MultiTaskNERModel
    censor_model = MultiTaskNERModel(CFG.CENSOR_ADV, NUM_LABELS)
    # Load saved weights (may be pytorch_model.bin or model.safetensors)
    pt_path = os.path.join(censor_dir, "pytorch_model.bin")
    st_path = os.path.join(censor_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        censor_model.load_state_dict(load_file(st_path))
    elif os.path.exists(pt_path):
        censor_model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"No censor weights found in {censor_dir}")
    censor_model = censor_model.float().to(DEVICE).eval()

    log.info(f"Loading Model 2 hallucinator from {halluc_dir}")
    halluc_tok = AutoTokenizer.from_pretrained(halluc_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.HALLUC_ADV, device_map="auto", torch_dtype=torch.float16)
    halluc_model = PeftModel.from_pretrained(base_model, halluc_dir)
    halluc_model.eval()

    from model2_advanced import anonymize_advanced
    consistency = EntityConsistency()
    def anonymize_fn(text):
        return anonymize_advanced(text, censor_model, censor_tok,
                                  halluc_model, halluc_tok,
                                  consistency=consistency)
    return anonymize_fn, {"censor": (censor_model, censor_tok),
                          "halluc": (halluc_model, halluc_tok)}


def load_model3(ckpt_dir):
    """Load Model 3 (Rephraser) from checkpoint directory."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    from peft import PeftModel

    # Model 3 reuses Model 1's censor + hallucinator
    censor_raw = os.path.join(ckpt_dir, "model1_baseline", "censor")
    halluc_raw = os.path.join(ckpt_dir, "model1_baseline", "hallucinator")
    reph_raw = os.path.join(ckpt_dir, "model3_rephraser", "rephraser")

    # Fallback: if run_model13 saved censor under model3 too
    if not os.path.isdir(censor_raw):
        censor_raw = os.path.join(ckpt_dir, "model3_rephraser", "censor")
        halluc_raw = os.path.join(ckpt_dir, "model3_rephraser", "hallucinator")

    censor_dir = _resolve_model_dir(censor_raw)
    halluc_dir = _resolve_model_dir(halluc_raw)
    reph_dir = _resolve_model_dir(reph_raw)

    log.info(f"Loading Model 3 censor from {censor_dir}")
    censor_tok = AutoTokenizer.from_pretrained(censor_dir)
    censor_model = AutoModelForTokenClassification.from_pretrained(
        censor_dir, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE).eval()
    from common import fix_deberta_params
    censor_model = fix_deberta_params(censor_model)

    log.info(f"Loading Model 3 hallucinator from {halluc_dir}")
    halluc_tok = AutoTokenizer.from_pretrained(halluc_dir)
    base_halluc = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.HALLUC_BASE, device_map="auto", torch_dtype=torch.float16)
    halluc_model = PeftModel.from_pretrained(base_halluc, halluc_dir)
    halluc_model.eval()

    log.info(f"Loading Model 3 rephraser from {reph_dir}")
    reph_tok = AutoTokenizer.from_pretrained(reph_dir)
    base_reph = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.REPHRASER_BASE, device_map="auto", torch_dtype=torch.float16)
    reph_model = PeftModel.from_pretrained(base_reph, reph_dir)
    reph_model.eval()

    from model1_baseline import anonymize_baseline
    from model3_rephraser import anonymize_rephraser
    def anonymize_fn(text):
        return anonymize_rephraser(text, censor_model, censor_tok,
                                   halluc_model, halluc_tok,
                                   reph_model, reph_tok)
    return anonymize_fn, {"censor": (censor_model, censor_tok),
                          "halluc": (halluc_model, halluc_tok),
                          "rephraser": (reph_model, reph_tok)}


def load_model4(ckpt_dir):
    """Load Model 4 (Semantic Paraphraser) from checkpoint directory."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    para_dir = _resolve_model_dir(
        os.path.join(ckpt_dir, "model4_semantic", "paraphraser"))

    log.info(f"Loading Model 4 paraphraser from {para_dir}")
    para_tok = AutoTokenizer.from_pretrained(para_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        CFG.PARAPHRASER_BASE, device_map="auto", torch_dtype=torch.float16)
    para_model = PeftModel.from_pretrained(base_model, para_dir)
    para_model.eval()

    from model4_semantic import anonymize_semantic
    def anonymize_fn(text):
        return anonymize_semantic(text, para_model, para_tok)
    return anonymize_fn, {"paraphraser": (para_model, para_tok)}


MODEL_LOADERS = {
    1: ("Model 1 (Baseline)", load_model1),
    2: ("Model 2 (Advanced)", load_model2),
    3: ("Model 3 (Rephraser)", load_model3),
    4: ("Model 4 (Semantic)", load_model4),
}


# ═══════════════════════════════════════════════════════════════════════════
# Extended Evaluation — Multi-Granularity CRR
# ═══════════════════════════════════════════════════════════════════════════

def multi_granularity_crr(originals, anonymized):
    """Compute CRR at 2,3,4-gram levels for fine-grained privacy analysis."""
    results = {}
    for ng in [2, 3, 4]:
        crr = compute_contextual_reidentification_risk(
            originals, anonymized, n_grams=ng)
        results[f"crr_{ng}gram"] = crr["crr"]
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Per-Difficulty Leakage Breakdown
# ═══════════════════════════════════════════════════════════════════════════

def compute_difficulty_leakage(originals, anonymized):
    """Split examples by text complexity and compute leakage for each tier.
    Short texts (<20 words) are 'simple', medium (20-60) are 'moderate',
    long (>60) are 'complex'. This reveals whether models struggle more
    on longer, denser texts."""
    buckets = {"simple": ([], []), "moderate": ([], []), "complex": ([], [])}
    for orig, anon in zip(originals, anonymized):
        wc = len(orig.split())
        if wc < 20:
            buckets["simple"][0].append(orig)
            buckets["simple"][1].append(anon)
        elif wc < 60:
            buckets["moderate"][0].append(orig)
            buckets["moderate"][1].append(anon)
        else:
            buckets["complex"][0].append(orig)
            buckets["complex"][1].append(anon)

    results = {}
    for level, (origs, anons) in buckets.items():
        if origs:
            leak = compute_leakage(origs, anons)
            crr = compute_contextual_reidentification_risk(origs, anons)
            results[level] = {
                "n": len(origs),
                "entity_leak_rate": leak["entity_leak_rate"],
                "crr": crr["crr"],
            }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Consistency Score (measures if same entity → same replacement)
# ═══════════════════════════════════════════════════════════════════════════

def measure_entity_consistency(anonymize_fn, n_trials=30):
    """Feed the same text multiple times and measure how consistently
    the same entities get the same replacements. Higher = better."""
    test_texts = [
        "Contact John Smith at john.smith@email.com about meeting in London.",
        "Maria Garcia from 42 Oak Street, Chicago called at 555-0147.",
        "Dr. Wei Zhang, patient ID P-2024-001, Room 42B, hospital in Boston.",
    ]
    scores = []
    for text in test_texts:
        outputs = [anonymize_fn(text) for _ in range(n_trials)]
        # Measure pairwise similarity
        unique = len(set(outputs))
        # 1.0 = perfectly consistent (always same output), 0.0 = random each time
        consistency = 1.0 - (unique - 1) / max(n_trials - 1, 1)
        scores.append(consistency)
    return {
        "mean_consistency": round(float(np.mean(scores)), 3),
        "per_text": [round(s, 3) for s in scores],
    }


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX Table Exporter
# ═══════════════════════════════════════════════════════════════════════════

def export_latex_table(all_results, save_path):
    """Generate a publication-ready LaTeX comparison table."""
    models = list(all_results.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparative evaluation of four anonymization models.}",
        r"\label{tab:comparison}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
    ]

    # Header
    header = "Metric"
    for m in models:
        short = m.replace("Model ", "M").split("(")[0].strip()
        header += f" & {short}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Privacy metrics
    lines.append(r"\multicolumn{" + str(len(models) + 1) + r"}{l}{\textit{Privacy (lower $=$ better)}} \\")
    # Entity Leak Rate
    row = "Entity Leak (\\%)"
    for m in models:
        v = all_results[m].get("leakage", {}).get("entity_leak_rate", -1)
        best = min(all_results[m2].get("leakage", {}).get("entity_leak_rate", 999)
                   for m2 in models)
        cell = f"{v:.1f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    # CRR
    row = "CRR 3-gram (\\%)"
    for m in models:
        v = all_results[m].get("crr", {}).get("crr", -1)
        best = min(all_results[m2].get("crr", {}).get("crr", 999) for m2 in models)
        cell = f"{v:.1f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(models) + 1) + r"}{l}{\textit{Utility (higher $=$ better)}} \\")

    # ROUGE-L
    row = "ROUGE-L"
    for m in models:
        v = all_results[m].get("rouge", {}).get("rougeL", -1)
        best = max(all_results[m2].get("rouge", {}).get("rougeL", -999) for m2 in models)
        cell = f"{v:.4f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    # BLEU
    row = "BLEU"
    for m in models:
        v = all_results[m].get("bleu", -1)
        best = max(all_results[m2].get("bleu", -999) for m2 in models)
        cell = f"{v:.2f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    # BERTScore
    row = "BERTScore F1"
    for m in models:
        v = all_results[m].get("bertscore_f1", -1)
        best = max(all_results[m2].get("bertscore_f1", -999) for m2 in models)
        cell = f"{v:.4f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(models) + 1) + r"}{l}{\textit{Curated Eval (37 examples)}} \\")

    # Curated PII rate
    row = "PII Anonymised (\\%)"
    for m in models:
        v = all_results[m].get("curated", {}).get("pii_rate", -1)
        best = max(all_results[m2].get("curated", {}).get("pii_rate", -999) for m2 in models)
        cell = f"{v:.1f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    # Curated No-PII rate
    row = "No-PII Preserved (\\%)"
    for m in models:
        v = all_results[m].get("curated", {}).get("nopii_rate", -1)
        best = max(all_results[m2].get("curated", {}).get("nopii_rate", -999) for m2 in models)
        cell = f"{v:.1f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    # Contextual
    row = "Contextual Gen. (\\%)"
    for m in models:
        v = all_results[m].get("curated", {}).get("contextual_rate", -1)
        best = max(all_results[m2].get("curated", {}).get("contextual_rate", -999) for m2 in models)
        cell = f"{v:.1f}"
        if v == best and v >= 0:
            cell = r"\textbf{" + cell + "}"
        row += f" & {cell}"
    row += r" \\"
    lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(tex)
    log.info(f"LaTeX table saved → {save_path}")
    return tex


# ═══════════════════════════════════════════════════════════════════════════
# Publication-Quality Comparison Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_comprehensive_comparison(all_results, save_dir):
    """Generate multi-panel comparison plots showing model gradations."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    names = list(all_results.keys())
    short_names = [n.replace("Model ", "M").split("(")[0].strip() for n in names]
    n = len(names)
    x = np.arange(n)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][:n]

    # ── Figure 1: Privacy vs Utility trade-off (the key figure) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Entity Leak Rate
    ax = axes[0, 0]
    vals = [all_results[m]["leakage"]["entity_leak_rate"] for m in names]
    bars = ax.bar(x, vals, 0.6, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Entity Leak Rate (%)"); ax.set_title("Entity Leak Rate ↓", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
                ha="center", fontsize=9)

    # (0,1) CRR
    ax = axes[0, 1]
    vals = [all_results[m].get("crr", {}).get("crr", 0) for m in names]
    bars = ax.bar(x, vals, 0.6, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("CRR (%)"); ax.set_title("Contextual Re-ID Risk ↓", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}%",
                ha="center", fontsize=9)

    # (0,2) Per-Entity Leakage heatmap-style grouped bar
    ax = axes[0, 2]
    pel_types = set()
    for m in names:
        pel_types.update(all_results[m].get("per_entity_leakage", {}).keys())
    pel_types = sorted(pel_types)
    if pel_types:
        bar_w = 0.8 / n
        for i, m in enumerate(names):
            pel = all_results[m].get("per_entity_leakage", {})
            vals = [pel.get(et, {}).get("rate", 0) for et in pel_types]
            ax.barh(np.arange(len(pel_types)) + i * bar_w, vals,
                    bar_w, label=short_names[i], color=colors[i], alpha=0.85)
        ax.set_yticks(np.arange(len(pel_types)) + bar_w * (n-1) / 2)
        ax.set_yticklabels(pel_types, fontsize=8)
        ax.set_xlabel("Leak Rate (%)")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
    ax.set_title("Per-Entity Leakage ↓", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # (1,0) ROUGE-L + BLEU
    ax = axes[1, 0]
    bar_w = 0.35
    rl = [all_results[m].get("rouge", {}).get("rougeL", 0) * 100 for m in names]
    bl = [all_results[m].get("bleu", 0) for m in names]
    ax.bar(x - bar_w/2, rl, bar_w, label="ROUGE-L", color="#3498db", alpha=0.85)
    ax.bar(x + bar_w/2, bl, bar_w, label="BLEU", color="#1abc9c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Score"); ax.set_title("Text Quality ↑", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # (1,1) BERTScore
    ax = axes[1, 1]
    vals = [all_results[m].get("bertscore_f1", 0) * 100 for m in names]
    bars = ax.bar(x, vals, 0.6, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("BERTScore F1 (%)"); ax.set_title("Semantic Preservation ↑", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}",
                ha="center", fontsize=9)

    # (1,2) Curated eval (PII / No-PII / Contextual)
    ax = axes[1, 2]
    bar_w = 0.25
    pii = [all_results[m].get("curated", {}).get("pii_rate", 0) for m in names]
    nopii = [all_results[m].get("curated", {}).get("nopii_rate", 0) for m in names]
    ctx = [all_results[m].get("curated", {}).get("contextual_rate", 0) for m in names]
    ax.bar(x - bar_w, pii, bar_w, label="PII Anonymised", color="#e74c3c", alpha=0.85)
    ax.bar(x, nopii, bar_w, label="No-PII Preserved", color="#2ecc71", alpha=0.85)
    ax.bar(x + bar_w, ctx, bar_w, label="Contextual Gen.", color="#9b59b6", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Rate (%)"); ax.set_title("Curated Eval (37 examples) ↑", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Privacy-Preserving Text Anonymization: 4-Model Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "full_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    log.info(f"Full comparison plot → {path}")

    # ── Figure 2: Privacy-Utility Pareto plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    leak_rates = [all_results[m]["leakage"]["entity_leak_rate"] for m in names]
    bert_scores = [all_results[m].get("bertscore_f1", 0) * 100 for m in names]
    for i, (lr, bs) in enumerate(zip(leak_rates, bert_scores)):
        ax.scatter(lr, bs, s=200, c=colors[i], zorder=5, edgecolors="black", linewidth=1)
        ax.annotate(short_names[i], (lr, bs), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")
    ax.set_xlabel("Entity Leak Rate (%) ← lower is better", fontsize=12)
    ax.set_ylabel("BERTScore F1 (%) → higher is better", fontsize=12)
    ax.set_title("Privacy-Utility Trade-off (Pareto Front)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Arrow annotations
    ax.annotate("", xy=(0, ax.get_ylim()[1]), xytext=(ax.get_xlim()[1], ax.get_ylim()[0]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.text(0.02, 0.98, "← IDEAL", transform=ax.transAxes, fontsize=10,
            va="top", color="green", fontweight="bold")
    path = os.path.join(save_dir, "pareto_tradeoff.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    log.info(f"Pareto plot → {path}")

    # ── Figure 3: Difficulty breakdown (if available) ──
    has_diff = any("difficulty_leakage" in all_results[m] for m in names)
    if has_diff:
        fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        levels = ["simple", "moderate", "complex"]
        for idx, metric_key in enumerate(["entity_leak_rate", "crr"]):
            ax = axes2[idx]
            bar_w = 0.8 / n
            for i, m in enumerate(names):
                diff = all_results[m].get("difficulty_leakage", {})
                vals = [diff.get(lv, {}).get(metric_key, 0) for lv in levels]
                ax.bar(np.arange(len(levels)) + i * bar_w, vals,
                       bar_w, label=short_names[i], color=colors[i])
            ax.set_xticks(np.arange(len(levels)) + bar_w * (n-1) / 2)
            ax.set_xticklabels([l.capitalize() for l in levels])
            title = "Entity Leak ↓" if idx == 0 else "CRR ↓"
            ax.set_title(f"By Text Complexity: {title}", fontweight="bold")
            ax.set_ylabel("%"); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = os.path.join(save_dir, "difficulty_breakdown.png")
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        log.info(f"Difficulty breakdown plot → {path}")

    # ── Figure 4: Multi-granularity CRR (if available) ──
    has_mg = any("multi_crr" in all_results[m] for m in names)
    if has_mg:
        fig, ax = plt.subplots(figsize=(8, 5))
        grams = ["crr_2gram", "crr_3gram", "crr_4gram"]
        gram_labels = ["2-gram", "3-gram", "4-gram"]
        bar_w = 0.8 / n
        for i, m in enumerate(names):
            mg = all_results[m].get("multi_crr", {})
            vals = [mg.get(g, 0) for g in grams]
            ax.bar(np.arange(len(grams)) + i * bar_w, vals,
                   bar_w, label=short_names[i], color=colors[i])
        ax.set_xticks(np.arange(len(grams)) + bar_w * (n-1) / 2)
        ax.set_xticklabels(gram_labels)
        ax.set_ylabel("CRR (%)"); ax.set_title("Multi-Granularity CRR ↓", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        path = os.path.join(save_dir, "multi_crr.png")
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        log.info(f"Multi-CRR plot → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Detailed Comparison Table (console + file)
# ═══════════════════════════════════════════════════════════════════════════

def print_full_comparison(all_results, save_dir):
    """Print a comprehensive multi-section comparison table."""
    os.makedirs(save_dir, exist_ok=True)
    names = list(all_results.keys())
    col_w = max(12, max(len(n) for n in names) + 2)

    lines = []
    def _p(s):
        lines.append(s)
        log.info(s)

    _p("\n" + "═" * 90)
    _p("  COMPREHENSIVE MODEL COMPARISON")
    _p("═" * 90)

    # Section 1: Core metrics
    _p("\n  ┌─ CORE METRICS " + "─" * 70)
    header = f"  {'Metric':<30}"
    for name in names:
        header += f" {name:>{col_w}}"
    _p(header)
    _p("  " + "─" * (30 + col_w * len(names) + len(names)))

    metrics = [
        ("Entity Leak Rate ↓", lambda r: f"{r['leakage']['entity_leak_rate']:.1f}%"),
        ("Token Leak Rate ↓",  lambda r: f"{r['leakage']['leakage_rate']:.1f}%"),
        ("CRR (3-gram) ↓",    lambda r: f"{r.get('crr', {}).get('crr', 0):.1f}%"),
        ("ROUGE-1 ↑",         lambda r: f"{r.get('rouge', {}).get('rouge1', 0):.4f}"),
        ("ROUGE-2 ↑",         lambda r: f"{r.get('rouge', {}).get('rouge2', 0):.4f}"),
        ("ROUGE-L ↑",         lambda r: f"{r.get('rouge', {}).get('rougeL', 0):.4f}"),
        ("BLEU ↑",            lambda r: f"{r.get('bleu', 0):.2f}"),
        ("BERTScore F1 ↑",    lambda r: f"{r.get('bertscore_f1', 0):.4f}"),
        ("Avg Orig Words",     lambda r: f"{r.get('length_stats', {}).get('orig_mean_words', 0):.0f}"),
        ("Avg Anon Words",     lambda r: f"{r.get('length_stats', {}).get('anon_mean_words', 0):.0f}"),
        ("Length Ratio",       lambda r: f"{r.get('length_stats', {}).get('ratio_mean', 0):.3f}"),
    ]
    for label, fn in metrics:
        row = f"  {label:<30}"
        for name in names:
            row += f" {fn(all_results[name]):>{col_w}}"
        _p(row)

    # Section 2: Curated eval
    _p("\n  ┌─ CURATED EVALUATION (37 examples) " + "─" * 50)
    curated_metrics = [
        ("PII Anonymised ↑",    lambda r: f"{r.get('curated', {}).get('pii_rate', 0):.1f}%"),
        ("No-PII Preserved ↑",  lambda r: f"{r.get('curated', {}).get('nopii_rate', 0):.1f}%"),
        ("Contextual Gen. ↑",   lambda r: f"{r.get('curated', {}).get('contextual_rate', 0):.1f}%"),
    ]
    for label, fn in curated_metrics:
        row = f"  {label:<30}"
        for name in names:
            row += f" {fn(all_results[name]):>{col_w}}"
        _p(row)

    # Section 2b: Per-difficulty breakdown from curated
    for diff in ["easy", "medium", "hard"]:
        row = f"  Curated {diff.capitalize():<21}"
        for name in names:
            pd = all_results[name].get("curated", {}).get("per_difficulty", {}).get(diff, {})
            rate = pd.get("rate", 0)
            row += f" {rate:>{col_w-1}.1f}%"
        _p(row)

    # Section 3: Multi-granularity CRR
    if any("multi_crr" in all_results[m] for m in names):
        _p("\n  ┌─ MULTI-GRANULARITY CRR ↓ " + "─" * 58)
        for ng in ["crr_2gram", "crr_3gram", "crr_4gram"]:
            label = ng.replace("crr_", "CRR ").replace("gram", "-gram")
            row = f"  {label:<30}"
            for name in names:
                v = all_results[name].get("multi_crr", {}).get(ng, 0)
                row += f" {v:>{col_w-1}.1f}%"
            _p(row)

    # Section 4: Difficulty leakage
    if any("difficulty_leakage" in all_results[m] for m in names):
        _p("\n  ┌─ LEAKAGE BY TEXT COMPLEXITY ↓ " + "─" * 54)
        for level in ["simple", "moderate", "complex"]:
            row = f"  {level.capitalize()+' Leak%':<30}"
            for name in names:
                v = all_results[name].get("difficulty_leakage", {}).get(level, {}).get("entity_leak_rate", 0)
                row += f" {v:>{col_w-1}.1f}%"
            _p(row)
        for level in ["simple", "moderate", "complex"]:
            row = f"  {level.capitalize()+' CRR%':<30}"
            for name in names:
                v = all_results[name].get("difficulty_leakage", {}).get(level, {}).get("crr", 0)
                row += f" {v:>{col_w-1}.1f}%"
            _p(row)

    # Section 5: Consistency
    if any("consistency" in all_results[m] for m in names):
        _p("\n  ┌─ ENTITY CONSISTENCY " + "─" * 64)
        row = f"  {'Consistency Score ↑':<30}"
        for name in names:
            v = all_results[name].get("consistency", {}).get("mean_consistency", 0)
            row += f" {v:>{col_w}.3f}"
        _p(row)

    # Section 6: Model 2 extras
    if any("mia" in all_results[m] for m in names):
        _p("\n  ┌─ MODEL 2 SPECIFIC " + "─" * 65)
        row = f"  {'MIA AUC (0.5=no mem.) ↓':<30}"
        for name in names:
            v = all_results[name].get("mia", {}).get("mia_auc", -1)
            cell = f"{v:.4f}" if v >= 0 else "N/A"
            row += f" {cell:>{col_w}}"
        _p(row)

    # Section 7: Prompt injection (if available)
    if any("prompt_injection" in all_results[m] for m in names):
        _p("\n  ┌─ PROMPT INJECTION ROBUSTNESS " + "─" * 54)
        row = f"  {'Mean PII Leak Rate ↓':<30}"
        for name in names:
            pi = all_results[name].get("prompt_injection", {})
            avg = float(np.mean([v.get("pii_leak_rate", 0)
                                 for v in pi.values() if isinstance(v, dict)
                                 ])) if pi else 0
            row += f" {avg:>{col_w-1}.1f}%"
        _p(row)

    _p("\n" + "═" * 90)
    _p("  ↓ = lower is better (privacy)  |  ↑ = higher is better (utility/quality)")
    _p("═" * 90)

    # Save to file
    with open(os.path.join(save_dir, "comparison_table.txt"), "w") as f:
        f.write("\n".join(lines))
    log.info(f"Comparison table saved → {os.path.join(save_dir, 'comparison_table.txt')}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified post-training evaluation for all 4 models")
    p.add_argument("--model", nargs="*", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4],
                   help="Which model(s) to evaluate (default: all)")
    p.add_argument("--checkpoint-dir", type=str, default="outputs",
                   help="Root directory containing model checkpoint subdirs")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer eval examples")
    p.add_argument("--n-eval", type=int, default=None,
                   help="Number of test examples to evaluate (default: CFG.NUM_EVAL)")
    p.add_argument("--prompt-injection", action="store_true",
                   help="Also run prompt injection eval (150 adversarial examples)")
    p.add_argument("--skip-mia", action="store_true",
                   help="Skip MIA evaluation for Model 2 (saves time)")
    p.add_argument("--skip-cross-lingual", action="store_true",
                   help="Skip cross-lingual eval for Model 2")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.QUICK_MODE = args.quick
    models_to_eval = sorted(set(args.model))
    ckpt_dir = args.checkpoint_dir
    n_eval = args.n_eval or (50 if args.quick else CFG.NUM_EVAL)

    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║       UNIFIED POST-TRAINING EVALUATION — ALL MODELS            ║")
    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info(f"Models: {models_to_eval}  |  Checkpoint dir: {ckpt_dir}")
    log.info(f"N eval: {n_eval}  |  Quick: {args.quick}")
    log.info(f"Prompt injection: {args.prompt_injection}")

    # ── Load data ──
    ds = load_ai4privacy()
    _, _, test_set, lang_col = language_stratified_split(ds)
    if args.quick:
        test_set = quick_subsample(test_set, min(n_eval, len(test_set)))
    N = min(n_eval, len(test_set))
    originals = [get_source_text(test_set[i]) for i in range(N)]
    log.info(f"Test examples: {N}")

    save_dir = os.path.join(CFG.OUTPUT_DIR, "eval_comparison")
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    start = time.time()

    for model_id in models_to_eval:
        model_name, loader_fn = MODEL_LOADERS[model_id]
        log.info(f"\n{'━' * 70}")
        log.info(f"  Loading & evaluating {model_name}")
        log.info(f"{'━' * 70}")

        try:
            anonymize_fn, components = loader_fn(ckpt_dir)
        except Exception as e:
            log.error(f"  Failed to load {model_name}: {e}")
            log.error(f"  Skipping. Make sure {ckpt_dir} has the right structure.")
            continue

        # ── Anonymize test set ──
        log.info(f"  Anonymizing {N} test examples …")
        anonymized = []
        for o in tqdm(originals, desc=f"Eval {model_name}"):
            try:
                anonymized.append(anonymize_fn(o))
            except Exception as e:
                log.warning(f"  Anonymization failed for one example: {e}")
                anonymized.append(o)  # fallback: unchanged

        # ── Core evaluation ──
        metrics = evaluate_anonymization(originals, anonymized, model_name)

        # ── Curated evaluation ──
        curated = run_curated_eval(anonymize_fn, model_name)
        metrics["curated"] = curated

        # ── Multi-granularity CRR ──
        log.info("  Computing multi-granularity CRR …")
        metrics["multi_crr"] = multi_granularity_crr(originals, anonymized)

        # ── Difficulty-stratified leakage ──
        log.info("  Computing difficulty-stratified leakage …")
        metrics["difficulty_leakage"] = compute_difficulty_leakage(
            originals, anonymized)

        # ── Entity consistency ──
        log.info("  Measuring entity consistency …")
        metrics["consistency"] = measure_entity_consistency(anonymize_fn)

        # ── Model 2 extras ──
        if model_id == 2:
            if not args.skip_mia and "halluc" in components:
                from model2_advanced import membership_inference_attack
                halluc_model, halluc_tok = components["halluc"]
                _, half_b_data, _, _ = language_stratified_split(ds)
                mia = membership_inference_attack(
                    halluc_model, halluc_tok, half_b_data, test_set, n_samples=200)
                metrics["mia"] = mia

            if not args.skip_cross_lingual:
                from model2_advanced import cross_lingual_eval
                cl = cross_lingual_eval(anonymize_fn, test_set, lang_col)
                metrics["cross_lingual"] = cl

        # ── Prompt injection eval ──
        if args.prompt_injection:
            try:
                from eval_prompt_injection import run_prompt_injection_eval
                log.info(f"  Running prompt injection eval (150 examples) …")
                pi_results = run_prompt_injection_eval(anonymize_fn)
                metrics["prompt_injection"] = pi_results
            except ImportError:
                log.warning("  eval_prompt_injection.py not found — skipping")

        # ── Per-model plot ──
        model_save = os.path.join(save_dir, f"model{model_id}")
        plot_evaluation_summary(metrics, model_save)

        # ── Save per-model results ──
        with open(os.path.join(model_save, "results.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        all_results[model_name] = metrics

        # Free GPU for next model
        del anonymize_fn, components
        cleanup_gpu()
        log.info(f"  {model_name} evaluation complete ✓")

    if not all_results:
        log.error("No models were successfully evaluated.")
        sys.exit(1)

    # ── Comparative outputs (need ≥1 model, 2+ for full comparison) ──
    log.info(f"\n{'═' * 70}")
    log.info(f"  GENERATING COMPARATIVE RESULTS ({len(all_results)} models)")
    log.info(f"{'═' * 70}")

    # Print detailed comparison table
    print_full_comparison(all_results, save_dir)

    # Generate plots
    if len(all_results) >= 2:
        plot_comprehensive_comparison(all_results, save_dir)
        # Also the simpler built-in comparison
        plot_comparison(all_results, save_dir)

    # LaTeX table for the paper
    export_latex_table(all_results, os.path.join(save_dir, "comparison_table.tex"))

    # Save all results JSON
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"All results saved → {os.path.join(save_dir, 'all_results.json')}")

    elapsed = time.time() - start
    log.info(f"\nTotal evaluation time: {elapsed/60:.1f} minutes")
    log.info("Done ✓")


if __name__ == "__main__":
    main()

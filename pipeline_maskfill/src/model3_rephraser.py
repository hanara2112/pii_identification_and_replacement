# ==============================================================================
# model3_rephraser.py — Model 3 (A-type): Context-Aware Rephraser Pipeline
# ==============================================================================
# Architecture:  Baseline Censor  ➜  Baseline Hallucinator  ➜  Rephraser
# The 3rd stage Rephraser (Flan-T5-base QLoRA) takes entity-replaced text
# and further generalises the surrounding context to suppress
# re-identification signals that survive entity swapping.
#
# Key Insight:
#   "Elon Musk is the richest man in the world."
#   After L1 anonymisation → "John Smith is the richest man in the world."
#   Anyone can still re-identify from context alone.
#   After L2 generalisation → "An individual is a prominent figure."
# ==============================================================================

import os, torch, numpy as np, logging, json
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm.auto import tqdm

from common import (
    CFG, DEVICE, SEED, BIO_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    _BF16_OK,
    model_dir, log,
    find_text_and_labels, get_source_text,
    tokenize_seq2seq,
    build_seq2seq_qlora, train_seq2seq,
    make_train_val, quick_subsample,
    evaluate_anonymization, evaluate_masking_quality,
    run_curated_eval,
    compute_contextual_reidentification_risk, cleanup_gpu,
    GENERALIZER, plot_evaluation_summary,
)

# Import baseline model functions to reuse Censor + Hallucinator
from model1_baseline import (
    build_censor_baseline, train_censor, anonymize_baseline,
    build_hallucinator_baseline, train_hallucinator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Rephraser Training Data Creation
# ═══════════════════════════════════════════════════════════════════════════

def create_rephraser_dataset(ds):
    """Build (entity_replaced_text → generalised_text) pairs from AI4Privacy.

    The Rephraser never sees real entities — privacy wall is maintained.
    Input:  text after entity replacement (L1 anonymised output)
    Target: text after contextual generalisation (L2 privacy output)
    """
    log.info("Creating rephraser training data …")

    def _make_pair(example):
        return GENERALIZER.create_training_pair_rephraser(example)

    mapped = ds.map(_make_pair, remove_columns=ds.column_names)
    # Filter out pairs where generalisation made no change
    mapped = mapped.filter(lambda ex: ex["input_text"].split(": ", 1)[-1] != ex["target_text"])
    log.info(f"  Rephraser pairs: {len(mapped):,} (after filtering no-change)")
    return mapped


# ═══════════════════════════════════════════════════════════════════════════
# Build & Train Rephraser
# ═══════════════════════════════════════════════════════════════════════════

def build_rephraser(output_dir):
    """Flan-T5-base with QLoRA for the contextual generalisation stage."""
    return build_seq2seq_qlora(CFG.REPHRASER_BASE, output_dir)


def train_rephraser(model, tok, train_ds, val_ds, output_dir):
    """Train the rephraser on (entity_replaced → generalised) pairs."""
    log.info("Training Rephraser (contextual generalisation) …")
    _tok = lambda ex: tokenize_seq2seq(ex, tok)
    t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(_tok, batched=True, remove_columns=val_ds.column_names)
    return train_seq2seq(model, tok, t, v, output_dir, epochs=3)


# ═══════════════════════════════════════════════════════════════════════════
# 3-Stage Inference
# ═══════════════════════════════════════════════════════════════════════════

def anonymize_rephraser(text, censor_model, censor_tok,
                        halluc_model, halluc_tok,
                        rephraser_model, rephraser_tok):
    """Stage 1: NER → Stage 2: Hallucinate → Stage 3: Rephrase."""
    # Stages 1 & 2: Baseline anonymisation (entity replacement)
    l1_anon = anonymize_baseline(text, censor_model, censor_tok,
                                 halluc_model, halluc_tok)

    # Stage 3: Contextual generalisation
    prompt = f"Rewrite to remove contextual re-identification clues while preserving core meaning: {l1_anon}"
    inp = rephraser_tok(prompt, return_tensors="pt", truncation=True,
                        max_length=CFG.S2S_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = rephraser_model.generate(
            **inp, max_new_tokens=CFG.GEN_MAX_TOKENS,
            num_beams=CFG.GEN_NUM_BEAMS,
        )
    l2_anon = rephraser_tok.decode(out[0], skip_special_tokens=True)
    return l2_anon


# ═══════════════════════════════════════════════════════════════════════════
# Main run function
# ═══════════════════════════════════════════════════════════════════════════

def run_rephraser_pipeline(half_a, half_b, test_set, lang_col,
                           baseline_result=None):
    """Train & evaluate Model 3 (A-type: 3-stage with Rephraser).

    If baseline_result is provided, reuses its Censor and Hallucinator
    to avoid re-training.  Otherwise trains them from scratch.
    """
    log.info("=" * 70)
    log.info(" MODEL 3 — A-TYPE: CENSOR ➜ HALLUCINATOR ➜ REPHRASER")
    log.info("=" * 70)

    out = model_dir("model3_rephraser")

    if CFG.QUICK_MODE:
        half_a = quick_subsample(half_a, CFG.QUICK_N)
        half_b = quick_subsample(half_b, CFG.QUICK_N)
        test_set = quick_subsample(test_set, min(CFG.NUM_EVAL, len(test_set)))

    # ── Reuse or train Censor + Hallucinator (same as Model 1) ──
    if baseline_result and "censor" in baseline_result:
        log.info("Reusing Model 1 Censor + Hallucinator …")
        censor_model, censor_tok = baseline_result["censor"]
        halluc_model, halluc_tok = baseline_result["halluc"]
    else:
        log.info("Training Censor + Hallucinator from scratch …")
        censor_model, censor_tok = build_censor_baseline()
        tr, va = make_train_val(half_a)
        train_censor(censor_model, censor_tok, tr, va,
                     os.path.join(out, "censor"))

        halluc_model, halluc_tok = build_hallucinator_baseline(
            os.path.join(out, "hallucinator"))
        tr_b, va_b = make_train_val(half_b)
        train_hallucinator(halluc_model, halluc_tok, tr_b, va_b,
                           os.path.join(out, "hallucinator"))

    # ── Rephraser ──
    reph_dir = os.path.join(out, "rephraser")
    reph_adapter = os.path.join(reph_dir, "adapter_config.json")
    if os.path.exists(reph_adapter):
        log.info(f"Rephraser already trained — loading from {reph_dir}")
        from transformers import AutoTokenizer
        rephraser_tok = AutoTokenizer.from_pretrained(reph_dir)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _BF16_OK else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(
            CFG.REPHRASER_BASE, quantization_config=bnb, device_map="auto")
        rephraser_model = PeftModel.from_pretrained(base, reph_dir)
    else:
        # Create Rephraser training data (only if we need to train)
        from datasets import concatenate_datasets
        rephraser_data = concatenate_datasets([half_a, half_b])
        if CFG.QUICK_MODE:
            rephraser_data = quick_subsample(rephraser_data, CFG.QUICK_N)
        rephraser_pairs = create_rephraser_dataset(rephraser_data)
        tr_r, va_r = make_train_val(rephraser_pairs)
        rephraser_model, rephraser_tok = build_rephraser(reph_dir)
        train_rephraser(rephraser_model, rephraser_tok, tr_r, va_r, reph_dir)

    # ── Evaluate ──
    _anon = lambda t: anonymize_rephraser(
        t, censor_model, censor_tok,
        halluc_model, halluc_tok,
        rephraser_model, rephraser_tok)

    N = min(CFG.NUM_EVAL, len(test_set))
    originals = [get_source_text(test_set[i]) for i in range(N)]
    anonymized = []
    for o in tqdm(originals, desc="Anonymizing (Model 3)"):
        anonymized.append(_anon(o))

    metrics = evaluate_anonymization(originals, anonymized,
                                     "Model 3 (A-type Rephraser)")
    metrics["masking_quality"] = evaluate_masking_quality(
        test_set, censor_model, censor_tok, "Model 3 (Censor)")

    # Extra: Contextual Re-identification Risk (key metric for Model 3)
    crr = compute_contextual_reidentification_risk(originals, anonymized)
    metrics["crr_detailed"] = crr
    log.info(f"  Contextual Re-ID Risk: {crr['crr']:.2f}%")

    curated = run_curated_eval(_anon, "Model 3 (A-type Rephraser)")
    metrics["curated"] = curated

    # ── Evaluation summary plot ──
    plot_evaluation_summary(metrics, out)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "metrics": metrics,
        "censor": (censor_model, censor_tok),
        "halluc": (halluc_model, halluc_tok),
        "rephraser": (rephraser_model, rephraser_tok),
        "anonymize_fn": _anon,
    }

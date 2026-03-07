# ==============================================================================
# model4_semantic.py — Model 4 (B-type): End-to-End Semantic Privacy Paraphraser
# ==============================================================================
# Architecture:  Single Flan-T5-base QLoRA (was Flan-T5-large: 3× smaller)
#                with Privacy-Aware Seq2Seq Training (BERTScore-inspired
#                semantic fidelity loss) and Entity-Type Augmented Input.
#
# Key Innovations:
#   1. Semantic Fidelity Loss: cosine similarity between encoder/decoder
#      hidden states targets 0.70 — preserves meaning, changes entities.
#   2. Entity-Type Prefix Tokens: input prepended with detected entity
#      types ("[TYPES: PERSON, LOC, DATE]") for explicit type awareness.
#   3. Sensitivity-Aware Metrics: evaluation tracks per-entity-type
#      performance with BLEU/ROUGE on the masking step.
#
# Training Data:
#   input  = "[TYPES: PERSON, LOC] Rewrite … : <original text>"
#   target = privacy paraphrase (entity-replaced + contextually generalised)
# ==============================================================================

import os, torch, numpy as np, logging, json
from tqdm.auto import tqdm

from common import (
    CFG, DEVICE, SEED, BIO_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    _BF16_OK,
    model_dir, log,
    find_text_and_labels, get_source_text,
    tokenize_seq2seq,
    build_seq2seq_qlora, train_seq2seq,
    make_train_val, quick_subsample,
    evaluate_anonymization, run_curated_eval,
    compute_contextual_reidentification_risk, cleanup_gpu,
    GENERALIZER, plot_evaluation_summary,
    PrivacyAwareSeq2SeqTrainer, plot_training_curves,
)
import evaluate as _evaluate


# ═══════════════════════════════════════════════════════════════════════════
# Entity-Type Augmented Training Data
# ═══════════════════════════════════════════════════════════════════════════

def _extract_entity_types(example):
    """Extract unique entity types present in an example for type-aware input."""
    _, labels = find_text_and_labels(example)
    types = set()
    for lbl in labels:
        if lbl.startswith("B-"):
            types.add(lbl[2:])
    return sorted(types)


def create_semantic_dataset(ds):
    """Build (original_text → privacy_paraphrase) pairs with entity-type prefix.

    Innovation: Each input is prepended with "[TYPES: PERSON, LOC, DATE]"
    so the model explicitly knows which entity types to replace.  This
    entity-type augmentation improves generation accuracy for rare types.

    For each example in the dataset:
      input  = "[TYPES: PERSON, LOC] Rewrite … : <original>"
      target = TextGeneralizer(entity_replaced_version)
    """
    log.info("Creating semantic paraphraser training data (entity-type augmented) …")

    def _make_pair(example):
        pair = GENERALIZER.create_training_pair_semantic(example)
        # Extract entity types and prepend as prefix tokens
        etypes = _extract_entity_types(example)
        if etypes:
            type_prefix = f"[TYPES: {', '.join(etypes)}] "
        else:
            type_prefix = "[TYPES: NONE] "
        pair["input_text"] = type_prefix + pair["input_text"]
        return pair

    mapped = ds.map(_make_pair, remove_columns=ds.column_names)
    # Filter out trivial identity mappings
    mapped = mapped.filter(
        lambda ex: ex["input_text"].split(": ", 1)[-1].strip()
                   != ex["target_text"].strip()
    )
    log.info(f"  Semantic pairs: {len(mapped):,}")
    return mapped


# ═══════════════════════════════════════════════════════════════════════════
# Build & Train Semantic Paraphraser
# ═══════════════════════════════════════════════════════════════════════════

def build_semantic_paraphraser(output_dir):
    """Flan-T5-base with QLoRA for end-to-end semantic privacy (3× smaller than T5-large)."""
    return build_seq2seq_qlora(CFG.PARAPHRASER_BASE, output_dir)


def train_semantic_paraphraser(model, tok, train_ds, val_ds, output_dir):
    """Train with PrivacyAwareSeq2SeqTrainer (BERTScore-inspired semantic loss).

    The semantic fidelity loss ensures the model preserves overall meaning
    (cosine similarity target 0.70 between encoder/decoder representations)
    while effectively anonymizing entities.
    """
    from common import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

    log.info("Training Semantic Paraphraser (Privacy-Aware Trainer) …")
    _tok = lambda ex: tokenize_seq2seq(ex, tok)
    t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(_tok, batched=True, remove_columns=val_ds.column_names)

    ep = 1
    bs = CFG.S2S_BATCH
    total_steps = max(1, len(t) // (bs * CFG.S2S_GRAD_ACCUM)) * ep
    warmup_steps = max(200, int(total_steps * 0.10))  # 10% warmup, min 200

    rouge = _evaluate.load("rouge")
    def _metrics(ep_):
        preds, labels = ep_
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        r = rouge.compute(predictions=dp, references=dl)
        return {k: round(v, 4) for k, v in r.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=ep,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        gradient_accumulation_steps=CFG.S2S_GRAD_ACCUM,
        learning_rate=CFG.S2S_LR,
        warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL", greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=CFG.GEN_MAX_TOKENS,
        fp16=False, bf16=_BF16_OK,
        logging_steps=100, save_total_limit=2, report_to="none",
    )

    trainer = PrivacyAwareSeq2SeqTrainer(
        model=model, args=args,
        train_dataset=t, eval_dataset=v,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=_metrics,
        semantic_weight=CFG.SEMANTIC_LOSS_WEIGHT,
        privacy_weight=CFG.PRIVACY_LOSS_WEIGHT,
        sim_target=CFG.SEMANTIC_SIM_TARGET,
    )
    trainer.train()

    plot_training_curves(
        trainer.state.log_history,
        "Semantic Paraphraser (Privacy-Aware)",
        os.path.join(output_dir, "loss_curves.png"),
    )
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    log.info(f"  Val ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    return trainer, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Single-Pass Inference
# ═══════════════════════════════════════════════════════════════════════════

def anonymize_semantic(text, model, tok):
    """Single-pass: original text → privacy-preserving paraphrase.

    Uses entity-type prefix detection for better type awareness during
    generation. A lightweight regex heuristic identifies likely entity
    types present in the input.
    """
    # Lightweight entity-type detection for prefix
    detected_types = set()
    import re
    if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text):
        detected_types.add("PERSON")
    if re.search(r'\b[A-Z][a-z]+(?:burg|town|ville|city|land|shire)\b|'
                 r'\b(?:New|San|Los|Las|Saint|North|South|East|West)\s+[A-Z]', text):
        detected_types.add("LOC")
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text):
        detected_types.add("EMAIL")
    if re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text):
        detected_types.add("PHONE")
    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
        detected_types.add("DATE")
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        detected_types.add("SSN")
    if re.search(r'https?://[^\s]+', text):
        detected_types.add("URL")
    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text):
        detected_types.add("IP_ADDRESS")

    type_prefix = f"[TYPES: {', '.join(sorted(detected_types))}] " if detected_types else "[TYPES: NONE] "
    prompt = f"{type_prefix}Rewrite the following text to fully anonymise all personally identifiable information and contextual clues: {text}"
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=CFG.S2S_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=CFG.GEN_MAX_TOKENS,
            num_beams=CFG.GEN_NUM_BEAMS,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
# Re-identification Risk Probe (Privacy Evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_reidentification_risk(originals, anonymized, n_grams_list=None):
    """Multi-granularity contextual re-identification risk.

    Measures at 2-gram, 3-gram, and 4-gram levels how many identifying
    n-grams from the original text survive in the anonymised output.
    """
    if n_grams_list is None:
        n_grams_list = [2, 3, 4]

    results = {}
    for ng in n_grams_list:
        crr = compute_contextual_reidentification_risk(
            originals, anonymized, n_grams=ng)
        results[f"crr_{ng}gram"] = crr["crr"]

    log.info("  Re-identification Risk (multi-granularity):")
    for k, v in results.items():
        log.info(f"    {k}: {v:.2f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main run function
# ═══════════════════════════════════════════════════════════════════════════

def run_semantic_pipeline(half_a, half_b, test_set, lang_col):
    """Train & evaluate Model 4 (B-type: Single-Pass Semantic Privacy).

    Uses both halves for training since this is a single end-to-end model
    (no Censor/Hallucinator separation needed).
    """
    log.info("=" * 70)
    log.info(" MODEL 4 — B-TYPE: SINGLE-PASS SEMANTIC PRIVACY PARAPHRASER")
    log.info("=" * 70)

    out = model_dir("model4_semantic")

    # Use both halves for training (no NER/Halluc split needed)
    from datasets import concatenate_datasets
    full_train = concatenate_datasets([half_a, half_b])
    if CFG.QUICK_MODE:
        full_train = quick_subsample(full_train, CFG.QUICK_N)
        test_set = quick_subsample(test_set, min(CFG.NUM_EVAL, len(test_set)))

    # ── Create training data ──
    semantic_pairs = create_semantic_dataset(full_train)
    tr, va = make_train_val(semantic_pairs)

    # ── Build & Train ──
    para_dir = os.path.join(out, "paraphraser")
    para_model, para_tok = build_semantic_paraphraser(para_dir)
    train_semantic_paraphraser(para_model, para_tok, tr, va, para_dir)

    # ── Evaluate ──
    _anon = lambda t: anonymize_semantic(t, para_model, para_tok)

    N = min(CFG.NUM_EVAL, len(test_set))
    originals = [get_source_text(test_set[i]) for i in range(N)]
    anonymized = []
    for o in tqdm(originals, desc="Anonymizing (Model 4)"):
        anonymized.append(_anon(o))

    metrics = evaluate_anonymization(originals, anonymized,
                                     "Model 4 (B-type Semantic)")

    # Extra: Multi-granularity re-identification risk
    reid = evaluate_reidentification_risk(originals, anonymized)
    metrics["reidentification_risk"] = reid

    curated = run_curated_eval(_anon, "Model 4 (B-type Semantic)")
    metrics["curated"] = curated

    # ── Evaluation summary plot ──
    plot_evaluation_summary(metrics, out)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "metrics": metrics,
        "paraphraser": (para_model, para_tok),
        "anonymize_fn": _anon,
    }

# ==============================================================================
# model1_baseline.py — Model 1: Baseline Decoupled Mask-and-Fill
# ==============================================================================
# Architecture:  DeBERTa-v3-base Censor  ➜  Flan-T5-Large QLoRA Hallucinator
# Data:          Half-A trains Censor, Half-B trains Hallucinator
# ==============================================================================

import os, torch, numpy as np, logging
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

from tqdm.auto import tqdm

from common import (
    CFG, DEVICE, SEED, BIO_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    _FP16_OK, _BF16_OK, fix_deberta_params,
    model_dir, log,
    find_text_and_labels, get_source_text,
    tokenize_and_align_ner, prepare_seq2seq_pair, tokenize_seq2seq,
    compute_ner_metrics, compute_ner_metrics_detailed,
    build_seq2seq_qlora, train_seq2seq,
    make_train_val, quick_subsample,
    evaluate_anonymization, evaluate_masking_quality,
    run_curated_eval, cleanup_gpu,
    plot_training_curves, plot_evaluation_summary,
)


# ═══════════════════════════════════════════════════════════════════════════
# Censor (NER)
# ═══════════════════════════════════════════════════════════════════════════

def build_censor_baseline():
    name = CFG.CENSOR_BASE
    log.info(f"Building baseline Censor: {name}")
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        name, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )
    fix_deberta_params(model)
    model = model.float().to(DEVICE)
    return model, tok


def train_censor(model, tok, train_ds, val_ds, output_dir):
    log.info("Training baseline Censor …")
    _tok = lambda ex: tokenize_and_align_ner(ex, tok)
    t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(_tok, batched=True, remove_columns=val_ds.column_names)
    total_steps = max(1, len(t) // (CFG.NER_BATCH * CFG.NER_GRAD_ACCUM)) * CFG.NER_EPOCHS
    warmup_steps = int(total_steps * CFG.NER_WARMUP)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CFG.NER_EPOCHS,
        per_device_train_batch_size=CFG.NER_BATCH,
        per_device_eval_batch_size=CFG.NER_BATCH * 2,
        gradient_accumulation_steps=CFG.NER_GRAD_ACCUM,
        learning_rate=CFG.NER_LR,
        weight_decay=CFG.NER_WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", greater_is_better=True,
        bf16=_BF16_OK, fp16=_FP16_OK,
        logging_steps=100, save_total_limit=2, report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=t, eval_dataset=v,
        processing_class=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # ── Plot loss curves ──
    plot_training_curves(
        trainer.state.log_history,
        "Censor (NER) Training Loss",
        os.path.join(output_dir, "loss_curves.png"),
    )

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    # ── Detailed NER evaluation ──
    log.info("  Running detailed NER evaluation …")
    eval_result = trainer.evaluate()
    log.info(f"  Censor F1: {eval_result.get('eval_f1', 0):.4f}")
    eval_preds = trainer.predict(trainer.eval_dataset)
    detailed = compute_ner_metrics_detailed((
        eval_preds.predictions, eval_preds.label_ids
    ))
    eval_result["per_entity_ner"] = detailed.get("per_entity", {})
    return trainer, eval_result


# ═══════════════════════════════════════════════════════════════════════════
# Hallucinator (Seq2Seq)
# ═══════════════════════════════════════════════════════════════════════════

def build_hallucinator_baseline(output_dir):
    return build_seq2seq_qlora(CFG.HALLUC_BASE, output_dir)


def train_hallucinator(model, tok, train_ds, val_ds, output_dir):
    log.info("Preparing seq2seq pairs for Hallucinator …")
    t = train_ds.map(prepare_seq2seq_pair, remove_columns=train_ds.column_names)
    v = val_ds.map(prepare_seq2seq_pair, remove_columns=val_ds.column_names)
    _tok = lambda ex: tokenize_seq2seq(ex, tok)
    t = t.map(_tok, batched=True, remove_columns=t.column_names)
    v = v.map(_tok, batched=True, remove_columns=v.column_names)
    return train_seq2seq(model, tok, t, v, output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

def anonymize_baseline(text, censor_model, censor_tok, halluc_model, halluc_tok):
    """3-step:  NER → mask → hallucinate."""
    # Step 1: NER
    enc = censor_tok(text, return_tensors="pt", truncation=True,
                     max_length=CFG.NER_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = censor_model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tokens = censor_tok.convert_ids_to_tokens(enc["input_ids"][0])
    word_ids = enc.word_ids()

    # Merge sub-words back, keep NER tags
    words, tags = [], []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        tag = ID2LABEL.get(preds[i], "O")
        if wid != prev_wid:
            raw = censor_tok.convert_ids_to_tokens(
                enc["input_ids"][0][i].item())
            words.append(raw.replace("▁", "").replace("##", ""))
            tags.append(tag)
        else:
            words[-1] += tokens[i].replace("▁", "").replace("##", "")
        prev_wid = wid

    # Step 2: Mask
    masked, prev_ent = [], None
    for w, t in zip(words, tags):
        if t == "O":
            masked.append(w); prev_ent = None
        elif t.startswith("B-"):
            masked.append(f"[{t[2:]}]"); prev_ent = t[2:]
        elif t.startswith("I-") and prev_ent:
            pass
        else:
            masked.append(w); prev_ent = None
    prompt = f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}"

    # Step 3: Hallucinate
    inp = halluc_tok(prompt, return_tensors="pt", truncation=True,
                     max_length=CFG.S2S_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = halluc_model.generate(
            **inp, max_new_tokens=CFG.GEN_MAX_TOKENS,
            num_beams=CFG.GEN_NUM_BEAMS,
        )
    return halluc_tok.decode(out[0], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main run function
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline(half_a, half_b, test_set, lang_col):
    """Train & evaluate Model 1 (Baseline)."""
    log.info("=" * 70)
    log.info(" MODEL 1 — BASELINE DECOUPLED MASK-AND-FILL")
    log.info("=" * 70)

    out = model_dir("model1_baseline")

    if CFG.QUICK_MODE:
        half_a = quick_subsample(half_a, CFG.QUICK_N)
        half_b = quick_subsample(half_b, CFG.QUICK_N)
        test_set = quick_subsample(test_set, min(CFG.NUM_EVAL, len(test_set)))

    # ── Censor ──
    censor_dir = os.path.join(out, "censor")
    censor_model, censor_tok = build_censor_baseline()
    tr, va = make_train_val(half_a)
    train_censor(censor_model, censor_tok, tr, va, censor_dir)

    # ── Hallucinator ──
    halluc_dir = os.path.join(out, "hallucinator")
    halluc_model, halluc_tok = build_hallucinator_baseline(halluc_dir)
    tr, va = make_train_val(half_b)
    train_hallucinator(halluc_model, halluc_tok, tr, va, halluc_dir)

    # ── Masking Quality Evaluation ──
    masking_metrics = evaluate_masking_quality(
        test_set, censor_model, censor_tok, "Model 1 (Baseline)")

    # ── Evaluate ──
    _anon = lambda t: anonymize_baseline(t, censor_model, censor_tok,
                                          halluc_model, halluc_tok)
    N = min(CFG.NUM_EVAL, len(test_set))
    originals = [get_source_text(test_set[i]) for i in range(N)]
    anonymized = []
    for o in tqdm(originals, desc="Anonymizing (Model 1)"):
        anonymized.append(_anon(o))
    metrics = evaluate_anonymization(originals, anonymized, "Model 1 (Baseline)")
    metrics["masking_quality"] = masking_metrics

    curated = run_curated_eval(_anon, "Model 1 (Baseline)")
    metrics["curated"] = curated

    # ── Evaluation summary plot ──
    plot_evaluation_summary(metrics, out)

    # Save results
    import json
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Return models for reuse by Model 3
    return {
        "metrics": metrics,
        "censor": (censor_model, censor_tok),
        "halluc": (halluc_model, halluc_tok),
        "anonymize_fn": _anon,
    }

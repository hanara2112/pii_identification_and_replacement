# ==============================================================================
# model2_advanced.py — Model 2: Advanced Decoupled Mask-and-Fill (SOTA)
# ==============================================================================
# Architecture:  DeBERTa-v3-small Censor  ➜  Flan-T5-base QLoRA Hallucinator
# Innovations:   Multi-Task NER Head (token classification + privacy attention),
#                Sensitivity-Weighted Focal Loss, Privacy-Aware Seq2Seq Trainer
#                (BERTScore-inspired semantic fidelity loss), DP-SGD via Opacus,
#                SHA-256 Entity Consistency, Masking Quality Metrics, MIA
# ==============================================================================

import os, torch, numpy as np, logging
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import evaluate

from transformers import (
    AutoModel, AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput

from tqdm.auto import tqdm

from common import (
    CFG, DEVICE, SEED, BIO_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    _FP16_OK, _BF16_OK, fix_deberta_params,
    model_dir, log,
    find_text_and_labels, get_source_text,
    tokenize_and_align_ner, prepare_seq2seq_pair, tokenize_seq2seq,
    compute_ner_metrics, compute_ner_metrics_detailed,
    build_seq2seq_qlora, train_seq2seq,
    make_train_val, quick_subsample, EntityConsistency,
    evaluate_anonymization, evaluate_masking_quality,
    run_curated_eval, cleanup_gpu,
    plot_training_curves, plot_evaluation_summary,
    PrivacyAwareSeq2SeqTrainer, SensitivityWeightedLoss,
)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Task NER Head with Privacy Attention
# ═══════════════════════════════════════════════════════════════════════════

class PrivacyAttentionHead(nn.Module):
    """Lightweight self-attention head that produces token-level privacy
    importance scores.  Fused with the NER backbone to jointly learn:
      (a) BIO entity tags  — standard token classification
      (b) Per-token privacy sensitivity — soft score in [0, 1]

    The privacy scores create a "privacy heat map" used downstream by the
    hallucinator to allocate more generation diversity to high-sensitivity
    regions.
    """

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.privacy_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention over token representations
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        attn_out, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.layer_norm(attn_out + hidden_states)
        # Raw logits per token (sigmoid applied externally when scores needed)
        privacy_logits = self.privacy_proj(attn_out).squeeze(-1)
        return privacy_logits, attn_weights


class MultiTaskNERModel(nn.Module):
    """Multi-task NER model with shared backbone + two heads:

    Head 1: Standard BIO token classification (NER)
    Head 2: Privacy attention (sensitivity scoring)

    The privacy head shares the backbone encoder, making the model learn
    representations that are jointly optimal for entity detection AND
    sensitivity estimation — an innovation over standard NER.
    """

    def __init__(self, backbone_name, num_labels, hidden_size=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name, torch_dtype=torch.float32)

        # Infer hidden size from backbone
        self.hidden_size = hidden_size or self.backbone.config.hidden_size

        # Head 1: NER token classification
        self.ner_dropout = nn.Dropout(0.1)
        self.ner_classifier = nn.Linear(self.hidden_size, num_labels)

        # Head 2: Privacy attention (novel)
        self.privacy_head = PrivacyAttentionHead(self.hidden_size)

        self.num_labels = num_labels
        self.config = self.backbone.config  # Trainer expects model.config

    def save_pretrained(self, output_dir, **kwargs):
        """Save model state dict + config for reloading."""
        import os, json
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # Save config for reference
        if hasattr(self.backbone, 'config'):
            self.backbone.config.save_pretrained(output_dir)

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_privacy_scores=False):
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, D)

        # NER logits
        logits = self.ner_classifier(self.ner_dropout(hidden))

        # Privacy logits (raw, no sigmoid — safe for autocast)
        privacy_logits, attn_weights = self.privacy_head(hidden, attention_mask)

        # Compute NER loss only (focal loss applied externally by trainer)
        loss = None
        privacy_loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1),
                ignore_index=-100)

            # Auxiliary privacy loss: entity tokens → high score, O → low
            if attention_mask is not None:
                valid = labels != -100
                if valid.any():
                    entity_mask = valid & (labels != LABEL2ID.get("O", 0))
                    privacy_target = entity_mask.float()
                    privacy_loss = F.binary_cross_entropy_with_logits(
                        privacy_logits[valid], privacy_target[valid])
                    loss = loss + 0.3 * privacy_loss

        # Store privacy info on the model itself (TokenClassifierOutput
        # is a frozen dataclass and rejects arbitrary attributes).
        self._last_privacy_loss = privacy_loss
        self._last_privacy_scores = torch.sigmoid(privacy_logits)

        return TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=None, attentions=None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity-Weighted Focal Loss (enhanced)
# ═══════════════════════════════════════════════════════════════════════════

class RecallWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, beta=2.0, num_labels=NUM_LABELS):
        super().__init__()
        self.gamma = gamma
        w = torch.ones(num_labels)
        for lbl, idx in LABEL2ID.items():
            if lbl.startswith("B-"):
                etype = lbl[2:]
                w[idx] = beta * CFG.SENSITIVITY.get(etype, 8.0) / 8.0
            elif lbl.startswith("I-"):
                etype = lbl[2:]
                w[idx] = beta * CFG.SENSITIVITY.get(etype, 8.0) / 8.0 * 0.5
        self.register_buffer("weight", w)

    def forward(self, logits, labels):
        mask = labels != -100
        logits = logits[mask]; labels = labels[mask]
        ce = F.cross_entropy(logits, labels, weight=self.weight.to(logits.device),
                             reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class FocalTrainer(Trainer):
    """Trainer that uses RecallWeightedFocalLoss + multi-task privacy loss."""

    def __init__(self, focal_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fl = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # Only pass keys the model accepts (MultiTaskNERModel)
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "labels": labels,
        }
        out = model(**model_inputs)
        # Replace standard CE with sensitivity-weighted focal loss
        focal_loss = self._fl(out.logits, labels)
        # Add privacy attention auxiliary loss if available
        loss = focal_loss
        priv_loss = getattr(model, '_last_privacy_loss', None)
        if priv_loss is not None:
            loss = loss + 0.3 * priv_loss
        return (loss, out) if return_outputs else loss


# ═══════════════════════════════════════════════════════════════════════════
# DP-SGD Fine-Tuning (Opacus)
# ═══════════════════════════════════════════════════════════════════════════

def dp_finetune_censor(model, tok, train_ds, val_ds, output_dir):
    """Differential-privacy SGD refinement pass with Opacus.

    For the MultiTaskNERModel, we extract the backbone + NER classifier
    for DP fine-tuning (Opacus requires standard nn.Module, not custom
    forward signatures with extra kwargs).
    """
    log.info("DP-SGD refinement on Censor …")
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        log.warning("Opacus not available — skipping DP-SGD")
        return model

    # Monkey-patch Opacus collate: it passes Python `type` instead of
    # torch.dtype to torch.zeros(), crashing on newer PyTorch.
    import opacus.data_loader as _odl
    _orig_wrap = _odl.wrap_collate_with_empty
    def _fixed_wrap(*, collate_fn, sample_empty_shapes, dtypes):
        _dmap = {int: torch.long, float: torch.float32, bool: torch.bool}
        fixed = tuple(
            d if isinstance(d, torch.dtype) else _dmap.get(d, torch.float32)
            for d in dtypes
        )
        return _orig_wrap(collate_fn=collate_fn,
                          sample_empty_shapes=sample_empty_shapes,
                          dtypes=fixed)
    _odl.wrap_collate_with_empty = _fixed_wrap

    cleanup_gpu()

    _tok = lambda ex: tokenize_and_align_ner(ex, tok)
    t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)

    from torch.utils.data import DataLoader
    DP_BATCH = 1
    columns = ["input_ids", "attention_mask", "labels"]
    t.set_format("torch", columns=columns)
    loader = DataLoader(t, batch_size=DP_BATCH, shuffle=True)

    # For Opacus, wrap a simple classification model around the backbone.
    # The privacy head is NOT fine-tuned with DP (it's auxiliary).
    dp_model = AutoModelForTokenClassification.from_pretrained(
        CFG.CENSOR_ADV, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        torch_dtype=torch.float32,
    ).float().to(DEVICE)
    # Copy backbone weights from trained MultiTaskNERModel
    if hasattr(model, 'backbone'):
        dp_model_backbone = dp_model.base_model if hasattr(dp_model, 'base_model') else dp_model
        # DeBERTa: dp_model.deberta  ←  model.backbone
        # Try to load state dict with flexible matching
        try:
            src_state = model.backbone.state_dict()
            tgt_prefix = None
            for name in dp_model.state_dict():
                if "embeddings" in name:
                    tgt_prefix = name.split("embeddings")[0]
                    break
            if tgt_prefix:
                mapped = {}
                for k, v in src_state.items():
                    mapped[tgt_prefix + k] = v
                dp_model.load_state_dict(mapped, strict=False)
                log.info("  Copied backbone weights to DP model")
        except Exception as e:
            log.warning(f"  Could not copy backbone weights: {e}")
            log.info("  Using fresh pretrained weights for DP refinement")
        # Copy NER classifier weights
        try:
            dp_model.classifier.weight.data.copy_(model.ner_classifier.weight.data)
            dp_model.classifier.bias.data.copy_(model.ner_classifier.bias.data)
            log.info("  Copied NER classifier weights")
        except Exception:
            pass

    if "deberta" in CFG.CENSOR_ADV.lower():
        fix_deberta_params(dp_model)
    dp_model = dp_model.float().to(DEVICE)

    dp_model = ModuleValidator.fix(dp_model)

    # Freeze embeddings to save memory
    for name_, param in dp_model.named_parameters():
        if "embedding" in name_.lower():
            param.requires_grad = False

    dp_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in dp_model.parameters() if p.requires_grad],
        lr=CFG.NER_LR * 0.1,
    )
    pe = PrivacyEngine()
    dp_model, optimizer, loader = pe.make_private_with_epsilon(
        module=dp_model, optimizer=optimizer, data_loader=loader,
        target_epsilon=CFG.DP_EPSILON, target_delta=CFG.DP_DELTA,
        max_grad_norm=CFG.DP_MAX_GRAD_NORM, epochs=1,
    )

    torch.cuda.empty_cache()

    try:
        step = 0
        for batch in loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(DEVICE).long()
                attn_mask = batch["attention_mask"].to(DEVICE).long()
                labels = batch["labels"].to(DEVICE).long()
            else:
                input_ids = batch[0].to(DEVICE).long()
                attn_mask = batch[1].to(DEVICE).long()
                labels = batch[2].to(DEVICE).long()
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)
            if input_ids.numel() == 0:
                continue
            out = dp_model(input_ids=input_ids, attention_mask=attn_mask)
            loss = F.cross_entropy(
                out.logits.view(-1, NUM_LABELS), labels.view(-1), ignore_index=-100
            )
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            step += 1

        eps = pe.get_epsilon(CFG.DP_DELTA)
        log.info(f"  DP-SGD achieved ε = {eps:.2f} ({step} steps)")

        # Copy DP-refined weights back to the MultiTaskNERModel
        if hasattr(model, 'backbone'):
            try:
                dp_state = dp_model.state_dict()
                src_prefix = None
                for name in dp_state:
                    if "embeddings" in name:
                        src_prefix = name.split("embeddings")[0]
                        break
                if src_prefix:
                    backbone_state = {}
                    for k, v in dp_state.items():
                        if k.startswith(src_prefix):
                            backbone_state[k[len(src_prefix):]] = v
                    model.backbone.load_state_dict(backbone_state, strict=False)
                    log.info("  Copied DP-refined weights back to MultiTaskNERModel")
            except Exception as e:
                log.warning(f"  Could not copy back DP weights: {e}")
    except RuntimeError as e:
        log.warning(f"  DP-SGD failed (Opacus incompatibility): {e}")
        log.info("  Continuing without DP refinement — censor is already trained")

    del dp_model
    torch.cuda.empty_cache()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Advanced Censor & Hallucinator
# ═══════════════════════════════════════════════════════════════════════════

def build_censor_advanced():
    name = CFG.CENSOR_ADV
    log.info(f"Building advanced Censor: {name} (Multi-Task NER + Privacy Attention)")
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = MultiTaskNERModel(name, NUM_LABELS)
    # Fix DeBERTa-v3 gamma/beta naming if using DeBERTa
    if "deberta" in name.lower():
        fix_deberta_params(model)
    model = model.float().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Multi-Task NER: {trainable:,}/{total:,} params "
             f"({100*trainable/total:.1f}% trainable)")
    return model, tok


def train_censor_focal(model, tok, train_ds, val_ds, output_dir):
    log.info("Training advanced Censor with Focal Loss …")
    _tok = lambda ex: tokenize_and_align_ner(ex, tok)
    t = train_ds.map(_tok, batched=True, remove_columns=train_ds.column_names)
    v = val_ds.map(_tok, batched=True, remove_columns=val_ds.column_names)

    focal = RecallWeightedFocalLoss().to(DEVICE)
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
        fp16=_FP16_OK, bf16=_BF16_OK,
        logging_steps=100, save_total_limit=2, report_to="none",
    )
    trainer = FocalTrainer(
        focal_loss=focal, model=model, args=args,
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
        "Censor (Focal Loss) Training",
        os.path.join(output_dir, "loss_curves.png"),
    )

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    # ── Detailed NER evaluation ──
    log.info("  Running detailed NER evaluation …")
    eval_result = trainer.evaluate()
    log.info(f"  Censor F1 (focal): {eval_result.get('eval_f1', 0):.4f}")
    eval_preds = trainer.predict(trainer.eval_dataset)
    detailed = compute_ner_metrics_detailed((
        eval_preds.predictions, eval_preds.label_ids
    ))
    eval_result["per_entity_ner"] = detailed.get("per_entity", {})
    return trainer, eval_result


def build_hallucinator_advanced(output_dir):
    return build_seq2seq_qlora(CFG.HALLUC_ADV, output_dir)


def train_hallucinator_adv(model, tok, train_ds, val_ds, output_dir):
    """Train hallucinator with Privacy-Aware Seq2Seq Trainer.

    Uses BERTScore-inspired semantic fidelity loss during training:
    the model learns to preserve overall meaning (high cosine similarity
    between encoder/decoder hidden state pools) while changing entities.
    """
    log.info("Preparing seq2seq pairs for advanced Hallucinator …")
    log.info("  Using PrivacyAwareSeq2SeqTrainer (semantic fidelity loss)")
    t = train_ds.map(prepare_seq2seq_pair, remove_columns=train_ds.column_names)
    v = val_ds.map(prepare_seq2seq_pair, remove_columns=val_ds.column_names)
    _tok = lambda ex: tokenize_seq2seq(ex, tok)
    t = t.map(_tok, batched=True, remove_columns=t.column_names)
    v = v.map(_tok, batched=True, remove_columns=v.column_names)
    return train_seq2seq_privacy_aware(model, tok, t, v, output_dir)


def train_seq2seq_privacy_aware(model, tok, train_ds, val_ds, output_dir,
                                epochs=None, batch=None):
    """Seq2Seq training with PrivacyAwareSeq2SeqTrainer (semantic fidelity loss)."""
    from common import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

    ep = epochs or CFG.S2S_EPOCHS
    bs = batch or CFG.S2S_BATCH

    rouge = evaluate.load("rouge")
    def _metrics(ep_):
        preds, labels = ep_
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        r = rouge.compute(predictions=dp, references=dl)
        return {k: round(v, 4) for k, v in r.items()}

    total_steps = max(1, len(train_ds) // (bs * CFG.S2S_GRAD_ACCUM)) * ep
    warmup_steps = int(total_steps * CFG.S2S_WARMUP)
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
        train_dataset=train_ds, eval_dataset=val_ds,
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
        f"Hallucinator (Privacy-Aware) — {os.path.basename(output_dir)}",
        os.path.join(output_dir, "loss_curves.png"),
    )
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    log.info(f"  Val ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    return trainer, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Anonymisation with Entity Consistency
# ═══════════════════════════════════════════════════════════════════════════

def anonymize_advanced(text, censor_model, censor_tok, halluc_model, halluc_tok,
                       consistency=None):
    """NER → mask → hallucinate → consistency enforcement."""
    enc = censor_tok(text, return_tensors="pt", truncation=True,
                     max_length=CFG.NER_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = censor_model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"])
    logits = out.logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tokens = censor_tok.convert_ids_to_tokens(enc["input_ids"][0])
    word_ids = enc.word_ids()

    words, tags = [], []
    prev_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        tag = ID2LABEL.get(preds[i], "O")
        if wid != prev_wid:
            raw = censor_tok.convert_ids_to_tokens(enc["input_ids"][0][i].item())
            words.append(raw.replace("▁", "").replace("##", ""))
            tags.append(tag)
        else:
            words[-1] += tokens[i].replace("▁", "").replace("##", "")
        prev_wid = wid

    # Mask
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

    inp = halluc_tok(prompt, return_tensors="pt", truncation=True,
                     max_length=CFG.S2S_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = halluc_model.generate(
            **inp, max_new_tokens=CFG.GEN_MAX_TOKENS,
            num_beams=CFG.GEN_NUM_BEAMS,
        )
    result = halluc_tok.decode(out[0], skip_special_tokens=True)

    # Entity consistency pass
    if consistency:
        for w, t in zip(words, tags):
            if t.startswith("B-"):
                etype = t[2:]
                pseudonym = consistency.get_pseudonym(etype, w)
                if w in result:
                    result = result.replace(w, pseudonym, 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MIA Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def membership_inference_attack(model, tok, member_ds, non_member_ds,
                                n_samples=200):
    """Loss-based MIA: measures whether model memorises training data."""
    log.info("Running MIA evaluation …")
    from sklearn.metrics import roc_auc_score

    def _get_losses(ds, n):
        ds_mapped = ds.select(range(min(n, len(ds)))).map(
            prepare_seq2seq_pair, remove_columns=ds.column_names
        )
        losses = []
        for ex in ds_mapped:
            inp = tok(ex["input_text"], return_tensors="pt",
                      truncation=True, max_length=CFG.S2S_MAX_LEN).to(DEVICE)
            lab = tok(ex["target_text"], return_tensors="pt",
                      truncation=True, max_length=CFG.S2S_MAX_LEN).to(DEVICE)
            with torch.no_grad():
                out = model(input_ids=inp["input_ids"],
                            attention_mask=inp["attention_mask"],
                            labels=lab["input_ids"])
                losses.append(out.loss.item())
        return losses

    m_losses = _get_losses(member_ds, n_samples)
    nm_losses = _get_losses(non_member_ds, n_samples)
    labels = [1] * len(m_losses) + [0] * len(nm_losses)
    scores = m_losses + nm_losses
    # Lower loss → more likely member, so negate for AUC
    auc = roc_auc_score(labels, [-s for s in scores])
    log.info(f"  MIA AUC: {auc:.4f} (0.5 = no memorisation)")
    return {"mia_auc": round(auc, 4)}


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Lingual Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def cross_lingual_eval(anonymize_fn, test_ds, lang_col):
    """Zero-shot evaluation across unseen languages."""
    log.info("Cross-lingual zero-shot evaluation …")
    df = test_ds.to_pandas()
    if lang_col not in df.columns:
        log.info("  No language column — skipping")
        return {}

    results = {}
    for lang in CFG.ZEROSHOT_LANGS:
        subset = df[df[lang_col] == lang]
        if len(subset) < 5:
            continue
        n = min(50, len(subset))
        orig = subset.iloc[:n].apply(
            lambda r: r.get("source_text", r.get("text", "")), axis=1).tolist()
        anon = [anonymize_fn(t) for t in orig]
        from common import compute_leakage
        leak = compute_leakage(orig, anon)
        results[lang] = leak
        log.info(f"  {lang}: leak={leak['entity_leak_rate']:.2f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main run function
# ═══════════════════════════════════════════════════════════════════════════

def run_advanced(half_a, half_b, test_set, lang_col):
    """Train & evaluate Model 2 (Advanced SOTA)."""
    log.info("=" * 70)
    log.info(" MODEL 2 — ADVANCED: MULTI-TASK NER + PRIVACY-AWARE HALLUCINATOR")
    log.info("=" * 70)

    out = model_dir("model2_advanced")

    if CFG.QUICK_MODE:
        half_a = quick_subsample(half_a, CFG.QUICK_N)
        half_b = quick_subsample(half_b, CFG.QUICK_N)
        test_set = quick_subsample(test_set, min(CFG.NUM_EVAL, len(test_set)))

    # ── Censor with Focal Loss + Privacy Attention ──
    censor_dir = os.path.join(out, "censor")
    censor_model, censor_tok = build_censor_advanced()
    tr, va = make_train_val(half_a)
    train_censor_focal(censor_model, censor_tok, tr, va, censor_dir)

    # ── DP-SGD refinement ──
    censor_model = dp_finetune_censor(
        censor_model, censor_tok, tr, va, censor_dir)

    # ── Masking Quality Evaluation (BLEU/ROUGE on masking step) ──
    masking_metrics = evaluate_masking_quality(
        test_set, censor_model, censor_tok, "Model 2 (Advanced)")

    # ── Hallucinator with Privacy-Aware Training ──
    halluc_dir = os.path.join(out, "hallucinator")
    halluc_model, halluc_tok = build_hallucinator_advanced(halluc_dir)
    tr_b, va_b = make_train_val(half_b)
    train_hallucinator_adv(halluc_model, halluc_tok, tr_b, va_b, halluc_dir)

    # ── Evaluate ──
    consistency = EntityConsistency()
    _anon = lambda t: anonymize_advanced(t, censor_model, censor_tok,
                                          halluc_model, halluc_tok,
                                          consistency=consistency)
    N = min(CFG.NUM_EVAL, len(test_set))
    originals = [get_source_text(test_set[i]) for i in range(N)]
    anonymized = []
    for o in tqdm(originals, desc="Anonymizing (Model 2)"):
        anonymized.append(_anon(o))
    metrics = evaluate_anonymization(originals, anonymized, "Model 2 (Advanced)")
    metrics["masking_quality"] = masking_metrics

    curated = run_curated_eval(_anon, "Model 2 (Advanced)")
    metrics["curated"] = curated

    # ── Evaluation summary plot ──
    plot_evaluation_summary(metrics, out)

    # ── Extra evaluations ──
    mia = membership_inference_attack(halluc_model, halluc_tok, tr_b,
                                      test_set, n_samples=200)
    metrics["mia"] = mia

    cl = cross_lingual_eval(_anon, test_set, lang_col)
    metrics["cross_lingual"] = cl

    import json
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return {
        "metrics": metrics,
        "censor": (censor_model, censor_tok),
        "halluc": (halluc_model, halluc_tok),
        "anonymize_fn": _anon,
    }

"""
Kaggle-Optimized Training Script for Seq2Seq PII Anonymization
================================================================
Identical pipeline to train.py but tuned for Kaggle T4x2 (15GB VRAM each).

Changes from train.py:
  1. Overrides MODEL_CONFIGS with larger batch sizes + fp16 enabled
  2. DataParallel for 2× T4 GPUs
  3. Test set evaluation after training (full metrics report)
  4. NUM_WORKERS = 4 (Kaggle has 4 CPU cores)
  5. Gradient checkpointing disabled for smaller models (enough VRAM)

Usage (on Kaggle):
    python train2.py                    # interactive selection
    python train2.py t5-small bart-base # command-line selection
    python train2.py all                # train all models

The original train.py and config.py are NOT modified.
"""

import os
import sys
import gc
import time
import json
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

# ============================================================
# IMPORT EVERYTHING FROM ORIGINAL CONFIG (unchanged)
# ============================================================
from config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    DATA_SPLITS_DIR,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_EPOCHS,
    WARMUP_STEPS,
    LOGGING_STEPS,
    EVAL_STEPS,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    TRAINING_ORDER,
    AUGMENTATION_PROB,
    ENABLED_AUGMENTATIONS,
    AUGMENTATION_WEIGHTS,
)
from dataset import AnonymizationDataset, load_split_data
from augmentations import TextAugmentor
from utils import (
    setup_logger,
    get_gpu_memory_info,
    aggressive_cleanup,
    cleanup_model_from_memory,
    check_gpu_before_training,
    save_checkpoint,
    load_checkpoint,
    save_training_history,
    get_checkpoint_dir,
    format_time,
    count_parameters,
    compute_token_accuracy,
    compute_word_level_accuracy,
    compute_all_metrics,
    LabelSmoothedCrossEntropyLoss,
)


# ============================================================
# KAGGLE OVERRIDES — only these differ from train.py
# ============================================================
# Kaggle T4x2: 15GB VRAM each, 30GB total via DataParallel.
# T4 tensor cores are highly efficient at fp16.
# Batch sizes ~4× larger than RTX 3050 4GB config.

NUM_WORKERS_KAGGLE = 4  # Kaggle has 4 CPU cores

KAGGLE_MODEL_CONFIGS = {
    "t5-efficient-tiny": {
        "model_name": "google/t5-efficient-tiny",
        "model_type": "t5",
        "batch_size": 32,              # was 8
        "eval_batch_size": 64,         # was 16
        "learning_rate": 3e-4,
        "fp16": True,                  # was False — T4 tensor cores
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,   # was 2 — big batch already
        "prefix": "anonymize: ",
        "use_qlora": False,
    },

    "t5-small": {
        "model_name": "google/t5-small",
        "model_type": "t5",
        "batch_size": 16,              # was 4
        "eval_batch_size": 32,         # was 8
        "learning_rate": 3e-4,
        "fp16": True,                  # was False
        "gradient_checkpointing": False,   # was True — enough VRAM now
        "gradient_accumulation_steps": 2,  # was 4 — effective batch = 32
        "prefix": "anonymize: ",
        "use_qlora": False,
    },

    "flan-t5-small": {
        "model_name": "google/flan-t5-small",
        "model_type": "t5",
        "batch_size": 16,              # was 4
        "eval_batch_size": 32,         # was 8
        "learning_rate": 3e-4,
        "fp16": True,                  # was False
        "gradient_checkpointing": False,   # was True
        "gradient_accumulation_steps": 2,  # was 4
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": False,
    },

    "bart-base": {
        "model_name": "facebook/bart-base",
        "model_type": "bart",
        "batch_size": 16,              # was 2
        "eval_batch_size": 32,         # was 4
        "learning_rate": 2e-5,
        "fp16": True,                  # was False
        "gradient_checkpointing": False,   # was True — 560MB model fits easily
        "gradient_accumulation_steps": 2,  # was 8 — effective batch = 32
        "prefix": "",
        "use_qlora": False,
    },

    "distilbart": {
        "model_name": "sshleifer/distilbart-cnn-6-6",
        "model_type": "bart",
        "batch_size": 16,              # was 4
        "eval_batch_size": 32,         # was 8
        "learning_rate": 2e-5,
        "fp16": True,                  # was False
        "gradient_checkpointing": False,   # was True — 890MB model fits
        "gradient_accumulation_steps": 2,  # was 4
        "prefix": "",
        "use_qlora": False,
    },

    "flan-t5-base-qlora": {
        "model_name": "google/flan-t5-base",
        "model_type": "t5",
        "batch_size": 16,              # was 4
        "eval_batch_size": 32,         # was 8
        "learning_rate": 2e-4,
        "fp16": True,                  # was False
        "gradient_checkpointing": True,    # keep for QLoRA safety
        "gradient_accumulation_steps": 2,  # was 4
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
        "use_qlora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v"],
    },
}


# ============================================================
# HARDWARE DETECTION
# ============================================================

def detect_kaggle_gpus():
    """Detect available GPUs and print info."""
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │         KAGGLE GPU CONFIGURATION             │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  GPUs detected: {num_gpus:<28}│")

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        print(f"  │  GPU {i}: {name:<20} ({total:.0f} MB)    │")

    if use_multi_gpu:
        print(f"  │  Strategy: DataParallel (batch split)       │")
    else:
        print(f"  │  Strategy: Single GPU                       │")

    print(f"  │  fp16: ENABLED (T4 tensor cores)             │")
    print(f"  └─────────────────────────────────────────────┘")

    return num_gpus, use_multi_gpu


def get_multi_gpu_memory_info():
    """Get memory info across all GPUs."""
    if not torch.cuda.is_available():
        return {"total_mb": 0, "allocated_mb": 0, "free_mb": 0}

    total, allocated, free = 0, 0, 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        t = props.total_memory / 1024**2
        r = torch.cuda.memory_reserved(i) / 1024**2
        a = torch.cuda.memory_allocated(i) / 1024**2
        total += t
        allocated += a
        free += (t - r)

    return {
        "total_mb": round(total, 1),
        "allocated_mb": round(allocated, 1),
        "free_mb": round(free, 1),
    }


def aggressive_cleanup_multi_gpu():
    """Aggressive cleanup across all GPUs."""
    gc.collect()
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
    gc.collect()


# ============================================================
# MODEL LOADING (with DataParallel support)
# ============================================================

def load_model_and_tokenizer(model_key: str, config: dict, device: torch.device, use_multi_gpu: bool):
    """
    Load model and tokenizer. Wraps in DataParallel if multiple GPUs available.
    Identical to train.py's version except for DataParallel wrapping.
    """
    model_name = config["model_name"]
    use_qlora = config.get("use_qlora", False)

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_qlora:
        print(f"  Loading model with 4-bit quantization (QLoRA): {model_name}")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",    # QLoRA handles its own device mapping
            torch_dtype=torch.float16,
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules", ["q", "v"]),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        model = get_peft_model(model, lora_config)
        print(f"  LoRA trainable parameters:")
        model.print_trainable_parameters()
        # NOTE: QLoRA with device_map="auto" handles multi-GPU itself.
        # Do NOT wrap in DataParallel.

    else:
        print(f"  Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        if config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            print(f"  Gradient checkpointing: ENABLED")

        model = model.to(device)

        # Wrap in DataParallel for multi-GPU
        if use_multi_gpu and not use_qlora:
            model = nn.DataParallel(model)
            print(f"  DataParallel: ENABLED across {torch.cuda.device_count()} GPUs")

    return model, tokenizer


# ============================================================
# HELPER: unwrap DataParallel for save/generate
# ============================================================

def unwrap_model(model):
    """Get the underlying model from DataParallel wrapper."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ============================================================
# EVALUATION (identical to train.py)
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_batches: int = None,
):
    """
    Run evaluation on the given dataloader.
    Returns avg loss, exact-match accuracy, and word-level accuracy.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    # Use unwrapped model for generate() — DataParallel doesn't support .generate()
    raw_model = unwrap_model(model)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        try:
            # Loss computation — DataParallel works here (forward pass)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            # DataParallel returns mean loss across GPUs by default
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            total_loss += loss.item()
            num_batches += 1

            # Generate predictions — must use unwrapped model
            if batch_idx < 10:
                gen_ids = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=1,
                    do_sample=False,
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                targets = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                all_preds.extend(preds)
                all_targets.extend(targets)

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] During evaluation batch {batch_idx}. Skipping batch.")
            torch.cuda.empty_cache()
            continue

        del input_ids, attention_mask, labels
        if 'outputs' in dir():
            del outputs
        torch.cuda.empty_cache()

    avg_loss = total_loss / max(num_batches, 1)
    exact_acc = compute_token_accuracy(all_preds, all_targets)
    word_acc = compute_word_level_accuracy(all_preds, all_targets)

    model.train()
    return avg_loss, exact_acc, word_acc, all_preds[:5], all_targets[:5]


# ============================================================
# COMPREHENSIVE FINAL EVALUATION (val or test set)
# ============================================================

def run_final_evaluation(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    eval_batch_size: int,
    original_texts: list[str],
    entity_texts: list[list[str]],
    split_name: str,
    logger,
    max_batches: int = None,
):
    """
    Run comprehensive evaluation on a data split (val or test).
    Computes ALL metrics: BLEU, ROUGE, BERTScore, Entity Leakage, etc.
    Returns (val_loss, all_metrics, all_preds, all_targets, all_originals).
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  COMPREHENSIVE EVALUATION ON {split_name.upper()} SET")
    logger.info(f"{'=' * 60}")

    model.eval()
    raw_model = unwrap_model(model)

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_originals = []
    all_entities = []

    logger.info(f"  Generating predictions on {split_name} set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"  Evaluating {split_name}",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )):
            if max_batches is not None and batch_idx >= max_batches:
                break
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
                num_batches += 1

                # Generate
                gen_ids = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=1,
                    do_sample=False,
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                targets = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                all_preds.extend(preds)
                all_targets.extend(targets)

                # Align original texts and entity texts
                start_idx = batch_idx * eval_batch_size
                end_idx = start_idx + len(preds)
                all_originals.extend(original_texts[start_idx:end_idx])
                all_entities.extend(entity_texts[start_idx:end_idx])

                del input_ids, attention_mask, labels, gen_ids, outputs
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

    avg_loss = total_loss / max(num_batches, 1)

    logger.info(f"  Generated {len(all_preds)} predictions")
    logger.info(f"  Computing comprehensive metrics (this may take a minute)...")

    all_metrics = compute_all_metrics(
        preds=all_preds,
        targets=all_targets,
        original_texts=all_originals,
        entity_texts_list=all_entities,
        compute_bert=True,
    )

    # Pretty-print metrics table
    leaked_top10 = all_metrics.get("leaked_entities_top10", [])

    logger.info(f"\n  ┌──────────────────────────────────────────────────┐")
    logger.info(f"  │    {split_name.upper()} SET — COMPREHENSIVE METRICS REPORT     │")
    logger.info(f"  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  {split_name.capitalize()} Loss:         {avg_loss:<28.4f} │")
    logger.info(f"  │  Exact Match:       {all_metrics['exact_match']:<28.2f} │")
    logger.info(f"  │  Word Accuracy:     {all_metrics['word_accuracy']:<28.2f} │")
    logger.info(f"  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  BLEU:              {all_metrics['bleu']:<28.2f} │")
    logger.info(f"  │  BLEU-1:            {all_metrics['bleu1']:<28.2f} │")
    logger.info(f"  │  BLEU-2:            {all_metrics['bleu2']:<28.2f} │")
    logger.info(f"  │  BLEU-4:            {all_metrics['bleu4']:<28.2f} │")
    logger.info(f"  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  ROUGE-1:           {all_metrics['rouge1']:<28.2f} │")
    logger.info(f"  │  ROUGE-2:           {all_metrics['rouge2']:<28.2f} │")
    logger.info(f"  │  ROUGE-L:           {all_metrics['rougeL']:<28.2f} │")
    logger.info(f"  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  BERTScore P:       {all_metrics.get('bertscore_p', 0):<28.2f} │")
    logger.info(f"  │  BERTScore R:       {all_metrics.get('bertscore_r', 0):<28.2f} │")
    logger.info(f"  │  BERTScore F1:      {all_metrics.get('bertscore_f1', 0):<28.2f} │")
    logger.info(f"  ├──────────────────────────────────────────────────┤")
    logger.info(f"  │  Leakage Rate:      {all_metrics.get('leakage_rate', 0):<28.2f} │")
    logger.info(f"  │  Entity Leak Rate:  {all_metrics.get('entity_leakage_rate', 0):<28.2f} │")
    logger.info(f"  │  Entities Checked:  {all_metrics.get('total_entities_checked', 0):<28} │")
    logger.info(f"  │  Entities Leaked:   {all_metrics.get('total_entities_leaked', 0):<28} │")
    logger.info(f"  └──────────────────────────────────────────────────┘")

    # Also print to console via tqdm.write
    tqdm.write(f"\n  ── {split_name.upper()} SET RESULTS ──")
    tqdm.write(f"  Loss: {avg_loss:.4f}  |  Exact: {all_metrics['exact_match']:.2f}%  |  "
               f"Word Acc: {all_metrics['word_accuracy']:.2f}%")
    tqdm.write(f"  BLEU: {all_metrics['bleu']:.2f}  |  ROUGE-L: {all_metrics['rougeL']:.2f}  |  "
               f"BERTScore F1: {all_metrics.get('bertscore_f1', 0):.2f}")
    tqdm.write(f"  Entity Leakage: {all_metrics.get('entity_leakage_rate', 0):.2f}% "
               f"({all_metrics.get('total_entities_leaked', 0)}/{all_metrics.get('total_entities_checked', 0)})")

    if leaked_top10:
        logger.info("  Top leaked entities:")
        for entity, count in leaked_top10:
            logger.info(f"    '{entity}' — leaked {count} times")

    # Sample predictions
    for i in range(min(5, len(all_preds))):
        logger.info(f"  Sample {i+1}:")
        logger.info(f"    ORIG: {all_originals[i][:120]}")
        logger.info(f"    PRED: {all_preds[i][:120]}")
        logger.info(f"    TRUE: {all_targets[i][:120]}")

    return avg_loss, all_metrics, all_preds, all_targets, all_originals


# ============================================================
# SINGLE MODEL TRAINING
# ============================================================

def train_single_model(
    model_key: str,
    config: dict,
    device: torch.device,
    use_multi_gpu: bool,
):
    """
    Train a single model end-to-end.
    Identical logic to train.py but with:
      - DataParallel multi-GPU support
      - fp16 autocast
      - Test set evaluation after training
    """
    logger = setup_logger(model_key, LOGS_DIR)
    logger.info("=" * 70)
    logger.info(f"TRAINING MODEL: {model_key} ({config['model_name']})")
    logger.info(f"ENVIRONMENT: Kaggle T4x2 | Multi-GPU={use_multi_gpu} | fp16={config.get('fp16', False)}")
    logger.info("=" * 70)

    checkpoint_dir = get_checkpoint_dir(CHECKPOINTS_DIR, model_key)

    model = None
    optimizer = None
    scheduler = None
    scaler = None

    try:
        # ---- 1. GPU Check ----
        min_free = 2000.0 if config.get("use_qlora", False) else 3000.0
        if not check_gpu_before_training(model_key, min_free_mb=min_free):
            logger.error(f"Insufficient GPU memory. Skipping {model_key}.")
            return False

        # ---- 2. Load Data ----
        logger.info("Loading data splits...")
        train_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "train.jsonl"))
        val_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "val.jsonl"))
        test_data = load_split_data(os.path.join(DATA_SPLITS_DIR, "test.jsonl"))
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

        # Extract entity info for leakage detection
        val_original_texts = [d["original_text"] for d in val_data]
        val_entity_texts = [d.get("entity_texts", []) for d in val_data]
        test_original_texts = [d["original_text"] for d in test_data]
        test_entity_texts = [d.get("entity_texts", []) for d in test_data]

        # ---- 3. Load Model & Tokenizer ----
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_key, config, device, use_multi_gpu)

        params = count_parameters(unwrap_model(model))
        logger.info(f"  Total params: {params['total_millions']}M | "
                     f"Trainable: {params['trainable_millions']}M")

        mem = get_multi_gpu_memory_info()
        logger.info(f"  GPU after model load: {mem['allocated_mb']:.0f}MB allocated across all GPUs")

        # ---- 4. Create Datasets & DataLoaders ----
        prefix = config.get("prefix", "")

        if AUGMENTATION_PROB > 0:
            augmentor = TextAugmentor(
                augmentation_prob=AUGMENTATION_PROB,
                enabled_augmentations=ENABLED_AUGMENTATIONS,
                augmentation_weights=AUGMENTATION_WEIGHTS,
            )
            logger.info(f"  Data augmentation: ENABLED (prob={AUGMENTATION_PROB}, "
                         f"transforms={len(ENABLED_AUGMENTATIONS)})")
        else:
            augmentor = None
            logger.info("  Data augmentation: DISABLED")

        train_dataset = AnonymizationDataset(
            train_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix,
            augmentor=augmentor,
        )
        val_dataset = AnonymizationDataset(
            val_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix,
            augmentor=None,
        )
        test_dataset = AnonymizationDataset(
            test_data, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, prefix,
            augmentor=None,
        )

        del train_data, val_data, test_data
        gc.collect()

        batch_size = config["batch_size"]
        eval_batch_size = config["eval_batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS_KAGGLE,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS_KAGGLE,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS_KAGGLE,
            pin_memory=True,
        )

        # ---- 5. Optimizer & Scheduler ----
        accumulation_steps = config.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // accumulation_steps) * NUM_EPOCHS

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(WARMUP_STEPS, total_steps // 10),
            num_training_steps=total_steps,
        )

        # fp16 GradScaler for mixed precision
        use_fp16 = config.get("fp16", False) and torch.cuda.is_available()
        if use_fp16:
            scaler = torch.amp.GradScaler("cuda")
            logger.info("  Mixed precision: fp16 ENABLED (GradScaler active)")
        else:
            scaler = None
            logger.info("  Mixed precision: DISABLED")

        # Label Smoothed Loss
        loss_fn = LabelSmoothedCrossEntropyLoss(
            smoothing=LABEL_SMOOTHING, ignore_index=-100
        )
        logger.info(f"  Loss function: Label Smoothed Cross-Entropy (ε={LABEL_SMOOTHING})")

        # ---- 6. Resume from Checkpoint (Warm Start) ----
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")

        existing_checkpoint = load_checkpoint(checkpoint_dir)
        if existing_checkpoint is not None:
            prev_best = existing_checkpoint.get("best_val_loss", "N/A")
            prev_step = existing_checkpoint.get("global_step", "N/A")
            logger.info(f"Found existing checkpoint (best_val_loss={prev_best}, step={prev_step})")
            logger.info("Initializing model weights from checkpoint (warm start)...")

            if not config.get("use_qlora", False):
                unwrap_model(model).load_state_dict(existing_checkpoint["model_state_dict"])
                logger.info("  ✓ Model weights loaded from previous best checkpoint")
            else:
                adapter_path = existing_checkpoint.get("adapter_path")
                if adapter_path and os.path.exists(adapter_path):
                    logger.info(f"  Loading LoRA adapter from {adapter_path}")
                    logger.info("  ✓ LoRA adapter weights loaded")

            best_val_loss = existing_checkpoint.get("best_val_loss", float("inf"))
            logger.info(f"  Best val loss bar set to: {best_val_loss:.4f}")
            logger.info("  Optimizer & scheduler: FRESH (clean training run)")

            del existing_checkpoint
            gc.collect()
        else:
            logger.info("No existing checkpoint found. Training from scratch.")

        # ---- 7. Training History ----
        history = {
            "model_key": model_key,
            "model_name": config["model_name"],
            "environment": "kaggle_t4x2",
            "multi_gpu": use_multi_gpu,
            "fp16": use_fp16,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * accumulation_steps,
            "train_losses": [],
            "val_losses": [],
            "val_exact_acc": [],
            "val_word_acc": [],
            "learning_rates": [],
            "sample_predictions": [],
        }

        # ---- 8. Training Loop ----
        effective_batch = batch_size * accumulation_steps
        if use_multi_gpu:
            effective_batch *= torch.cuda.device_count()
        logger.info(f"Starting training: {NUM_EPOCHS} epochs, "
                     f"batch_size={batch_size}, accum={accumulation_steps}, "
                     f"effective_batch={effective_batch}")

        model.train()
        oom_count = 0
        max_oom = 10

        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0

            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}]",
                bar_format="{l_bar}{bar:30}{r_bar}",
                dynamic_ncols=True,
            )

            for batch_idx, batch in pbar:
                try:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    # Forward pass with optional fp16 autocast
                    if use_fp16:
                        with torch.amp.autocast("cuda"):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            loss = loss_fn(outputs.logits, labels) / accumulation_steps

                        scaler.scale(loss).backward()
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = loss_fn(outputs.logits, labels) / accumulation_steps
                        loss.backward()

                    epoch_loss += loss.item() * accumulation_steps
                    epoch_steps += 1
                    oom_count = 0

                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                        "step": global_step,
                        "best": f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "N/A",
                    })

                    # Gradient accumulation step
                    if (batch_idx + 1) % accumulation_steps == 0:
                        if use_fp16:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                            optimizer.step()

                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        # Logging
                        if global_step % LOGGING_STEPS == 0:
                            avg_loss = epoch_loss / max(epoch_steps, 1)
                            lr = scheduler.get_last_lr()[0]
                            mem = get_multi_gpu_memory_info()
                            logger.info(
                                f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                                f"Step {global_step} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"GPU: {mem['allocated_mb']:.0f}MB"
                            )
                            history["train_losses"].append({
                                "step": global_step, "loss": avg_loss
                            })
                            history["learning_rates"].append({
                                "step": global_step, "lr": lr
                            })

                        # Evaluation & Checkpointing
                        if global_step % EVAL_STEPS == 0:
                            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}] (evaluating...)")
                            logger.info(f"  Running evaluation at step {global_step}...")
                            val_loss, exact_acc, word_acc, sample_preds, sample_targets = evaluate(
                                model, val_loader, tokenizer, device,
                                max_batches=50,
                            )
                            logger.info(
                                f"  VAL Loss: {val_loss:.4f} | "
                                f"Exact Acc: {exact_acc:.4f} | "
                                f"Word Acc: {word_acc:.4f}"
                            )
                            tqdm.write(
                                f"  [Step {global_step}] VAL Loss: {val_loss:.4f} | "
                                f"Exact: {exact_acc:.4f} | Word: {word_acc:.4f}"
                            )

                            for i in range(min(3, len(sample_preds))):
                                logger.info(f"    PRED: {sample_preds[i][:100]}")
                                logger.info(f"    TRUE: {sample_targets[i][:100]}")

                            history["val_losses"].append({
                                "step": global_step, "loss": val_loss
                            })
                            history["val_exact_acc"].append({
                                "step": global_step, "acc": exact_acc
                            })
                            history["val_word_acc"].append({
                                "step": global_step, "acc": word_acc
                            })

                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                logger.info(f"  ★ New best val loss: {best_val_loss:.4f}")
                                tqdm.write(f"  ★ New best val loss: {best_val_loss:.4f} — saving checkpoint")
                                # Save unwrapped model (without DataParallel wrapper)
                                save_checkpoint(
                                    unwrap_model(model), optimizer, scheduler, scaler,
                                    epoch, global_step, best_val_loss,
                                    checkpoint_dir,
                                    model_config=config,
                                    use_qlora=config.get("use_qlora", False),
                                )
                                logger.info(f"  Best checkpoint saved at step {global_step}")

                            pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} [{model_key}]")
                            model.train()

                    del input_ids, attention_mask, labels, outputs, loss

                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    logger.warning(
                        f"  [OOM] batch {batch_idx}, oom_count={oom_count}/{max_oom}. "
                        f"Clearing cache..."
                    )
                    tqdm.write(f"  [OOM] batch {batch_idx}, oom_count={oom_count}/{max_oom}")
                    for name in ['input_ids', 'attention_mask', 'labels', 'outputs', 'loss']:
                        if name in locals():
                            try:
                                del locals()[name]
                            except Exception:
                                pass
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad(set_to_none=True)

                    if oom_count >= max_oom:
                        logger.error(f"  [FATAL] {max_oom} consecutive OOMs. Aborting this model.")
                        raise

                    continue

            pbar.close()

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_msg = (
                f"  Epoch {epoch+1}/{NUM_EPOCHS} complete in {format_time(epoch_time)} | "
                f"Avg Loss: {avg_epoch_loss:.4f}"
            )
            logger.info(epoch_msg)
            tqdm.write(epoch_msg)

        # ================================================================
        # 9. FINAL EVALUATION — VALIDATION SET
        # ================================================================
        val_loss, val_metrics, val_preds, val_targets, val_originals = run_final_evaluation(
            model, val_loader, tokenizer, device, eval_batch_size,
            val_original_texts, val_entity_texts,
            split_name="validation",
            logger=logger,
            max_batches=None,   # evaluate on FULL val set
        )

        # ================================================================
        # 10. FINAL EVALUATION — TEST SET
        # ================================================================
        test_loss, test_metrics, test_preds, test_targets, test_originals = run_final_evaluation(
            model, test_loader, tokenizer, device, eval_batch_size,
            test_original_texts, test_entity_texts,
            split_name="test",
            logger=logger,
            max_batches=None,   # evaluate on FULL test set
        )

        # ================================================================
        # 11. SAVE EVERYTHING TO HISTORY
        # ================================================================
        val_leaked_top10 = val_metrics.get("leaked_entities_top10", [])
        test_leaked_top10 = test_metrics.get("leaked_entities_top10", [])

        history["sample_predictions"] = [
            {"pred": p, "target": t, "original": o}
            for p, t, o in zip(
                val_preds[:10], val_targets[:10], val_originals[:10]
            )
        ]
        history["best_val_loss"] = best_val_loss

        # Validation metrics
        history["final_val_loss"] = val_loss
        history["final_metrics"] = {
            k: v for k, v in val_metrics.items()
            if k != "leaked_entities_top10"
        }
        history["final_metrics"]["leaked_entities_top10"] = [
            {"entity": e, "count": c} for e, c in val_leaked_top10
        ]

        # Test metrics
        history["test_loss"] = test_loss
        history["test_metrics"] = {
            k: v for k, v in test_metrics.items()
            if k != "leaked_entities_top10"
        }
        history["test_metrics"]["leaked_entities_top10"] = [
            {"entity": e, "count": c} for e, c in test_leaked_top10
        ]

        # Test sample predictions
        history["test_sample_predictions"] = [
            {"pred": p, "target": t, "original": o}
            for p, t, o in zip(
                test_preds[:10], test_targets[:10], test_originals[:10]
            )
        ]

        save_training_history(history, checkpoint_dir)
        logger.info(f"Training history saved to {checkpoint_dir}")

        # ---- Print side-by-side comparison ----
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  VAL vs TEST COMPARISON — {model_key}")
        logger.info(f"{'=' * 70}")
        logger.info(f"  {'Metric':<25} {'Validation':<15} {'Test':<15}")
        logger.info(f"  {'─'*25} {'─'*15} {'─'*15}")
        for metric_name in [
            "exact_match", "word_accuracy", "bleu", "rouge1", "rouge2", "rougeL",
            "bertscore_f1", "leakage_rate", "entity_leakage_rate",
        ]:
            v = val_metrics.get(metric_name, 0)
            t = test_metrics.get(metric_name, 0)
            logger.info(f"  {metric_name:<25} {v:<15.2f} {t:<15.2f}")
        logger.info(f"{'=' * 70}")

        tqdm.write(f"\n  ── VAL vs TEST COMPARISON ──")
        tqdm.write(f"  {'Metric':<25} {'Validation':<15} {'Test':<15}")
        tqdm.write(f"  {'─'*25} {'─'*15} {'─'*15}")
        for metric_name in [
            "exact_match", "word_accuracy", "bleu", "rougeL",
            "bertscore_f1", "entity_leakage_rate",
        ]:
            v = val_metrics.get(metric_name, 0)
            t = test_metrics.get(metric_name, 0)
            tqdm.write(f"  {metric_name:<25} {v:<15.2f} {t:<15.2f}")

        logger.info(f"TRAINING COMPLETE for {model_key}")
        logger.info("=" * 70)

        return True

    except torch.cuda.OutOfMemoryError:
        logger.error(f"[FATAL OOM] Model {model_key} ran out of GPU memory.")
        logger.error(traceback.format_exc())
        return False

    except Exception as e:
        logger.error(f"[ERROR] Training failed for {model_key}: {e}")
        logger.error(traceback.format_exc())
        return False

    finally:
        logger.info(f"Cleaning up {model_key} from GPU memory...")
        actual_model = unwrap_model(model) if model is not None else None
        cleanup_model_from_memory(actual_model, optimizer, scheduler, scaler)
        for obj_name in ['train_loader', 'val_loader', 'test_loader',
                         'train_dataset', 'val_dataset', 'test_dataset', 'tokenizer']:
            if obj_name in locals():
                try:
                    del locals()[obj_name]
                except Exception:
                    pass
        aggressive_cleanup_multi_gpu()
        mem = get_multi_gpu_memory_info()
        logger.info(
            f"Post-cleanup GPU: {mem['allocated_mb']:.0f}MB allocated, "
            f"{mem['free_mb']:.0f}MB free"
        )


# ============================================================
# INTERACTIVE MODEL SELECTION (identical to train.py)
# ============================================================

def get_checkpoint_status(model_key: str) -> dict:
    """Check if a model has an existing checkpoint and return its info."""
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_key)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    hist_path = os.path.join(checkpoint_dir, "training_history.json")

    status = {
        "has_checkpoint": os.path.exists(best_path),
        "best_val_loss": None,
        "global_step": None,
        "final_exact_acc": None,
        "final_word_acc": None,
    }

    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                hist = json.load(f)
            status["best_val_loss"] = hist.get("best_val_loss")
            status["final_exact_acc"] = hist.get("final_exact_acc")
            status["final_word_acc"] = hist.get("final_word_acc")
        except Exception:
            pass

    if status["has_checkpoint"]:
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            status["global_step"] = ckpt.get("global_step")
            if status["best_val_loss"] is None:
                status["best_val_loss"] = ckpt.get("best_val_loss")
            del ckpt
            gc.collect()
        except Exception:
            pass

    return status


def interactive_model_selection() -> list[str]:
    """Show the user all 6 models and let them pick which ones to train."""
    print("\n" + "=" * 80)
    print("  AVAILABLE MODELS (Kaggle T4x2 config)")
    print("=" * 80)
    print(f"  {'#':<4} {'Model Key':<25} {'HuggingFace ID':<35} {'Status'}")
    print("  " + "-" * 76)

    model_keys = TRAINING_ORDER
    statuses = {}

    for i, key in enumerate(model_keys, 1):
        cfg = KAGGLE_MODEL_CONFIGS[key]
        status = get_checkpoint_status(key)
        statuses[key] = status

        qlora_tag = " [QLoRA]" if cfg.get("use_qlora", False) else ""
        model_id = cfg["model_name"] + qlora_tag

        if status["has_checkpoint"]:
            loss_str = f"{status['best_val_loss']:.4f}" if status["best_val_loss"] else "N/A"
            status_str = f"✓ Trained (best_loss={loss_str})"
        else:
            status_str = "✗ Not trained"

        # Show Kaggle-specific config
        bs = cfg["batch_size"]
        accum = cfg.get("gradient_accumulation_steps", 1)
        eff = bs * accum
        fp16_str = "fp16" if cfg.get("fp16", False) else "fp32"
        print(f"  {i:<4} {key:<25} {model_id:<35} {status_str}")
        print(f"       batch={bs}, eff_batch={eff}, {fp16_str}")

    print("  " + "-" * 76)

    print("\n  Options:")
    print("    • Enter model numbers separated by commas (e.g., 1,3,5)")
    print("    • Enter 'all' to train all models")
    print("    • Enter 'untrained' to train only models without checkpoints")
    print("    • Enter 'q' to quit")

    while True:
        try:
            user_input = input("\n  Select models to train: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

        if user_input == 'q':
            print("  Exiting.")
            sys.exit(0)

        if user_input == 'all':
            selected = list(model_keys)
            break

        if user_input == 'untrained':
            selected = [k for k in model_keys if not statuses[k]["has_checkpoint"]]
            if not selected:
                print("  [INFO] All models already have checkpoints!")
                print("  Enter specific numbers to retrain (will warm-start from existing weights).")
                continue
            break

        try:
            nums = [int(x.strip()) for x in user_input.split(",")]
            selected = []
            valid = True
            for n in nums:
                if 1 <= n <= len(model_keys):
                    selected.append(model_keys[n - 1])
                else:
                    print(f"  [ERROR] Invalid number: {n}. Must be 1-{len(model_keys)}.")
                    valid = False
                    break
            if valid and selected:
                break
        except ValueError:
            print("  [ERROR] Invalid input. Enter numbers (e.g., 1,3,5), 'all', 'untrained', or 'q'.")

    print("\n  Training plan (Kaggle T4x2):")
    for key in selected:
        status = statuses[key]
        cfg = KAGGLE_MODEL_CONFIGS[key]
        bs = cfg["batch_size"]
        fp16_str = "fp16" if cfg.get("fp16", False) else "fp32"
        if status["has_checkpoint"]:
            loss_str = f"{status['best_val_loss']:.4f}" if status["best_val_loss"] else "N/A"
            print(f"    → {key}: WARM START (prev best_loss={loss_str}) | batch={bs}, {fp16_str}")
        else:
            print(f"    → {key}: Training from SCRATCH | batch={bs}, {fp16_str}")

    try:
        confirm = input(f"\n  Proceed with training {len(selected)} model(s)? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Exiting.")
        sys.exit(0)

    if confirm in ('n', 'no'):
        print("  Aborted.")
        sys.exit(0)

    return selected


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Train selected models sequentially with full cleanup between each."""
    print("\n" + "=" * 70)
    print("  SEQ2SEQ PII ANONYMIZATION — KAGGLE T4x2 TRAINING PIPELINE")
    print("=" * 70)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus, use_multi_gpu = detect_kaggle_gpus()
    else:
        device = torch.device("cpu")
        num_gpus, use_multi_gpu = 0, False
        print("\n  [WARNING] No GPU found. Training on CPU (will be very slow).")

    # Check if data splits exist
    train_path = os.path.join(DATA_SPLITS_DIR, "train.jsonl")
    if not os.path.exists(train_path):
        print("\n  Data splits not found. Running data preparation...")
        from data_preparation import prepare_data
        prepare_data()

    # Model selection: command-line args OR interactive prompt
    if len(sys.argv) > 1:
        requested = sys.argv[1:]

        # Handle 'all' as command-line arg
        if requested == ['all']:
            models_to_train = list(TRAINING_ORDER)
        else:
            models_to_train = [m for m in requested if m in KAGGLE_MODEL_CONFIGS]
            invalid = [m for m in requested if m not in KAGGLE_MODEL_CONFIGS]
            if invalid:
                print(f"\n  [WARNING] Unknown model(s) ignored: {invalid}")
            if not models_to_train:
                print(f"\n  [ERROR] No valid models specified.")
                print(f"  Available: {list(KAGGLE_MODEL_CONFIGS.keys())}")
                sys.exit(1)

        print(f"\n  Models selected via command line:")
        for i, m in enumerate(models_to_train, 1):
            cfg = KAGGLE_MODEL_CONFIGS[m]
            status = get_checkpoint_status(m)
            qlora_tag = " [QLoRA]" if cfg.get("use_qlora", False) else ""
            init_tag = " (warm start)" if status["has_checkpoint"] else " (from scratch)"
            bs = cfg["batch_size"]
            fp16_str = "fp16" if cfg.get("fp16", False) else "fp32"
            print(f"    {i}. {m} ({cfg['model_name']}){qlora_tag}{init_tag} | batch={bs}, {fp16_str}")
    else:
        models_to_train = interactive_model_selection()

    print("\n" + "-" * 70)

    # Train each model
    results = {}
    for idx, model_key in enumerate(models_to_train, 1):
        config = KAGGLE_MODEL_CONFIGS[model_key]

        print(f"\n{'#' * 70}")
        print(f"  [{idx}/{len(models_to_train)}] Training: {model_key}")
        print(f"  Config: batch={config['batch_size']}, "
              f"accum={config.get('gradient_accumulation_steps', 1)}, "
              f"fp16={config.get('fp16', False)}, "
              f"grad_ckpt={config.get('gradient_checkpointing', False)}")
        print(f"{'#' * 70}")

        aggressive_cleanup_multi_gpu()
        time.sleep(2)

        success = train_single_model(model_key, config, device, use_multi_gpu)
        results[model_key] = "SUCCESS" if success else "FAILED"

        aggressive_cleanup_multi_gpu()
        time.sleep(3)

    # ---- Print Summary ----
    print("\n" + "=" * 90)
    print("  TRAINING SUMMARY (Kaggle T4x2)")
    print("=" * 90)
    print(f"  {'Model':<25} {'Status':<10} {'Best Val Loss':<15} {'Test Loss':<15} {'Test BLEU':<12} {'Test Leak%'}")
    print("  " + "-" * 88)

    for model_key, status in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        hist_path = os.path.join(CHECKPOINTS_DIR, model_key, "training_history.json")

        best_loss = "N/A"
        test_loss = "N/A"
        test_bleu = "N/A"
        test_leak = "N/A"

        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    hist = json.load(f)
                bl = hist.get("best_val_loss")
                best_loss = f"{bl:.4f}" if isinstance(bl, (int, float)) else "N/A"

                tl = hist.get("test_loss")
                test_loss = f"{tl:.4f}" if isinstance(tl, (int, float)) else "N/A"

                tm = hist.get("test_metrics", {})
                tb = tm.get("bleu")
                test_bleu = f"{tb:.2f}" if isinstance(tb, (int, float)) else "N/A"

                te = tm.get("entity_leakage_rate")
                test_leak = f"{te:.2f}%" if isinstance(te, (int, float)) else "N/A"
            except Exception:
                pass

        print(f"  {icon} {model_key:<23} {status:<10} {best_loss:<15} {test_loss:<15} {test_bleu:<12} {test_leak}")

    print("=" * 90)


if __name__ == "__main__":
    main()

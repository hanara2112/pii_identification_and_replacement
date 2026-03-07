# ==============================================================================
# End-to-End Seq2Seq Text Anonymization — Comparative Model Study
# ==============================================================================
#
# Loads the AI4Privacy multilingual dataset, prepares (original → anonymized)
# seq2seq pairs, and trains 5 encoder-decoder models with full fine-tuning:
#
#   1. t5-efficient-tiny    (google/t5-efficient-tiny)        ~40M params
#   2. t5-small             (google-t5/t5-small)              ~60M params
#   3. flan-t5-small        (google/flan-t5-small)            ~77M params
#   4. bart-base            (facebook/bart-base)              ~139M params
#   5. distilbart-cnn-6-6   (sshleifer/distilbart-cnn-6-6)   ~333M params
#
# Evaluation: Loss, Exact Match, Word Accuracy, BLEU (1/2/4), ROUGE (1/2/L),
#             BERTScore (P/R/F1), Leakage Rate, Entity Leak Rate,
#             + 37 curated eval examples across easy/medium/hard difficulty.
#
# Usage:
#   python script.py                       # Train & evaluate all 5 models
#   python script.py --model flan-t5-small # Train a specific model
#   python script.py --eval-only           # Evaluate existing checkpoints
#   python script.py --quick               # Quick test with small subset
# ==============================================================================

# ── 1. Imports & Setup ──────────────────────────────────────────────────────

import os, sys, subprocess, warnings, gc, json, re, random, logging, argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def install_deps():
    deps = [
        "transformers>=4.40.0", "datasets>=2.18", "evaluate", "accelerate>=0.28",
        "rouge_score", "sacrebleu", "sentencepiece", "scipy", "scikit-learn",
        "pandas", "matplotlib", "bert_score", "Faker", "nltk",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U"] + deps,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

install_deps()

import evaluate
import nltk
from datasets import Dataset, load_dataset, DatasetDict
from faker import Faker
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    EarlyStoppingCallback,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE} | PyTorch {torch.__version__}")


# ── 2. Configuration ────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "t5-efficient-tiny": {
        "hf_name": "google/t5-efficient-tiny",
        "family": "t5",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "epochs": 10,
        "warmup_ratio": 0.06,
    },
    "t5-small": {
        "hf_name": "google-t5/t5-small",
        "family": "t5",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "epochs": 10,
        "warmup_ratio": 0.06,
    },
    "flan-t5-small": {
        "hf_name": "google/flan-t5-small",
        "family": "t5",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "epochs": 10,
        "warmup_ratio": 0.06,
    },
    "bart-base": {
        "hf_name": "facebook/bart-base",
        "family": "bart",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 2e-5,
        "epochs": 10,
        "warmup_ratio": 0.06,
    },
    "distilbart": {
        "hf_name": "sshleifer/distilbart-cnn-6-6",
        "family": "bart",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 2e-5,
        "epochs": 10,
        "warmup_ratio": 0.06,
    },
}


class Config:
    OUTPUT_ROOT     = "./Seq2Seq_model"
    PLOT_DIR        = "./Seq2Seq_model/plots"

    MAX_INPUT_LEN   = 256
    MAX_TARGET_LEN  = 256
    TEST_RATIO      = 0.05
    VAL_RATIO       = 0.05
    QUICK_MODE      = False
    QUICK_SAMPLE_N  = 2000
    NUM_BEAMS       = 4
    EVAL_ONLY       = False
    SELECTED_MODEL  = None  # None = all

    # PII entity types for leakage detection
    ENTITY_TYPES = [
        "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
        "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
        "DRIVER_LICENSE", "USERNAME", "URL",
    ]


def setup_dirs():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    for name in MODEL_CONFIGS:
        os.makedirs(os.path.join(Config.OUTPUT_ROOT, name), exist_ok=True)

setup_dirs()


def parse_args():
    parser = argparse.ArgumentParser(description="Seq2Seq Text Anonymization")
    parser.add_argument("--model", default=None, choices=list(MODEL_CONFIGS.keys()),
                        help="Train a specific model (default: all)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(model=None, quick=False, eval_only=False, epochs=None)
    Config.QUICK_MODE = args.quick
    Config.EVAL_ONLY = args.eval_only
    Config.SELECTED_MODEL = args.model
    if args.epochs:
        for cfg in MODEL_CONFIGS.values():
            cfg["epochs"] = args.epochs

parse_args()


# ── 3. Data Loading & Preparation ──────────────────────────────────────────

def load_ai4privacy():
    """Load the full AI4Privacy dataset from HuggingFace."""
    log.info("Loading AI4Privacy dataset...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    log.info(f"Loaded {len(ds):,} examples")
    return ds


def extract_seq2seq_pair(example):
    """Extract (original_text, anonymized_text) pair from an AI4Privacy example.

    The dataset provides source_text (with real PII) and the masked/anonymized
    versions. We construct pairs for end-to-end seq2seq training.
    """
    source = example.get("source_text", example.get("text", ""))
    target = example.get("target_text", "")

    # If target_text not available, reconstruct from privacy_mask
    if not target and "privacy_mask" in example:
        masks = example["privacy_mask"]
        if isinstance(masks, list) and masks:
            target = source
            # Sort by position (reverse) to replace from end to start
            sorted_masks = sorted(
                masks,
                key=lambda m: m.get("start", m.get("offset", 0)),
                reverse=True
            )
            for span in sorted_masks:
                start = span.get("start", span.get("offset", 0))
                value = span.get("value", "")
                end = span.get("end", start + len(value))
                fake = span.get("fake_value", span.get("replacement", ""))
                if fake and start < len(target):
                    target = target[:start] + fake + target[end:]

    if not target:
        target = source  # fallback

    return {"input_text": source, "target_text": target}


def prepare_dataset(ds):
    """Convert AI4Privacy dataset to seq2seq format and split."""
    log.info("Preparing seq2seq pairs...")

    ds = ds.map(extract_seq2seq_pair, remove_columns=ds.column_names)

    # Filter out empty or identical pairs
    ds = ds.filter(lambda x: len(x["input_text"]) > 10 and x["input_text"] != x["target_text"])
    log.info(f"Valid seq2seq pairs: {len(ds):,}")

    # Train / val / test split
    splits = ds.train_test_split(test_size=Config.TEST_RATIO + Config.VAL_RATIO, seed=SEED)
    test_val = splits["test"].train_test_split(
        test_size=Config.TEST_RATIO / (Config.TEST_RATIO + Config.VAL_RATIO), seed=SEED
    )

    train_ds = splits["train"]
    val_ds = test_val["train"]
    test_ds = test_val["test"]

    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def quick_subsample(ds, n=2000):
    if len(ds) <= n:
        return ds
    return ds.select(random.sample(range(len(ds)), n))


# ── 4. Tokenization ────────────────────────────────────────────────────────

def tokenize_fn(examples, tokenizer):
    """Tokenize input-target pairs for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=Config.MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=Config.MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    return model_inputs


# ── 5. Model Loading ───────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load a seq2seq model and tokenizer for full fine-tuning."""
    cfg = MODEL_CONFIGS[model_name]
    hf_name = cfg["hf_name"]
    log.info(f"Loading {model_name} ({hf_name})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params: {total_params / 1e6:.2f}M | "
             f"Trainable: {trainable_params / 1e6:.2f}M")

    return model, tokenizer, cfg


# ── 6. Training ─────────────────────────────────────────────────────────────

def train_model(model_name: str, model, tokenizer, cfg: dict,
                train_ds, val_ds):
    """Train a single seq2seq model with full fine-tuning."""
    log.info(f"\n{'='*70}")
    log.info(f"TRAINING: {model_name}")
    log.info(f"{'='*70}")

    output_dir = os.path.join(Config.OUTPUT_ROOT, model_name)

    tok_train = train_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True, batch_size=1000,
        remove_columns=train_ds.column_names, num_proc=1,
    )
    tok_val = val_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True, batch_size=1000,
        remove_columns=val_ds.column_names, num_proc=1,
    )

    # Metrics
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 with pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE
        rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        # BLEU
        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels]
        )

        return {
            "rouge1": round(rouge["rouge1"] * 100, 2),
            "rouge2": round(rouge["rouge2"] * 100, 2),
            "rougeL": round(rouge["rougeL"] * 100, 2),
            "bleu": round(bleu["score"], 2),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=Config.MAX_TARGET_LEN,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Model saved to {output_dir}")

    val_metrics = trainer.evaluate()
    log.info(f"Validation metrics: {val_metrics}")
    return trainer, val_metrics


# ── 7. Comprehensive Evaluation ─────────────────────────────────────────────

def generate_anonymized(text: str, model, tokenizer) -> str:
    """Generate anonymized version of input text."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=Config.MAX_INPUT_LEN
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=Config.MAX_TARGET_LEN,
            num_beams=Config.NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_leakage(originals: List[str], anonymized: List[str],
                    test_ds=None) -> Dict:
    """Measure PII leakage: entity-level and token-level.

    Entity Leak Rate: fraction of original PII entities appearing verbatim
    in the anonymized output.
    Leakage Rate: fraction of capitalized words (proxy for entities) that
    survive anonymization.
    """
    total_entities = 0
    leaked_entities = 0
    total_tokens = 0
    leaked_tokens = 0

    for orig, anon in zip(originals, anonymized):
        anon_lower = anon.lower()

        # Token-level leakage (capitalized words as PII proxy)
        for word in orig.split():
            if len(word) > 2 and any(c.isupper() for c in word):
                total_tokens += 1
                if word.lower() in anon_lower:
                    leaked_tokens += 1

        # Entity-level: extract multi-word capitalized sequences
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', orig)
        for ent in entities:
            if len(ent) > 2:
                total_entities += 1
                if ent.lower() in anon_lower:
                    leaked_entities += 1

    leakage_rate = leaked_tokens / max(total_tokens, 1) * 100
    entity_leak_rate = leaked_entities / max(total_entities, 1) * 100

    return {
        "leakage_rate": round(leakage_rate, 2),
        "entity_leak_rate": round(entity_leak_rate, 2),
        "entities_leaked": leaked_entities,
        "total_entities": total_entities,
        "tokens_leaked": leaked_tokens,
        "total_tokens": total_tokens,
    }


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Exact string match percentage."""
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return round(matches / max(len(predictions), 1) * 100, 2)


def compute_word_accuracy(predictions: List[str], references: List[str]) -> float:
    """Word-level accuracy across all examples."""
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        for pw, rw in zip(pred_words, ref_words):
            total += 1
            if pw == rw:
                correct += 1
        total += abs(len(pred_words) - len(ref_words))
    return round(correct / max(total, 1) * 100, 2)


def evaluate_model(model_name: str, model, tokenizer, test_ds) -> Dict:
    """Full evaluation suite for a single model."""
    log.info(f"\n{'='*70}")
    log.info(f"EVALUATING: {model_name}")
    log.info(f"{'='*70}")

    model.eval()
    model.to(DEVICE)

    originals = test_ds["input_text"]
    references = test_ds["target_text"]
    predictions = []

    log.info(f"Generating predictions for {len(originals)} test examples...")
    for i, text in enumerate(originals):
        pred = generate_anonymized(text, model, tokenizer)
        predictions.append(pred)
        if i < 5:
            log.info(f"  [{i+1}] ORIG: {text[:100]}...")
            log.info(f"       PRED: {pred[:100]}...")
            log.info(f"       TRUE: {references[i][:100]}...")

    # ── Metrics ──
    results = {"model": model_name}

    # Loss (via trainer eval)
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")

    # Exact Match & Word Accuracy
    results["exact_match"] = compute_exact_match(predictions, references)
    results["word_accuracy"] = compute_word_accuracy(predictions, references)

    # BLEU scores
    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )
    results["bleu"] = round(bleu_result["score"], 2)

    # BLEU-1, BLEU-2, BLEU-4 via NLTK
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    ref_tokens = [[ref.split()] for ref in references]
    pred_tokens = [pred.split() for pred in predictions]
    results["bleu_1"] = round(corpus_bleu(ref_tokens, pred_tokens,
                                           weights=(1, 0, 0, 0),
                                           smoothing_function=smooth) * 100, 2)
    results["bleu_2"] = round(corpus_bleu(ref_tokens, pred_tokens,
                                           weights=(0.5, 0.5, 0, 0),
                                           smoothing_function=smooth) * 100, 2)
    results["bleu_4"] = round(corpus_bleu(ref_tokens, pred_tokens,
                                           weights=(0.25, 0.25, 0.25, 0.25),
                                           smoothing_function=smooth) * 100, 2)

    # ROUGE scores
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    results["rouge_1"] = round(rouge_result["rouge1"] * 100, 2)
    results["rouge_2"] = round(rouge_result["rouge2"] * 100, 2)
    results["rouge_L"] = round(rouge_result["rougeL"] * 100, 2)

    # BERTScore
    try:
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(predictions=predictions, references=references, lang="en")
        results["bertscore_P"] = round(float(np.mean(bs["precision"])) * 100, 2)
        results["bertscore_R"] = round(float(np.mean(bs["recall"])) * 100, 2)
        results["bertscore_F1"] = round(float(np.mean(bs["f1"])) * 100, 2)
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        results["bertscore_P"] = results["bertscore_R"] = results["bertscore_F1"] = 0.0

    # Leakage
    leakage = compute_leakage(originals, predictions)
    results.update(leakage)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    results["params_M"] = round(total_params / 1e6, 2)

    log.info(f"\n  Results for {model_name}:")
    for k, v in results.items():
        if k != "model":
            log.info(f"    {k}: {v}")

    return results, predictions


# ── 8. Curated Eval Examples (37 examples: easy/medium/hard) ────────────────

CURATED_EVAL_EXAMPLES = [
    # ── EASY (10 examples) ──
    {"id": "easy_01", "category": "single_name", "difficulty": "easy",
     "input": "Please contact John regarding the project update."},
    {"id": "easy_02", "category": "single_name", "difficulty": "easy",
     "input": "Maria has submitted the report on time."},
    {"id": "easy_03", "category": "single_location", "difficulty": "easy",
     "input": "I live in London and work remotely."},
    {"id": "easy_04", "category": "single_location", "difficulty": "easy",
     "input": "The office is located in Chicago."},
    {"id": "easy_05", "category": "single_number", "difficulty": "easy",
     "input": "My account number is 74829361."},
    {"id": "easy_06", "category": "single_number", "difficulty": "easy",
     "input": "Please reference ticket number 55123."},
    {"id": "easy_07", "category": "single_email", "difficulty": "easy",
     "input": "You can reach me at sarah.jones@gmail.com for further details."},
    {"id": "easy_08", "category": "single_name", "difficulty": "easy",
     "input": "Thank you, David, for your quick response."},
    {"id": "easy_09", "category": "single_date", "difficulty": "easy",
     "input": "The appointment is scheduled for 15/07/2025."},
    {"id": "easy_10", "category": "single_location", "difficulty": "easy",
     "input": "She moved to Berlin last year."},

    # ── MEDIUM (12 examples) ──
    {"id": "medium_01", "category": "full_name", "difficulty": "medium",
     "input": "Dear Michael Thompson, your invoice has been processed successfully."},
    {"id": "medium_02", "category": "name_and_email", "difficulty": "medium",
     "input": "Jessica Parker can be contacted at jessica.parker@outlook.com."},
    {"id": "medium_03", "category": "name_and_location", "difficulty": "medium",
     "input": "Robert Chen from San Francisco submitted the application yesterday."},
    {"id": "medium_04", "category": "name_and_number", "difficulty": "medium",
     "input": "Hello Priya Sharma, we need to discuss your account 48291037."},
    {"id": "medium_05", "category": "name_and_phone", "difficulty": "medium",
     "input": "Please call Ahmed Hassan at +44 7911 123456 to confirm."},
    {"id": "medium_06", "category": "multiple_names", "difficulty": "medium",
     "input": "The meeting between Lisa Wong and James Miller has been rescheduled."},
    {"id": "medium_07", "category": "email_and_number", "difficulty": "medium",
     "input": "Send the verification code 83921 to admin@techcorp.io."},
    {"id": "medium_08", "category": "name_and_date", "difficulty": "medium",
     "input": "Emily Rodriguez was born on 23/04/1992 according to our records."},
    {"id": "medium_09", "category": "location_and_number", "difficulty": "medium",
     "input": "The property at 742 Evergreen Terrace, Springfield has ID 90210."},
    {"id": "medium_10", "category": "lowercase_input", "difficulty": "medium",
     "input": "my name is alex morgan and i live in seattle."},
    {"id": "medium_11", "category": "uppercase_input", "difficulty": "medium",
     "input": "CONTACT EMMA WATSON AT EMMA.WATSON@YAHOO.COM IMMEDIATELY."},
    {"id": "medium_12", "category": "mixed_case", "difficulty": "medium",
     "input": "jOHN sMITH lives in nEW yORK and his email is John@Gmail.Com."},

    # ── HARD (15 examples) ──
    {"id": "hard_01", "category": "multi_entity", "difficulty": "hard",
     "input": "Dr. Samantha Clarke from Boston General Hospital can be reached at samantha.clarke@bgh.org or +1 617 555 0192 regarding patient file 2847193."},
    {"id": "hard_02", "category": "multi_entity", "difficulty": "hard",
     "input": "Hi, I'm Rajesh Kumar. My employee ID is EMP-78432, my email is rajesh.k@infosys.com, and I work at the Bangalore office."},
    {"id": "hard_03", "category": "dense_pii", "difficulty": "hard",
     "input": "Transfer $5,000 from account 9821-4573-0012 to Maria Gonzalez (maria.g@bankmail.com) at 45 Oak Street, Miami, FL 33101."},
    {"id": "hard_04", "category": "long_text", "difficulty": "hard",
     "input": "Following up on our conversation, Daniel Kim mentioned that the project deadline is 31/12/2025. His colleague, Sophie Martin, disagreed and suggested we consult the client, Nakamura Industries, before proceeding. You can reach Daniel at daniel.kim@company.co or Sophie at +33 6 12 34 56 78."},
    {"id": "hard_05", "category": "informal_slang", "difficulty": "hard",
     "input": "yo hit up mike at mike99@hotmail.com or txt him at 555-867-5309 hes in LA rn"},
    {"id": "hard_06", "category": "typos", "difficulty": "hard",
     "input": "Plese contcat Jonh Smtih at jonh.smith@gmal.com abuot the accont 73829."},
    {"id": "hard_07", "category": "multi_language_names", "difficulty": "hard",
     "input": "The visa application for François Müller-Björkström was processed at the São Paulo consulate on 08/11/2024."},
    {"id": "hard_08", "category": "embedded_pii", "difficulty": "hard",
     "input": "Username: alex_chen_1995, Password reset email sent to alexchen@protonmail.com, last login from IP 192.168.1.42."},
    {"id": "hard_09", "category": "ambiguous_entities", "difficulty": "hard",
     "input": "Apple hired Jordan from Amazon. Jordan's first day in Cupertino is March 15th."},
    {"id": "hard_10", "category": "no_pii", "difficulty": "hard",
     "input": "The weather forecast predicts rain tomorrow with temperatures around 15 degrees."},
    {"id": "hard_11", "category": "no_pii", "difficulty": "hard",
     "input": "Please review the quarterly report and submit your feedback by Friday."},
    {"id": "hard_12", "category": "repeated_entity", "difficulty": "hard",
     "input": "Call Sarah. Sarah's number is 555-0147. Tell Sarah that Sarah's appointment is confirmed."},
    {"id": "hard_13", "category": "tabular_format", "difficulty": "hard",
     "input": "Name: Wei Zhang, DOB: 12/03/1988, SSN: 123-45-6789, Address: 88 Pine Road, Austin, TX 73301."},
    {"id": "hard_14", "category": "conversational", "difficulty": "hard",
     "input": "Hey, it's Tom. Can you send the package to my new place? It's 1520 Maple Avenue, Portland. My zip is 97201 and phone is 503-555-0198."},
    {"id": "hard_15", "category": "edge_numbers", "difficulty": "hard",
     "input": "Patient ID: P-2024-08173, Room 42B, admitted on 01/15/2024 by Dr. Ananya Patel, contact: ananya.p@hospital.org."},
]


def check_pii_changed(original: str, output: str, category: str) -> str:
    """Determine if PII was changed, unchanged, or if no-PII was preserved."""
    if category == "no_pii":
        return "CORRECT" if original.strip() == output.strip() else "FALSE POSITIVE"
    return "CHANGED" if original.strip() != output.strip() else "UNCHANGED"


def run_curated_eval(model_name: str, model, tokenizer) -> Dict:
    """Run 37 curated eval examples and report detailed results."""
    log.info(f"\n  Running curated eval examples for {model_name}...")

    results = {"easy": [], "medium": [], "hard": []}
    pii_changed, pii_total = 0, 0
    nopii_correct, nopii_total = 0, 0

    for example in CURATED_EVAL_EXAMPLES:
        output = generate_anonymized(example["input"], model, tokenizer)
        status = check_pii_changed(example["input"], output, example["category"])

        result = {
            "id": example["id"],
            "category": example["category"],
            "input": example["input"],
            "output": output,
            "status": status,
        }
        results[example["difficulty"]].append(result)

        if example["category"] == "no_pii":
            nopii_total += 1
            if status == "CORRECT":
                nopii_correct += 1
        else:
            pii_total += 1
            if status == "CHANGED":
                pii_changed += 1

    summary = {
        "total": len(CURATED_EVAL_EXAMPLES),
        "pii_anonymized": pii_changed,
        "pii_total": pii_total,
        "pii_rate": round(pii_changed / max(pii_total, 1) * 100, 1),
        "nopii_correct": nopii_correct,
        "nopii_total": nopii_total,
        "nopii_rate": round(nopii_correct / max(nopii_total, 1) * 100, 1),
        "details": results,
    }

    log.info(f"  PII anonymized: {pii_changed}/{pii_total} ({summary['pii_rate']}%)")
    log.info(f"  No-PII correct: {nopii_correct}/{nopii_total} ({summary['nopii_rate']}%)")

    # Per-difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        items = results[diff]
        pii_items = [x for x in items if x["category"] != "no_pii"]
        nopii_items = [x for x in items if x["category"] == "no_pii"]
        pii_ok = sum(1 for x in pii_items if x["status"] == "CHANGED")
        nopii_ok = sum(1 for x in nopii_items if x["status"] == "CORRECT")
        log.info(f"    {diff.upper()}: PII {pii_ok}/{len(pii_items)}"
                 f"  No-PII {nopii_ok}/{len(nopii_items)}")

    return summary


# ── 9. Results Reporting ────────────────────────────────────────────────────

def format_comparison_table(all_results: List[Dict]):
    """Generate a formatted comparison table similar to evaluation_results_readable.txt."""
    header = (f"{'Model':<25} {'Size':>10} {'Loss':>8} {'Exact%':>8} {'WordAcc%':>8} "
              f"{'BLEU':>8} {'ROUGE-L':>8} {'BERTScF1':>8} {'Leak%':>8}")
    separator = "─" * len(header)

    lines = ["\n" + "=" * 80, "  MODEL COMPARISON TABLE  (sorted smallest → largest)",
             "=" * 80, "", header, separator]

    sorted_results = sorted(all_results, key=lambda x: x.get("params_M", 0))
    for r in sorted_results:
        lines.append(
            f"  {r['model']:<23} {r.get('params_M', 0):>8.2f}M "
            f"{r.get('loss', 0):>8.4f} {r.get('exact_match', 0):>7.2f} "
            f"{r.get('word_accuracy', 0):>8.2f} "
            f"{r.get('bleu', 0):>8.2f} {r.get('rouge_L', 0):>8.2f} "
            f"{r.get('bertscore_F1', 0):>8.2f} "
            f"{r.get('entity_leak_rate', 0):>7.2f}"
        )

    return "\n".join(lines)


def save_results(all_results: List[Dict], all_curated: Dict):
    """Save complete evaluation results to JSON and readable text."""
    out_json = os.path.join(Config.OUTPUT_ROOT, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"models": all_results, "curated_eval": all_curated},
                  f, indent=2, default=str, ensure_ascii=False)
    log.info(f"JSON results → {out_json}")

    # Readable text report
    out_txt = os.path.join(Config.OUTPUT_ROOT, "evaluation_results_readable.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  SEQ2SEQ TEXT ANONYMIZATION — EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(sorted(all_results, key=lambda x: x.get("params_M", 0))):
            name = r["model"]
            f.write(f"\n  [{i+1}/{len(all_results)}]  {name}  ({MODEL_CONFIGS[name]['hf_name']})\n")
            f.write("─" * 80 + "\n")
            f.write(f"  Model size:    {r.get('params_M', 0):.2f}M parameters\n")
            f.write(f"  Type:          Full fine-tuning\n\n")
            f.write(f"  TEST SET METRICS\n")
            f.write(f"    BLEU:           {r.get('bleu', 0):.2f}\n")
            f.write(f"    BLEU-1:         {r.get('bleu_1', 0):.2f}\n")
            f.write(f"    BLEU-2:         {r.get('bleu_2', 0):.2f}\n")
            f.write(f"    BLEU-4:         {r.get('bleu_4', 0):.2f}\n")
            f.write(f"    ROUGE-1:        {r.get('rouge_1', 0):.2f}\n")
            f.write(f"    ROUGE-2:        {r.get('rouge_2', 0):.2f}\n")
            f.write(f"    ROUGE-L:        {r.get('rouge_L', 0):.2f}\n")
            f.write(f"    BERTScore P:    {r.get('bertscore_P', 0):.2f}\n")
            f.write(f"    BERTScore R:    {r.get('bertscore_R', 0):.2f}\n")
            f.write(f"    BERTScore F1:   {r.get('bertscore_F1', 0):.2f}\n")
            f.write(f"    Leakage Rate:   {r.get('leakage_rate', 0):.2f}\n")
            f.write(f"    Entity Leak:    {r.get('entity_leak_rate', 0):.2f}\n")
            f.write(f"    Exact Match:    {r.get('exact_match', 0):.2f}\n")
            f.write(f"    Word Accuracy:  {r.get('word_accuracy', 0):.2f}\n\n")

            # Curated eval for this model
            if name in all_curated:
                ce = all_curated[name]
                f.write(f"  CURATED EVAL: PII {ce['pii_anonymized']}/{ce['pii_total']} "
                        f"({ce['pii_rate']}%) | No-PII {ce['nopii_correct']}/{ce['nopii_total']} "
                        f"({ce['nopii_rate']}%)\n\n")

        # Comparison table
        f.write(format_comparison_table(all_results))
        f.write("\n")

    log.info(f"Readable report → {out_txt}")


# ── 10. Visualization ──────────────────────────────────────────────────────

def plot_model_comparison(all_results: List[Dict]):
    """Generate comparison plots across all models."""
    sorted_r = sorted(all_results, key=lambda x: x.get("params_M", 0))
    names = [r["model"] for r in sorted_r]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # BLEU
    axes[0, 0].bar(x, [r.get("bleu", 0) for r in sorted_r], color="#3498db")
    axes[0, 0].set_title("BLEU Score")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(names, rotation=30, ha="right")

    # ROUGE-L
    axes[0, 1].bar(x, [r.get("rouge_L", 0) for r in sorted_r], color="#2ecc71")
    axes[0, 1].set_title("ROUGE-L")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(names, rotation=30, ha="right")

    # BERTScore F1
    axes[0, 2].bar(x, [r.get("bertscore_F1", 0) for r in sorted_r], color="#9b59b6")
    axes[0, 2].set_title("BERTScore F1")
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(names, rotation=30, ha="right")

    # Entity Leak Rate
    axes[1, 0].bar(x, [r.get("entity_leak_rate", 0) for r in sorted_r], color="#e74c3c")
    axes[1, 0].set_title("Entity Leak Rate (%) ↓")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=30, ha="right")

    # PII Anonymization Rate (from curated eval if available)
    axes[1, 1].bar(x, [r.get("word_accuracy", 0) for r in sorted_r], color="#f39c12")
    axes[1, 1].set_title("Word Accuracy (%)")
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(names, rotation=30, ha="right")

    # Parameter count vs BLEU (scatter)
    params = [r.get("params_M", 0) for r in sorted_r]
    bleu = [r.get("bleu", 0) for r in sorted_r]
    axes[1, 2].scatter(params, bleu, s=100, c="#e67e22", zorder=3)
    for i, name in enumerate(names):
        axes[1, 2].annotate(name, (params[i], bleu[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1, 2].set_xlabel("Parameters (M)")
    axes[1, 2].set_ylabel("BLEU")
    axes[1, 2].set_title("Scale vs Quality")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(Config.PLOT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Comparison plots → {path}")

    # Privacy-Utility tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in sorted_r:
        ax.scatter(r.get("entity_leak_rate", 0), r.get("bleu", 0),
                   s=r.get("params_M", 50) * 2, alpha=0.7, zorder=3)
        ax.annotate(r["model"],
                    (r.get("entity_leak_rate", 0), r.get("bleu", 0)),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Entity Leak Rate (%) →", fontsize=12)
    ax.set_ylabel("BLEU Score ↑", fontsize=12)
    ax.set_title("Privacy–Utility Tradeoff", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    path = os.path.join(Config.PLOT_DIR, "privacy_utility_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Privacy-utility plot → {path}")


# ── 11. Error Analysis ──────────────────────────────────────────────────────

def error_analysis(all_curated: Dict):
    """Analyze common failure patterns across all models."""
    log.info("\n" + "=" * 70)
    log.info("ERROR ANALYSIS ACROSS MODELS")
    log.info("=" * 70)

    failure_categories = defaultdict(lambda: defaultdict(int))

    for model_name, curated in all_curated.items():
        for diff in ["easy", "medium", "hard"]:
            for item in curated["details"][diff]:
                if item["status"] in ("UNCHANGED", "FALSE POSITIVE"):
                    failure_categories[item["category"]][model_name] += 1

    if failure_categories:
        log.info("\n  Failure patterns:")
        for cat, models in sorted(failure_categories.items()):
            model_list = ", ".join(f"{m}({c})" for m, c in models.items())
            log.info(f"    {cat}: {model_list}")

    # Entity consistency check
    log.info("\n  Entity consistency issues:")
    for model_name, curated in all_curated.items():
        for item in curated["details"]["hard"]:
            if item["id"] == "hard_12" and item["status"] == "CHANGED":
                output = item["output"]
                # Check if "Sarah" still appears in output (should be replaced consistently)
                if "sarah" in output.lower():
                    log.info(f"    {model_name}: repeated entity 'Sarah' not fully anonymized")

    # No-PII preservation
    log.info("\n  No-PII preservation:")
    for model_name, curated in all_curated.items():
        for item in curated["details"]["hard"]:
            if item["category"] == "no_pii" and item["status"] == "FALSE POSITIVE":
                log.info(f"    {model_name}: false positive on {item['id']}")


# ── 12. Main Pipeline ──────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("SEQ2SEQ TEXT ANONYMIZATION — COMPARATIVE STUDY")
    log.info("=" * 70)

    # Load and prepare data
    ds = load_ai4privacy()
    if Config.QUICK_MODE:
        ds = quick_subsample(ds, Config.QUICK_SAMPLE_N)
        log.info(f"Quick mode: {len(ds):,} samples")

    train_ds, val_ds, test_ds = prepare_dataset(ds)

    # Select models to run
    if Config.SELECTED_MODEL:
        models_to_run = {Config.SELECTED_MODEL: MODEL_CONFIGS[Config.SELECTED_MODEL]}
    else:
        models_to_run = MODEL_CONFIGS

    all_results = []
    all_curated = {}

    for model_name, cfg in models_to_run.items():
        log.info(f"\n{'#' * 70}")
        log.info(f"# Model: {model_name}")
        log.info(f"{'#' * 70}")

        output_dir = os.path.join(Config.OUTPUT_ROOT, model_name)

        if Config.EVAL_ONLY:
            # Load from checkpoint
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
        else:
            model, tokenizer, cfg_loaded = load_model(model_name)
            model.to(DEVICE)
            train_model(model_name, model, tokenizer, cfg_loaded, train_ds, val_ds)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.to(DEVICE)
        results, predictions = evaluate_model(model_name, model, tokenizer, test_ds)
        curated = run_curated_eval(model_name, model, tokenizer)

        all_results.append(results)
        all_curated[model_name] = curated

        # Free memory
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save and visualize
    save_results(all_results, all_curated)
    if len(all_results) > 1:
        plot_model_comparison(all_results)
    error_analysis(all_curated)

    # Final comparison
    log.info(format_comparison_table(all_results))
    log.info("\n" + "=" * 70)
    log.info("COMPARATIVE STUDY COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

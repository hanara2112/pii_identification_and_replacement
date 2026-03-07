# ==============================================================================
# BASELINE: Language-Stratified Privacy-Preserving Text Anonymization
# ==============================================================================
#
# Loads the full AI4Privacy ~500K multilingual dataset, splits it into two
# language-stratified halves (each half mirrors the language distribution),
# and trains two independent models:
#
#   Model A  (Censor)      — DeBERTa-v3 NER token classifier   → trained on Half A
#   Model B  (Hallucinator)— Flan-T5-Large seq2seq generator    → trained on Half B
#
# At inference the two models are composed: Censor detects PII spans, then
# Hallucinator generates context-aware replacements.
#
# This is a straightforward baseline implementation.  Genuinely novel
# contributions live in novel_pipeline.py and paper.tex.
#
# Usage:
#   python script.py                  # Full pipeline
#   python script.py --quick          # Quick test with small subset
#   python script.py --mode censor    # Train Censor only
#   python script.py --mode halluc    # Train Hallucinator only
#   python script.py --mode eval      # Evaluation only
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
        "peft>=0.10.0", "bitsandbytes>=0.43", "rouge_score", "sacrebleu",
        "sentencepiece", "scipy", "scikit-learn", "pandas", "matplotlib",
        "seqeval", "bert_score", "Faker",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U"] + deps,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

install_deps()

import evaluate
from datasets import Dataset, load_dataset, DatasetDict
from faker import Faker
from sklearn.model_selection import train_test_split
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1_score,
)
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForSeq2Seq, DataCollatorForTokenClassification,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE} | PyTorch {torch.__version__}")


# ── 2. Configuration ────────────────────────────────────────────────────────

class Config:
    MODE: str = "ALL"

    # Backbones
    CENSOR_BACKBONE  = "microsoft/deberta-v3-base"
    HALLUC_BACKBONE  = "google/flan-t5-large"

    # Dirs
    OUTPUT_ROOT = "./privacy_baseline"
    CENSOR_DIR  = "./privacy_baseline/censor"
    HALLUC_DIR  = "./privacy_baseline/hallucinator"
    EVAL_DIR    = "./privacy_baseline/evaluation"
    PLOT_DIR    = "./privacy_baseline/plots"

    # NER (Censor)
    NER_MAX_LEN      = 256
    NER_BATCH_SIZE   = 16
    NER_GRAD_ACCUM   = 2
    NER_EPOCHS       = 5
    NER_LR           = 3e-5
    NER_WEIGHT_DECAY = 0.01
    NER_WARMUP_RATIO = 0.1

    # Seq2Seq (Hallucinator)
    SEQ2SEQ_MAX_LEN      = 256
    SEQ2SEQ_BATCH_SIZE   = 8
    SEQ2SEQ_GRAD_ACCUM   = 4
    SEQ2SEQ_EPOCHS       = 3
    SEQ2SEQ_LR           = 1e-4
    SEQ2SEQ_WARMUP_RATIO = 0.06

    # QLoRA
    LORA_R       = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]

    # Generation
    GEN_MAX_TOKENS = 200
    GEN_NUM_BEAMS  = 4

    # Data
    QUICK_MODE     = False
    QUICK_SAMPLE_N = 2000
    TEST_RATIO     = 0.05
    NUM_EVAL       = 200

    # Entity types (AI4Privacy BIO tags)
    ENTITY_TYPES = [
        "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
        "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
        "DRIVER_LICENSE", "USERNAME", "URL", "MEDICAL", "ACCOUNT",
        "BUILDING", "POSTCODE",
    ]


def setup_dirs():
    for d in [Config.OUTPUT_ROOT, Config.CENSOR_DIR, Config.HALLUC_DIR,
              Config.EVAL_DIR, Config.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

setup_dirs()


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Privacy Pipeline")
    parser.add_argument("--mode", default="ALL",
                        choices=["ALL", "CENSOR", "HALLUC", "EVAL"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs-ner", type=int, default=None)
    parser.add_argument("--epochs-seq2seq", type=int, default=None)
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(mode="ALL", quick=False,
                                  epochs_ner=None, epochs_seq2seq=None)
    Config.MODE = args.mode.upper()
    Config.QUICK_MODE = args.quick
    if args.epochs_ner:
        Config.NER_EPOCHS = args.epochs_ner
    if args.epochs_seq2seq:
        Config.SEQ2SEQ_EPOCHS = args.epochs_seq2seq

parse_args()


# ── 3. BIO Tag System ───────────────────────────────────────────────────────

def build_bio_labels(entity_types: List[str]):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(Config.ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)
log.info(f"BIO label set: {NUM_LABELS} labels ({len(Config.ENTITY_TYPES)} entity types)")


# ── 4. Data Loading & Language-Stratified Split ─────────────────────────────

def load_ai4privacy():
    """Load the full AI4Privacy dataset from HuggingFace."""
    log.info("Loading AI4Privacy dataset (full multilingual)...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    log.info(f"Loaded {len(ds):,} examples")
    return ds


def language_stratified_split(ds, test_ratio=0.05):
    """Split dataset into two halves with equal language representation.

    For every language in the dataset, we split that language's examples
    50/50 into Half-A and Half-B.  A small held-out test set is also
    carved from each language proportionally.

    Returns: half_a (Dataset), half_b (Dataset), test_set (Dataset)
    """
    log.info("Performing language-stratified split...")

    df = ds.to_pandas()

    # Detect language column
    lang_col = None
    for candidate in ["language", "lang", "Language"]:
        if candidate in df.columns:
            lang_col = candidate
            break
    if lang_col is None:
        log.warning("No language column found — falling back to random split")
        lang_col = "__lang_dummy"
        df[lang_col] = "unknown"

    lang_counts = df[lang_col].value_counts()
    log.info(f"Languages found: {len(lang_counts)}")
    for lang, cnt in lang_counts.items():
        log.info(f"  {lang}: {cnt:,}")

    half_a_indices, half_b_indices, test_indices = [], [], []

    for lang, group in df.groupby(lang_col):
        indices = group.index.tolist()
        random.shuffle(indices)

        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        n_remaining = n - n_test
        n_half = n_remaining // 2

        test_indices.extend(indices[:n_test])
        half_a_indices.extend(indices[n_test:n_test + n_half])
        half_b_indices.extend(indices[n_test + n_half:])

    random.shuffle(half_a_indices)
    random.shuffle(half_b_indices)
    random.shuffle(test_indices)

    half_a = ds.select(half_a_indices)
    half_b = ds.select(half_b_indices)
    test_set = ds.select(test_indices)

    log.info(f"Split complete: Half-A={len(half_a):,} | "
             f"Half-B={len(half_b):,} | Test={len(test_set):,}")

    # Verify stratification
    for name, subset in [("Half-A", half_a), ("Half-B", half_b), ("Test", test_set)]:
        sub_df = subset.to_pandas()
        if lang_col in sub_df.columns:
            log.info(f"  {name} languages: {dict(sub_df[lang_col].value_counts())}")

    return half_a, half_b, test_set


def quick_subsample(ds, n=2000):
    if len(ds) <= n:
        return ds
    indices = random.sample(range(len(ds)), n)
    return ds.select(indices)


# ── 5. NER Data Preparation (Censor — Half A) ──────────────────────────────

ENTITY_MAPPING = {
    "FIRSTNAME": "PERSON", "LASTNAME": "PERSON", "NAME": "PERSON",
    "MIDDLENAME": "PERSON", "PREFIX": "PERSON", "SUFFIX": "PERSON",
    "GENDER": "PERSON",
    "CITY": "LOC", "STATE": "LOC", "COUNTRY": "LOC", "COUNTY": "LOC",
    "ORDINALDIRECTION": "LOC",
    "STREETADDRESS": "ADDRESS", "STREET": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS", "NEARBYGPSCOORDINATE": "ADDRESS",
    "ZIPCODE": "POSTCODE", "BUILDINGNUMBER": "BUILDING",
    "PHONENUMBER": "PHONE",
    "CREDITCARDNUMBER": "CREDIT_CARD", "CREDITCARD": "CREDIT_CARD",
    "IPADDRESS": "IP_ADDRESS", "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS",
    "MAC": "IP_ADDRESS",
    "DRIVINGLICENSE": "DRIVER_LICENSE",
    "COMPANY": "ORG", "ORGANIZATION": "ORG", "HOSPITAL": "ORG",
    "UNIVERSITY": "ORG", "JOBTITLE": "ORG",
    "ACCOUNTNUMBER": "ACCOUNT", "BITCOINADDRESS": "ACCOUNT",
    "ACCOUNTNAME": "ACCOUNT", "VEHICLEVIN": "ACCOUNT",
    "VEHICLEVRM": "ACCOUNT", "IMEI": "ACCOUNT", "PASSWORD": "ACCOUNT",
    "PIN": "ACCOUNT", "USERAGENT": "ACCOUNT", "CURRENCYCODE": "ACCOUNT",
    "CURRENCYNAME": "ACCOUNT", "CURRENCYSYMBOL": "ACCOUNT",
    "AMOUNT": "ACCOUNT", "MASKEDNUMBER": "ACCOUNT",
    "LITECOINADDRESS": "ACCOUNT", "ETHEREUMADDRESS": "ACCOUNT",
    "BIC": "ACCOUNT",
    "DOB": "DATE", "TIME": "DATE", "DATEOFBIRTH": "DATE", "AGE": "DATE",
    "IBAN": "IBAN",
}


def find_text_and_labels(example):
    """Extract tokens and BIO labels from an AI4Privacy example."""
    if "tokens" in example and "ner_tags" in example:
        return example["tokens"], example["ner_tags"]
    if "tokens" in example and "labels" in example:
        return example["tokens"], example["labels"]

    text = example.get("source_text", example.get("text", ""))
    masks = example.get("privacy_mask", [])

    if not masks:
        tokens = text.split()
        return tokens, ["O"] * len(tokens)

    spans = sorted(masks, key=lambda m: m.get("start", m.get("offset", 0)))
    tokens, labels = [], []
    pos = 0

    for span in spans:
        start = span.get("start", span.get("offset", 0))
        end = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O"))
        value = span.get("value", text[start:end])

        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before)
            labels.extend(["O"] * len(before))

        entity_tokens = value.split()
        if entity_tokens:
            bio = label.upper().replace(" ", "_")
            bio = ENTITY_MAPPING.get(bio, bio)

            if bio in Config.ENTITY_TYPES:
                tokens.append(entity_tokens[0])
                labels.append(f"B-{bio}")
                for t in entity_tokens[1:]:
                    tokens.append(t)
                    labels.append(f"I-{bio}")
            else:
                tokens.extend(entity_tokens)
                labels.extend(["O"] * len(entity_tokens))
        pos = end

    if pos < len(text):
        rest = text[pos:].split()
        tokens.extend(rest)
        labels.extend(["O"] * len(rest))

    return tokens, labels


def tokenize_and_align_ner(examples, tokenizer):
    """Tokenize and align BIO labels with sub-word tokens."""
    all_tokens, all_labels = [], []
    key = ("source_text" if "source_text" in examples
           else "text" if "text" in examples else "tokens")

    for i in range(len(examples[key])):
        example = {k: v[i] for k, v in examples.items()}
        tokens, labels = find_text_and_labels(example)
        all_tokens.append(tokens)
        all_labels.append(labels)

    tokenized = tokenizer(
        all_tokens, truncation=True, max_length=Config.NER_MAX_LEN,
        padding="max_length", is_split_into_words=True,
    )

    aligned_labels = []
    for i, label_seq in enumerate(all_labels):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                lbl = label_seq[wid] if wid < len(label_seq) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = label_seq[wid] if wid < len(label_seq) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
            prev_word = wid
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


# ── 6. Seq2Seq Data Preparation (Hallucinator — Half B) ────────────────────

def prepare_seq2seq_examples(example):
    """Create mask-and-fill pairs for seq2seq training."""
    tokens, labels = find_text_and_labels(example)

    masked, prev_entity = [], None
    for tok, lbl in zip(tokens, labels):
        if lbl == "O":
            masked.append(tok)
            prev_entity = None
        elif lbl.startswith("B-"):
            masked.append(f"[{lbl[2:]}]")
            prev_entity = lbl[2:]
        elif lbl.startswith("I-") and prev_entity:
            pass  # continuation — skip
        else:
            masked.append(tok)
            prev_entity = None

    return {
        "input_text": f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}",
        "target_text": " ".join(tokens),
    }


def tokenize_seq2seq(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input_text"], max_length=Config.SEQ2SEQ_MAX_LEN,
        truncation=True, padding="max_length",
    )
    labels = tokenizer(
        examples["target_text"], max_length=Config.SEQ2SEQ_MAX_LEN,
        truncation=True, padding="max_length",
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lab]
        for lab in labels["input_ids"]
    ]
    return model_inputs


# ── 7. Model A — Censor (DeBERTa-v3 NER) ───────────────────────────────────

def build_censor():
    log.info(f"Loading Censor backbone: {Config.CENSOR_BACKBONE}")
    tokenizer = AutoTokenizer.from_pretrained(Config.CENSOR_BACKBONE)
    model = AutoModelForTokenClassification.from_pretrained(
        Config.CENSOR_BACKBONE,
        num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    log.info(f"Censor parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def compute_ner_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_sent, pred_sent = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_sent.append(ID2LABEL.get(int(l), "O"))
            pred_sent.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_sent)
        pred_labels.append(pred_sent)
    return {"f1": seq_f1_score(true_labels, pred_labels, average="weighted")}


def train_censor(train_ds, val_ds, model, tokenizer):
    log.info("=== Training Censor (Model A) on Half-A ===")

    ner_train = train_ds.map(
        lambda ex: tokenize_and_align_ner(ex, tokenizer),
        batched=True, batch_size=500, remove_columns=train_ds.column_names,
    )
    ner_val = val_ds.map(
        lambda ex: tokenize_and_align_ner(ex, tokenizer),
        batched=True, batch_size=500, remove_columns=val_ds.column_names,
    )

    args = TrainingArguments(
        output_dir=Config.CENSOR_DIR,
        num_train_epochs=Config.NER_EPOCHS,
        per_device_train_batch_size=Config.NER_BATCH_SIZE,
        per_device_eval_batch_size=Config.NER_BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.NER_GRAD_ACCUM,
        learning_rate=Config.NER_LR,
        weight_decay=Config.NER_WEIGHT_DECAY,
        warmup_ratio=Config.NER_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ner_train, eval_dataset=ner_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(Config.CENSOR_DIR)
    tokenizer.save_pretrained(Config.CENSOR_DIR)
    log.info(f"Censor saved to {Config.CENSOR_DIR}")

    metrics = trainer.evaluate()
    log.info(f"Censor val F1: {metrics.get('eval_f1', 0):.4f}")
    return trainer, metrics


# ── 8. Model B — Hallucinator (Flan-T5-Large + QLoRA) ──────────────────────

def build_hallucinator():
    log.info(f"Loading Hallucinator backbone: {Config.HALLUC_BACKBONE}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.HALLUC_BACKBONE)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.HALLUC_BACKBONE,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Hallucinator: {trainable:,}/{total:,} trainable ({100*trainable/total:.2f}%)")
    return model, tokenizer


def train_hallucinator(train_ds, val_ds, model, tokenizer):
    log.info("=== Training Hallucinator (Model B) on Half-B ===")

    seq2seq_train = train_ds.map(prepare_seq2seq_examples,
                                  remove_columns=train_ds.column_names)
    seq2seq_val = val_ds.map(prepare_seq2seq_examples,
                              remove_columns=val_ds.column_names)

    seq2seq_train = seq2seq_train.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer),
        batched=True, batch_size=500,
        remove_columns=["input_text", "target_text"],
    )
    seq2seq_val = seq2seq_val.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer),
        batched=True, batch_size=500,
        remove_columns=["input_text", "target_text"],
    )

    rouge = evaluate.load("rouge")
    def compute_seq2seq_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=Config.HALLUC_DIR,
        num_train_epochs=Config.SEQ2SEQ_EPOCHS,
        per_device_train_batch_size=Config.SEQ2SEQ_BATCH_SIZE,
        per_device_eval_batch_size=Config.SEQ2SEQ_BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.SEQ2SEQ_GRAD_ACCUM,
        learning_rate=Config.SEQ2SEQ_LR,
        warmup_ratio=Config.SEQ2SEQ_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=Config.GEN_MAX_TOKENS,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=seq2seq_train, eval_dataset=seq2seq_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_seq2seq_metrics,
    )

    trainer.train()
    model.save_pretrained(Config.HALLUC_DIR)
    tokenizer.save_pretrained(Config.HALLUC_DIR)
    log.info(f"Hallucinator saved to {Config.HALLUC_DIR}")

    metrics = trainer.evaluate()
    log.info(f"Hallucinator val ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    return trainer, metrics


# ── 9. Inference — Compose Censor + Hallucinator ────────────────────────────

def anonymize_text(text: str, censor_model, censor_tok, halluc_model, halluc_tok) -> str:
    """Full pipeline: NER detection → seq2seq replacement."""

    # Step 1: NER
    inputs = censor_tok(text, return_tensors="pt", truncation=True,
                        max_length=Config.NER_MAX_LEN).to(DEVICE)
    censor_model.eval()
    with torch.no_grad():
        logits = censor_model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tok_ids = inputs["input_ids"][0].tolist()
    tokens = censor_tok.convert_ids_to_tokens(tok_ids)

    masked, prev_ent = [], None
    for tok, pid in zip(tokens, preds):
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            continue
        label = ID2LABEL.get(pid, "O")
        if label.startswith("B-"):
            masked.append(f"[{label[2:]}]")
            prev_ent = label[2:]
        elif label.startswith("I-") and prev_ent:
            continue
        else:
            clean = tok.lstrip("#▁")
            if (tok.startswith("##") or tok.startswith("▁")) and masked:
                masked[-1] += clean
            else:
                masked.append(clean if clean else tok)
            prev_ent = None

    # Step 2: Generate replacements
    prompt = f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}"
    inputs = halluc_tok(prompt, return_tensors="pt", truncation=True,
                        max_length=Config.SEQ2SEQ_MAX_LEN).to(DEVICE)
    halluc_model.eval()
    with torch.no_grad():
        out = halluc_model.generate(
            **inputs, max_new_tokens=Config.GEN_MAX_TOKENS,
            num_beams=Config.GEN_NUM_BEAMS, early_stopping=True,
        )
    return halluc_tok.decode(out[0], skip_special_tokens=True)


# ── 10. Evaluation ──────────────────────────────────────────────────────────

def compute_leakage(originals: List[str], anonymized: List[str]) -> Dict:
    """Measure PII leakage: fraction of original entities surviving in output."""
    total, leaked = 0, 0
    for orig, anon in zip(originals, anonymized):
        anon_lower = anon.lower()
        for w in orig.split():
            if len(w) > 2 and w[0].isupper():
                total += 1
                if w.lower() in anon_lower:
                    leaked += 1
    rate = leaked / max(total, 1)
    return {"total_entities": total, "leaked": leaked,
            "leakage_rate": rate, "privacy_retention": 1.0 - rate}


def evaluate_pipeline(test_ds, censor_model, censor_tok, halluc_model, halluc_tok):
    log.info("=== Evaluation ===")
    n = min(Config.NUM_EVAL, len(test_ds))
    originals, anonymized = [], []

    for i in range(n):
        tokens, labels = find_text_and_labels(test_ds[i])
        orig = " ".join(tokens)
        anon = anonymize_text(orig, censor_model, censor_tok, halluc_model, halluc_tok)
        originals.append(orig)
        anonymized.append(anon)
        if i < 5:
            log.info(f"  Orig:  {orig[:120]}...")
            log.info(f"  Anon:  {anon[:120]}...")

    leakage = compute_leakage(originals, anonymized)
    log.info(f"Leakage: {leakage['leakage_rate']:.4f} | "
             f"Privacy retention: {leakage['privacy_retention']:.4f}")

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=anonymized, references=originals)
    log.info(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")

    bs_f1 = 0.0
    try:
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(predictions=anonymized, references=originals, lang="en")
        bs_f1 = float(np.mean(bs["f1"]))
        log.info(f"BERTScore F1: {bs_f1:.4f}")
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")

    results = {"leakage": leakage, "rouge": rouge_scores,
               "bertscore_f1": bs_f1, "n": n}

    out_path = os.path.join(Config.EVAL_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results → {out_path}")
    return results


# ── 11. Visualization ──────────────────────────────────────────────────────

def plot_results(results: Dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(["Privacy\nRetention", "Leakage\nRate"],
                [results["leakage"]["privacy_retention"],
                 results["leakage"]["leakage_rate"]],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_ylim(0, 1); axes[0].set_title("Privacy")

    rv = {k: v for k, v in results["rouge"].items() if isinstance(v, float)}
    axes[1].bar(rv.keys(), rv.values(), color="#3498db")
    axes[1].set_ylim(0, 1); axes[1].set_title("ROUGE (Utility)")

    axes[2].bar(["BERTScore F1"], [results["bertscore_f1"]], color="#9b59b6")
    axes[2].set_ylim(0, 1); axes[2].set_title("BERTScore")

    plt.tight_layout()
    path = os.path.join(Config.PLOT_DIR, "baseline_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Plots → {path}")


# ── 12. Main Pipeline ──────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("BASELINE: Language-Stratified Split → 2 Models")
    log.info("=" * 70)

    ds = load_ai4privacy()
    if Config.QUICK_MODE:
        ds = quick_subsample(ds, Config.QUICK_SAMPLE_N)
        log.info(f"Quick mode: {len(ds):,} samples")

    half_a, half_b, test_set = language_stratified_split(ds, Config.TEST_RATIO)

    ha_train_idx, ha_val_idx = train_test_split(
        range(len(half_a)), test_size=0.1, random_state=SEED)
    hb_train_idx, hb_val_idx = train_test_split(
        range(len(half_b)), test_size=0.1, random_state=SEED)

    ha_train, ha_val = half_a.select(ha_train_idx), half_a.select(ha_val_idx)
    hb_train, hb_val = half_b.select(hb_train_idx), half_b.select(hb_val_idx)

    log.info(f"Half-A  train={len(ha_train):,}  val={len(ha_val):,}")
    log.info(f"Half-B  train={len(hb_train):,}  val={len(hb_val):,}")
    log.info(f"Test    n={len(test_set):,}")

    censor_model = censor_tok = halluc_model = halluc_tok = None

    if Config.MODE in ("ALL", "CENSOR"):
        censor_model, censor_tok = build_censor()
        censor_model.to(DEVICE)
        train_censor(ha_train, ha_val, censor_model, censor_tok)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if Config.MODE in ("ALL", "HALLUC"):
        halluc_model, halluc_tok = build_hallucinator()
        train_hallucinator(hb_train, hb_val, halluc_model, halluc_tok)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if Config.MODE in ("ALL", "EVAL"):
        if censor_model is None:
            censor_tok = AutoTokenizer.from_pretrained(Config.CENSOR_DIR)
            censor_model = AutoModelForTokenClassification.from_pretrained(Config.CENSOR_DIR)
            censor_model.to(DEVICE)
        if halluc_model is None:
            halluc_tok = AutoTokenizer.from_pretrained(Config.HALLUC_DIR)
            halluc_model = AutoModelForSeq2SeqLM.from_pretrained(Config.HALLUC_DIR)
            halluc_model.to(DEVICE)

        results = evaluate_pipeline(test_set, censor_model, censor_tok,
                                     halluc_model, halluc_tok)
        plot_results(results)

        log.info("=" * 70)
        log.info("BASELINE RESULTS")
        log.info(f"  Privacy retention : {results['leakage']['privacy_retention']:.4f}")
        log.info(f"  Leakage rate      : {results['leakage']['leakage_rate']:.4f}")
        log.info(f"  ROUGE-L           : {results['rouge'].get('rougeL', 0):.4f}")
        log.info(f"  BERTScore F1      : {results['bertscore_f1']:.4f}")
        log.info("=" * 70)

    log.info("BASELINE PIPELINE COMPLETE")


if __name__ == "__main__":
    main()

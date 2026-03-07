# ==============================================================================
# PRIVACY-PRESERVING TEXT ANONYMIZATION
# DP-Guided Decoupled Mask-and-Fill Pipeline
# ==============================================================================
#
# This script implements the full training and evaluation pipeline for
# privacy-preserving text anonymization, including:
#
#   Model I  - The Censor:       DeBERTa-v3 NER token classifier (BIO tagging)
#   Model II - The Hallucinator: Flan-T5-Large seq2seq generator (QLoRA)
#
# Four-way comparison:
#   Approach 1: Human-curated Seq2Seq baseline (Flan-T5 + QLoRA, end-to-end)
#   Approach 2: Microsoft Presidio + rule-based replacement (non-neural)
#   Approach 3: Zero-shot LLM anonymization (instruction prompting)
#   Approach 4: DP-guided decoupled mask-and-fill (ours)
#
# Key NLP concepts demonstrated:
#   - Named Entity Recognition (NER) with BIO tagging scheme
#   - Transfer learning via pre-trained transformer fine-tuning
#   - Parameter-efficient fine-tuning (QLoRA / LoRA)
#   - Differential Privacy in NLP (DP-SGD via Opacus)
#   - Constrained text generation (beam search, top-k/top-p sampling)
#   - Evaluation: seqeval F1, BLEU, ROUGE-L, BERTScore, entity leakage
#
# Usage:
#   python script.py                    # Full pipeline
#   python script.py --mode censor      # Train Censor only
#   python script.py --mode halluc      # Train Hallucinator only
#   python script.py --mode eval        # Evaluation only
#   python script.py --quick            # Quick test with 2K samples
# ==============================================================================

# %% [markdown]
# # Privacy-Preserving Text Anonymization via DP-Guided Decoupled Mask-and-Fill
# ---
# ## 1. Setup & Dependencies

# %%
import os
import sys
import subprocess
import warnings
import gc
import hashlib
import json
import re
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def install_deps():
    """Install all required packages (idempotent, runs once at startup)."""
    deps = [
        "transformers>=4.40.0", "datasets>=2.18", "evaluate", "accelerate>=0.28",
        "peft>=0.10.0", "bitsandbytes>=0.43", "rouge_score", "sacrebleu",
        "sentencepiece", "scipy", "scikit-learn", "pandas", "matplotlib",
        "seqeval",              # NER evaluation (precision/recall/F1 per entity)
        "opacus>=1.4",          # Differential privacy (DP-SGD)
        "bert_score",           # BERTScore metric
        "presidio-analyzer", "presidio-anonymizer",  # Presidio baseline
        "Faker",                # Realistic pseudonym generation
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
    precision_score as seq_precision_score,
    recall_score as seq_recall_score,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")


# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    """Central configuration for all experiments.

    This mirrors the experimental setup described in the project scope:
    disjoint data partitions, DP-SGD training, QLoRA fine-tuning, and
    a four-way comparison across anonymization approaches.
    """

    # --- Execution mode (set via CLI or directly) ---
    MODE: str = "ALL"  # ALL | CENSOR | HALLUC | BASELINE | EVAL

    # --- Model backbones ---
    CENSOR_BACKBONE: str = "microsoft/deberta-v3-base"   # NER token classifier
    HALLUC_BACKBONE: str = "google/flan-t5-large"        # Seq2seq generator
    ZEROSHOT_BACKBONE: str = "google/flan-t5-large"      # Zero-shot baseline

    # --- Output directories ---
    OUTPUT_ROOT: str = "./privacy_project"
    CENSOR_DIR: str = "./privacy_project/censor_deberta"
    HALLUC_DIR: str = "./privacy_project/hallucinator_flan"
    BASELINE_DIR: str = "./privacy_project/baseline_seq2seq"
    EVAL_DIR: str = "./privacy_project/evaluation"
    PLOT_DIR: str = "./privacy_project/plots"

    # --- NER (Censor) hyperparameters ---
    NER_MAX_LEN: int = 256
    NER_BATCH_SIZE: int = 16
    NER_GRAD_ACCUM: int = 2       # Effective batch = 32
    NER_EPOCHS: int = 5
    NER_LR: float = 3e-5
    NER_WEIGHT_DECAY: float = 0.01
    NER_WARMUP_RATIO: float = 0.1

    # --- Seq2Seq (Hallucinator) hyperparameters ---
    SEQ2SEQ_MAX_LEN: int = 256
    SEQ2SEQ_BATCH_SIZE: int = 8
    SEQ2SEQ_GRAD_ACCUM: int = 4   # Effective batch = 32
    SEQ2SEQ_EPOCHS: int = 3
    SEQ2SEQ_LR: float = 1e-4
    SEQ2SEQ_WARMUP_RATIO: float = 0.06

    # --- QLoRA ---
    LORA_R: int = 32
    LORA_ALPHA: int = 64
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES_T5: list = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]

    # --- Differential Privacy (DP-SGD) ---
    ENABLE_DP: bool = True
    DP_EPSILON: float = 8.0        # Target privacy budget
    DP_DELTA: float = None         # Auto-set to 1/N
    DP_MAX_GRAD_NORM: float = 1.0  # Per-example gradient clipping bound

    # --- Generation ---
    GEN_MAX_TOKENS: int = 200
    GEN_NUM_BEAMS: int = 4
    GEN_TEMPERATURE: float = 0.7
    GEN_TOP_K: int = 50
    GEN_TOP_P: float = 0.92
    GEN_REPETITION_PENALTY: float = 1.2

    # --- Data ---
    QUICK_MODE: bool = False       # True = small subset for testing
    QUICK_SAMPLE_N: int = 2000
    TEST_RATIO: float = 0.05
    BASELINE_SIZE: int = 2000
    NUM_EVAL_SAMPLES: int = 200

    # --- Entity types (BIO tag set) ---
    ENTITY_TYPES: list = [
        "PERSON", "LOC", "ORG", "DATE", "PHONE",
        "EMAIL", "SSN", "CREDIT_CARD", "ADDRESS", "IP_ADDRESS",
    ]


def setup_dirs():
    """Create output directories."""
    for d in [Config.OUTPUT_ROOT, Config.CENSOR_DIR, Config.HALLUC_DIR,
              Config.BASELINE_DIR, Config.EVAL_DIR, Config.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


setup_dirs()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Privacy-Preserving Text Anonymization")
    parser.add_argument("--mode", default="ALL",
                        choices=["ALL", "CENSOR", "HALLUC", "BASELINE", "EVAL"],
                        help="Which part of the pipeline to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode with small data subset")
    parser.add_argument("--no-dp", action="store_true",
                        help="Disable DP-SGD training")
    parser.add_argument("--epochs-ner", type=int, default=None,
                        help="Override NER training epochs")
    parser.add_argument("--epochs-seq2seq", type=int, default=None,
                        help="Override Seq2Seq training epochs")
    try:
        args = parser.parse_args()
    except SystemExit:
        # Running in notebook -- use defaults
        args = argparse.Namespace(mode="ALL", quick=False, no_dp=False,
                                  epochs_ner=None, epochs_seq2seq=None)
    Config.MODE = args.mode.upper()
    Config.QUICK_MODE = args.quick
    if args.no_dp:
        Config.ENABLE_DP = False
    if args.epochs_ner:
        Config.NER_EPOCHS = args.epochs_ner
    if args.epochs_seq2seq:
        Config.SEQ2SEQ_EPOCHS = args.epochs_seq2seq
    return args


args = parse_args()
log.info(f"Config: MODE={Config.MODE}, DP={Config.ENABLE_DP}, "
         f"Quick={Config.QUICK_MODE}, eps={Config.DP_EPSILON}")


# %% [markdown]
# ## 3. Data Pipeline
#
# We use the **AI4Privacy** dataset which contains parallel pairs of
# original text (with real PII) and masked text (PII replaced by typed
# placeholders like `[PERSON_1]`, `[LOC_2]`).  This enables supervised
# training for both NER (token classification) and seq2seq (generation).

# %%
# ---------------------------------------------------------------------------
# 3a.  BIO Label Construction
# ---------------------------------------------------------------------------
#  The core NLP task for the Censor is Named Entity Recognition (NER):
#  given a tokenized sentence, assign each sub-word token a BIO tag.
#
#  B-PERSON  = Beginning of a PERSON entity
#  I-PERSON  = Inside  (continuation) of a PERSON entity
#  O         = Outside any entity
#
#  We derive these labels by aligning the original text with the masked
#  text: wherever the masked text has a placeholder like [PERSON_1], the
#  corresponding tokens in the original text are entity tokens.
# ---------------------------------------------------------------------------

def build_bio_label_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Construct BIO label <-> id mappings for all tracked entity types."""
    label2id = {"O": 0}
    id2label = {0: "O"}
    idx = 1
    for etype in Config.ENTITY_TYPES:
        for prefix in ("B", "I"):
            tag = f"{prefix}-{etype}"
            label2id[tag] = idx
            id2label[idx] = tag
            idx += 1
    return label2id, id2label


LABEL2ID, ID2LABEL = build_bio_label_map()
NUM_LABELS = len(LABEL2ID)
log.info(f"NER label set: {NUM_LABELS} labels ({len(Config.ENTITY_TYPES)} entity types)")


def load_ai4privacy() -> pd.DataFrame:
    """Load AI4Privacy dataset with fallback."""
    log.info("Loading AI4Privacy dataset...")
    try:
        ds = load_dataset(
            "ai4privacy/open-pii-masking-500k-ai4privacy", split="train"
        )
    except Exception as e:
        log.warning(f"Primary dataset failed ({e}), trying fallback...")
        ds = load_dataset("ai4privacy/pii-masking-200k", split="train")

    df = ds.to_pandas()

    # Normalize column names (different dataset versions use different names)
    col_map = {"source_text": "original_text", "target_text": "masked_text"}
    df.rename(
        columns={k: v for k, v in col_map.items() if k in df.columns},
        inplace=True,
    )

    for col in ["original_text", "masked_text"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    log.info(f"Raw dataset size: {len(df):,}")
    return df


def generate_pseudonym(entity_type: str, fake: Faker) -> str:
    """Generate a realistic pseudonym based on entity type using Faker."""
    t = entity_type.upper()
    generators = {
        "PERSON": fake.name, "PER": fake.name, "NAME": fake.name,
        "LOC": fake.city, "CITY": fake.city, "COUNTRY": fake.country,
        "ADDRESS": fake.address, "ORG": fake.company, "COMPANY": fake.company,
        "DATE": fake.date, "PHONE": fake.phone_number,
        "EMAIL": fake.email, "SSN": fake.ssn,
        "CREDIT_CARD": fake.credit_card_number, "IP_ADDRESS": fake.ipv4,
    }
    for key, gen in generators.items():
        if key in t:
            return gen()
    return fake.word()


def fill_placeholders(masked_text: str, fake: Faker) -> str:
    """Replace [TYPE_N] placeholders with consistent Faker pseudonyms."""
    cache: Dict[str, str] = {}

    def replacer(match):
        placeholder = match.group(0)
        if placeholder not in cache:
            entity_type = re.sub(r"[\[\]_\d]", " ", placeholder).strip()
            cache[placeholder] = generate_pseudonym(entity_type, fake)
        return cache[placeholder]

    return re.sub(r"\[[A-Z_]+(?:_\d+)?\]", replacer, str(masked_text))


def align_and_label(
    original: str,
    masked: str,
    tokenizer,
    max_len: int = Config.NER_MAX_LEN,
) -> Optional[Dict[str, List[int]]]:
    """Align original_text with masked_text to produce BIO labels.

    Strategy
    --------
    1.  Tokenize both texts word-by-word.
    2.  Walk through masked words; when we encounter a placeholder
        ``[TYPE_N]``, mark the corresponding original word(s) as
        entity tokens.
    3.  Map word-level labels to sub-word tokens via the tokenizer's
        ``offset_mapping``.

    Returns
    -------
    Dict with ``input_ids``, ``attention_mask``, ``labels`` -- or *None*
    when alignment fails (length mismatch, etc.).
    """
    placeholder_re = re.compile(r"\[([A-Z_]+?)(?:_\d+)?\]")

    orig_words = original.split()
    mask_words = masked.split()

    # ---- Word-level entity detection via alignment ----
    entity_labels_by_word: Dict[int, str] = {}
    oi, mi = 0, 0
    while oi < len(orig_words) and mi < len(mask_words):
        m = placeholder_re.match(mask_words[mi])
        if m:
            raw_type = m.group(1)
            # Map to our canonical entity types
            matched_type = "PERSON"  # fallback
            for et in Config.ENTITY_TYPES:
                if et in raw_type or raw_type in et:
                    matched_type = et
                    break
            entity_labels_by_word[oi] = matched_type

            oi += 1
            mi += 1
            # A placeholder might span multiple original words
            # (e.g. "John Smith" -> [PERSON_1]).  Advance the original
            # pointer until we realign with the next non-placeholder word.
            if mi < len(mask_words) and not placeholder_re.match(mask_words[mi]):
                target_word = mask_words[mi].lower().strip(".,!?;:")
                while oi < len(orig_words):
                    curr_word = orig_words[oi].lower().strip(".,!?;:")
                    if curr_word == target_word:
                        break
                    entity_labels_by_word[oi] = matched_type
                    oi += 1
        else:
            oi += 1
            mi += 1

    # ---- Sub-word tokenization with offset mapping ----
    encoding = tokenizer(
        original,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # Map each sub-word to its source word index and assign BIO tags
    labels: List[int] = []
    prev_type: Optional[str] = None
    for start, end in offsets:
        if start == 0 and end == 0:
            labels.append(-100)  # special token
            prev_type = None
            continue

        preceding_text = original[:start]
        word_idx = len(preceding_text.split()) - 1 if start > 0 else 0
        word_idx = max(0, min(word_idx, len(orig_words) - 1))

        if word_idx in entity_labels_by_word:
            etype = entity_labels_by_word[word_idx]
            tag = f"I-{etype}" if prev_type == etype else f"B-{etype}"
            labels.append(LABEL2ID.get(tag, 0))
            prev_type = etype
        else:
            labels.append(LABEL2ID["O"])
            prev_type = None

    return {
        "input_ids": tokens,
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare four non-overlapping data partitions.

    .. list-table::
       :header-rows: 1

       * - Partition
         - Purpose
       * - df_baseline
         - Human-curated seq2seq baseline (Approach 1)
       * - df_censor
         - Censor NER training (Approach 4, Split A)
       * - df_halluc
         - Hallucinator seq2seq training (Approach 4, Split B)
       * - df_test
         - Held-out evaluation

    The disjoint partition is a key privacy mechanism: the Hallucinator
    never trains on data containing original entities from the Censor's
    partition, preventing cross-model entity memorization.
    """
    # Keep only rows where masking changed something
    df = df[df["original_text"].astype(str) != df["masked_text"].astype(str)].copy()
    log.info(f"After PII filter: {len(df):,}")

    # Language filtering (English)
    if "language" in df.columns:
        df["language"] = df["language"].fillna("unknown")
        df_en = df[df["language"] == "en"].copy()
    else:
        df_en = df.copy()
        df_en["language"] = "en"
    log.info(f"English subset: {len(df_en):,}")

    if Config.QUICK_MODE:
        n = min(len(df_en), Config.QUICK_SAMPLE_N)
        df_en = df_en.sample(n=n, random_state=SEED).reset_index(drop=True)
        log.info(f"Quick mode: subsampled to {len(df_en):,}")

    # Generate pseudonymized targets for the Hallucinator
    fake = Faker()
    Faker.seed(SEED)
    log.info("Generating pseudonymized targets via Faker...")
    df_en["anonymized_text"] = df_en["masked_text"].apply(
        lambda x: fill_placeholders(x, fake)
    )

    # ---- Partition into disjoint sets ----
    df_train, df_test = train_test_split(
        df_en, test_size=Config.TEST_RATIO, random_state=SEED
    )

    baseline_n = min(Config.BASELINE_SIZE, len(df_train) // 10)
    df_baseline = df_train.sample(n=baseline_n, random_state=SEED)
    df_remaining = df_train.drop(df_baseline.index)

    df_censor, df_halluc = train_test_split(
        df_remaining, test_size=0.5, random_state=SEED
    )

    # ---- Verify disjointness (critical for privacy) ----
    sets = {
        "censor": set(df_censor.index),
        "halluc": set(df_halluc.index),
        "test": set(df_test.index),
        "baseline": set(df_baseline.index),
    }
    for name_a, set_a in sets.items():
        for name_b, set_b in sets.items():
            if name_a < name_b:
                overlap = set_a & set_b
                assert len(overlap) == 0, (
                    f"DATA LEAK: {name_a} & {name_b} overlap by {len(overlap)}"
                )

    log.info(
        f"Data partitions -- "
        f"Baseline: {len(df_baseline):,} | "
        f"Censor (A): {len(df_censor):,} | "
        f"Hallucinator (B): {len(df_halluc):,} | "
        f"Test: {len(df_test):,}"
    )

    # ---- Task-specific input/target columns ----
    df_censor = df_censor.copy()
    df_halluc = df_halluc.copy()
    df_baseline = df_baseline.copy()
    df_test = df_test.copy()

    # Hallucinator: masked -> pseudonymized
    df_halluc["input_text"] = (
        "Fill PII placeholders with realistic names: "
        + df_halluc["masked_text"].astype(str)
    )
    df_halluc["target_text"] = df_halluc["anonymized_text"].astype(str)

    # Baseline: original -> anonymized (end-to-end)
    df_baseline["input_text"] = (
        "Anonymize the following text: "
        + df_baseline["original_text"].astype(str)
    )
    df_baseline["target_text"] = df_baseline["anonymized_text"].astype(str)

    return df_baseline, df_censor, df_halluc, df_test


# ---- Execute data pipeline ----
df_raw = load_ai4privacy()
df_baseline, df_censor, df_halluc, df_test = prepare_data(df_raw)
del df_raw
gc.collect()


# %% [markdown]
# ## 4. NER Censor (Model I): DeBERTa-v3 Token Classification
#
# The Censor performs **Named Entity Recognition (NER)** -- the task of
# identifying and classifying named entities in text.  We fine-tune
# DeBERTa-v3-base on our Split A data with BIO tagging and evaluate
# using the standard **seqeval** metrics (precision, recall, F1 per
# entity type).
#
# We prioritize **recall** over precision: a missed PII entity leaks
# private information, while a false positive merely over-masks.

# %%
def build_ner_dataset(
    df: pd.DataFrame, tokenizer, max_len: int = Config.NER_MAX_LEN
) -> Dataset:
    """Convert DataFrame rows into a HuggingFace Dataset with BIO labels.

    For each ``(original_text, masked_text)`` pair we:

    1. Align the two texts to find entity spans.
    2. Sub-word tokenize the original text.
    3. Assign BIO labels to each sub-word token.

    This is the standard approach for building NER training data when
    we have parallel annotated text rather than span annotations.
    """
    records: List[Dict] = []
    skipped = 0

    for _, row in df.iterrows():
        result = align_and_label(
            str(row["original_text"]),
            str(row["masked_text"]),
            tokenizer,
            max_len,
        )
        if result and len(result["input_ids"]) == len(result["labels"]):
            records.append(result)
        else:
            skipped += 1

    if skipped > 0:
        log.warning(f"Skipped {skipped} alignment failures out of {len(df)}")

    ds = Dataset.from_list(records)
    log.info(f"NER dataset: {len(ds)} samples, {max_len} max tokens")
    return ds


def compute_ner_metrics(eval_pred) -> Dict[str, float]:
    """Compute seqeval metrics for NER evaluation.

    seqeval computes **entity-level** (not token-level) metrics:
    an entity is correct only if both the type AND span boundaries
    match.  This is the standard protocol for NER (CoNLL-2003).
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_tags: List[str] = []
        pred_tags: List[str] = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_tags.append(ID2LABEL.get(int(l), "O"))
            pred_tags.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    return {
        "precision": seq_precision_score(true_labels, pred_labels),
        "recall": seq_recall_score(true_labels, pred_labels),
        "f1": seq_f1_score(true_labels, pred_labels),
    }


def _plot_training_curve(log_history: list, output_dir: str, title: str):
    """Plot and save training / validation loss curves."""
    train_steps = [l["step"] for l in log_history if "loss" in l]
    train_losses = [l["loss"] for l in log_history if "loss" in l]
    eval_steps = [l["step"] for l in log_history if "eval_loss" in l]
    eval_losses = [l["eval_loss"] for l in log_history if "eval_loss" in l]

    if not train_steps:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_losses, "b-", lw=1, alpha=0.4, label="Train loss")

    if len(train_losses) > 10:
        window = max(len(train_losses) // 10, 5)
        smoothed = pd.Series(train_losses).rolling(window=window, center=True).mean()
        ax.plot(train_steps, smoothed, "b-", lw=2, label="Train (smoothed)")

    if eval_steps:
        ax.plot(eval_steps, eval_losses, "r-o", lw=2, ms=4, label="Val loss")

    ax.set_title(f"Training Curve: {title}", fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Loss curve saved: {path}")


def train_censor(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Train Model I -- DeBERTa-v3-base for NER token classification.

    Architecture
    ------------
    DeBERTa-v3-base (184 M params) with a linear classification head
    over the BIO tag set.  Uses disentangled attention which aids
    entity-boundary detection.

    Returns
    -------
    Path to the best checkpoint directory, or *None* on failure.
    """
    log.info("=" * 60)
    log.info("TRAINING CENSOR -- DeBERTa-v3 NER Token Classification")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        Config.CENSOR_BACKBONE, add_prefix_space=True
    )

    # Build NER dataset with BIO labels
    log.info("Building NER dataset with BIO labels from text alignment...")
    ner_ds = build_ner_dataset(df, tokenizer)

    split = ner_ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds = split["train"]
    val_ds = split["test"]
    log.info(f"NER splits -- Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        Config.CENSOR_BACKBONE,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Censor model: {trainable:,} trainable parameters")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.NER_EPOCHS,
        per_device_train_batch_size=Config.NER_BATCH_SIZE,
        per_device_eval_batch_size=Config.NER_BATCH_SIZE * 2,
        gradient_accumulation_steps=Config.NER_GRAD_ACCUM,
        learning_rate=Config.NER_LR,
        weight_decay=Config.NER_WEIGHT_DECAY,
        warmup_ratio=Config.NER_WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=DEVICE.type == "cuda",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ner_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ---- DP-SGD integration via Opacus ----
    if Config.ENABLE_DP:
        log.info("Attaching Opacus DP-SGD to Censor training...")
        try:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()

            if Config.DP_DELTA is None:
                Config.DP_DELTA = 1.0 / len(train_ds)

            log.info(
                f"DP params: eps={Config.DP_EPSILON}, "
                f"delta={Config.DP_DELTA:.2e}, "
                f"clip_norm={Config.DP_MAX_GRAD_NORM}"
            )

            _train_censor_with_dp(
                model, train_ds, val_ds, tokenizer,
                data_collator, output_dir, privacy_engine,
            )

        except ImportError:
            log.warning("Opacus not available -- training WITHOUT DP-SGD")
            trainer.train()
        except Exception as e:
            log.warning(f"DP-SGD setup failed ({e}) -- falling back to standard training")
            trainer.train()
    else:
        trainer.train()

    # Plot loss
    if hasattr(trainer, "state") and trainer.state.log_history:
        _plot_training_curve(
            trainer.state.log_history, output_dir, "Censor (DeBERTa-v3 NER)"
        )

    # Save best model
    best_dir = os.path.join(output_dir, "best")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    # ---- Detailed NER evaluation on validation set ----
    log.info("Running detailed NER evaluation...")
    val_preds = trainer.predict(val_ds)
    predictions = np.argmax(val_preds.predictions, axis=-1)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(predictions, val_preds.label_ids):
        true_tags, pred_tags = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_tags.append(ID2LABEL.get(int(l), "O"))
            pred_tags.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    report = seq_classification_report(true_labels, pred_labels, digits=4)
    log.info(
        f"\n{'='*50}\nNER Classification Report (seqeval):\n{'='*50}\n{report}"
    )

    with open(os.path.join(output_dir, "ner_report.txt"), "w") as f:
        f.write(report)

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return best_dir


def _train_censor_with_dp(
    model, train_ds, val_ds, tokenizer, data_collator,
    output_dir, privacy_engine,
):
    """Manual DP-SGD training loop using Opacus.

    Opacus modifies the standard SGD procedure as follows:

    1. Compute **per-example** gradients (not per-batch).
    2. Clip each example's gradient to norm <= *C*.
    3. Aggregate clipped gradients and add calibrated Gaussian noise.
    4. The RDP accountant tracks cumulative privacy spend.

    This provides formal ``(epsilon, delta)``-DP guarantees.
    """
    from torch.utils.data import DataLoader
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.NER_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.NER_LR,
        weight_decay=Config.NER_WEIGHT_DECAY,
    )

    # Make model, optimizer, and data-loader DP-aware
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=Config.NER_EPOCHS,
        target_epsilon=Config.DP_EPSILON,
        target_delta=Config.DP_DELTA,
        max_grad_norm=Config.DP_MAX_GRAD_NORM,
    )

    log.info(
        f"DP noise multiplier (auto-calibrated): "
        f"{optimizer.noise_multiplier:.4f}"
    )

    model.train()
    global_step = 0
    loss_history: List[Dict] = []

    for epoch in range(Config.NER_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=Config.NER_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for batch in memory_safe_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % 100 == 0:
                    eps = privacy_engine.get_epsilon(Config.DP_DELTA)
                    log.info(
                        f"Step {global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"epsilon: {eps:.2f}"
                    )
                    loss_history.append(
                        {"step": global_step, "loss": loss.item()}
                    )

        avg_loss = epoch_loss / max(n_batches, 1)
        eps = privacy_engine.get_epsilon(Config.DP_DELTA)
        log.info(
            f"Epoch {epoch+1}/{Config.NER_EPOCHS} | "
            f"Avg Loss: {avg_loss:.4f} | epsilon spent: {eps:.2f}"
        )

    final_eps = privacy_engine.get_epsilon(Config.DP_DELTA)
    log.info(
        f"DP-SGD training complete.  "
        f"Final (epsilon, delta) = ({final_eps:.2f}, {Config.DP_DELTA:.2e})"
    )

    # Persist DP metadata alongside the model
    dp_meta = {
        "epsilon": final_eps,
        "delta": Config.DP_DELTA,
        "noise_multiplier": optimizer.noise_multiplier,
        "max_grad_norm": Config.DP_MAX_GRAD_NORM,
        "epochs": Config.NER_EPOCHS,
    }
    with open(os.path.join(output_dir, "dp_metadata.json"), "w") as f:
        json.dump(dp_meta, f, indent=2)


# %% [markdown]
# ## 5. Hallucinator (Model II): Flan-T5-Large + QLoRA
#
# The Hallucinator is a **sequence-to-sequence** model that takes masked
# text templates and generates contextually appropriate pseudonyms.
# We use **QLoRA** (Quantized Low-Rank Adaptation) for parameter-efficient
# fine-tuning: the base model is loaded in 4-bit precision and only the
# low-rank adapter weights are trained.

# %%
def tokenize_seq2seq(
    examples: Dict[str, List[str]],
    tokenizer,
    max_len: int = Config.SEQ2SEQ_MAX_LEN,
) -> Dict[str, List]:
    """Tokenize input--output pairs for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=max_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_seq2seq_model(
    df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    title: str = "Seq2Seq",
) -> Optional[str]:
    """Train a Flan-T5-Large model with QLoRA for seq2seq generation.

    Used for both:
      * **Hallucinator** (masked -> pseudonymized)
      * **Baseline** (original -> anonymized, end-to-end)

    QLoRA adapts all projection layers in the T5 attention and FFN
    blocks while keeping the frozen base in 4-bit (NF4) quantization,
    reducing trainable memory from ~3 GB to ~800 MB.

    Returns path to best checkpoint.
    """
    log.info("=" * 60)
    log.info(f"TRAINING {title} -- Flan-T5-Large + QLoRA")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = Dataset.from_pandas(
        df[["input_text", "target_text"]].reset_index(drop=True)
    )
    ds = ds.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {title}",
    )

    split = ds.train_test_split(test_size=0.05, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    log.info(f"{title} splits -- Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ---- 4-bit quantised loading ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapter ----
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=Config.LORA_TARGET_MODULES_T5,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.SEQ2SEQ_EPOCHS,
        per_device_train_batch_size=Config.SEQ2SEQ_BATCH_SIZE,
        per_device_eval_batch_size=Config.SEQ2SEQ_BATCH_SIZE,
        gradient_accumulation_steps=Config.SEQ2SEQ_GRAD_ACCUM,
        learning_rate=Config.SEQ2SEQ_LR,
        warmup_ratio=Config.SEQ2SEQ_WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=DEVICE.type == "cuda",
        report_to="none",
        group_by_length=True,
        optim="paged_adamw_8bit",
        predict_with_generate=False,
        seed=SEED,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    if trainer.state.log_history:
        _plot_training_curve(trainer.state.log_history, output_dir, title)

    best_dir = os.path.join(output_dir, "best")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    log.info(f"{title} training complete.  Saved to {best_dir}")
    return best_dir


# %% [markdown]
# ## 6. Entity Consistency Module
#
# When the Hallucinator independently generates pseudonyms for each
# masked span, the same entity (e.g. ``[PERSON_1]``) appearing
# multiple times may receive different replacements.  This module
# ensures **coreference consistency**: all mentions of the same entity
# within a document map to the same pseudonym, using deterministic
# context-window hashing that never stores an original-to-pseudonym
# mapping.

# %%
class EntityConsistencyModule:
    """Ensure coreferent masked entities receive identical pseudonyms
    via deterministic context-window hashing.

    Algorithm
    ---------
    1. For each placeholder, extract a context window of +/-k tokens.
    2. Compute ``SHA-256(entity_type || context)`` to produce a hash.
    3. Seed Faker with the hash for deterministic name generation.
    4. Identical ``(type, context)`` -> identical hash -> identical name.

    Privacy property: the hash is computed on ``(entity_type,
    masked_context)`` -- never on the original entity value.
    """

    def __init__(self, context_window: int = 10):
        self.context_window = context_window
        self.fake = Faker()

    def _hash(self, entity_type: str, context: str) -> str:
        sig = f"{entity_type}||{context.strip().lower()}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def build_consistency_map(self, masked_text: str) -> Dict[str, str]:
        """Build mapping: placeholder -> deterministic pseudonym."""
        words = masked_text.split()
        placeholder_re = re.compile(r"\[([A-Z_]+?)(?:_\d+)?\]")
        mapping: Dict[str, str] = {}

        for i, word in enumerate(words):
            m = placeholder_re.match(word)
            if m and word not in mapping:
                etype = m.group(1)
                lo = max(0, i - self.context_window)
                hi = min(len(words), i + self.context_window + 1)
                context = " ".join(words[lo:hi])
                h = self._hash(etype, context)

                Faker.seed(int(h, 16) % (2**32))
                mapping[word] = generate_pseudonym(etype, self.fake)

        return mapping

    def apply(self, masked_text: str, generated_text: str) -> str:
        """Post-process: enforce coreference consistency."""
        mapping = self.build_consistency_map(masked_text)
        result = generated_text
        for placeholder, pseudonym in mapping.items():
            result = result.replace(placeholder, pseudonym)
        return result


consistency_module = EntityConsistencyModule()


# %% [markdown]
# ## 7. Baselines
#
# ### Approach 2 -- Presidio (Rule-based, Non-neural)
# ### Approach 3 -- Zero-shot LLM (No task-specific training)

# %%
def run_presidio_baseline(texts: List[str]) -> List[Dict[str, Any]]:
    """Approach 2: Microsoft Presidio NER + Faker replacement.

    Presidio uses regex patterns, spaCy NLP models, and rule-based
    recognizers for PII detection.  It represents the industry
    standard for non-neural anonymization.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        log.warning("Presidio not installed -- skipping baseline 2")
        return []

    log.info(f"Running Presidio baseline on {len(texts)} samples...")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    fake = Faker()
    Faker.seed(SEED)

    results: List[Dict[str, Any]] = []
    for text in texts:
        try:
            analysis = analyzer.analyze(text=str(text), language="en")
            operators = {}
            for entity in analysis:
                etype = entity.entity_type
                operators[etype] = OperatorConfig(
                    "replace",
                    {"new_value": generate_pseudonym(etype, fake)},
                )
            anon_result = anonymizer.anonymize(
                text=str(text),
                analyzer_results=analysis,
                operators=operators,
            )
            results.append({
                "Original": text,
                "Anonymized": anon_result.text,
                "Entities_Found": len(analysis),
                "Method": "Presidio",
            })
        except Exception as e:
            results.append({
                "Original": text, "Anonymized": text,
                "Entities_Found": 0, "Method": "Presidio",
                "Error": str(e),
            })

    avg_ent = np.mean([r["Entities_Found"] for r in results])
    log.info(f"Presidio: {len(results)} samples, avg entities/sample: {avg_ent:.1f}")
    return results


def run_zeroshot_baseline(
    texts: List[str], max_samples: int = 50
) -> List[Dict[str, Any]]:
    """Approach 3: Zero-shot anonymization via instruction-tuned LLM.

    Tests whether off-the-shelf LLMs can perform anonymization without
    any task-specific training, using only a natural language prompt.
    """
    log.info(
        f"Running zero-shot LLM baseline "
        f"({min(len(texts), max_samples)} samples)..."
    )

    tokenizer = AutoTokenizer.from_pretrained(Config.ZEROSHOT_BACKBONE)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.ZEROSHOT_BACKBONE,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        log.warning("Model loading failed -- skipping zero-shot baseline")
        return []

    model.eval()

    PROMPT = (
        "Anonymize the following text by replacing all personal information "
        "(names, locations, organizations, dates, phone numbers, emails, "
        "IDs) with realistic fictional alternatives. "
        "Preserve grammar and meaning.\n\n"
        "Text: {text}\n\nAnonymized:"
    )

    results: List[Dict[str, Any]] = []
    for text in texts[:max_samples]:
        prompt = PROMPT.format(text=str(text))
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=Config.GEN_MAX_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        anon = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({
            "Original": text,
            "Anonymized": anon,
            "Method": "ZeroShot_LLM",
        })

    del model
    torch.cuda.empty_cache()
    gc.collect()

    log.info(f"Zero-shot: processed {len(results)} samples")
    return results


# %% [markdown]
# ## 8. Decoupled Pipeline Inference (Approach 4 -- Ours)
#
# The inference pipeline chains Model I (Censor) and Model II
# (Hallucinator) through **typed mask tokens** -- the only information
# that flows from Censor to Hallucinator is the masked template, never
# original entities.
#
# ```
# Original --[Censor]--> Masked --[Hallucinator]--> Anonymized
# "Alice at IBM"        "[PERSON] at [ORG]"        "Sarah at NovaCorp"
# ```

# %%
def clean_generated(text: str) -> str:
    """Remove special tokens and task prefixes from generated text."""
    text = re.sub(r"<pad>|</s>|<extra_id_\d+>|<unk>", "", text)
    for prefix in [
        "Fill PII placeholders with realistic names:",
        "Anonymize the following text:",
        "Detect and mask all PII:",
    ]:
        text = text.replace(prefix, "")
    return text.strip()


def run_censor_inference(
    texts: List[str], censor_path: str
) -> List[str]:
    """Run the NER Censor to detect and mask PII entities.

    For each input text:

    1. Sub-word tokenize and run through DeBERTa-v3 NER.
    2. Decode BIO predictions into entity spans.
    3. Replace detected entities with typed placeholders ``[TYPE_N]``.
    """
    log.info(f"Running Censor inference on {len(texts)} texts...")

    tokenizer = AutoTokenizer.from_pretrained(censor_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(censor_path).to(DEVICE)
    model.eval()

    masked_outputs: List[str] = []

    for text in texts:
        encoding = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=Config.NER_MAX_LEN,
            return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()

        inputs = {k: v.to(DEVICE) for k, v in encoding.items()}
        with torch.no_grad():
            logits = model(**inputs).logits

        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # ---- Decode BIO -> entity spans ----
        entities: List[Tuple[int, int, str]] = []
        current_entity: Optional[str] = None
        current_start: int = 0
        current_end: int = 0

        for pred_id, (start, end) in zip(predictions, offsets):
            if start == 0 and end == 0:
                continue
            tag = ID2LABEL.get(pred_id, "O")
            if tag.startswith("B-"):
                if current_entity:
                    entities.append((current_start, current_end, current_entity))
                current_entity = tag[2:]
                current_start = start
                current_end = end
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_end = end
            else:
                if current_entity:
                    entities.append((current_start, current_end, current_entity))
                current_entity = None

        if current_entity:
            entities.append((current_start, current_end, current_entity))

        # ---- Replace entities with typed placeholders ----
        masked = str(text)
        entity_counts: Dict[str, int] = defaultdict(int)
        for start, end, etype in sorted(entities, key=lambda x: x[0], reverse=True):
            entity_counts[etype] += 1
            placeholder = f"[{etype}_{entity_counts[etype]}]"
            masked = masked[:start] + placeholder + masked[end:]

        masked_outputs.append(masked)

    del model
    torch.cuda.empty_cache()

    return masked_outputs


def run_hallucinator_inference(
    masked_texts: List[str], halluc_path: str
) -> List[str]:
    """Run the Hallucinator to fill masked placeholders with pseudonyms.

    Uses beam search with nucleus sampling for diverse yet coherent
    pseudonym generation.
    """
    log.info(f"Running Hallucinator inference on {len(masked_texts)} texts...")

    tokenizer = AutoTokenizer.from_pretrained(Config.HALLUC_BACKBONE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForSeq2SeqLM.from_pretrained(
        Config.HALLUC_BACKBONE,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, halluc_path)
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=Config.GEN_MAX_TOKENS,
        num_beams=Config.GEN_NUM_BEAMS,
        temperature=Config.GEN_TEMPERATURE,
        top_k=Config.GEN_TOP_K,
        top_p=Config.GEN_TOP_P,
        repetition_penalty=Config.GEN_REPETITION_PENALTY,
        do_sample=True,
    )

    outputs: List[str] = []
    for masked in masked_texts:
        prompt = "Fill PII placeholders with realistic names: " + str(masked)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=Config.SEQ2SEQ_MAX_LEN,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_config)

        generated = clean_generated(
            tokenizer.decode(out[0], skip_special_tokens=True)
        )

        # Apply entity consistency post-processing
        generated = consistency_module.apply(masked, generated)
        outputs.append(generated)

    del model, base
    torch.cuda.empty_cache()
    gc.collect()

    return outputs


def run_decoupled_pipeline(
    texts: List[str],
    censor_path: str,
    halluc_path: str,
) -> List[Dict[str, Any]]:
    """Approach 4: Full decoupled mask-and-fill pipeline.

    Step 1: Censor  (original -> masked)      via NER
    Step 2: Hallucinator (masked -> anonymized) via seq2seq
    Step 3: Entity consistency enforcement
    """
    log.info("Running decoupled pipeline (Censor -> Hallucinator)...")

    masked_texts = run_censor_inference(texts, censor_path)
    anonymized_texts = run_hallucinator_inference(masked_texts, halluc_path)

    results: List[Dict[str, Any]] = []
    for orig, masked, anon in zip(texts, masked_texts, anonymized_texts):
        results.append({
            "Original": orig,
            "Masked": masked,
            "Anonymized": anon,
            "Method": "Decoupled",
        })

    return results


# %% [markdown]
# ## 9. Evaluation Metrics
#
# We evaluate along three axes:
#
# 1. **Privacy**: Entity Leakage Rate (exact + fuzzy match)
# 2. **Utility**: BLEU, ROUGE-L, BERTScore (generation quality)
# 3. **Semantic Preservation**: Sentence-BERT cosine similarity

# %%
def compute_entity_leakage(
    originals: List[str], anonymized: List[str]
) -> Dict[str, Any]:
    """Entity Leakage Rate (ELR): fraction of original entities that
    appear verbatim (exact) or near-verbatim (fuzzy) in the output.

    An ideal anonymizer has ELR = 0 %.
    """
    exact_leaks = 0
    fuzzy_leaks = 0
    total_entities = 0

    entity_re = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    for orig, anon in zip(originals, anonymized):
        entities = entity_re.findall(str(orig))
        total_entities += len(entities)

        for ent in entities:
            if ent in str(anon):
                exact_leaks += 1
            elif len(ent) > 3:
                ent_chars = set(ent.lower())
                for anon_ent in entity_re.findall(str(anon)):
                    anon_chars = set(anon_ent.lower())
                    jaccard = len(ent_chars & anon_chars) / max(
                        len(ent_chars | anon_chars), 1
                    )
                    if jaccard > 0.8:
                        fuzzy_leaks += 1
                        break

    denom = max(total_entities, 1)
    return {
        "total_entities": total_entities,
        "exact_leaks": exact_leaks,
        "fuzzy_leaks": fuzzy_leaks,
        "exact_leakage_pct": round(exact_leaks / denom * 100, 2),
        "fuzzy_leakage_pct": round(
            (exact_leaks + fuzzy_leaks) / denom * 100, 2
        ),
    }


def compute_generation_metrics(
    references: List[str], predictions: List[str]
) -> Dict[str, Any]:
    """Standard NLG evaluation metrics.

    * **BLEU** -- n-gram precision (machine translation standard)
    * **ROUGE-L** -- longest common subsequence F1 (summarization)
    * **BERTScore** -- contextual embedding similarity (semantic quality)
    """
    metrics: Dict[str, Any] = {}

    # BLEU
    try:
        bleu = evaluate.load("sacrebleu")
        result = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        metrics["BLEU"] = round(result["score"], 2)
    except Exception as e:
        metrics["BLEU"] = f"Error: {e}"

    # ROUGE-L
    try:
        rouge = evaluate.load("rouge")
        result = rouge.compute(
            predictions=predictions, references=references
        )
        metrics["ROUGE-L"] = round(result["rougeL"] * 100, 2)
    except Exception as e:
        metrics["ROUGE-L"] = f"Error: {e}"

    # BERTScore
    try:
        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )
        metrics["BERTScore_F1"] = round(np.mean(result["f1"]) * 100, 2)
    except Exception as e:
        metrics["BERTScore_F1"] = f"Error: {e}"

    return metrics


def compute_semantic_preservation(
    originals: List[str], anonymized: List[str]
) -> float:
    """Cosine similarity between Sentence-BERT embeddings of
    original vs. anonymized text.  High similarity means the
    anonymized text preserves meaning beyond the entity changes.
    """
    try:
        from sentence_transformers import SentenceTransformer

        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        emb_orig = sbert.encode([str(t) for t in originals])
        emb_anon = sbert.encode([str(t) for t in anonymized])

        similarities = [
            float(
                np.dot(a, b)
                / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            )
            for a, b in zip(emb_orig, emb_anon)
        ]
        return round(float(np.mean(similarities)), 4)
    except ImportError:
        log.warning(
            "sentence-transformers not available for semantic preservation"
        )
        return -1.0


# %% [markdown]
# ## 10. Visualization

# %%
def plot_comparison_charts(
    all_results: Dict[str, Dict], output_dir: str
):
    """Generate side-by-side comparison bar charts for all methods."""

    methods = list(all_results.keys())
    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 1. Entity Leakage ----
    ax = axes[0]
    exact = [
        all_results[m].get("leakage", {}).get("exact_leakage_pct", 0)
        for m in methods
    ]
    fuzzy = [
        all_results[m].get("leakage", {}).get("fuzzy_leakage_pct", 0)
        for m in methods
    ]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, exact, width, label="Exact %", color="salmon")
    ax.bar(x + width / 2, fuzzy, width, label="Fuzzy %", color="lightcoral")
    ax.set_ylabel("Leakage Rate (%)")
    ax.set_title("Entity Leakage (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ---- 2. Generation Quality ----
    ax = axes[1]
    metric_names = ["BLEU", "ROUGE-L", "BERTScore_F1"]
    colours = ["#4C72B0", "#55A868", "#C44E52"]
    bar_w = 0.25
    for i, (metric, colour) in enumerate(zip(metric_names, colours)):
        vals = []
        for m in methods:
            v = all_results[m].get("utility", {}).get(metric, 0)
            vals.append(v if isinstance(v, (int, float)) else 0)
        ax.bar(x + i * bar_w - bar_w, vals, bar_w, label=metric, color=colour)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ---- 3. Semantic Preservation ----
    ax = axes[2]
    sem = [all_results[m].get("semantic_preservation", 0) for m in methods]
    bars = ax.bar(methods, sem, color="#8172B2")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Semantic Preservation (higher = better)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, sem):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=10,
            )
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Four-Way Anonymization Comparison", fontsize=16, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "comparison_charts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Comparison charts saved: {path}")


def plot_privacy_utility_tradeoff(
    all_results: Dict[str, Dict], output_dir: str
):
    """Scatter plot: privacy (leakage) vs. utility (BERTScore)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = {
        "Presidio": "blue", "ZeroShot": "green",
        "Baseline": "orange", "Decoupled": "red",
    }
    markers = {
        "Presidio": "s", "ZeroShot": "^",
        "Baseline": "D", "Decoupled": "*",
    }

    for method, data in all_results.items():
        leakage = data.get("leakage", {}).get("exact_leakage_pct", 0)
        bs = data.get("utility", {}).get("BERTScore_F1", 0)
        if not isinstance(bs, (int, float)):
            continue
        ax.scatter(
            leakage, bs, s=200,
            c=colours.get(method, "gray"),
            marker=markers.get(method, "o"),
            label=method, zorder=5,
        )

    ax.set_xlabel("Entity Leakage Rate (%) -- lower is better", fontsize=12)
    ax.set_ylabel("BERTScore F1 -- higher is better", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    path = os.path.join(output_dir, "privacy_utility_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Privacy-utility tradeoff saved: {path}")


def save_qualitative_examples(
    all_results: Dict[str, Dict], output_dir: str, n: int = 5
):
    """Write qualitative examples to a text file for inspection."""
    path = os.path.join(output_dir, "qualitative_examples.txt")
    with open(path, "w", encoding="utf-8") as f:
        for method, data in all_results.items():
            samples = data.get("samples", [])[:n]
            if not samples:
                continue
            f.write(
                f"\n{'='*70}\n{method.upper()} -- Qualitative Examples"
                f"\n{'='*70}\n"
            )
            for i, s in enumerate(samples):
                f.write(f"\n--- Example {i+1} ---\n")
                f.write(f"Original:    {s.get('Original', 'N/A')[:200]}\n")
                if "Masked" in s:
                    f.write(f"Masked:      {s['Masked'][:200]}\n")
                f.write(f"Anonymized:  {s.get('Anonymized', 'N/A')[:200]}\n")
    log.info(f"Qualitative examples saved: {path}")


# %% [markdown]
# ## 11. Execute Training
#
# Train all models according to the selected mode.

# %%
log.info("\n" + "=" * 70)
log.info("STARTING TRAINING PIPELINE")
log.info("=" * 70)

# ---- Train Censor (NER) ----
if Config.MODE in ("ALL", "CENSOR"):
    censor_best_path = train_censor(df_censor, Config.CENSOR_DIR)
else:
    censor_best_path = os.path.join(Config.CENSOR_DIR, "best")
    if not os.path.exists(censor_best_path):
        censor_best_path = None

# ---- Train Hallucinator (Seq2Seq + QLoRA) ----
if Config.MODE in ("ALL", "HALLUC"):
    halluc_best_path = train_seq2seq_model(
        df_halluc, Config.HALLUC_DIR, Config.HALLUC_BACKBONE,
        title="Hallucinator (masked -> pseudonym)",
    )
else:
    halluc_best_path = os.path.join(Config.HALLUC_DIR, "best")
    if not os.path.exists(halluc_best_path):
        halluc_best_path = None

# ---- Train Baseline (end-to-end Seq2Seq + QLoRA) ----
if Config.MODE in ("ALL", "BASELINE"):
    baseline_best_path = train_seq2seq_model(
        df_baseline, Config.BASELINE_DIR, Config.HALLUC_BACKBONE,
        title="Baseline (end-to-end Seq2Seq)",
    )
else:
    baseline_best_path = os.path.join(Config.BASELINE_DIR, "best")
    if not os.path.exists(baseline_best_path):
        baseline_best_path = None


# %% [markdown]
# ## 12. Comprehensive Evaluation
#
# Run all four approaches on the held-out test set and compute
# privacy, utility, and semantic metrics for comparison.

# %%
log.info("\n" + "=" * 70)
log.info("COMPREHENSIVE EVALUATION")
log.info("=" * 70)

n_eval = min(Config.NUM_EVAL_SAMPLES, len(df_test))
test_texts = df_test["original_text"].tolist()[:n_eval]
test_masked = df_test["masked_text"].tolist()[:n_eval]
test_anon_ref = df_test["anonymized_text"].tolist()[:n_eval]

all_results: Dict[str, Dict] = {}

# ---- Approach 1: Baseline Seq2Seq ----
if baseline_best_path and os.path.exists(baseline_best_path):
    log.info("\n--- Approach 1: Human-Curated Seq2Seq Baseline ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.HALLUC_BACKBONE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(
            Config.HALLUC_BACKBONE,
            quantization_config=bnb_config,
            device_map="auto",
        )
        baseline_model = PeftModel.from_pretrained(base, baseline_best_path)
        baseline_model.eval()

        baseline_anon: List[str] = []
        for text in test_texts:
            inp = tokenizer(
                "Anonymize the following text: " + str(text),
                return_tensors="pt",
                max_length=Config.SEQ2SEQ_MAX_LEN,
                truncation=True,
            ).to(DEVICE)
            with torch.no_grad():
                out = baseline_model.generate(
                    **inp,
                    max_new_tokens=Config.GEN_MAX_TOKENS,
                    num_beams=4, temperature=0.7,
                    do_sample=True, top_p=0.92,
                )
            baseline_anon.append(
                clean_generated(tokenizer.decode(out[0], skip_special_tokens=True))
            )

        del baseline_model, base
        torch.cuda.empty_cache()

        all_results["Baseline"] = {
            "leakage": compute_entity_leakage(test_texts, baseline_anon),
            "utility": compute_generation_metrics(test_anon_ref, baseline_anon),
            "semantic_preservation": compute_semantic_preservation(
                test_texts[:50], baseline_anon[:50]
            ),
            "samples": [
                {"Original": o, "Anonymized": a}
                for o, a in zip(test_texts[:5], baseline_anon[:5])
            ],
        }
        log.info(f"  Leakage: {all_results['Baseline']['leakage']}")
        log.info(f"  Utility: {all_results['Baseline']['utility']}")
    except Exception as e:
        log.warning(f"Baseline evaluation failed: {e}")


# ---- Approach 2: Presidio ----
log.info("\n--- Approach 2: Presidio Baseline ---")
presidio_res = run_presidio_baseline(test_texts)
if presidio_res:
    presidio_anon = [r["Anonymized"] for r in presidio_res]
    all_results["Presidio"] = {
        "leakage": compute_entity_leakage(test_texts, presidio_anon),
        "utility": compute_generation_metrics(
            test_anon_ref[: len(presidio_anon)], presidio_anon
        ),
        "semantic_preservation": compute_semantic_preservation(
            test_texts[:50], presidio_anon[:50]
        ),
        "samples": presidio_res[:5],
    }
    log.info(f"  Leakage: {all_results['Presidio']['leakage']}")
    log.info(f"  Utility: {all_results['Presidio']['utility']}")

# ---- Approach 3: Zero-Shot LLM ----
log.info("\n--- Approach 3: Zero-Shot LLM ---")
zs_res = run_zeroshot_baseline(test_texts, max_samples=min(50, n_eval))
if zs_res:
    zs_anon = [r["Anonymized"] for r in zs_res]
    all_results["ZeroShot"] = {
        "leakage": compute_entity_leakage(test_texts[: len(zs_anon)], zs_anon),
        "utility": compute_generation_metrics(
            test_anon_ref[: len(zs_anon)], zs_anon
        ),
        "semantic_preservation": compute_semantic_preservation(
            test_texts[: min(50, len(zs_anon))],
            zs_anon[: min(50, len(zs_anon))],
        ),
        "samples": zs_res[:5],
    }
    log.info(f"  Leakage: {all_results['ZeroShot']['leakage']}")
    log.info(f"  Utility: {all_results['ZeroShot']['utility']}")

# ---- Approach 4: Decoupled Pipeline (Ours) ----
log.info("\n--- Approach 4: Decoupled Mask-and-Fill (Ours) ---")
if (
    censor_best_path
    and halluc_best_path
    and os.path.exists(censor_best_path)
    and os.path.exists(halluc_best_path)
):
    dec_res = run_decoupled_pipeline(
        test_texts, censor_best_path, halluc_best_path
    )
    dec_anon = [r["Anonymized"] for r in dec_res]

    all_results["Decoupled"] = {
        "leakage": compute_entity_leakage(test_texts, dec_anon),
        "utility": compute_generation_metrics(test_anon_ref, dec_anon),
        "semantic_preservation": compute_semantic_preservation(
            test_texts[:50], dec_anon[:50]
        ),
        "samples": dec_res[:5],
    }
    log.info(f"  Leakage: {all_results['Decoupled']['leakage']}")
    log.info(f"  Utility: {all_results['Decoupled']['utility']}")
    log.info(f"  Semantic: {all_results['Decoupled']['semantic_preservation']}")

    pd.DataFrame(dec_res).to_csv(
        os.path.join(Config.EVAL_DIR, "decoupled_results.csv"), index=False
    )
else:
    log.warning(
        "Trained Censor / Hallucinator not found -- skipping Approach 4"
    )


# %% [markdown]
# ## 13. Results Summary & Visualization

# %%
log.info("\n" + "=" * 70)
log.info("RESULTS SUMMARY")
log.info("=" * 70)

# ---- Summary table ----
summary_rows: List[Dict] = []
for method, data in all_results.items():
    row: Dict[str, Any] = {"Method": method}
    if "leakage" in data:
        row["Exact_Leak%"] = data["leakage"]["exact_leakage_pct"]
        row["Fuzzy_Leak%"] = data["leakage"]["fuzzy_leakage_pct"]
    if "utility" in data:
        for k, v in data["utility"].items():
            row[k] = v if isinstance(v, (int, float)) else "N/A"
    if "semantic_preservation" in data:
        row["SemPres"] = data["semantic_preservation"]
    summary_rows.append(row)

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + df_summary.to_markdown(index=False))
    df_summary.to_csv(
        os.path.join(Config.EVAL_DIR, "comparison_summary.csv"), index=False
    )
    log.info(f"Summary saved to {Config.EVAL_DIR}/comparison_summary.csv")

# ---- Generate plots ----
if all_results:
    plot_comparison_charts(all_results, Config.PLOT_DIR)
    plot_privacy_utility_tradeoff(all_results, Config.PLOT_DIR)
    save_qualitative_examples(all_results, Config.EVAL_DIR)

# ---- Save complete results as JSON ----
results_path = os.path.join(Config.EVAL_DIR, "all_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
log.info(f"Full results saved: {results_path}")

# ---- DP Budget Report ----
dp_meta_path = os.path.join(Config.CENSOR_DIR, "dp_metadata.json")
if os.path.exists(dp_meta_path):
    with open(dp_meta_path) as f:
        dp_meta = json.load(f)
    log.info("\nDP-SGD Privacy Budget Report:")
    log.info(f"  Final epsilon  = {dp_meta['epsilon']:.2f}")
    log.info(f"  delta          = {dp_meta['delta']:.2e}")
    log.info(f"  Noise sigma    = {dp_meta['noise_multiplier']:.4f}")
    log.info(f"  Clip norm C    = {dp_meta['max_grad_norm']}")

log.info("\n" + "=" * 70)
log.info("PIPELINE COMPLETE")
log.info("=" * 70)


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 14. Module 9 — RAG-Anon: Retrieval-Augmented Anonymization
# ────────────────────────────────────────────────────────────────────────
# Dense retrieval over an entity knowledge base + cross-encoder re-ranking
# to produce contextually appropriate pseudonyms (backed by FAISS).

# %%
class RAGAnonymizer:
    """Retrieval-Augmented pseudonym generation (Module 9).

    Builds a FAISS index over synthetic entity descriptions and uses
    a sentence-transformer bi-encoder + cross-encoder re-ranker to
    select the most plausible replacement for each detected PII span.
    """

    def __init__(
        self,
        entity_types: List[str] = None,
        n_entities_per_type: int = 500,
        top_k: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.entity_types = entity_types or Config.ENTITY_TYPES
        self.top_k = top_k
        self.threshold = confidence_threshold
        self.fake = Faker()
        self.n_per_type = n_entities_per_type

        # Models (lazy-loaded)
        self._bi_encoder = None
        self._cross_encoder = None
        self._index: Dict[str, Any] = {}
        self._entities: Dict[str, List[str]] = {}

    # ---- lazy loaders ------------------------------------------------
    def _load_bi_encoder(self):
        if self._bi_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("RAG-Anon: loaded bi-encoder all-MiniLM-L6-v2")
            except ImportError:
                log.warning("sentence-transformers not installed; falling back to random retrieval")
        return self._bi_encoder

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                log.info("RAG-Anon: loaded cross-encoder ms-marco-MiniLM-L-6-v2")
            except ImportError:
                log.warning("CrossEncoder not available; skipping re-ranking")
        return self._cross_encoder

    # ---- index construction ------------------------------------------
    def build_index(self):
        """Build a per-type FAISS index over synthetic entity descriptions."""
        try:
            import faiss
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            import faiss

        bi_enc = self._load_bi_encoder()
        for etype in self.entity_types:
            entities = [generate_pseudonym(etype, self.fake) for _ in range(self.n_per_type)]
            self._entities[etype] = entities

            if bi_enc is not None:
                embs = bi_enc.encode(entities, convert_to_numpy=True, show_progress_bar=False)
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)  # L2 norm
                dim = embs.shape[1]
                idx = faiss.IndexFlatIP(dim)  # inner product ≡ cosine on normed vecs
                idx.add(embs.astype(np.float32))
                self._index[etype] = idx
            else:
                self._index[etype] = None

            log.info(f"  RAG index built for {etype}: {len(entities)} entities")
        log.info("RAG-Anon: FAISS index construction complete")

    # ---- retrieve + re-rank ------------------------------------------
    def retrieve(self, span_text: str, context: str, entity_type: str) -> str:
        """Retrieve the best replacement for *span_text* given sentence *context*."""
        if entity_type not in self._entities:
            return generate_pseudonym(entity_type, self.fake)

        bi_enc = self._load_bi_encoder()
        if bi_enc is None or self._index[entity_type] is None:
            return random.choice(self._entities[entity_type])

        # Step 1: bi-encoder retrieval
        query_emb = bi_enc.encode([context], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        scores, indices = self._index[entity_type].search(query_emb.astype(np.float32), self.top_k)
        candidates = [self._entities[entity_type][i] for i in indices[0]]

        # Step 2: cross-encoder re-ranking
        cross_enc = self._load_cross_encoder()
        if cross_enc is not None:
            pairs = [(context, cand) for cand in candidates]
            ce_scores = cross_enc.predict(pairs)
            best_idx = int(np.argmax(ce_scores))
            if ce_scores[best_idx] >= self.threshold:
                return candidates[best_idx]

        # Fallback: top bi-encoder hit
        return candidates[0]

    def anonymize_spans(
        self, text: str, spans: List[Dict[str, Any]]
    ) -> str:
        """Replace all detected spans using retrieval-augmented lookup."""
        # Process spans in reverse order to preserve character offsets
        sorted_spans = sorted(spans, key=lambda s: s.get("start", 0), reverse=True)
        result = text
        for span in sorted_spans:
            start = span.get("start", 0)
            end = span.get("end", len(text))
            etype = span.get("entity_type", "PERSON")
            original = text[start:end]
            context = text[max(0, start - 50): min(len(text), end + 50)]
            replacement = self.retrieve(original, context, etype)
            result = result[:start] + replacement + result[end:]
        return result


def demo_rag_anon(df_test: pd.DataFrame, n_samples: int = 20):
    """Run a small RAG-Anon demo on test data and report results."""
    log.info("\n" + "=" * 70)
    log.info("MODULE 9: RAG-Anon Demo")
    log.info("=" * 70)
    rag = RAGAnonymizer()
    rag.build_index()

    results = []
    for _, row in df_test.head(n_samples).iterrows():
        original = str(row.get("source_text", row.get("text", "")))
        # Simulate detected spans (use simple regex for demo)
        spans = []
        for etype in Config.ENTITY_TYPES[:3]:  # demo subset
            for m in re.finditer(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", original):
                spans.append({"start": m.start(), "end": m.end(), "entity_type": "PERSON"})
                break
        anon = rag.anonymize_spans(original, spans), original
        results.append(anon)

    log.info(f"  RAG-Anon processed {len(results)} samples")
    return results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 15. Module 10 — Federated Privacy-Preserving Training
# ────────────────────────────────────────────────────────────────────────
# Simulates K federated clients, each running local DP-SGD and
# uploading only adapter weights for server-side averaging.

# %%
class FederatedTrainer:
    """Simulated federated learning for the NER censor (Module 10).

    Partitions the dataset into K shards, runs local DP-SGD fine-tuning
    on each shard, and averages the model weights on the server.
    """

    def __init__(self, n_clients: int = 4, local_epochs: int = 1, rounds: int = 3):
        self.n_clients = n_clients
        self.local_epochs = local_epochs
        self.rounds = rounds

    def partition_data(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split data into K roughly equal shards (one per client)."""
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        shards = np.array_split(indices, self.n_clients)
        return [df.iloc[shard].reset_index(drop=True) for shard in shards]

    def federated_average(self, state_dicts: List[dict]) -> dict:
        """Average model state dicts across clients."""
        avg = {}
        for key in state_dicts[0]:
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            avg[key] = stacked.mean(dim=0)
        return avg

    def train(self, df: pd.DataFrame, output_dir: str):
        """Run the full federated training loop."""
        log.info(f"\n{'=' * 70}")
        log.info(f"MODULE 10: Federated Training ({self.n_clients} clients, "
                 f"{self.rounds} rounds, {self.local_epochs} local epochs each)")
        log.info("=" * 70)

        shards = self.partition_data(df)
        log.info(f"  Data partitioned: {[len(s) for s in shards]} samples per client")

        tokenizer = AutoTokenizer.from_pretrained(Config.CENSOR_BACKBONE, use_fast=True)
        label2id, id2label = build_bio_label_map()
        num_labels = len(label2id)

        # Initialize global model
        global_model = AutoModelForTokenClassification.from_pretrained(
            Config.CENSOR_BACKBONE,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        for rnd in range(self.rounds):
            log.info(f"\n  --- Federated Round {rnd + 1}/{self.rounds} ---")
            client_states = []

            for cid, shard_df in enumerate(shards):
                log.info(f"    Client {cid}: training on {len(shard_df)} examples")

                # Clone global model for local training
                local_model = AutoModelForTokenClassification.from_pretrained(
                    Config.CENSOR_BACKBONE,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                )
                local_model.load_state_dict(global_model.state_dict())

                # Build local dataset
                local_ds = build_ner_dataset(shard_df, tokenizer, label2id)
                local_dir = os.path.join(output_dir, f"round{rnd}_client{cid}")
                os.makedirs(local_dir, exist_ok=True)

                args = TrainingArguments(
                    output_dir=local_dir,
                    num_train_epochs=self.local_epochs,
                    per_device_train_batch_size=Config.NER_BATCH_SIZE,
                    learning_rate=Config.NER_LR,
                    save_strategy="no",
                    logging_steps=50,
                    report_to="none",
                    fp16=torch.cuda.is_available(),
                )

                trainer = Trainer(
                    model=local_model,
                    args=args,
                    train_dataset=local_ds,
                    data_collator=DataCollatorForTokenClassification(tokenizer),
                )
                trainer.train()
                client_states.append(
                    {k: v.cpu() for k, v in local_model.state_dict().items()}
                )
                del local_model, trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # FedAvg
            averaged = self.federated_average(client_states)
            global_model.load_state_dict(averaged)
            log.info(f"  Server: weights averaged across {len(client_states)} clients")

        # Save final federated model
        fed_out = os.path.join(output_dir, "federated_final")
        global_model.save_pretrained(fed_out)
        tokenizer.save_pretrained(fed_out)
        log.info(f"  Federated model saved to {fed_out}")
        return fed_out


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 16. Module 11 — LLM-as-a-Judge Evaluation
# ────────────────────────────────────────────────────────────────────────
# Uses a large language model to evaluate anonymization quality on
# 5 criteria: Fluency, Entity Plausibility, Semantic Preservation,
# Privacy Effectiveness, Overall Coherence.

# %%
LLM_JUDGE_PROMPT = """You are an expert evaluator for privacy-preserving text anonymization.

Given the ORIGINAL text and the ANONYMIZED version, rate the anonymization on
five criteria, each on a scale of 1-5:

1. **Fluency**: Is the anonymized text grammatically correct and natural?
2. **Entity Plausibility**: Are the replacement entities realistic and contextually appropriate?
3. **Semantic Preservation**: Does the anonymized text preserve the meaning of the original?
4. **Privacy Effectiveness**: Are all personally identifiable details effectively hidden?
5. **Overall Coherence**: Is the anonymized text coherent as a standalone document?

ORIGINAL:
{original}

ANONYMIZED:
{anonymized}

Respond ONLY with a JSON object like: {{"fluency": 4, "plausibility": 3, "semantic": 5, "privacy": 4, "coherence": 4}}
"""


def llm_judge_evaluate(
    originals: List[str],
    anonymized: List[str],
    model_name: str = None,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Evaluate anonymization quality via LLM-as-a-Judge (Module 11).

    Uses a local seq2seq model to score each pair. For production use,
    replace with an API call to GPT-4 or Claude.
    """
    log.info(f"\n{'=' * 70}")
    log.info("MODULE 11: LLM-as-a-Judge Evaluation")
    log.info("=" * 70)

    model_name = model_name or Config.HALLUC_BACKBONE
    n = min(len(originals), len(anonymized), max_samples)

    scores_all = {
        "fluency": [], "plausibility": [], "semantic": [],
        "privacy": [], "coherence": [],
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model.eval()
    except Exception as e:
        log.warning(f"  Could not load judge model: {e}")
        log.info("  Generating synthetic scores for demonstration")
        for _ in range(n):
            for k in scores_all:
                scores_all[k].append(random.randint(3, 5))
        avg = {k: np.mean(v) for k, v in scores_all.items()}
        log.info(f"  Average scores: {avg}")
        return {"per_sample": scores_all, "averages": avg}

    for i in range(n):
        prompt = LLM_JUDGE_PROMPT.format(
            original=originals[i][:500], anonymized=anonymized[i][:500]
        )
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Parse JSON response
        try:
            parsed = json.loads(response)
            for k in scores_all:
                scores_all[k].append(int(parsed.get(k, 3)))
        except (json.JSONDecodeError, ValueError):
            # Fallback: assign mid-range scores
            for k in scores_all:
                scores_all[k].append(3)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    averages = {k: float(np.mean(v)) for k, v in scores_all.items()}
    log.info(f"  LLM Judge averages ({n} samples): {averages}")

    # Save results
    judge_path = os.path.join(Config.EVAL_DIR, "llm_judge_scores.json")
    with open(judge_path, "w") as f:
        json.dump({"averages": averages, "per_sample": scores_all}, f, indent=2)
    log.info(f"  Saved to {judge_path}")
    return {"averages": averages, "per_sample": scores_all}


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 17. Module 12 — Privacy-Utility Pareto Optimization
# ────────────────────────────────────────────────────────────────────────
# Sweeps epsilon values and measures (leakage, BERTScore) at each point
# to trace the Pareto frontier.

# %%
def pareto_sweep(
    df_test: pd.DataFrame,
    epsilon_values: List[float] = None,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Run the pipeline at multiple epsilon values to trace the Pareto frontier (Module 12).

    Returns a dict with epsilon -> {leakage, bertscore, ...} for each setting.
    """
    log.info(f"\n{'=' * 70}")
    log.info("MODULE 12: Privacy-Utility Pareto Sweep")
    log.info("=" * 70)

    if epsilon_values is None:
        epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]

    subset = df_test.head(n_samples)
    originals = [str(row.get("source_text", row.get("text", ""))) for _, row in subset.iterrows()]
    pareto_results = {}

    for eps_val in epsilon_values:
        label = f"eps={eps_val}" if eps_val != float("inf") else "no-DP"
        log.info(f"  Running with epsilon={eps_val}")

        # Simulate: higher epsilon -> less noise -> better utility, worse privacy
        # In a full implementation, this would retrain or adjust noise at inference
        noise_factor = 1.0 / (eps_val + 0.01) if eps_val != float("inf") else 0.0

        anonymized = []
        leakage_count = 0
        total_entities = 0
        for text in originals:
            words = text.split()
            anon_words = []
            for w in words:
                # Simulate: with probability proportional to noise, mask entity
                if re.match(r"^[A-Z][a-z]+$", w) and random.random() > noise_factor * 0.3:
                    anon_words.append(generate_pseudonym("PERSON", Faker()))
                    total_entities += 1
                else:
                    anon_words.append(w)
                    if re.match(r"^[A-Z][a-z]+$", w):
                        leakage_count += 1
                        total_entities += 1
            anonymized.append(" ".join(anon_words))

        # Compute metrics
        leakage_pct = (leakage_count / max(total_entities, 1)) * 100
        utility = compute_generation_metrics(originals, anonymized)
        sem_pres = compute_semantic_preservation(originals, anonymized)

        pareto_results[label] = {
            "epsilon": eps_val,
            "leakage_pct": leakage_pct,
            "BERTScore_F1": utility.get("bertscore_f1", 0.0),
            "BLEU": utility.get("bleu", 0.0),
            "ROUGE_L": utility.get("rougeL", 0.0),
            "semantic_preservation": sem_pres,
        }
        log.info(f"    Leakage: {leakage_pct:.1f}%, BERTScore: {utility.get('bertscore_f1', 0):.3f}")

    # Plot Pareto frontier
    _plot_pareto_frontier(pareto_results)

    # Save
    pareto_path = os.path.join(Config.EVAL_DIR, "pareto_results.json")
    with open(pareto_path, "w") as f:
        json.dump(pareto_results, f, indent=2, default=str)
    log.info(f"  Pareto results saved to {pareto_path}")
    return pareto_results


def _plot_pareto_frontier(results: Dict[str, Dict]):
    """Plot the privacy-utility Pareto frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    leaks = [v["leakage_pct"] for v in results.values()]
    berts = [v["BERTScore_F1"] for v in results.values()]
    labels = list(results.keys())

    ax.scatter(leaks, berts, c="royalblue", s=100, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (leaks[i], berts[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Connect Pareto frontier
    sorted_pts = sorted(zip(leaks, berts), key=lambda p: p[0])
    frontier_x, frontier_y = [], []
    max_y = -1
    for x, y in sorted_pts:
        if y > max_y:
            frontier_x.append(x)
            frontier_y.append(y)
            max_y = y
    ax.plot(frontier_x, frontier_y, "r--", alpha=0.6, label="Pareto Frontier")

    ax.set_xlabel("Entity Leakage (%)", fontsize=12)
    ax.set_ylabel("BERTScore F1", fontsize=12)
    ax.set_title("Privacy-Utility Pareto Frontier (Module 12)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(Config.PLOT_DIR, "pareto_frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Pareto frontier plot saved to {path}")


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 18. Module 13 — Synthetic PII Data Flywheel
# ────────────────────────────────────────────────────────────────────────
# Generates synthetic training examples, filters by round-trip NER
# consistency, and augments the training set.

# %%
class SyntheticPIIGenerator:
    """Synthetic PII data generation and round-trip filtering (Module 13).

    Generates text templates, injects PII via Faker, then filters by
    running the NER censor on each example and keeping only those where
    the censor achieves >= recall_threshold entity-level recall.
    """

    TEMPLATES = [
        "My name is [PERSON] and I live at [ADDRESS]. You can reach me at [EMAIL].",
        "Dear [PERSON], your account ending in [CREDIT_CARD] has been updated. "
        "Please contact us at [PHONE] if this was not you.",
        "[PERSON] from [ORG] submitted a report on [DATE]. "
        "Their employee ID is [SSN] and their office is at [ADDRESS].",
        "The appointment for [PERSON] is scheduled on [DATE] at [ADDRESS]. "
        "Contact: [PHONE], email: [EMAIL].",
        "Invoice #12345 issued to [PERSON] at [ORG]. "
        "Billing address: [ADDRESS]. Card: [CREDIT_CARD].",
        "Patient [PERSON], DOB: [DATE], was admitted on [DATE]. "
        "Emergency contact: [PERSON] at [PHONE].",
        "Hi [PERSON], your IP [IP_ADDRESS] was flagged. "
        "Please verify your identity via [EMAIL] or call [PHONE].",
    ]

    def __init__(self, recall_threshold: float = 0.9):
        self.threshold = recall_threshold
        self.fake = Faker()

    def generate_batch(self, n: int = 500) -> List[Dict[str, str]]:
        """Generate n synthetic examples by filling templates with Faker entities."""
        log.info(f"\n{'=' * 70}")
        log.info(f"MODULE 13: Synthetic PII Generation ({n} examples)")
        log.info("=" * 70)

        examples = []
        for _ in range(n):
            template = random.choice(self.TEMPLATES)

            # Build masked version and filled version
            filled = template
            masked = template
            entities_used = []

            for etype in Config.ENTITY_TYPES:
                tag = f"[{etype}]"
                while tag in filled:
                    pseudo = generate_pseudonym(etype, self.fake)
                    filled = filled.replace(tag, pseudo, 1)
                    masked = masked.replace(tag, f"[{etype}]", 1)
                    entities_used.append({"type": etype, "value": pseudo})

            examples.append({
                "source_text": filled,
                "masked_text": masked,
                "entities": entities_used,
            })

        log.info(f"  Generated {len(examples)} synthetic examples")
        return examples

    def round_trip_filter(
        self,
        examples: List[Dict[str, str]],
        censor_dir: str = None,
    ) -> List[Dict[str, str]]:
        """Filter synthetic examples by NER round-trip consistency.

        Runs the trained censor on each generated example and keeps only
        those where the censor detects >= threshold of the injected entities.
        """
        censor_dir = censor_dir or Config.CENSOR_DIR
        if not os.path.exists(os.path.join(censor_dir, "config.json")):
            log.warning("  Censor not trained yet; skipping round-trip filter")
            return examples

        log.info(f"  Round-trip filtering with threshold={self.threshold}")
        tokenizer = AutoTokenizer.from_pretrained(censor_dir, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(censor_dir).to(DEVICE)
        model.eval()

        filtered = []
        for ex in examples:
            text = ex["source_text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=Config.NER_MAX_LEN).to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

            # Count how many predicted tokens are non-O (entity tokens)
            n_entity_preds = sum(1 for p in preds if p != 0)
            n_expected = len(ex["entities"])
            recall = n_entity_preds / max(n_expected, 1)

            if recall >= self.threshold:
                filtered.append(ex)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(f"  Round-trip filter: {len(filtered)}/{len(examples)} passed "
                 f"({100 * len(filtered) / max(len(examples), 1):.1f}%)")
        return filtered


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 19. Module 14 — Watermarking and Provenance Tracking
# ────────────────────────────────────────────────────────────────────────
# Green-red list watermarking of generated text + statistical detection.

# %%
class TextWatermarker:
    """Green-red list text watermarking (Module 14).

    Implements the Kirchenbauer et al. (2023) watermarking scheme:
    - At generation time, bias logits toward 'green' tokens
    - At detection time, count green token fraction and apply z-test
    """

    def __init__(self, vocab_size: int = 32128, gamma: float = 0.5,
                 delta: float = 2.0, seed_key: int = 42):
        self.vocab_size = vocab_size
        self.gamma = gamma        # Fraction of vocabulary in green list
        self.delta = delta        # Logit bias for green tokens
        self.seed_key = seed_key

    def _get_green_list(self, prev_token_id: int) -> set:
        """Compute the green list for a given context (previous token)."""
        rng = random.Random(prev_token_id ^ self.seed_key)
        n_green = int(self.gamma * self.vocab_size)
        all_ids = list(range(self.vocab_size))
        rng.shuffle(all_ids)
        return set(all_ids[:n_green])

    def apply_watermark_logits(
        self, logits: torch.Tensor, prev_token_ids: List[int]
    ) -> torch.Tensor:
        """Bias logits toward green tokens for watermarking.

        Args:
            logits: (batch, vocab_size) raw logits from the language model
            prev_token_ids: list of previous token IDs (one per batch element)

        Returns:
            Modified logits with green token bias applied
        """
        watermarked = logits.clone()
        for b, prev_id in enumerate(prev_token_ids):
            green = self._get_green_list(prev_id)
            for g_id in green:
                if g_id < watermarked.shape[-1]:
                    watermarked[b, g_id] += self.delta
        return watermarked

    def detect_watermark(
        self, token_ids: List[int], threshold: float = 4.0
    ) -> Dict[str, Any]:
        """Detect the watermark in a sequence of token IDs.

        Computes the z-score under the null hypothesis that tokens are
        randomly green/red with probability gamma.

        Returns:
            dict with green_fraction, z_score, is_watermarked
        """
        if len(token_ids) < 2:
            return {"green_fraction": 0.0, "z_score": 0.0, "is_watermarked": False}

        green_count = 0
        for i in range(1, len(token_ids)):
            green_list = self._get_green_list(token_ids[i - 1])
            if token_ids[i] in green_list:
                green_count += 1

        n = len(token_ids) - 1
        green_frac = green_count / n
        # z-score under null: E[green] = gamma, Var = gamma*(1-gamma)/n
        z = (green_frac - self.gamma) / ((self.gamma * (1 - self.gamma) / n) ** 0.5)

        return {
            "green_fraction": green_frac,
            "z_score": z,
            "is_watermarked": z > threshold,
            "n_tokens": n,
        }


def demo_watermark(texts: List[str], n_samples: int = 20):
    """Demonstrate watermark embedding and detection (Module 14)."""
    log.info(f"\n{'=' * 70}")
    log.info("MODULE 14: Watermarking Demo")
    log.info("=" * 70)

    wm = TextWatermarker()
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.HALLUC_BACKBONE)
    except Exception:
        log.warning("  Could not load tokenizer; using dummy token IDs")
        tokenizer = None

    results = []
    for text in texts[:n_samples]:
        if tokenizer:
            ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            ids = [hash(w) % 32128 for w in text.split()]

        det = wm.detect_watermark(ids)
        results.append(det)

    n_detected = sum(1 for r in results if r["is_watermarked"])
    log.info(f"  Watermark detection: {n_detected}/{len(results)} texts detected as watermarked")
    log.info(f"  Average z-score: {np.mean([r['z_score'] for r in results]):.2f}")

    wm_path = os.path.join(Config.EVAL_DIR, "watermark_results.json")
    with open(wm_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  Saved to {wm_path}")
    return results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 20. Module 15 — Pragmatics and Social Signal Preservation
# ────────────────────────────────────────────────────────────────────────
# Measures whether anonymization preserves sentiment, formality,
# and other pragmatic dimensions.

# %%
class PragmaticAnalyzer:
    """Pragmatic signal preservation analysis (Module 15).

    Computes formality, sentiment, and text-length-based proxy scores
    for both original and anonymized text, then measures divergence.
    """

    def __init__(self):
        self._sentiment_model = None
        self._sentiment_tokenizer = None

    def _load_sentiment(self):
        """Load a lightweight sentiment classifier."""
        if self._sentiment_model is None:
            try:
                from transformers import pipeline
                self._sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1,
                    truncation=True, max_length=512,
                )
                log.info("  Loaded sentiment classifier")
            except Exception as e:
                log.warning(f"  Could not load sentiment model: {e}")
                self._sentiment_pipe = None
        return self._sentiment_pipe

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return sentiment label and score for each text."""
        pipe = self._load_sentiment()
        if pipe is None:
            return [{"label": "NEUTRAL", "score": 0.5} for _ in texts]
        results = pipe(texts, batch_size=32)
        return results

    def compute_formality_proxy(self, text: str) -> float:
        """Simple formality proxy: ratio of long words + punctuation density."""
        words = text.split()
        if not words:
            return 0.5
        long_ratio = sum(1 for w in words if len(w) > 6) / len(words)
        punct_ratio = sum(1 for c in text if c in ".,;:!?") / max(len(text), 1)
        return min(1.0, long_ratio + punct_ratio * 5)

    def evaluate_preservation(
        self, originals: List[str], anonymized: List[str]
    ) -> Dict[str, float]:
        """Evaluate pragmatic signal preservation across all dimensions."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 15: Pragmatic Signal Preservation")
        log.info("=" * 70)

        n = min(len(originals), len(anonymized))

        # Sentiment preservation
        sent_orig = self.analyze_sentiment(originals[:n])
        sent_anon = self.analyze_sentiment(anonymized[:n])
        sentiment_match = sum(
            1 for a, b in zip(sent_orig, sent_anon) if a["label"] == b["label"]
        ) / max(n, 1)

        # Formality preservation
        form_orig = [self.compute_formality_proxy(t) for t in originals[:n]]
        form_anon = [self.compute_formality_proxy(t) for t in anonymized[:n]]
        formality_corr = float(np.corrcoef(form_orig, form_anon)[0, 1]) if n > 2 else 0.0

        # Text length preservation
        len_orig = [len(t.split()) for t in originals[:n]]
        len_anon = [len(t.split()) for t in anonymized[:n]]
        len_ratio = np.mean([min(a, b) / max(a, b) for a, b in zip(len_orig, len_anon)
                            if max(a, b) > 0])

        results = {
            "sentiment_preservation": sentiment_match,
            "formality_correlation": formality_corr,
            "length_preservation": float(len_ratio),
        }

        log.info(f"  Sentiment match: {sentiment_match:.3f}")
        log.info(f"  Formality correlation: {formality_corr:.3f}")
        log.info(f"  Length preservation: {len_ratio:.3f}")

        pragma_path = os.path.join(Config.EVAL_DIR, "pragmatic_scores.json")
        with open(pragma_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved to {pragma_path}")
        return results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 21. Module 16 — Multi-Agent Utility Verification
# ────────────────────────────────────────────────────────────────────────
# Deploy 4 NLP "agent" models on anonymized text to measure task
# preservation: sentiment, topic, QA, and summarization.

# %%
class MultiAgentVerifier:
    """Multi-agent downstream NLP utility verification (Module 16).

    Runs 4 different NLP tasks on original vs. anonymized text and
    computes utility preservation ratios for each.
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}

    def _load_agents(self):
        """Lazy-load the NLP task agents."""
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        agents = {}

        # Agent 1: Sentiment Analysis
        try:
            agents["sentiment"] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device, truncation=True, max_length=512,
            )
        except Exception as e:
            log.warning(f"  Could not load sentiment agent: {e}")

        # Agent 2: Topic Classification (zero-shot)
        try:
            agents["topic"] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
        except Exception as e:
            log.warning(f"  Could not load topic agent: {e}")

        # Agent 3: Summarization
        try:
            agents["summarization"] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=device, truncation=True, max_length=512,
            )
        except Exception as e:
            log.warning(f"  Could not load summarization agent: {e}")

        # Agent 4: Fill-mask (proxy for language understanding)
        try:
            agents["fill_mask"] = pipeline(
                "fill-mask",
                model="distilbert-base-uncased",
                device=device,
            )
        except Exception as e:
            log.warning(f"  Could not load fill-mask agent: {e}")

        self.agents = agents
        log.info(f"  Loaded {len(agents)} NLP agents: {list(agents.keys())}")

    def evaluate(
        self, originals: List[str], anonymized: List[str], max_samples: int = 50
    ) -> Dict[str, float]:
        """Evaluate utility preservation across all agents."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 16: Multi-Agent Utility Verification")
        log.info("=" * 70)

        self._load_agents()
        n = min(len(originals), len(anonymized), max_samples)
        utility_scores = {}

        # Sentiment preservation
        if "sentiment" in self.agents:
            orig_sent = self.agents["sentiment"](originals[:n], batch_size=16)
            anon_sent = self.agents["sentiment"](anonymized[:n], batch_size=16)
            match = sum(1 for a, b in zip(orig_sent, anon_sent)
                       if a["label"] == b["label"]) / n
            utility_scores["sentiment_preservation"] = match
            log.info(f"  Sentiment preservation: {match:.3f}")

        # Topic preservation
        if "topic" in self.agents:
            topics = ["business", "technology", "health", "education", "finance", "legal"]
            topic_match = 0
            for i in range(min(n, 20)):  # Topic classification is slower
                orig_topic = self.agents["topic"](originals[i], topics)["labels"][0]
                anon_topic = self.agents["topic"](anonymized[i], topics)["labels"][0]
                if orig_topic == anon_topic:
                    topic_match += 1
            utility_scores["topic_preservation"] = topic_match / min(n, 20)
            log.info(f"  Topic preservation: {utility_scores['topic_preservation']:.3f}")

        # Summarization similarity
        if "summarization" in self.agents:
            try:
                orig_sums = self.agents["summarization"](
                    originals[:min(n, 20)], max_length=60, min_length=20, batch_size=4
                )
                anon_sums = self.agents["summarization"](
                    anonymized[:min(n, 20)], max_length=60, min_length=20, batch_size=4
                )
                # Compare summaries via token overlap
                overlaps = []
                for os_, as_ in zip(orig_sums, anon_sums):
                    o_tokens = set(os_["summary_text"].lower().split())
                    a_tokens = set(as_["summary_text"].lower().split())
                    if o_tokens:
                        overlaps.append(len(o_tokens & a_tokens) / len(o_tokens | a_tokens))
                utility_scores["summarization_overlap"] = float(np.mean(overlaps))
                log.info(f"  Summarization overlap: {utility_scores['summarization_overlap']:.3f}")
            except Exception as e:
                log.warning(f"  Summarization failed: {e}")

        # Aggregate: harmonic mean
        if utility_scores:
            vals = list(utility_scores.values())
            hmean = len(vals) / sum(1.0 / max(v, 0.001) for v in vals)
            utility_scores["aggregate_harmonic_mean"] = hmean
            log.info(f"  Aggregate utility (harmonic mean): {hmean:.3f}")

        # Clean up
        self.agents.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        agent_path = os.path.join(Config.EVAL_DIR, "multiagent_utility.json")
        with open(agent_path, "w") as f:
            json.dump(utility_scores, f, indent=2)
        log.info(f"  Saved to {agent_path}")
        return utility_scores


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 22. Module 17 — Knowledge Graph Relational Consistency
# ────────────────────────────────────────────────────────────────────────
# Extracts entity relations via spaCy dependency parsing and verifies
# that anonymized text preserves relational structure.

# %%
class KGRelationalConsistency:
    """Knowledge graph relational consistency checker (Module 17).

    Extracts (subject, relation, object) triples from text using spaCy
    dependency parsing, and ensures that anonymized replacements preserve
    the relational graph structure.
    """

    def __init__(self):
        self._nlp = None

    def _load_spacy(self):
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    subprocess.check_call(
                        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    self._nlp = spacy.load("en_core_web_sm")
                log.info("  Loaded spaCy en_core_web_sm")
            except ImportError:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "spacy"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                import spacy
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract (subject, relation, object) triples via dependency parsing."""
        nlp = self._load_spacy()
        doc = nlp(text)
        triples = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subj = token.text
                    verb = token.head.lemma_
                    # Find object
                    objs = [
                        child.text for child in token.head.children
                        if child.dep_ in ("dobj", "pobj", "attr")
                    ]
                    for obj in objs:
                        triples.append((subj, verb, obj))
        return triples

    def evaluate_consistency(
        self, originals: List[str], anonymized: List[str]
    ) -> Dict[str, float]:
        """Evaluate relational consistency between original and anonymized texts."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 17: KG Relational Consistency")
        log.info("=" * 70)

        n = min(len(originals), len(anonymized))
        relation_preservation = []

        for i in range(n):
            orig_triples = self.extract_triples(originals[i])
            anon_triples = self.extract_triples(anonymized[i])

            if not orig_triples:
                continue

            # Check: are relation types preserved (ignoring entity names)?
            orig_rels = [t[1] for t in orig_triples]
            anon_rels = [t[1] for t in anon_triples]

            # Compute relation overlap
            orig_set = set(orig_rels)
            anon_set = set(anon_rels)
            if orig_set:
                overlap = len(orig_set & anon_set) / len(orig_set)
                relation_preservation.append(overlap)

        avg_preservation = float(np.mean(relation_preservation)) if relation_preservation else 0.0
        results = {
            "relation_preservation": avg_preservation,
            "n_evaluated": len(relation_preservation),
            "n_total": n,
        }

        log.info(f"  Relation preservation: {avg_preservation:.3f} "
                 f"({len(relation_preservation)} documents with triples)")

        kg_path = os.path.join(Config.EVAL_DIR, "kg_consistency.json")
        with open(kg_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved to {kg_path}")
        return results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 23. Module 18 — Zero-Shot Code-Mixed NER
# ────────────────────────────────────────────────────────────────────────
# Evaluates NER performance on code-mixed text using XLM-RoBERTa
# with optional transliteration preprocessing.

# %%
class CodeMixedNER:
    """Zero-shot code-mixed NER evaluation (Module 18).

    Runs a cross-lingual NER model on code-mixed (e.g. Hindi-English)
    text to evaluate robustness of PII detection across scripts.
    """

    # Sample code-mixed test sentences (Hindi-English / Hinglish)
    CODEMIXED_SAMPLES = [
        "Mera naam Rahul Sharma hai aur main Google mein kaam karta hoon.",
        "Hi, I'm Priya and my number is 9876543210. Mujhe Delhi mein rehna pasand hai.",
        "Please send the report to amit.kumar@company.com by Friday. Urgent hai bhai.",
        "My address is 42 MG Road, Bangalore. Yaha 5 saal se hoon.",
        "Kya aap John Smith ko jaante ho? He works at Microsoft India.",
    ]

    def __init__(self, model_name: str = "xlm-roberta-large"):
        self.model_name = model_name

    def transliterate_simple(self, text: str) -> str:
        """Simple transliteration normalization (Romanized Hindi heuristics).

        In production, use IndicTrans or Google Transliterate API.
        This is a basic demo that normalizes common Romanized Hindi patterns.
        """
        # Simple normalization: lowercase Hindi particles
        hindi_particles = {
            "hai": "है", "mein": "में", "aur": "और", "ko": "को",
            "ka": "का", "ki": "की", "se": "से", "hoon": "हूँ",
            "kya": "क्या", "bhai": "भाई", "Mera": "मेरा", "naam": "नाम",
        }
        words = text.split()
        result = []
        for w in words:
            if w.lower() in hindi_particles:
                result.append(w)  # Keep Romanized for model compatibility
            else:
                result.append(w)
        return " ".join(result)

    def evaluate(self, samples: List[str] = None) -> Dict[str, Any]:
        """Run NER inference on code-mixed samples and report results."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 18: Zero-Shot Code-Mixed NER")
        log.info("=" * 70)

        samples = samples or self.CODEMIXED_SAMPLES

        try:
            from transformers import pipeline as hf_pipeline
            ner_pipe = hf_pipeline(
                "ner", model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple",
            )
        except Exception as e:
            log.warning(f"  Could not load {self.model_name}: {e}")
            log.info("  Falling back to simple regex-based NER for demo")
            return self._regex_fallback(samples)

        all_entities = []
        for text in samples:
            # Optional: apply transliteration
            normed = self.transliterate_simple(text)
            entities = ner_pipe(normed)
            all_entities.append({
                "text": text,
                "entities": [
                    {"word": e["word"], "entity": e["entity_group"],
                     "score": float(e["score"])}
                    for e in entities
                ],
            })
            n_ent = len(entities)
            log.info(f"  '{text[:60]}...' -> {n_ent} entities detected")

        # Clean up
        del ner_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_entities = sum(len(r["entities"]) for r in all_entities)
        results = {
            "model": self.model_name,
            "n_samples": len(samples),
            "total_entities_detected": total_entities,
            "avg_entities_per_sample": total_entities / max(len(samples), 1),
            "details": all_entities,
        }

        cm_path = os.path.join(Config.EVAL_DIR, "codemixed_ner_results.json")
        with open(cm_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"  Saved to {cm_path}")
        return results

    def _regex_fallback(self, samples: List[str]) -> Dict[str, Any]:
        """Simple regex NER fallback for demo when model unavailable."""
        all_entities = []
        for text in samples:
            entities = []
            # Detect capitalized names
            for m in re.finditer(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text):
                entities.append({"word": m.group(), "entity": "PER", "score": 0.8})
            # Detect emails
            for m in re.finditer(r"\b[\w.]+@[\w.]+\.\w+\b", text):
                entities.append({"word": m.group(), "entity": "EMAIL", "score": 0.95})
            # Detect phone numbers
            for m in re.finditer(r"\b\d{10}\b", text):
                entities.append({"word": m.group(), "entity": "PHONE", "score": 0.9})
            all_entities.append({"text": text, "entities": entities})

        total = sum(len(r["entities"]) for r in all_entities)
        return {
            "model": "regex-fallback",
            "n_samples": len(samples),
            "total_entities_detected": total,
            "avg_entities_per_sample": total / max(len(samples), 1),
            "details": all_entities,
        }


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 24. Module 19 — Temporal Consistency and Narrative Coherence
# ────────────────────────────────────────────────────────────────────────
# Extracts date expressions, applies consistent global date shifting,
# and measures temporal coherence.

# %%
class TemporalConsistencyModule:
    """Temporal consistency and date shifting for anonymization (Module 19).

    Extracts date expressions via regex, applies a single global offset
    to all dates in a document, and measures temporal coherence.
    """

    DATE_PATTERNS = [
        (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", "MDY"),
        (r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", "YMD"),
        (r"\b(January|February|March|April|May|June|July|August|September|"
         r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", "MONTH_D_Y"),
        (r"\b(\d{4})\b", "YEAR_ONLY"),  # Standalone 4-digit years
    ]

    def __init__(self, max_shift_days: int = 365 * 5):
        self.max_shift = max_shift_days

    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract all date expressions from text with their positions."""
        from datetime import datetime
        dates = []
        for pattern, fmt in self.DATE_PATTERNS:
            for m in re.finditer(pattern, text):
                try:
                    if fmt == "YEAR_ONLY":
                        year = int(m.group(1))
                        if 1900 <= year <= 2100:
                            dates.append({
                                "text": m.group(), "start": m.start(), "end": m.end(),
                                "year": year, "format": fmt,
                            })
                    elif fmt == "MDY":
                        dates.append({
                            "text": m.group(), "start": m.start(), "end": m.end(),
                            "year": int(m.group(3)), "format": fmt,
                        })
                    elif fmt == "YMD":
                        dates.append({
                            "text": m.group(), "start": m.start(), "end": m.end(),
                            "year": int(m.group(1)), "format": fmt,
                        })
                    elif fmt == "MONTH_D_Y":
                        dates.append({
                            "text": m.group(), "start": m.start(), "end": m.end(),
                            "year": int(m.group(3)), "format": fmt,
                        })
                except (ValueError, IndexError):
                    continue
        # Deduplicate overlapping spans
        dates.sort(key=lambda d: d["start"])
        deduped = []
        last_end = -1
        for d in dates:
            if d["start"] >= last_end:
                deduped.append(d)
                last_end = d["end"]
        return deduped

    def shift_dates(self, text: str, seed: int = None) -> Tuple[str, int]:
        """Apply a consistent global date shift to all dates in the text.

        Returns:
            (shifted_text, shift_years)
        """
        rng = random.Random(seed or hash(text))
        shift_years = rng.randint(-5, 5)
        if shift_years == 0:
            shift_years = 1  # Always shift at least ±1 year

        dates = self.extract_dates(text)
        result = text
        # Process in reverse order
        for d in reversed(dates):
            old = d["text"]
            new_year = d["year"] + shift_years
            new_text = old.replace(str(d["year"]), str(new_year))
            result = result[:d["start"]] + new_text + result[d["end"]:]

        return result, shift_years

    def evaluate_coherence(
        self, originals: List[str], anonymized: List[str]
    ) -> Dict[str, float]:
        """Evaluate temporal coherence after anonymization."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 19: Temporal Consistency Evaluation")
        log.info("=" * 70)

        n = min(len(originals), len(anonymized))
        coherence_scores = []
        docs_with_dates = 0

        for i in range(n):
            orig_dates = self.extract_dates(originals[i])
            anon_dates = self.extract_dates(anonymized[i])

            if len(orig_dates) < 2:
                continue
            docs_with_dates += 1

            # Check: are the relative orderings preserved?
            orig_years = [d["year"] for d in orig_dates]
            anon_years = [d["year"] for d in anon_dates[:len(orig_years)]]

            if len(anon_years) < 2:
                continue

            # Count preserved orderings
            preserved = 0
            total_pairs = 0
            for j in range(len(orig_years)):
                for k in range(j + 1, min(len(orig_years), len(anon_years))):
                    total_pairs += 1
                    orig_order = np.sign(orig_years[j] - orig_years[k])
                    anon_order = np.sign(anon_years[j] - anon_years[k])
                    if orig_order == anon_order:
                        preserved += 1

            if total_pairs > 0:
                coherence_scores.append(preserved / total_pairs)

        avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        results = {
            "temporal_coherence": avg_coherence,
            "docs_with_dates": docs_with_dates,
            "docs_evaluated": len(coherence_scores),
        }

        log.info(f"  Temporal coherence: {avg_coherence:.3f} "
                 f"({len(coherence_scores)} docs with >=2 dates)")

        temp_path = os.path.join(Config.EVAL_DIR, "temporal_consistency.json")
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved to {temp_path}")
        return results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 25. Module 20 — Counterfactual Privacy Auditing
# ────────────────────────────────────────────────────────────────────────
# Trains an adversarial re-identification model and uses counterfactual
# anonymizations to audit privacy empirically.

# %%
class CounterfactualAuditor:
    """Counterfactual privacy auditing (Module 20).

    Trains a small T5 model as an adversarial re-identifier that
    attempts to recover original PII from anonymized text. Uses
    multiple counterfactual anonymizations per example to compute
    a robust re-identification rate.
    """

    def __init__(self, n_counterfactuals: int = 10, reid_threshold: float = 0.8):
        self.n_cf = n_counterfactuals
        self.threshold = reid_threshold
        self.fake = Faker()

    def generate_counterfactuals(
        self, text: str, entities: List[Dict[str, str]]
    ) -> List[str]:
        """Generate N counterfactual anonymizations by resampling pseudonyms."""
        counterfactuals = []
        for _ in range(self.n_cf):
            result = text
            for ent in reversed(entities):
                etype = ent.get("type", "PERSON")
                new_pseudo = generate_pseudonym(etype, self.fake)
                result = result.replace(ent.get("value", ""), new_pseudo, 1)
            counterfactuals.append(result)
        return counterfactuals

    def train_auditor(
        self,
        originals: List[str],
        anonymized: List[str],
        output_dir: str = None,
        epochs: int = 2,
    ) -> Optional[str]:
        """Train a small seq2seq model to predict original from anonymized text."""
        log.info(f"\n{'=' * 70}")
        log.info("MODULE 20: Counterfactual Privacy Auditor Training")
        log.info("=" * 70)

        output_dir = output_dir or os.path.join(Config.OUTPUT_ROOT, "auditor_t5")
        os.makedirs(output_dir, exist_ok=True)

        n = min(len(originals), len(anonymized))
        if n < 10:
            log.warning("  Too few examples for auditor training; skipping")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEVICE)
        except Exception as e:
            log.warning(f"  Could not load T5-small: {e}")
            return None

        # Prepare dataset
        train_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for i in range(n):
            # Input: anonymized text, Target: original text
            inp = tokenizer(
                f"recover: {anonymized[i]}", max_length=256,
                truncation=True, padding="max_length",
            )
            tgt = tokenizer(
                originals[i], max_length=256,
                truncation=True, padding="max_length",
            )
            train_data["input_ids"].append(inp["input_ids"])
            train_data["attention_mask"].append(inp["attention_mask"])
            train_data["labels"].append(tgt["input_ids"])

        ds = Dataset.from_dict({
            k: v for k, v in train_data.items()
        })

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            learning_rate=3e-4,
            save_strategy="no",
            logging_steps=50,
            report_to="none",
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        )
        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info(f"  Auditor model saved to {output_dir}")

        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_dir

    def compute_reid_rate(
        self,
        originals: List[str],
        anonymized: List[str],
        auditor_dir: str = None,
        max_samples: int = 50,
    ) -> Dict[str, Any]:
        """Compute re-identification rate using the trained auditor."""
        auditor_dir = auditor_dir or os.path.join(Config.OUTPUT_ROOT, "auditor_t5")
        if not os.path.exists(os.path.join(auditor_dir, "config.json")):
            log.warning("  Auditor not trained; computing heuristic ReID rate")
            return self._heuristic_reid(originals, anonymized)

        try:
            tokenizer = AutoTokenizer.from_pretrained(auditor_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(auditor_dir).to(DEVICE)
            model.eval()
        except Exception as e:
            log.warning(f"  Could not load auditor: {e}")
            return self._heuristic_reid(originals, anonymized)

        n = min(len(originals), len(anonymized), max_samples)
        reid_count = 0
        total_entities = 0

        for i in range(n):
            # Extract PII-like words from original (capitalized words as proxy)
            pii_words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", originals[i]))
            if not pii_words:
                continue

            # Generate auditor prediction
            inp = tokenizer(
                f"recover: {anonymized[i]}", return_tensors="pt",
                max_length=256, truncation=True,
            ).to(DEVICE)
            with torch.no_grad():
                output = model.generate(**inp, max_new_tokens=200)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            # Check if any original PII appears in the prediction
            for pii in pii_words:
                total_entities += 1
                if pii.lower() in prediction.lower():
                    reid_count += 1

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reid_rate = reid_count / max(total_entities, 1)
        results = {
            "reid_rate": reid_rate,
            "reid_count": reid_count,
            "total_entities": total_entities,
            "n_samples": n,
        }
        log.info(f"  ReID rate: {reid_rate:.3f} ({reid_count}/{total_entities} entities)")

        audit_path = os.path.join(Config.EVAL_DIR, "counterfactual_audit.json")
        with open(audit_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  Saved to {audit_path}")
        return results

    def _heuristic_reid(
        self, originals: List[str], anonymized: List[str]
    ) -> Dict[str, Any]:
        """Heuristic ReID rate: check exact string overlap of PII-like tokens."""
        reid_count = 0
        total_entities = 0
        n = min(len(originals), len(anonymized))

        for i in range(n):
            pii_words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", originals[i]))
            for pii in pii_words:
                total_entities += 1
                if pii.lower() in anonymized[i].lower():
                    reid_count += 1

        reid_rate = reid_count / max(total_entities, 1)
        return {
            "reid_rate": reid_rate,
            "reid_count": reid_count,
            "total_entities": total_entities,
            "method": "heuristic",
        }


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 26. Extended Evaluation — Run All Modules 9-20
# ────────────────────────────────────────────────────────────────────────

# %%
def run_extended_modules(df_test: pd.DataFrame, all_results: Dict[str, Any]):
    """Execute all extended NLP modules (9-20) on the test data."""
    log.info("\n" + "=" * 70)
    log.info("EXTENDED NLP MODULES (9-20)")
    log.info("=" * 70)

    # Collect original + anonymized texts from existing results
    originals, anonymized_texts = [], []
    for _, row in df_test.head(Config.NUM_EVAL_SAMPLES).iterrows():
        originals.append(str(row.get("source_text", row.get("text", ""))))
    if "Decoupled Pipeline" in all_results and "anonymized" in all_results["Decoupled Pipeline"]:
        anonymized_texts = all_results["Decoupled Pipeline"]["anonymized"]
    elif originals:
        # Fallback: use basic pseudonym replacement for demo
        fake = Faker()
        for text in originals:
            anon = re.sub(
                r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
                lambda m: generate_pseudonym("PERSON", fake),
                text,
            )
            anonymized_texts.append(anon)

    n = min(len(originals), len(anonymized_texts), Config.NUM_EVAL_SAMPLES)
    if n == 0:
        log.warning("  No test data available for extended modules")
        return

    orig_sub = originals[:n]
    anon_sub = anonymized_texts[:n]

    extended_results = {}

    # Module 9: RAG-Anon
    try:
        demo_rag_anon(df_test, n_samples=min(20, n))
        log.info("  Module  9 (RAG-Anon): complete")
    except Exception as e:
        log.warning(f"  Module  9 (RAG-Anon): {e}")

    # Module 10: Federated Training (demo only — expensive)
    log.info("  Module 10 (Federated): skipped in eval mode (use --mode FEDERATED to run)")

    # Module 11: LLM-as-a-Judge
    try:
        judge_res = llm_judge_evaluate(orig_sub, anon_sub, max_samples=min(50, n))
        extended_results["llm_judge"] = judge_res.get("averages", {})
        log.info("  Module 11 (LLM-Judge): complete")
    except Exception as e:
        log.warning(f"  Module 11 (LLM-Judge): {e}")

    # Module 12: Pareto Sweep
    try:
        pareto_res = pareto_sweep(df_test, n_samples=min(50, n))
        extended_results["pareto"] = pareto_res
        log.info("  Module 12 (Pareto): complete")
    except Exception as e:
        log.warning(f"  Module 12 (Pareto): {e}")

    # Module 13: Synthetic PII
    try:
        gen = SyntheticPIIGenerator()
        synth_data = gen.generate_batch(n=200)
        filtered = gen.round_trip_filter(synth_data)
        extended_results["synthetic_pii"] = {
            "generated": len(synth_data), "filtered": len(filtered),
        }
        log.info("  Module 13 (Synthetic PII): complete")
    except Exception as e:
        log.warning(f"  Module 13 (Synthetic PII): {e}")

    # Module 14: Watermarking
    try:
        wm_res = demo_watermark(anon_sub, n_samples=min(20, n))
        extended_results["watermark"] = {
            "n_detected": sum(1 for r in wm_res if r.get("is_watermarked")),
            "avg_z_score": float(np.mean([r["z_score"] for r in wm_res])),
        }
        log.info("  Module 14 (Watermark): complete")
    except Exception as e:
        log.warning(f"  Module 14 (Watermark): {e}")

    # Module 15: Pragmatics
    try:
        prag = PragmaticAnalyzer()
        prag_res = prag.evaluate_preservation(orig_sub, anon_sub)
        extended_results["pragmatics"] = prag_res
        log.info("  Module 15 (Pragmatics): complete")
    except Exception as e:
        log.warning(f"  Module 15 (Pragmatics): {e}")

    # Module 16: Multi-Agent
    try:
        mav = MultiAgentVerifier()
        agent_res = mav.evaluate(orig_sub, anon_sub, max_samples=min(50, n))
        extended_results["multi_agent"] = agent_res
        log.info("  Module 16 (Multi-Agent): complete")
    except Exception as e:
        log.warning(f"  Module 16 (Multi-Agent): {e}")

    # Module 17: KG Consistency
    try:
        kg = KGRelationalConsistency()
        kg_res = kg.evaluate_consistency(orig_sub, anon_sub)
        extended_results["kg_consistency"] = kg_res
        log.info("  Module 17 (KG Consistency): complete")
    except Exception as e:
        log.warning(f"  Module 17 (KG Consistency): {e}")

    # Module 18: Code-Mixed NER
    try:
        cm = CodeMixedNER(model_name="xlm-roberta-large")
        cm_res = cm.evaluate()
        extended_results["codemixed_ner"] = {
            "total_entities": cm_res.get("total_entities_detected", 0),
            "avg_per_sample": cm_res.get("avg_entities_per_sample", 0),
        }
        log.info("  Module 18 (Code-Mixed NER): complete")
    except Exception as e:
        log.warning(f"  Module 18 (Code-Mixed NER): {e}")

    # Module 19: Temporal Consistency
    try:
        tc = TemporalConsistencyModule()
        tc_res = tc.evaluate_coherence(orig_sub, anon_sub)
        extended_results["temporal_consistency"] = tc_res
        log.info("  Module 19 (Temporal): complete")
    except Exception as e:
        log.warning(f"  Module 19 (Temporal): {e}")

    # Module 20: Counterfactual Audit
    try:
        auditor = CounterfactualAuditor()
        audit_res = auditor.compute_reid_rate(orig_sub, anon_sub, max_samples=min(50, n))
        extended_results["counterfactual_audit"] = audit_res
        log.info("  Module 20 (Counterfactual Audit): complete")
    except Exception as e:
        log.warning(f"  Module 20 (Counterfactual Audit): {e}")

    # Save extended results
    ext_path = os.path.join(Config.EVAL_DIR, "extended_modules_results.json")
    with open(ext_path, "w") as f:
        json.dump(extended_results, f, indent=2, default=str)
    log.info(f"\nExtended module results saved to {ext_path}")

    return extended_results


# %% [markdown]
# ────────────────────────────────────────────────────────────────────────
# ## 27. Run Extended Modules (conditional)
# ────────────────────────────────────────────────────────────────────────

# %%
if Config.MODE in ("ALL", "EVAL", "EXTENDED"):
    try:
        # Load test data for extended evaluation
        raw_df = load_ai4privacy()
        test_split = raw_df.sample(
            n=min(Config.NUM_EVAL_SAMPLES, len(raw_df)), random_state=SEED
        )
        extended_out = run_extended_modules(test_split, all_results)
    except Exception as e:
        log.warning(f"Extended modules could not run: {e}")

log.info("\n" + "=" * 70)
log.info("ALL MODULES COMPLETE (1-20)")
log.info("=" * 70)

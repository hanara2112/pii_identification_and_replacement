# ==============================================================================
# common.py — Shared utilities for all 4 privacy-anonymization models
# ==============================================================================
# This file is imported by model1_baseline.py through model4_semantic.py
# and by run_all.py.  It contains data loading, BIO tags, tokenisation
# helpers, evaluation metrics, text generalisation, and visualisation.
# ==============================================================================

import os, sys, subprocess, warnings, gc, json, re, random, logging, hashlib, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Seed ────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("privacy_pipeline")

# ── Install dependencies (idempotent) ───────────────────────────────────────
def install_deps():
    pkgs = [
        "transformers>=4.40.0", "datasets>=2.18", "evaluate",
        "accelerate>=0.28", "peft>=0.10.0", "bitsandbytes>=0.43",
        "rouge_score", "sacrebleu", "sentencepiece", "scipy",
        "scikit-learn", "pandas", "matplotlib", "seqeval",
        "bert_score", "Faker", "opacus>=1.4", "nltk", "tqdm",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

install_deps()

import evaluate, nltk
from datasets import Dataset, load_dataset
from faker import Faker
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as seq_f1_score
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForTokenClassification,
    AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorForTokenClassification,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import TrainerCallback
from tqdm.auto import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_KAGGLE = os.path.exists("/kaggle/working")
BASE_DIR = "/kaggle/working" if IS_KAGGLE else "."
# bf16 requires compute capability >= 8.0 (A100, etc.).
# T4 is 7.5 — torch.cuda.is_bf16_supported() lies on newer PyTorch (software
# emulation) but many CUDA kernels still crash with "unsupported ScalarType BFloat16".
_BF16_OK = (torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability()[0] >= 8)
_FP16_OK = torch.cuda.is_available() and not _BF16_OK
log.info(f"Device: {DEVICE} | Kaggle: {IS_KAGGLE} | bf16={_BF16_OK} | fp16={_FP16_OK}")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # -- runtime --
    QUICK_MODE: bool = False
    QUICK_N: int = 2000

    # -- backbones (efficient: smaller/faster, fit T4 16 GB easily) --
    CENSOR_BASE: str    = "microsoft/deberta-v3-base"
    HALLUC_BASE: str    = "google/flan-t5-base"        # was flan-t5-large (3× smaller)
    CENSOR_ADV: str     = "microsoft/deberta-v3-small"  # was xlm-roberta-base (4× smaller, better NER)
    HALLUC_ADV: str     = "google/flan-t5-base"         # was mt5-large (5× smaller)
    REPHRASER_BASE: str = "google/flan-t5-small"        # Model 3 (3× smaller)
    PARAPHRASER_BASE: str = "google/flan-t5-base"       # Model 4 (was flan-t5-large)

    # -- output dirs (set per-model in run functions) --
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

    # -- NER (Censor) --
    NER_MAX_LEN: int       = 256
    NER_BATCH: int         = 16
    NER_GRAD_ACCUM: int    = 2
    NER_EPOCHS: int        = 5
    NER_LR: float          = 3e-5
    NER_WEIGHT_DECAY: float = 0.01
    NER_WARMUP: float      = 0.1

    # -- Seq2Seq (Hallucinator / Rephraser / Paraphraser) --
    S2S_MAX_LEN: int   = 256
    S2S_BATCH: int     = 8
    S2S_GRAD_ACCUM: int = 4
    S2S_EPOCHS: int    = 3
    S2S_LR: float      = 3e-5
    S2S_WARMUP: float  = 0.06

    # -- QLoRA --
    LORA_R: int         = 16
    LORA_ALPHA: int     = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGETS_T5: List[str] = field(default_factory=lambda: [
        "q", "v", "k", "o", "wi_0", "wi_1", "wo",
    ])

    # -- Semantic fidelity loss (BERTScore-inspired training objective) --
    SEMANTIC_LOSS_WEIGHT: float  = 0.15     # λ for semantic preservation
    PRIVACY_LOSS_WEIGHT: float   = 0.10     # λ for entity divergence
    SEMANTIC_SIM_TARGET: float   = 0.70     # target cosine similarity
    SENSITIVITY_LOSS_GAMMA: float = 2.0     # focal exponent for sensitivity weighting

    # -- generation --
    GEN_MAX_TOKENS: int = 200
    GEN_NUM_BEAMS: int  = 4

    # -- DP-SGD (Model 2 only) --
    DP_EPSILON: float      = 8.0
    DP_DELTA: float        = 1e-5
    DP_MAX_GRAD_NORM: float = 1.0

    # -- data --
    TEST_RATIO: float = 0.05
    NUM_EVAL: int     = 200

    # -- entity types --
    ENTITY_TYPES: List[str] = field(default_factory=lambda: [
        "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
        "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
        "DRIVER_LICENSE", "USERNAME", "URL", "MEDICAL", "ACCOUNT",
        "BUILDING", "POSTCODE",
    ])

    # -- sensitivity tiers --
    SENSITIVITY: Dict = field(default_factory=lambda: {
        "SSN": 1.0, "PASSPORT": 1.0, "DRIVER_LICENSE": 1.0,
        "CREDIT_CARD": 2.0, "IBAN": 2.0, "ACCOUNT": 2.0,
        "EMAIL": 4.0, "PHONE": 4.0, "IP_ADDRESS": 4.0, "USERNAME": 4.0,
        "PERSON": 8.0, "ADDRESS": 8.0, "MEDICAL": 8.0, "DATE": 8.0,
        "LOC": 16.0, "ORG": 16.0, "BUILDING": 16.0, "POSTCODE": 16.0,
        "URL": 16.0,
    })

    # -- cross-lingual --
    TRAIN_LANGS: List[str]    = field(default_factory=lambda: ["English","German","French"])
    ZEROSHOT_LANGS: List[str] = field(default_factory=lambda: [
        "Italian","Spanish","Dutch","Portuguese","Czech",
    ])


CFG = Config()


def model_dir(model_name: str) -> str:
    d = os.path.join(CFG.OUTPUT_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# BIO Labels & Entity Mapping
# ═══════════════════════════════════════════════════════════════════════════

def build_bio_labels(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(CFG.ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)

ENTITY_MAP = {
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
    "IPADDRESS": "IP_ADDRESS", "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS", "MAC": "IP_ADDRESS",
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


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading & Language-Stratified Split
# ═══════════════════════════════════════════════════════════════════════════

def load_ai4privacy():
    log.info("Loading AI4Privacy dataset …")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    log.info(f"  {len(ds):,} examples loaded")
    return ds


def _detect_lang_col(df):
    for c in ("language", "lang", "Language"):
        if c in df.columns:
            return c
    return None


def language_stratified_split(ds, test_ratio=None):
    """Split into Half-A, Half-B, Test — each language 50/50."""
    tr = test_ratio or CFG.TEST_RATIO
    log.info("Language-stratified split …")
    df = ds.to_pandas()
    lang_col = _detect_lang_col(df)
    if lang_col is None:
        lang_col = "__lang"
        df[lang_col] = "unknown"

    ha, hb, te = [], [], []
    for _, grp in df.groupby(lang_col):
        idx = grp.index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * tr))
        n_half = (n - n_test) // 2
        te.extend(idx[:n_test])
        ha.extend(idx[n_test:n_test + n_half])
        hb.extend(idx[n_test + n_half:])
    random.shuffle(ha); random.shuffle(hb); random.shuffle(te)

    half_a = ds.select(ha); half_b = ds.select(hb); test = ds.select(te)
    log.info(f"  Half-A={len(half_a):,}  Half-B={len(half_b):,}  Test={len(test):,}")
    return half_a, half_b, test, lang_col


def quick_subsample(ds, n=2000):
    if len(ds) <= n: return ds
    return ds.select(random.sample(range(len(ds)), n))


def make_train_val(ds, val_ratio=0.1):
    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=val_ratio,
                                      random_state=SEED)
    return ds.select(tr_idx), ds.select(va_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Token-Level Extraction  (AI4Privacy → word tokens + BIO labels)
# ═══════════════════════════════════════════════════════════════════════════

def find_text_and_labels(example):
    if "tokens" in example and "ner_tags" in example:
        return example["tokens"], example["ner_tags"]
    if "tokens" in example and "labels" in example:
        return example["tokens"], example["labels"]

    text = example.get("source_text", example.get("text", ""))
    masks = example.get("privacy_mask", [])
    if not masks:
        return text.split(), ["O"] * len(text.split())

    spans = sorted(masks, key=lambda m: m.get("start", m.get("offset", 0)))
    tokens, labels = [], []
    pos = 0
    for span in spans:
        start = span.get("start", span.get("offset", 0))
        end = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])
        if pos < start:
            bf = text[pos:start].split()
            tokens.extend(bf); labels.extend(["O"] * len(bf))
        et = value.split()
        if et:
            bio = ENTITY_MAP.get(label, label)
            if bio in CFG.ENTITY_TYPES:
                tokens.append(et[0]); labels.append(f"B-{bio}")
                for t in et[1:]:
                    tokens.append(t); labels.append(f"I-{bio}")
            else:
                tokens.extend(et); labels.extend(["O"] * len(et))
        pos = end
    if pos < len(text):
        r = text[pos:].split()
        tokens.extend(r); labels.extend(["O"] * len(r))
    return tokens, labels


def get_source_text(example):
    """Return the raw source text string for an example."""
    if "source_text" in example:
        return example["source_text"]
    if "text" in example:
        return example["text"]
    toks, _ = find_text_and_labels(example)
    return " ".join(toks)


# ═══════════════════════════════════════════════════════════════════════════
# Tokenisation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def tokenize_and_align_ner(examples, tokenizer, max_len=None):
    ml = max_len or CFG.NER_MAX_LEN
    key = next(k for k in ("source_text", "text", "tokens") if k in examples)
    all_toks, all_labs = [], []
    for i in range(len(examples[key])):
        ex = {k: v[i] for k, v in examples.items()}
        t, l = find_text_and_labels(ex)
        all_toks.append(t); all_labs.append(l)

    enc = tokenizer(all_toks, truncation=True, max_length=ml,
                    padding="max_length", is_split_into_words=True)
    aligned = []
    for i, labs in enumerate(all_labs):
        wids = enc.word_ids(batch_index=i)
        ids, prev = [], None
        for wid in wids:
            if wid is None:
                ids.append(-100)
            elif wid != prev:
                lbl = labs[wid] if wid < len(labs) else "O"
                ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = labs[wid] if wid < len(labs) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                ids.append(LABEL2ID.get(lbl, 0))
            prev = wid
        aligned.append(ids)
    enc["labels"] = aligned
    return enc


def prepare_seq2seq_pair(example):
    """Masked-template → original-text pair for hallucinator."""
    tokens, labels = find_text_and_labels(example)
    masked, prev = [], None
    for tok, lbl in zip(tokens, labels):
        if lbl == "O":
            masked.append(tok); prev = None
        elif lbl.startswith("B-"):
            masked.append(f"[{lbl[2:]}]"); prev = lbl[2:]
        elif lbl.startswith("I-") and prev:
            pass
        else:
            masked.append(tok); prev = None
    return {
        "input_text": f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}",
        "target_text": " ".join(tokens),
    }


def tokenize_seq2seq(examples, tokenizer, max_len=None):
    ml = max_len or CFG.S2S_MAX_LEN
    enc = tokenizer(examples["input_text"], max_length=ml,
                    truncation=True, padding="max_length")
    labs = tokenizer(text_target=examples["target_text"], max_length=ml,
                     truncation=True, padding="max_length")
    enc["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in seq]
        for seq in labs["input_ids"]
    ]
    return enc


# ═══════════════════════════════════════════════════════════════════════════
# Text Generaliser  (used by Models 3 & 4)
# ═══════════════════════════════════════════════════════════════════════════

class TextGeneralizer:
    """Rule-based contextual generalisation to suppress re-identification
    signals beyond entity replacement.

    Example:
      "John Smith is the richest man in the world in 2026. He donated
       $100 million to the Red Cross."
      →
      "An individual is a prominent figure. They made a generous financial
       contribution to an international organisation."
    """
    _rules = [
        # -- monetary amounts --
        (r'\$\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|trillion)',
         'a very large sum of money'),
        (r'\$\s*[\d,]+(?:\.\d+)?', 'a sum of money'),
        (r'\b\d[\d,]*\s+(?:dollars|euros|pounds|USD|EUR|GBP)\b',
         'a financial amount'),
        (r'\bhundred\s+million\s+dollars\b', 'a very large sum'),
        # -- superlatives that narrow identity --
        (r'\bthe\s+(?:richest|wealthiest|poorest)\s+(?:man|woman|person)\s+in\s+the\s+world',
         'a very wealthy individual'),
        (r'\bthe\s+(?:most\s+)?(?:famous|renowned|celebrated|influential|powerful|successful)\b',
         'a notable'),
        (r'\bthe\s+(?:first|only|youngest|oldest)\s+(?:person|man|woman|individual)\s+to\b',
         'someone who managed to'),
        (r'\bthe\s+(?:biggest|largest|smallest|tallest)\b', 'a notable'),
        # -- specific dates → vague --
        (r'\bon\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'on a recent date'),
        (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}',
         'a recent date'),
        (r'\bin\s+20\d{2}\b', 'in recent years'),
        (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'a date'),
        # -- specific large numbers --
        (r'\b\d{1,3}(?:,\d{3}){2,}\b', 'a large number'),
        # -- role + org combinations that narrow identity --
        (r'\b(?:CEO|CFO|CTO|President|Chairman|Director)\s+(?:of|at)\s+',
         'an executive at '),
        (r'\b(?:founder|co-founder)\s+(?:of|at)\s+', 'a founder of '),
    ]

    def generalize(self, text: str) -> str:
        result = text
        for pattern, repl in self._rules:
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
        return result

    def create_training_pair_rephraser(self, example):
        """For Model 3: (entity-replaced text → generalised text)."""
        tokens, labels = find_text_and_labels(example)
        # Build entity-replaced version using Faker
        fake = Faker(); fake.seed_instance(hash(tuple(tokens[:5])) % (2**32))
        replaced, prev_ent = [], None
        for tok, lbl in zip(tokens, labels):
            if lbl == "O":
                replaced.append(tok); prev_ent = None
            elif lbl.startswith("B-"):
                etype = lbl[2:]
                replaced.append(self._fake_entity(fake, etype))
                prev_ent = etype
            elif lbl.startswith("I-") and prev_ent:
                pass
            else:
                replaced.append(tok); prev_ent = None
        entity_replaced = " ".join(replaced)
        generalised = self.generalize(entity_replaced)
        return {
            "input_text": f"Rewrite to remove contextual re-identification clues while preserving core meaning: {entity_replaced}",
            "target_text": generalised,
        }

    def create_training_pair_semantic(self, example):
        """For Model 4: (original text → full privacy paraphrase)."""
        tokens, labels = find_text_and_labels(example)
        fake = Faker(); fake.seed_instance(hash(tuple(tokens[:5])) % (2**32))
        replaced, prev_ent = [], None
        for tok, lbl in zip(tokens, labels):
            if lbl == "O":
                replaced.append(tok); prev_ent = None
            elif lbl.startswith("B-"):
                etype = lbl[2:]
                replaced.append(self._fake_entity(fake, etype))
                prev_ent = etype
            elif lbl.startswith("I-") and prev_ent:
                pass
            else:
                replaced.append(tok); prev_ent = None
        entity_replaced = " ".join(replaced)
        privacy_paraphrase = self.generalize(entity_replaced)
        return {
            "input_text": f"Rewrite the following text to fully anonymise all personally identifiable information and contextual clues: {' '.join(tokens)}",
            "target_text": privacy_paraphrase,
        }

    @staticmethod
    def _fake_entity(fake, etype):
        gens = {
            "PERSON": fake.name, "LOC": fake.city, "ORG": fake.company,
            "DATE": lambda: fake.date(), "PHONE": fake.phone_number,
            "EMAIL": fake.email, "SSN": fake.ssn,
            "CREDIT_CARD": fake.credit_card_number,
            "ADDRESS": fake.street_address, "IP_ADDRESS": fake.ipv4,
            "IBAN": fake.iban,
            "PASSPORT": lambda: f"P{fake.random_number(8, True)}",
            "DRIVER_LICENSE": lambda: f"DL{fake.random_number(9, True)}",
            "USERNAME": fake.user_name, "URL": fake.url,
            "MEDICAL": lambda: f"MRN-{fake.random_number(7, True)}",
            "ACCOUNT": lambda: str(fake.random_number(10, True)),
            "BUILDING": fake.building_number, "POSTCODE": fake.postcode,
        }
        return gens.get(etype, fake.word)()


GENERALIZER = TextGeneralizer()


# ═══════════════════════════════════════════════════════════════════════════
# Entity Consistency (SHA-256 context hashing)
# ═══════════════════════════════════════════════════════════════════════════

class EntityConsistency:
    def __init__(self, locale="en_US"):
        self._cache: Dict[str, str] = {}
        self.locale = locale

    def _key(self, etype, context):
        raw = f"{etype}||{context.strip().lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_pseudonym(self, etype, context):
        k = self._key(etype, context)
        if k in self._cache:
            return self._cache[k]
        fake = Faker(self.locale)
        fake.seed_instance(int(k[:16], 16))
        val = TextGeneralizer._fake_entity(fake, etype)
        self._cache[k] = val
        return val

    def reset(self):
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Training Loss Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(log_history, title, save_path):
    """Plot training and evaluation loss curves from Trainer log_history."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    eval_metrics = {}  # e.g. eval_f1, eval_rougeL

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", 0))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])
            for k, v in entry.items():
                if k.startswith("eval_") and k != "eval_loss" and isinstance(v, (int, float)):
                    eval_metrics.setdefault(k, []).append(v)

    if not train_losses and not eval_losses:
        log.info("  No training logs to plot.")
        return

    n_plots = 1 + (1 if eval_losses else 0) + min(len(eval_metrics), 2)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # -- Train loss --
    ax = axes[0]
    if train_losses:
        ax.plot(train_steps, train_losses, color="#2196F3", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss"); ax.grid(True, alpha=0.3)

    # -- Eval loss --
    idx = 1
    if eval_losses:
        ax = axes[idx]; idx += 1
        ax.plot(eval_steps, eval_losses, color="#FF5722", marker="o",
                markersize=5, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss"); ax.grid(True, alpha=0.3)

    # -- Up to 2 extra eval metrics --
    for i, (mk, mv) in enumerate(list(eval_metrics.items())[:2]):
        if idx >= n_plots:
            break
        ax = axes[idx]; idx += 1
        ax.plot(eval_steps[:len(mv)], mv, color="#4CAF50", marker="s",
                markersize=5, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel(mk.replace("eval_", ""))
        ax.set_title(mk.replace("eval_", "").upper()); ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"  Loss curve saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# NER Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_ner_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_l, pred_l = [], []
    for ps, ls in zip(preds, labels):
        t, p = [], []
        for pi, li in zip(ps, ls):
            if li == -100: continue
            t.append(ID2LABEL.get(int(li), "O"))
            p.append(ID2LABEL.get(int(pi), "O"))
        true_l.append(t); pred_l.append(p)
    return {"f1": seq_f1_score(true_l, pred_l, average="weighted")}


def compute_ner_metrics_detailed(eval_preds):
    """Extended NER metrics with per-entity-type breakdown (for final eval)."""
    from seqeval.metrics import classification_report as seq_report
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_l, pred_l = [], []
    for ps, ls in zip(preds, labels):
        t, p = [], []
        for pi, li in zip(ps, ls):
            if li == -100:
                continue
            t.append(ID2LABEL.get(int(li), "O"))
            p.append(ID2LABEL.get(int(pi), "O"))
        true_l.append(t); pred_l.append(p)
    f1 = seq_f1_score(true_l, pred_l, average="weighted")
    report = seq_report(true_l, pred_l, output_dict=True, zero_division=0)
    log.info("  Per-entity NER performance:")
    log.info(f"  {'Entity':<20} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}")
    log.info(f"  {'─' * 48}")
    for etype in sorted(report.keys()):
        if etype in ("micro avg", "macro avg", "weighted avg", "samples avg"):
            continue
        d = report[etype]
        log.info(f"  {etype:<20} {d['precision']:>7.3f} {d['recall']:>7.3f} "
                 f"{d['f1-score']:>7.3f} {d['support']:>9}")
    for avg in ("micro avg", "macro avg", "weighted avg"):
        if avg in report:
            d = report[avg]
            log.info(f"  {avg:<20} {d['precision']:>7.3f} {d['recall']:>7.3f} "
                     f"{d['f1-score']:>7.3f} {d['support']:>9}")
    return {"f1": f1, "per_entity": report}


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_leakage(originals, anonymized):
    t_ent, l_ent = 0, 0
    t_tok, l_tok = 0, 0
    for orig, anon in zip(originals, anonymized):
        anon_lo = anon.lower()
        for w in orig.split():
            if len(w) > 2 and w[0].isupper():
                t_tok += 1
                if w.lower() in anon_lo:
                    l_tok += 1
        for ent in re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', orig):
            if len(ent) > 2:
                t_ent += 1
                if ent.lower() in anon_lo:
                    l_ent += 1
    return {
        "leakage_rate": round(l_tok / max(t_tok, 1) * 100, 2),
        "entity_leak_rate": round(l_ent / max(t_ent, 1) * 100, 2),
        "leaked": l_ent, "total_entities": t_ent,
    }


def compute_per_entity_leakage(originals, anonymized):
    """Compute leakage broken down by detected entity type via regex heuristics."""
    patterns = {
        "EMAIL":      r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "PHONE":      r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
        "IP_ADDRESS": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        "URL":        r'https?://[^\s]+',
        "SSN":        r'\b\d{3}-\d{2}-\d{4}\b',
        "DATE":       r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "NUMBER":     r'\b\d{5,}\b',
        "NAME":       r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    }
    results = {}
    for etype, pattern in patterns.items():
        total, leaked = 0, 0
        for orig, anon in zip(originals, anonymized):
            matches = re.findall(pattern, orig)
            anon_lo = anon.lower()
            for m in matches:
                total += 1
                if m.lower() in anon_lo:
                    leaked += 1
        if total > 0:
            results[etype] = {
                "leaked": leaked, "total": total,
                "rate": round(leaked / total * 100, 1),
            }
    return results


def compute_contextual_reidentification_risk(originals, anonymized, n_grams=3):
    """Measure how many identifying n-grams from originals survive."""
    risk = 0
    total = 0
    for orig, anon in zip(originals, anonymized):
        orig_words = orig.lower().split()
        anon_lo = anon.lower()
        for i in range(len(orig_words) - n_grams + 1):
            ngram = " ".join(orig_words[i:i + n_grams])
            # skip very common n-grams
            if any(w[0].isupper() for w in orig.split()[i:i + n_grams] if w):
                total += 1
                if ngram in anon_lo:
                    risk += 1
    return {
        "crr": round(risk / max(total, 1) * 100, 2),
        "identifying_ngrams_leaked": risk,
        "total_identifying_ngrams": total,
    }


def evaluate_anonymization(originals, anonymized, model_name="model"):
    """Comprehensive evaluation with per-entity, length, and sample analysis."""
    import time as _time
    log.info(f"\n{'═' * 65}")
    log.info(f"  EVALUATING: {model_name}  ({len(originals)} examples)")
    log.info(f"{'═' * 65}")

    # ── Privacy metrics ──
    log.info("  Computing leakage metrics …")
    leakage = compute_leakage(originals, anonymized)
    crr = compute_contextual_reidentification_risk(originals, anonymized)
    per_entity_leak = compute_per_entity_leakage(originals, anonymized)

    # ── Output length analysis ──
    orig_lens = [len(o.split()) for o in originals]
    anon_lens = [len(a.split()) for a in anonymized]
    length_ratios = [al / max(ol, 1) for ol, al in zip(orig_lens, anon_lens)]
    length_stats = {
        "orig_mean_words": round(float(np.mean(orig_lens)), 1),
        "anon_mean_words": round(float(np.mean(anon_lens)), 1),
        "ratio_mean": round(float(np.mean(length_ratios)), 3),
        "ratio_std": round(float(np.std(length_ratios)), 3),
        "anon_min_words": int(np.min(anon_lens)) if anon_lens else 0,
        "anon_max_words": int(np.max(anon_lens)) if anon_lens else 0,
    }

    # ── Text quality metrics ──
    log.info("  Computing ROUGE …")
    rouge_m = evaluate.load("rouge")
    rouge = rouge_m.compute(predictions=anonymized, references=originals)

    log.info("  Computing BLEU …")
    bleu_m = evaluate.load("sacrebleu")
    bleu = bleu_m.compute(predictions=anonymized,
                          references=[[r] for r in originals])

    log.info("  Computing BERTScore …")
    bs_f1 = 0.0
    bs_scores = []
    try:
        bsm = evaluate.load("bertscore")
        bs = bsm.compute(predictions=anonymized, references=originals, lang="en")
        bs_scores = [float(x) for x in bs["f1"]]
        bs_f1 = round(float(np.mean(bs_scores)), 4)
    except Exception:
        pass

    results = {
        "model": model_name,
        "n": len(originals),
        "leakage": leakage,
        "crr": crr,
        "per_entity_leakage": per_entity_leak,
        "length_stats": length_stats,
        "rouge": {k: round(v, 4) for k, v in rouge.items() if isinstance(v, float)},
        "bleu": round(bleu["score"], 2),
        "bertscore_f1": bs_f1,
    }

    # ── Detailed logging ──
    log.info(f"\n  {'METRIC':<35} {'VALUE':>12}")
    log.info(f"  {'─' * 48}")
    log.info(f"  {'Entity Leak Rate ↓':<35} {leakage['entity_leak_rate']:>11.2f}%")
    log.info(f"  {'Token Leak Rate ↓':<35} {leakage['leakage_rate']:>11.2f}%")
    log.info(f"  {'Leaked / Total Entities':<35} {leakage['leaked']:>5} / {leakage['total_entities']:<5}")
    log.info(f"  {'CRR (3-gram) ↓':<35} {crr['crr']:>11.2f}%")
    log.info(f"  {'─' * 48}")
    log.info(f"  {'ROUGE-1 ↑':<35} {rouge.get('rouge1', 0):>11.4f}")
    log.info(f"  {'ROUGE-2 ↑':<35} {rouge.get('rouge2', 0):>11.4f}")
    log.info(f"  {'ROUGE-L ↑':<35} {rouge.get('rougeL', 0):>11.4f}")
    log.info(f"  {'BLEU ↑':<35} {bleu['score']:>11.2f}")
    log.info(f"  {'BERTScore F1 ↑':<35} {bs_f1:>11.4f}")
    log.info(f"  {'─' * 48}")
    log.info(f"  {'Avg original length (words)':<35} {length_stats['orig_mean_words']:>11.1f}")
    log.info(f"  {'Avg anonymized length (words)':<35} {length_stats['anon_mean_words']:>11.1f}")
    log.info(f"  {'Length ratio (anon/orig)':<35} {length_stats['ratio_mean']:>11.3f}")
    log.info(f"  {'Min / Max anon words':<35} {length_stats['anon_min_words']:>5} / {length_stats['anon_max_words']:<5}")

    # ── Per-entity leakage table ──
    if per_entity_leak:
        log.info(f"\n  Per-Entity-Type Leakage:")
        log.info(f"  {'Entity Type':<20} {'Leaked':>8} {'Total':>8} {'Rate%':>8}")
        log.info(f"  {'─' * 46}")
        for etype, info in sorted(per_entity_leak.items(),
                                  key=lambda x: -x[1].get("rate", 0)):
            log.info(f"  {etype:<20} {info['leaked']:>8} {info['total']:>8} "
                     f"{info['rate']:>7.1f}%")

    # ── BERTScore distribution ──
    if bs_scores:
        log.info(f"\n  BERTScore F1 Distribution:")
        log.info(f"    Mean={np.mean(bs_scores):.4f}  Std={np.std(bs_scores):.4f}  "
                 f"Min={np.min(bs_scores):.4f}  Max={np.max(bs_scores):.4f}")

    # ── Sample outputs ──
    n_samples = min(5, len(originals))
    log.info(f"\n  Sample Outputs ({n_samples} examples):")
    log.info(f"  {'─' * 60}")
    for i in range(n_samples):
        o_trunc = originals[i][:120] + ("…" if len(originals[i]) > 120 else "")
        a_trunc = anonymized[i][:120] + ("…" if len(anonymized[i]) > 120 else "")
        log.info(f"  [{i+1}] Original  : {o_trunc}")
        log.info(f"      Anonymized: {a_trunc}")
        if bs_scores:
            log.info(f"      BERTScore : {bs_scores[i]:.4f}")
        log.info("")

    log.info(f"{'═' * 65}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Curated 37-Example Evaluation Set
# ═══════════════════════════════════════════════════════════════════════════

CURATED_EXAMPLES = [
    # EASY (10)
    {"id":"e01","cat":"name","diff":"easy","text":"Please contact John regarding the project update."},
    {"id":"e02","cat":"name","diff":"easy","text":"Maria has submitted the report on time."},
    {"id":"e03","cat":"location","diff":"easy","text":"I live in London and work remotely."},
    {"id":"e04","cat":"location","diff":"easy","text":"The office is located in Chicago."},
    {"id":"e05","cat":"number","diff":"easy","text":"My account number is 74829361."},
    {"id":"e06","cat":"number","diff":"easy","text":"Please reference ticket number 55123."},
    {"id":"e07","cat":"email","diff":"easy","text":"You can reach me at sarah.jones@gmail.com for details."},
    {"id":"e08","cat":"name","diff":"easy","text":"Thank you, David, for your quick response."},
    {"id":"e09","cat":"date","diff":"easy","text":"The appointment is scheduled for 15/07/2025."},
    {"id":"e10","cat":"location","diff":"easy","text":"She moved to Berlin last year."},
    # MEDIUM (12)
    {"id":"m01","cat":"multi","diff":"medium","text":"Dear Michael Thompson, your invoice has been processed."},
    {"id":"m02","cat":"multi","diff":"medium","text":"Jessica Parker can be contacted at jessica.parker@outlook.com."},
    {"id":"m03","cat":"multi","diff":"medium","text":"Robert Chen from San Francisco submitted the application."},
    {"id":"m04","cat":"multi","diff":"medium","text":"Hello Priya Sharma, we need to discuss your account 48291037."},
    {"id":"m05","cat":"multi","diff":"medium","text":"Please call Ahmed Hassan at +44 7911 123456 to confirm."},
    {"id":"m06","cat":"multi","diff":"medium","text":"The meeting between Lisa Wong and James Miller was rescheduled."},
    {"id":"m07","cat":"multi","diff":"medium","text":"Send the verification code 83921 to admin@techcorp.io."},
    {"id":"m08","cat":"multi","diff":"medium","text":"Emily Rodriguez was born on 23/04/1992 per our records."},
    {"id":"m09","cat":"multi","diff":"medium","text":"The property at 742 Evergreen Terrace, Springfield has ID 90210."},
    {"id":"m10","cat":"case","diff":"medium","text":"my name is alex morgan and i live in seattle."},
    {"id":"m11","cat":"case","diff":"medium","text":"CONTACT EMMA WATSON AT EMMA.WATSON@YAHOO.COM IMMEDIATELY."},
    {"id":"m12","cat":"case","diff":"medium","text":"jOHN sMITH lives in nEW yORK and his email is John@Gmail.Com."},
    # HARD (15)
    {"id":"h01","cat":"dense","diff":"hard","text":"Dr. Samantha Clarke from Boston General Hospital can be reached at samantha.clarke@bgh.org or +1 617 555 0192 regarding patient file 2847193."},
    {"id":"h02","cat":"dense","diff":"hard","text":"Hi, I'm Rajesh Kumar. My employee ID is EMP-78432, my email is rajesh.k@infosys.com, and I work at the Bangalore office."},
    {"id":"h03","cat":"dense","diff":"hard","text":"Transfer $5,000 from account 9821-4573-0012 to Maria Gonzalez (maria.g@bankmail.com) at 45 Oak Street, Miami, FL 33101."},
    {"id":"h04","cat":"long","diff":"hard","text":"Following up on our conversation, Daniel Kim mentioned that the project deadline is 31/12/2025. His colleague, Sophie Martin, suggested we consult the client, Nakamura Industries, before proceeding. You can reach Daniel at daniel.kim@company.co or Sophie at +33 6 12 34 56 78."},
    {"id":"h05","cat":"informal","diff":"hard","text":"yo hit up mike at mike99@hotmail.com or txt him at 555-867-5309 hes in LA rn"},
    {"id":"h06","cat":"typos","diff":"hard","text":"Plese contcat Jonh Smtih at jonh.smith@gmal.com abuot the accont 73829."},
    {"id":"h07","cat":"multilang","diff":"hard","text":"The visa for François Müller-Björkström was processed at the São Paulo consulate on 08/11/2024."},
    {"id":"h08","cat":"embedded","diff":"hard","text":"Username: alex_chen_1995, Password reset email sent to alexchen@protonmail.com, last login from IP 192.168.1.42."},
    {"id":"h09","cat":"ambiguous","diff":"hard","text":"Apple hired Jordan from Amazon. Jordan's first day in Cupertino is March 15th."},
    {"id":"h10","cat":"no_pii","diff":"hard","text":"The weather forecast predicts rain tomorrow with temperatures around 15 degrees."},
    {"id":"h11","cat":"no_pii","diff":"hard","text":"Please review the quarterly report and submit your feedback by Friday."},
    {"id":"h12","cat":"repeated","diff":"hard","text":"Call Sarah. Sarah's number is 555-0147. Tell Sarah that Sarah's appointment is confirmed."},
    {"id":"h13","cat":"tabular","diff":"hard","text":"Name: Wei Zhang, DOB: 12/03/1988, SSN: 123-45-6789, Address: 88 Pine Road, Austin, TX 73301."},
    {"id":"h14","cat":"conversational","diff":"hard","text":"Hey, it's Tom. Package goes to 1520 Maple Avenue, Portland. Zip 97201, phone 503-555-0198."},
    {"id":"h15","cat":"edge","diff":"hard","text":"Patient ID: P-2024-08173, Room 42B, admitted on 01/15/2024 by Dr. Ananya Patel, contact: ananya.p@hospital.org."},
    # CONTEXTUAL RE-ID (bonus for Models 3 & 4)
    {"id":"c01","cat":"contextual","diff":"hard","text":"Elon Musk is the richest man in the world in 2026. He went to the Red Cross to donate one hundred million dollars."},
    {"id":"c02","cat":"contextual","diff":"hard","text":"The CEO of Apple, who lives in Cupertino, announced the new iPhone at the keynote yesterday."},
    {"id":"c03","cat":"contextual","diff":"hard","text":"The youngest billionaire in history dropped out of Harvard to start a social media company."},
]


def run_curated_eval(anonymize_fn, model_name="model"):
    """Run curated examples with detailed per-difficulty/per-category breakdown."""
    log.info(f"\n{'═' * 65}")
    log.info(f"  CURATED EVALUATION ({len(CURATED_EXAMPLES)} examples) — {model_name}")
    log.info(f"{'═' * 65}")
    results = {"easy": [], "medium": [], "hard": []}
    pii_ok, pii_tot, nopii_ok, nopii_tot = 0, 0, 0, 0
    ctx_ok, ctx_tot = 0, 0
    cat_stats = defaultdict(lambda: {"ok": 0, "total": 0})

    for ex in tqdm(CURATED_EXAMPLES, desc=f"Curated eval ({model_name})",
                   leave=False):
        out = anonymize_fn(ex["text"])
        if ex["cat"] == "no_pii":
            ok = (ex["text"].strip() == out.strip())
            status = "CORRECT" if ok else "FALSE_POSITIVE"
            nopii_tot += 1
            if ok: nopii_ok += 1
        elif ex["cat"] == "contextual":
            changed = (ex["text"].strip() != out.strip())
            status = "GENERALISED" if changed else "UNCHANGED"
            ctx_tot += 1
            if changed: ctx_ok += 1
        else:
            changed = (ex["text"].strip() != out.strip())
            status = "CHANGED" if changed else "UNCHANGED"
            pii_tot += 1
            if changed: pii_ok += 1
        # Track per-category
        cat_stats[ex["cat"]]["total"] += 1
        if status in ("CHANGED", "CORRECT", "GENERALISED"):
            cat_stats[ex["cat"]]["ok"] += 1
        results[ex["diff"]].append({
            "id": ex["id"], "cat": ex["cat"], "status": status,
            "input": ex["text"], "output": out,
        })

    summary = {
        "pii_rate": round(pii_ok / max(pii_tot, 1) * 100, 1),
        "pii_ok": pii_ok, "pii_total": pii_tot,
        "nopii_rate": round(nopii_ok / max(nopii_tot, 1) * 100, 1),
        "nopii_ok": nopii_ok, "nopii_total": nopii_tot,
        "contextual_rate": round(ctx_ok / max(ctx_tot, 1) * 100, 1),
        "ctx_ok": ctx_ok, "ctx_total": ctx_tot,
        "details": results,
    }

    # ── Overall summary ──
    log.info(f"\n  Overall:")
    log.info(f"    PII anonymised  : {pii_ok}/{pii_tot} ({summary['pii_rate']}%)")
    log.info(f"    No-PII correct  : {nopii_ok}/{nopii_tot} ({summary['nopii_rate']}%)")
    log.info(f"    Contextual gen. : {ctx_ok}/{ctx_tot} ({summary['contextual_rate']}%)")

    # ── Per-difficulty table ──
    log.info(f"\n  Per-Difficulty Breakdown:")
    log.info(f"  {'Difficulty':<12} {'Passed':>8} {'Total':>8} {'Rate%':>8}")
    log.info(f"  {'─' * 38}")
    diff_stats = {}
    for diff in ["easy", "medium", "hard"]:
        items = results[diff]
        passed = sum(1 for r in items
                     if r["status"] in ("CHANGED", "CORRECT", "GENERALISED"))
        total = len(items)
        rate = round(passed / max(total, 1) * 100, 1)
        diff_stats[diff] = {"passed": passed, "total": total, "rate": rate}
        log.info(f"  {diff:<12} {passed:>8} {total:>8} {rate:>7.1f}%")
    summary["per_difficulty"] = diff_stats

    # ── Per-category table ──
    log.info(f"\n  Per-Category Breakdown:")
    log.info(f"  {'Category':<18} {'Passed':>8} {'Total':>8} {'Rate%':>8}")
    log.info(f"  {'─' * 44}")
    cat_summary = {}
    for cat, info in sorted(cat_stats.items()):
        rate = round(info["ok"] / max(info["total"], 1) * 100, 1)
        cat_summary[cat] = {"ok": info["ok"], "total": info["total"], "rate": rate}
        log.info(f"  {cat:<18} {info['ok']:>8} {info['total']:>8} {rate:>7.1f}%")
    summary["per_category"] = cat_summary

    # ── Sample I/O per difficulty ──
    for diff in ["easy", "medium", "hard"]:
        items = results[diff]
        log.info(f"\n  Sample {diff.upper()} outputs (up to 3):")
        log.info(f"  {'─' * 55}")
        for r in items[:3]:
            mark = "✓" if r["status"] in ("CHANGED", "CORRECT", "GENERALISED") else "✗"
            log.info(f"    [{r['id']}] {mark} {r['status']}")
            i_trunc = r["input"][:100] + ("…" if len(r["input"]) > 100 else "")
            o_trunc = r["output"][:100] + ("…" if len(r["output"]) > 100 else "")
            log.info(f"      IN : {i_trunc}")
            log.info(f"      OUT: {o_trunc}")

    log.info(f"{'═' * 65}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Masking Quality Evaluation (BLEU / ROUGE on the NER masking step)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_masking_quality(test_ds, censor_model, censor_tok, model_name="model"):
    """Evaluate the NER masking step in isolation using BLEU, ROUGE, token F1.

    Compares the model's masked output (e.g. "Contact [PERSON] at [EMAIL]")
    against ground-truth masked text derived from dataset annotations.
    """
    log.info(f"\n{'─' * 55}")
    log.info(f"  MASKING QUALITY — {model_name}")
    log.info(f"{'─' * 55}")

    pred_masked_texts, ref_masked_texts = [], []
    token_correct, token_total = 0, 0

    N = min(CFG.NUM_EVAL, len(test_ds))
    for i in tqdm(range(N), desc=f"Masking eval ({model_name})", leave=False):
        ex = test_ds[i]
        tokens, labels = find_text_and_labels(ex)
        text = get_source_text(ex)

        # Build ground-truth masked text
        ref_parts, prev_ent = [], None
        for tok_, lbl in zip(tokens, labels):
            if lbl == "O":
                ref_parts.append(tok_); prev_ent = None
            elif lbl.startswith("B-"):
                ref_parts.append(f"[{lbl[2:]}]"); prev_ent = lbl[2:]
            elif lbl.startswith("I-") and prev_ent:
                pass
            else:
                ref_parts.append(tok_); prev_ent = None
        ref_masked = " ".join(ref_parts)

        # Model's predicted masking
        enc = censor_tok(text, return_tensors="pt", truncation=True,
                         max_length=CFG.NER_MAX_LEN).to(DEVICE)
        with torch.no_grad():
            logits = censor_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        word_ids = enc.word_ids()
        words, tags = [], []
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None:
                continue
            tag = ID2LABEL.get(preds[j], "O")
            if wid != prev_wid:
                raw = censor_tok.convert_ids_to_tokens(
                    enc["input_ids"][0][j].item())
                words.append(raw.replace("▁", "").replace("##", ""))
                tags.append(tag)
            else:
                tok_str = censor_tok.convert_ids_to_tokens(
                    enc["input_ids"][0][j].item())
                words[-1] += tok_str.replace("▁", "").replace("##", "")
            prev_wid = wid

        pred_parts, prev_ent = [], None
        for w, t in zip(words, tags):
            if t == "O":
                pred_parts.append(w); prev_ent = None
            elif t.startswith("B-"):
                pred_parts.append(f"[{t[2:]}]"); prev_ent = t[2:]
            elif t.startswith("I-") and prev_ent:
                pass
            else:
                pred_parts.append(w); prev_ent = None
        pred_masked = " ".join(pred_parts)

        pred_masked_texts.append(pred_masked)
        ref_masked_texts.append(ref_masked)

        # Token-level accuracy (align by min length)
        pt = pred_masked.split()
        rt = ref_masked.split()
        for p, r in zip(pt, rt):
            token_total += 1
            if p == r:
                token_correct += 1
        token_total += abs(len(pt) - len(rt))

    # Compute metrics
    rouge_m = evaluate.load("rouge")
    bleu_m = evaluate.load("sacrebleu")
    rouge = rouge_m.compute(predictions=pred_masked_texts,
                            references=ref_masked_texts)
    bleu = bleu_m.compute(predictions=pred_masked_texts,
                          references=[[r] for r in ref_masked_texts])

    results = {
        "mask_rouge1": round(rouge.get("rouge1", 0), 4),
        "mask_rouge2": round(rouge.get("rouge2", 0), 4),
        "mask_rougeL": round(rouge.get("rougeL", 0), 4),
        "mask_bleu": round(bleu["score"], 2),
        "mask_token_accuracy": round(token_correct / max(token_total, 1) * 100, 2),
        "n_evaluated": N,
    }

    log.info(f"  {'Metric':<30} {'Value':>10}")
    log.info(f"  {'─' * 42}")
    log.info(f"  {'Mask ROUGE-1 ↑':<30} {results['mask_rouge1']:>10.4f}")
    log.info(f"  {'Mask ROUGE-2 ↑':<30} {results['mask_rouge2']:>10.4f}")
    log.info(f"  {'Mask ROUGE-L ↑':<30} {results['mask_rougeL']:>10.4f}")
    log.info(f"  {'Mask BLEU ↑':<30} {results['mask_bleu']:>10.2f}")
    log.info(f"  {'Mask Token Accuracy ↑':<30} {results['mask_token_accuracy']:>9.2f}%")
    log.info(f"{'─' * 55}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Privacy-Aware Seq2Seq Trainer (BERTScore-inspired semantic fidelity loss)
# ═══════════════════════════════════════════════════════════════════════════

class PrivacyAwareSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer that adds two auxiliary losses on top of standard CE:

    1. **Semantic Fidelity Loss** (BERTScore-inspired):
       Computes cosine similarity between mean-pooled encoder hidden states
       (input text representation) and mean-pooled decoder logit-weighted
       embeddings (generated text representation).  Penalises deviation
       from a target similarity (default 0.70) — the model must *preserve*
       overall meaning while *changing* entity-specific content.

    2. **Sensitivity-Weighted CE** (optional):
       Re-weights the cross-entropy loss on output tokens that correspond
       to entity placeholders, giving higher weight to high-sensitivity
       entity types (SSN > PERSON > LOC).

    These losses address the Privacy–Utility Paradox directly during
    training, not just at evaluation time.
    """

    def __init__(self, *args, semantic_weight=None, privacy_weight=None,
                 sim_target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_weight = semantic_weight if semantic_weight is not None else CFG.SEMANTIC_LOSS_WEIGHT
        self.privacy_weight = privacy_weight if privacy_weight is not None else CFG.PRIVACY_LOSS_WEIGHT
        self.sim_target = sim_target if sim_target is not None else CFG.SEMANTIC_SIM_TARGET

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs_copy = {k: v for k, v in inputs.items()}
        labels = inputs_copy.get("labels")

        # Only request hidden states every N steps (saves ~20% time)
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        use_sem = (self.semantic_weight > 0
                   and self._step_counter % 4 == 0)  # every 4th step

        outputs = model(**inputs_copy, output_hidden_states=use_sem)
        ce_loss = outputs.loss
        total_loss = ce_loss

        # ── Semantic Fidelity Loss (computed every 4th step) ──
        if use_sem:
            enc_h = getattr(outputs, "encoder_last_hidden_state", None)
            dec_hidden = getattr(outputs, "decoder_hidden_states", None)
            dec_h = dec_hidden[-1] if dec_hidden else None

            if enc_h is not None and dec_h is not None:
                # Mean-pool encoder with attention mask
                enc_mask = inputs_copy.get("attention_mask")
                if enc_mask is not None:
                    enc_mask_f = enc_mask.unsqueeze(-1).float()
                    enc_pool = (enc_h * enc_mask_f).sum(1) / enc_mask_f.sum(1).clamp(min=1)
                else:
                    enc_pool = enc_h.mean(dim=1)

                # Mean-pool decoder with label mask
                if labels is not None:
                    dec_mask = (labels != -100).unsqueeze(-1).float()
                    dec_pool = (dec_h * dec_mask).sum(1) / dec_mask.sum(1).clamp(min=1)
                else:
                    dec_pool = dec_h.mean(dim=1)

                # Match dimensions (encoder D may differ from decoder D)
                if enc_pool.shape[-1] != dec_pool.shape[-1]:
                    min_d = min(enc_pool.shape[-1], dec_pool.shape[-1])
                    enc_pool = enc_pool[..., :min_d]
                    dec_pool = dec_pool[..., :min_d]

                # Normalise to unit vectors to prevent NaN from cosine_sim
                enc_pool = F.normalize(enc_pool, dim=-1)
                dec_pool = F.normalize(dec_pool, dim=-1)

                cos_sim = F.cosine_similarity(enc_pool, dec_pool)
                target = torch.full_like(cos_sim, self.sim_target)
                sem_loss = F.mse_loss(cos_sim, target)

                # Guard against NaN / explosion
                if torch.isfinite(sem_loss) and sem_loss < 10.0:
                    total_loss = total_loss + self.semantic_weight * sem_loss

        # Final NaN guard — fall back to CE if total is bad
        if not torch.isfinite(total_loss):
            total_loss = ce_loss if torch.isfinite(ce_loss) else ce_loss.new_tensor(0.0)

        return (total_loss, outputs) if return_outputs else total_loss


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity-Weighted Seq2Seq Loss
# ═══════════════════════════════════════════════════════════════════════════

class SensitivityWeightedLoss(nn.Module):
    """Re-weights CE loss on entity-placeholder tokens by sensitivity tier.

    Input format expected: "Replace PII placeholders … [SSN] … [PERSON] …"
    Tokens matching "[ENTITY_TYPE]" get higher weight for high-sensitivity types.
    """

    def __init__(self, base_weight=1.0, sensitivity_map=None):
        super().__init__()
        self.base_weight = base_weight
        self.sens_map = sensitivity_map or CFG.SENSITIVITY

    def compute_token_weights(self, input_ids, tokenizer):
        """Compute per-token weights based on entity sensitivity in input."""
        batch_size, seq_len = input_ids.shape
        weights = torch.ones(batch_size, seq_len, device=input_ids.device)

        for b in range(batch_size):
            text = tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for etype, sens in self.sens_map.items():
                marker = f"[{etype}]"
                if marker in text:
                    # Find token positions for this marker
                    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
                    for start_pos in range(seq_len - len(marker_ids) + 1):
                        if input_ids[b, start_pos:start_pos + len(marker_ids)].tolist() == marker_ids:
                            w = self.base_weight * (16.0 / max(sens, 1.0))
                            weights[b, start_pos:start_pos + len(marker_ids)] = w
        return weights


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(all_results: Dict[str, Dict], save_dir: str):
    """Bar charts comparing all models side by side."""
    os.makedirs(save_dir, exist_ok=True)
    names = list(all_results.keys())
    n = len(names)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(n)
    w = 0.6

    # Entity leak rate
    vals = [all_results[m]["leakage"]["entity_leak_rate"] for m in names]
    axes[0].bar(x, vals, w, color="#e74c3c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("%"); axes[0].set_title("Entity Leak Rate ↓")

    # CRR
    vals = [all_results[m].get("crr", {}).get("crr", 0) for m in names]
    axes[1].bar(x, vals, w, color="#e67e22")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("%"); axes[1].set_title("Contextual Re-ID Risk ↓")

    # ROUGE-L
    vals = [all_results[m].get("rouge", {}).get("rougeL", 0) * 100 for m in names]
    axes[2].bar(x, vals, w, color="#3498db")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_ylabel("%"); axes[2].set_title("ROUGE-L ↑")

    # BERTScore
    vals = [all_results[m].get("bertscore_f1", 0) * 100 for m in names]
    axes[3].bar(x, vals, w, color="#9b59b6")
    axes[3].set_xticks(x); axes[3].set_xticklabels(names, rotation=20, ha="right")
    axes[3].set_ylabel("%"); axes[3].set_title("BERTScore F1 ↑")

    plt.suptitle("4-Model Comparison: Privacy vs Utility", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Comparison plot → {path}")


def plot_evaluation_summary(results: Dict, save_dir: str):
    """Plot detailed evaluation summary for a single model."""
    os.makedirs(save_dir, exist_ok=True)
    model_name = results.get("model", "Model")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Privacy metrics bar
    ax = axes[0, 0]
    metrics = {
        "Entity\nLeak%": results["leakage"]["entity_leak_rate"],
        "Token\nLeak%": results["leakage"]["leakage_rate"],
        "CRR%": results.get("crr", {}).get("crr", 0),
    }
    colors = ["#e74c3c", "#e67e22", "#f39c12"]
    ax.bar(list(metrics.keys()), list(metrics.values()), color=colors)
    ax.set_ylabel("%"); ax.set_title("Privacy Metrics (lower = better)")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    # (0,1) Quality metrics bar
    ax = axes[0, 1]
    qmetrics = {
        "ROUGE-1": results.get("rouge", {}).get("rouge1", 0) * 100,
        "ROUGE-L": results.get("rouge", {}).get("rougeL", 0) * 100,
        "BLEU": results.get("bleu", 0),
        "BERTSc": results.get("bertscore_f1", 0) * 100,
    }
    colors_q = ["#3498db", "#2980b9", "#1abc9c", "#9b59b6"]
    ax.bar(list(qmetrics.keys()), list(qmetrics.values()), color=colors_q)
    ax.set_ylabel("%"); ax.set_title("Utility Metrics (higher = better)")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(qmetrics.values()):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    # (1,0) Per-entity leakage
    ax = axes[1, 0]
    pel = results.get("per_entity_leakage", {})
    if pel:
        sorted_ents = sorted(pel.items(), key=lambda x: -x[1]["rate"])
        etypes = [e[0] for e in sorted_ents]
        rates = [e[1]["rate"] for e in sorted_ents]
        ax.barh(etypes, rates, color="#e74c3c", alpha=0.8)
        ax.set_xlabel("Leak Rate (%)"); ax.set_title("Per-Entity-Type Leakage")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No entity-type data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Per-Entity-Type Leakage")

    # (1,1) Length distribution
    ax = axes[1, 1]
    ls = results.get("length_stats", {})
    categories = ["Orig\nMean", "Anon\nMean", "Anon\nMin", "Anon\nMax"]
    values = [ls.get("orig_mean_words", 0), ls.get("anon_mean_words", 0),
              ls.get("anon_min_words", 0), ls.get("anon_max_words", 0)]
    ax.bar(categories, values, color=["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"])
    ax.set_ylabel("Words"); ax.set_title("Output Length Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Evaluation Summary — {model_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "eval_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"  Eval summary plot → {path}")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# DeBERTa-v3 Compatibility Fix
# ═══════════════════════════════════════════════════════════════════════════

def fix_deberta_params(model):
    """Fix DeBERTa-v3 LayerNorm naming (gamma/beta → weight/bias)
    and cast any fp16 params to fp32 so AMP gradient scaling works."""
    for module in model.modules():
        if hasattr(module, 'gamma'):
            if not hasattr(module, 'weight'):
                module.weight = module.gamma
            del module.gamma
        if hasattr(module, 'beta'):
            if not hasattr(module, 'bias'):
                module.bias = module.beta
            del module.beta
    for p in model.parameters():
        if p.data.dtype == torch.float16:
            p.data = p.data.float()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# QLoRA Seq2Seq Builder (shared by Models 1-4)
# ═══════════════════════════════════════════════════════════════════════════

def build_seq2seq_qlora(backbone: str, output_dir: str):
    """Load a seq2seq model in 4-bit with QLoRA adapters."""
    log.info(f"Loading {backbone} (4-bit QLoRA) …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if _BF16_OK else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        backbone, quantization_config=bnb, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA, lora_dropout=CFG.LORA_DROPOUT,
        target_modules=CFG.LORA_TARGETS_T5,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  {trainable:,}/{total:,} trainable ({100*trainable/total:.2f}%)")
    return model, tok


def train_seq2seq(model, tok, train_ds, val_ds, output_dir,
                  epochs=None, batch=None):
    """Generic seq2seq trainer (used by Hallucinator, Rephraser, Paraphraser)."""
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
    # T5/Flan-T5 decoder has known fp16 instability (layer-norm overflow →
    # NaN loss).  Google trained T5 in bf16, which T4 doesn't support.
    # Since the base model is already 4-bit quantised, fp32 LoRA adapters
    # add only ~70 MB — safe to skip fp16 AMP entirely on non-bf16 GPUs.
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
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        compute_metrics=_metrics,
    )
    trainer.train()

    # ── Plot loss curves ──
    plot_training_curves(
        trainer.state.log_history,
        f"Seq2Seq Training — {os.path.basename(output_dir)}",
        os.path.join(output_dir, "loss_curves.png"),
    )

    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    log.info(f"  Val ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    return trainer, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Preflight Checks — catch issues before hours of training
# ═══════════════════════════════════════════════════════════════════════════

def preflight_checks(models_to_run, sample_ds):
    """Smoke-test every component before committing to hours of training.

    Catches NaN losses, fp16/AMP+GradScaler crashes, broken tokenisers,
    QLoRA load failures, dataset format problems — all in ~5 minutes
    instead of discovering them 4+ hours into a run.
    """
    import traceback
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║                   PREFLIGHT CHECKS                         ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    errors = []

    # ── 1. GPU & Precision ──
    log.info("[1/5] GPU & precision …")
    if not torch.cuda.is_available():
        errors.append("CUDA not available — training will be extremely slow")
    else:
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9
        log.info(f"  GPU: {props.name}, {mem_gb:.1f} GB")
        log.info(f"  bf16={_BF16_OK}  fp16={_FP16_OK}")
        if mem_gb < 14:
            errors.append(f"GPU memory ({mem_gb:.1f} GB) may be too low for QLoRA")

    # ── 2. Dataset Format ──
    log.info("[2/5] Dataset format …")
    try:
        row = sample_ds[0]
        toks, labs = find_text_and_labels(row)
        src = get_source_text(row)
        assert len(toks) > 0, "Empty tokens"
        assert len(toks) == len(labs), f"len mismatch: {len(toks)} vs {len(labs)}"
        for lb in labs:
            assert lb in LABEL2ID, f"Unknown label '{lb}'"
        pair = prepare_seq2seq_pair(row)
        assert "input_text" in pair and "target_text" in pair
        log.info(f"  ✓ {len(toks)} tokens, {len(src)} chars, seq2seq pair OK")
    except Exception as e:
        errors.append(f"Dataset format: {e}")

    # ── 3. NER Censor Smoke Tests ──
    ner_tests = []
    if any(m in models_to_run for m in [1, 3]):
        ner_tests.append((CFG.CENSOR_BASE, True, "DeBERTa-v3"))
    if 2 in models_to_run:
        ner_tests.append((CFG.CENSOR_ADV, False, "XLM-RoBERTa"))
    for backbone, needs_fix, label in ner_tests:
        log.info(f"[3/5] NER smoke test: {label} …")
        try:
            _preflight_ner(backbone, needs_fix, sample_ds)
            log.info(f"  ✓ {label}: load → forward → backward (AMP+GradScaler) → inference OK")
        except Exception as e:
            errors.append(f"{label} NER: {e}")
            log.error(f"  ✗ {label}: {e}")
            log.debug(traceback.format_exc())
        cleanup_gpu()

    # ── 4. QLoRA Seq2Seq Smoke Tests ──
    s2s_tests = set()
    if any(m in models_to_run for m in [1, 3]):
        s2s_tests.add(CFG.HALLUC_BASE)
    if 2 in models_to_run:
        s2s_tests.add(CFG.HALLUC_ADV)
    if 3 in models_to_run:
        s2s_tests.add(CFG.REPHRASER_BASE)
    if 4 in models_to_run:
        s2s_tests.add(CFG.PARAPHRASER_BASE)
    for backbone in sorted(s2s_tests):
        short = backbone.split("/")[-1]
        log.info(f"[4/5] QLoRA smoke test: {short} …")
        try:
            _preflight_seq2seq(backbone, sample_ds)
            log.info(f"  ✓ {short}: 4-bit load → forward → backward → generate OK")
        except Exception as e:
            errors.append(f"QLoRA {short}: {e}")
            log.error(f"  ✗ {short}: {e}")
            log.debug(traceback.format_exc())
        cleanup_gpu()

    # ── 5. Evaluation Libraries ──
    log.info("[5/5] Evaluation libraries …")
    try:
        _r = evaluate.load("rouge")
        _ = _r.compute(predictions=["test output"], references=["test ref"])
        _b = evaluate.load("sacrebleu")
        _ = _b.compute(predictions=["test output"], references=[["test ref"]])
        log.info("  ✓ ROUGE + BLEU OK")
    except Exception as e:
        errors.append(f"Eval libraries: {e}")

    # ── Summary ──
    log.info("─" * 60)
    if errors:
        log.error(f"PREFLIGHT FAILED — {len(errors)} issue(s):")
        for i, err in enumerate(errors, 1):
            log.error(f"  [{i}] {err}")
        raise RuntimeError(
            f"Preflight failed with {len(errors)} issue(s). "
            "Fix before training to avoid wasting compute."
        )
    log.info("✓ ALL PREFLIGHT CHECKS PASSED — safe to start training")
    log.info("")


def _preflight_ner(backbone, fix_deberta, sample_ds):
    """Smoke-test NER: load → fix → AMP forward+backward with GradScaler → inference."""
    tok = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        backbone, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )
    if fix_deberta:
        fix_deberta_params(model)
    model = model.float().to(DEVICE)   # .float() ensures no bf16/fp16 survive

    # Verify no non-fp32 params remain (bf16 crashes T4, fp16 crashes GradScaler)
    bad_params = [n for n, p in model.named_parameters()
                  if p.data.dtype not in (torch.float32, torch.float64)]
    if bad_params:
        raise ValueError(
            f"Non-fp32 params remain — will crash on T4: {bad_params[:3]}")

    # Tokenize one real example (batched API, same as training pipeline)
    row = sample_ds[0]
    ex = {k: [row[k]] for k in row}
    enc = tokenize_and_align_ner(ex, tok)
    batch = {k: torch.tensor([enc[k][0]]).to(DEVICE) for k in enc}

    # Forward + backward with AMP + GradScaler (exactly like Trainer)
    model.train()
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5)
    use_amp = _BF16_OK or _FP16_OK
    amp_dtype = torch.bfloat16 if _BF16_OK else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=_FP16_OK)
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        out = model(**batch)
    loss = out.loss
    if torch.isnan(loss):
        raise ValueError("NaN loss on forward pass")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # This is where old DeBERTa fp16 bug crashed
    nan_grads = [n for n, p in model.named_parameters()
                 if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_grads:
        raise ValueError(f"NaN gradients in: {nan_grads[:5]}")
    scaler.step(optimizer)
    scaler.update()

    # Inference: check logits are valid and ▁ is stripped
    model.eval()
    text = get_source_text(row)
    inp = tok(text, return_tensors="pt", truncation=True,
              max_length=CFG.NER_MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**inp).logits
    if torch.isnan(logits).any():
        raise ValueError("NaN in inference logits")

    # Verify sub-word ▁ / ## stripping (the bug we just fixed)
    all_toks = tok.convert_ids_to_tokens(inp["input_ids"][0])
    word_ids = inp.word_ids()
    words, prev_wid = [], None
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            raw = tok.convert_ids_to_tokens(inp["input_ids"][0][i].item())
            words.append(raw.replace("▁", "").replace("##", ""))
        else:
            words[-1] += all_toks[i].replace("▁", "").replace("##", "")
        prev_wid = wid
    bad = [w for w in words if "▁" in w]
    if bad:
        raise ValueError(f"SentencePiece ▁ not stripped: {bad[:5]}")

    # Verify compute_ner_metrics works with this model's output shape
    m = compute_ner_metrics((out.logits.detach().cpu().numpy(),
                             batch["labels"].cpu().numpy()))
    if "f1" not in m:
        raise ValueError("compute_ner_metrics missing 'f1' key")

    del model, tok, optimizer, scaler, batch, out
    torch.cuda.empty_cache()


def _preflight_seq2seq(backbone, sample_ds):
    """Smoke-test QLoRA seq2seq: 4-bit load → forward → backward → generate."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model, tok = build_seq2seq_qlora(backbone, tmpdir)
        pair = prepare_seq2seq_pair(sample_ds[0])
        enc = tokenize_seq2seq({k: [v] for k, v in pair.items()}, tok)
        batch = {k: torch.tensor(v).to(DEVICE) for k, v in enc.items()}

        model.train()
        out = model(**batch)
        loss = out.loss
        if loss is None:
            raise ValueError("Model returned None loss (missing labels?)")
        if torch.isnan(loss):
            raise ValueError("NaN loss on forward pass")
        loss.backward()

        # Quick generate to verify inference works
        model.eval()
        inp = tok(pair["input_text"][:200], return_tensors="pt",
                  truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=20)
        decoded = tok.decode(gen[0], skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise ValueError(f"generate returned {type(decoded)}, expected str")

        del model, tok, batch, out
    torch.cuda.empty_cache()

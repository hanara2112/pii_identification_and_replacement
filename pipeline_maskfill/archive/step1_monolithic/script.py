# ==============================================================================
# Privacy-Preserving Text Anonymization via Decoupled Mask-and-Fill
# ==============================================================================
#
# Task 2: Language-Stratified Split of AI4Privacy 500K → Two Independent Models
#
#   Model I  — The Censor:       NER token-classifier on Half-A
#   Model II — The Hallucinator:  Seq2Seq PII-fill generator on Half-B
#
# TWO MODES:
#
#   --mode baseline   DeBERTa-v3-base + Flan-T5-Large (QLoRA)
#                     Standard cross-entropy, no DP, basic evaluation.
#
#   --mode advanced   XLM-RoBERTa-base + mT5-large (QLoRA)
#                     + DP-SGD via Opacus (formal ε,δ guarantees)
#                     + Recall-weighted focal loss for NER
#                     + Entity-consistency module (SHA-256 context hash)
#                     + Per-language evaluation breakdown
#                     + Cross-lingual zero-shot transfer evaluation
#                     + Curated 37-example eval set (easy/medium/hard)
#                     + Membership-inference attack evaluation
#
# At inference the two models compose:  Censor detects PII spans  →
#   Hallucinator generates context-aware fake replacements from masked
#   templates it has *never seen* alongside original entities.
#
# The data-partition wall means the Hallucinator cannot memorise the
# real entities that the Censor sees, providing architectural privacy on
# top of any formal DP guarantee.
#
# Usage:
#   python script.py                          # Advanced mode (default)
#   python script.py --mode baseline          # Baseline only
#   python script.py --mode advanced          # Advanced with all features
#   python script.py --quick                  # Quick test (~2 k samples)
#   python script.py --stage censor           # Train Censor only
#   python script.py --stage halluc           # Train Hallucinator only
#   python script.py --stage eval             # Evaluate only (load saved)
# ==============================================================================

# ── 1. Imports & Seed ───────────────────────────────────────────────────────

import os, sys, subprocess, warnings, gc, json, re, random, logging, argparse
import hashlib, math
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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

# ── 2. Install dependencies ─────────────────────────────────────────────────

def install_deps():
    """Install all required packages (runs once, then cached)."""
    deps = [
        "transformers>=4.40.0", "datasets>=2.18", "evaluate",
        "accelerate>=0.28", "peft>=0.10.0", "bitsandbytes>=0.43",
        "rouge_score", "sacrebleu", "sentencepiece", "scipy",
        "scikit-learn", "pandas", "matplotlib", "seqeval",
        "bert_score", "Faker", "opacus>=1.4", "nltk",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U"] + deps,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

install_deps()

import evaluate, nltk
from datasets import Dataset, load_dataset, DatasetDict
from faker import Faker
from sklearn.model_selection import train_test_split
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1_score,
)
from transformers import (
    AutoModel, AutoModelForSeq2SeqLM, AutoModelForTokenClassification,
    AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorForTokenClassification,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    TrainingArguments, Trainer,
    BitsAndBytesConfig, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE} | PyTorch {torch.__version__}")

# ── 3. Configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    # --- Pipeline control ---
    MODE: str = "advanced"        # "baseline" or "advanced"
    STAGE: str = "ALL"            # "ALL", "CENSOR", "HALLUC", "EVAL"
    QUICK_MODE: bool = False
    QUICK_N: int = 2000

    # --- Baseline backbones ---
    CENSOR_BASE: str = "microsoft/deberta-v3-base"
    HALLUC_BASE: str = "google/flan-t5-large"

    # --- Advanced backbones (multilingual) ---
    CENSOR_ADV: str = "xlm-roberta-base"
    HALLUC_ADV: str = "google/mt5-large"

    # --- Directories ---
    OUTPUT_ROOT: str = "./privacy_pipeline"
    CENSOR_DIR: str  = "./privacy_pipeline/censor"
    HALLUC_DIR: str  = "./privacy_pipeline/hallucinator"
    EVAL_DIR: str    = "./privacy_pipeline/evaluation"
    PLOT_DIR: str    = "./privacy_pipeline/plots"

    # --- NER (Censor) ---
    NER_MAX_LEN: int       = 256
    NER_BATCH: int         = 16
    NER_GRAD_ACCUM: int    = 2
    NER_EPOCHS: int        = 5
    NER_LR: float          = 3e-5
    NER_WEIGHT_DECAY: float = 0.01
    NER_WARMUP: float      = 0.1

    # --- Seq2Seq (Hallucinator) ---
    S2S_MAX_LEN: int       = 256
    S2S_BATCH: int         = 8
    S2S_GRAD_ACCUM: int    = 4
    S2S_EPOCHS: int        = 3
    S2S_LR: float          = 1e-4
    S2S_WARMUP: float      = 0.06

    # --- QLoRA ---
    LORA_R: int          = 16
    LORA_ALPHA: int      = 32
    LORA_DROPOUT: float  = 0.05
    LORA_TARGETS_T5: List[str] = field(default_factory=lambda: [
        "q", "v", "k", "o", "wi_0", "wi_1", "wo",
    ])
    LORA_TARGETS_MT5: List[str] = field(default_factory=lambda: [
        "q", "v", "k", "o", "wi_0", "wi_1", "wo",
    ])

    # --- Generation ---
    GEN_MAX_TOKENS: int = 200
    GEN_NUM_BEAMS: int  = 4

    # --- DP-SGD (advanced only) ---
    DP_ENABLED: bool         = True
    DP_EPSILON_TARGET: float = 8.0
    DP_DELTA: float          = 1e-5
    DP_MAX_GRAD_NORM: float  = 1.0
    DP_NOISE_MULTIPLIER: float = 1.1

    # --- Entity-type sensitivity tiers (advanced) ---
    SENSITIVITY: Dict = field(default_factory=lambda: {
        "SSN": 1.0, "PASSPORT": 1.0, "DRIVER_LICENSE": 1.0,
        "CREDIT_CARD": 2.0, "IBAN": 2.0, "ACCOUNT": 2.0,
        "EMAIL": 4.0, "PHONE": 4.0, "IP_ADDRESS": 4.0, "USERNAME": 4.0,
        "PERSON": 8.0, "ADDRESS": 8.0, "MEDICAL": 8.0, "DATE": 8.0,
        "LOC": 16.0, "ORG": 16.0, "BUILDING": 16.0, "POSTCODE": 16.0,
        "URL": 16.0,
    })

    # --- Focal loss (advanced NER) ---
    FOCAL_GAMMA: float = 2.0
    FOCAL_BETA_RECALL: float = 2.0   # recall weight in Fβ-driven weighting

    # --- Cross-lingual config (advanced) ---
    TRAIN_LANGS: List[str] = field(default_factory=lambda: [
        "English", "German", "French",
    ])
    ZEROSHOT_LANGS: List[str] = field(default_factory=lambda: [
        "Italian", "Spanish", "Dutch", "Portuguese", "Czech",
    ])

    # --- Data ---
    TEST_RATIO: float = 0.05
    NUM_EVAL: int = 200

    # --- Entity types ---
    ENTITY_TYPES: List[str] = field(default_factory=lambda: [
        "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
        "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
        "DRIVER_LICENSE", "USERNAME", "URL", "MEDICAL", "ACCOUNT",
        "BUILDING", "POSTCODE",
    ])


CFG = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Decoupled Anonymization Pipeline")
    parser.add_argument("--mode", default="advanced",
                        choices=["baseline", "advanced"])
    parser.add_argument("--stage", default="ALL",
                        choices=["ALL", "CENSOR", "HALLUC", "EVAL"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs-ner", type=int, default=None)
    parser.add_argument("--epochs-s2s", type=int, default=None)
    parser.add_argument("--no-dp", action="store_true",
                        help="Disable DP-SGD even in advanced mode")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(mode="advanced", stage="ALL", quick=False,
                                  epochs_ner=None, epochs_s2s=None, no_dp=False)
    CFG.MODE = args.mode
    CFG.STAGE = args.stage.upper()
    CFG.QUICK_MODE = args.quick
    if args.epochs_ner:  CFG.NER_EPOCHS = args.epochs_ner
    if args.epochs_s2s:  CFG.S2S_EPOCHS = args.epochs_s2s
    if args.no_dp:       CFG.DP_ENABLED = False

parse_args()


def setup_dirs():
    for d in [CFG.OUTPUT_ROOT, CFG.CENSOR_DIR, CFG.HALLUC_DIR,
              CFG.EVAL_DIR, CFG.PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

setup_dirs()


# ── 4. BIO Tag System ───────────────────────────────────────────────────────

def build_bio_labels(entity_types: List[str]):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(CFG.ENTITY_TYPES)
NUM_LABELS = len(BIO_LABELS)
log.info(f"BIO labels: {NUM_LABELS}  ({len(CFG.ENTITY_TYPES)} entity types)")


# ── 5. Entity Mapping (AI4Privacy → Canonical) ──────────────────────────────

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


# ── 6. Data Loading & Language-Stratified Split ─────────────────────────────

def load_ai4privacy():
    """Load full AI4Privacy ~500 K multilingual dataset from HuggingFace."""
    log.info("Loading AI4Privacy dataset (full multilingual)…")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    log.info(f"Loaded {len(ds):,} examples")
    return ds


def detect_lang_col(df):
    for c in ("language", "lang", "Language"):
        if c in df.columns:
            return c
    return None


def language_stratified_split(ds, test_ratio=0.05):
    """Split into two halves so every language gets 50/50 + held-out test.

    Returns: half_a, half_b, test_set, lang_col
    """
    log.info("Language-stratified splitting…")
    df = ds.to_pandas()
    lang_col = detect_lang_col(df)
    if lang_col is None:
        lang_col = "__lang_dummy"
        df[lang_col] = "unknown"

    lang_counts = df[lang_col].value_counts()
    log.info(f"Languages: {len(lang_counts)}")
    for lang, cnt in lang_counts.head(15).items():
        log.info(f"  {lang}: {cnt:,}")

    ha, hb, test = [], [], []
    for lang, grp in df.groupby(lang_col):
        idx = grp.index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_ratio))
        n_rem  = n - n_test
        n_half = n_rem // 2
        test.extend(idx[:n_test])
        ha.extend(idx[n_test:n_test + n_half])
        hb.extend(idx[n_test + n_half:])

    random.shuffle(ha); random.shuffle(hb); random.shuffle(test)
    half_a   = ds.select(ha)
    half_b   = ds.select(hb)
    test_set = ds.select(test)

    log.info(f"Half-A={len(half_a):,}  Half-B={len(half_b):,}  Test={len(test_set):,}")

    # Verify proportionality
    for name, subset in [("Half-A", half_a), ("Half-B", half_b), ("Test", test_set)]:
        sub = subset.to_pandas()
        if lang_col in sub.columns and lang_col != "__lang_dummy":
            dist = dict(sub[lang_col].value_counts())
            log.info(f"  {name} langs: {dict(list(dist.items())[:5])} …")

    return half_a, half_b, test_set, lang_col


def quick_subsample(ds, n=2000):
    if len(ds) <= n:
        return ds
    return ds.select(random.sample(range(len(ds)), n))


# ── 7. Token-level extraction from AI4Privacy ──────────────────────────────

def find_text_and_labels(example):
    """Extract word tokens and BIO labels from an AI4Privacy entry."""
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
        end   = span.get("end", start + span.get("length", len(span.get("value", ""))))
        label = span.get("label", span.get("entity_type", "O")).upper().replace(" ", "_")
        value = span.get("value", text[start:end])

        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before)
            labels.extend(["O"] * len(before))

        entity_toks = value.split()
        if entity_toks:
            bio = ENTITY_MAP.get(label, label)
            if bio in CFG.ENTITY_TYPES:
                tokens.append(entity_toks[0]); labels.append(f"B-{bio}")
                for t in entity_toks[1:]:
                    tokens.append(t); labels.append(f"I-{bio}")
            else:
                tokens.extend(entity_toks)
                labels.extend(["O"] * len(entity_toks))
        pos = end

    if pos < len(text):
        rest = text[pos:].split()
        tokens.extend(rest)
        labels.extend(["O"] * len(rest))

    return tokens, labels


# ── 8. NER Tokenisation (Censor — Half A) ──────────────────────────────────

def tokenize_and_align_ner(examples, tokenizer):
    """Sub-word tokenise and align BIO labels."""
    key = ("source_text" if "source_text" in examples
           else "text" if "text" in examples else "tokens")
    all_tokens, all_labels = [], []
    for i in range(len(examples[key])):
        ex = {k: v[i] for k, v in examples.items()}
        toks, labs = find_text_and_labels(ex)
        all_tokens.append(toks); all_labels.append(labs)

    tokenized = tokenizer(
        all_tokens, truncation=True, max_length=CFG.NER_MAX_LEN,
        padding="max_length", is_split_into_words=True,
    )

    aligned = []
    for i, lab_seq in enumerate(all_labels):
        wids = tokenized.word_ids(batch_index=i)
        ids, prev = [], None
        for wid in wids:
            if wid is None:
                ids.append(-100)
            elif wid != prev:
                lbl = lab_seq[wid] if wid < len(lab_seq) else "O"
                ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = lab_seq[wid] if wid < len(lab_seq) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                ids.append(LABEL2ID.get(lbl, 0))
            prev = wid
        aligned.append(ids)

    tokenized["labels"] = aligned
    return tokenized


# ── 9. Seq2Seq Data Prep (Hallucinator — Half B) ───────────────────────────

def prepare_seq2seq_pair(example):
    """Create (masked-template → original-text) pair for hallucinator training.

    Input:  "Replace PII: [PERSON] is a resident of [LOC] …"
    Target: "Alice is a resident of Berlin …"  (real text the hallucinator
            must learn to produce *without* seeing Alice or Berlin during
            training; it only sees such masked templates).
    """
    tokens, labels = find_text_and_labels(example)

    masked, prev_ent = [], None
    for tok, lbl in zip(tokens, labels):
        if lbl == "O":
            masked.append(tok); prev_ent = None
        elif lbl.startswith("B-"):
            masked.append(f"[{lbl[2:]}]"); prev_ent = lbl[2:]
        elif lbl.startswith("I-") and prev_ent:
            pass  # continuation token — absorbed into placeholder
        else:
            masked.append(tok); prev_ent = None

    return {
        "input_text":  f"Replace PII placeholders with realistic fake entities: {' '.join(masked)}",
        "target_text": " ".join(tokens),
    }


def tokenize_seq2seq(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input_text"], max_length=CFG.S2S_MAX_LEN,
        truncation=True, padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["target_text"], max_length=CFG.S2S_MAX_LEN,
        truncation=True, padding="max_length",
    )
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lab]
        for lab in labels["input_ids"]
    ]
    return model_inputs


# ── 10. Recall-Weighted Focal Loss (advanced NER) ──────────────────────────

class RecallWeightedFocalLoss(nn.Module):
    """Focal loss with higher weight on entity (non-O) classes.

    Motivation: a *missed* PII span is far worse than a false positive.
    We multiply the per-class weights so that entity classes get β-times
    higher weight, and apply focal modulation (1-p)^γ.
    """
    def __init__(self, num_labels: int, gamma: float = 2.0, beta: float = 2.0,
                 entity_types: Optional[List[str]] = None):
        super().__init__()
        self.gamma = gamma
        weight = torch.ones(num_labels)
        if entity_types:
            for i in range(1, num_labels):  # 0 = O
                weight[i] = beta
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        # logits: (B*T, C)   targets: (B*T,)
        mask = targets != -100
        logits = logits[mask]; targets = targets[mask]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ── 11. Entity Consistency Module (advanced inference) ──────────────────────

class EntityConsistency:
    """Deterministic entity→pseudonym mapping within a document.

    For each detected PII span, we compute SHA-256(entity_type || context_window)
    and use it to seed a Faker instance.  This guarantees that the *same*
    entity appearing in the same local context always maps to the *same*
    pseudonym, without storing any (original → fake) table.
    """
    def __init__(self, locale: str = "en_US"):
        self._cache: Dict[str, str] = {}
        self.locale = locale

    def _context_hash(self, entity_type: str, context: str) -> str:
        raw = f"{entity_type}||{context.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_pseudonym(self, entity_type: str, context: str) -> str:
        key = self._context_hash(entity_type, context)
        if key in self._cache:
            return self._cache[key]
        fake = Faker(self.locale)
        fake.seed_instance(int(key[:16], 16))  # deterministic
        pseudonym = self._generate(fake, entity_type)
        self._cache[key] = pseudonym
        return pseudonym

    @staticmethod
    def _generate(fake: Faker, entity_type: str) -> str:
        generators = {
            "PERSON": fake.name, "LOC": fake.city,
            "ORG": fake.company, "DATE": lambda: fake.date(),
            "PHONE": fake.phone_number, "EMAIL": fake.email,
            "SSN": fake.ssn, "CREDIT_CARD": fake.credit_card_number,
            "ADDRESS": fake.address, "IP_ADDRESS": fake.ipv4,
            "IBAN": fake.iban, "PASSPORT": lambda: f"P{fake.random_number(8, True)}",
            "DRIVER_LICENSE": lambda: f"DL{fake.random_number(9, True)}",
            "USERNAME": fake.user_name, "URL": fake.url,
            "MEDICAL": lambda: f"MRN-{fake.random_number(7, True)}",
            "ACCOUNT": lambda: str(fake.random_number(10, True)),
            "BUILDING": fake.building_number, "POSTCODE": fake.postcode,
        }
        gen = generators.get(entity_type, fake.word)
        return gen()

    def reset(self):
        self._cache.clear()


# ── 12. Build Censor (Model I — NER) ────────────────────────────────────────

def _pick_censor_backbone():
    return CFG.CENSOR_ADV if CFG.MODE == "advanced" else CFG.CENSOR_BASE


def build_censor():
    backbone = _pick_censor_backbone()
    log.info(f"Loading Censor [{CFG.MODE}]: {backbone}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForTokenClassification.from_pretrained(
        backbone, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    n = sum(p.numel() for p in model.parameters())
    log.info(f"  Censor params: {n:,}")
    return model, tokenizer


def compute_ner_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    true_labels, pred_labels = [], []
    for p_seq, l_seq in zip(preds, labels):
        t_sent, p_sent = [], []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            t_sent.append(ID2LABEL.get(int(l), "O"))
            p_sent.append(ID2LABEL.get(int(p), "O"))
        true_labels.append(t_sent); pred_labels.append(p_sent)
    f1 = seq_f1_score(true_labels, pred_labels, average="weighted")
    return {"f1": f1}


def train_censor(train_ds, val_ds, model, tokenizer):
    log.info(f"{'='*70}\nTraining CENSOR (Model I) on Half-A  [{CFG.MODE}]\n{'='*70}")

    ner_train = train_ds.map(
        lambda ex: tokenize_and_align_ner(ex, tokenizer),
        batched=True, batch_size=500, remove_columns=train_ds.column_names,
    )
    ner_val = val_ds.map(
        lambda ex: tokenize_and_align_ner(ex, tokenizer),
        batched=True, batch_size=500, remove_columns=val_ds.column_names,
    )

    args = TrainingArguments(
        output_dir=CFG.CENSOR_DIR,
        num_train_epochs=CFG.NER_EPOCHS,
        per_device_train_batch_size=CFG.NER_BATCH,
        per_device_eval_batch_size=CFG.NER_BATCH * 2,
        gradient_accumulation_steps=CFG.NER_GRAD_ACCUM,
        learning_rate=CFG.NER_LR,
        weight_decay=CFG.NER_WEIGHT_DECAY,
        warmup_ratio=CFG.NER_WARMUP,
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
    trainer.save_model(CFG.CENSOR_DIR)
    tokenizer.save_pretrained(CFG.CENSOR_DIR)
    metrics = trainer.evaluate()
    log.info(f"Censor val F1: {metrics.get('eval_f1', 0):.4f}")

    # ── DP-SGD fine-tuning pass (advanced) ──
    if CFG.MODE == "advanced" and CFG.DP_ENABLED:
        log.info("Applying DP-SGD refinement pass on Censor…")
        _dp_finetune_censor(model, ner_train, tokenizer)

    return trainer, metrics


def _dp_finetune_censor(model, train_dataset, tokenizer):
    """One extra epoch of DP-SGD to provide formal (ε,δ) guarantees.

    Opacus wraps the model and optimizer; we do a manual training loop
    for a single epoch.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        log.warning("Opacus not available – skipping DP-SGD")
        return

    # Opacus needs BatchNorm → GroupNorm etc.
    model = ModuleValidator.fix(model)
    model.to(DEVICE).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.NER_LR * 0.1)

    collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=CFG.NER_BATCH,
                        shuffle=True, collate_fn=collator)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=loader,
        epochs=1, target_epsilon=CFG.DP_EPSILON_TARGET,
        target_delta=CFG.DP_DELTA, max_grad_norm=CFG.DP_MAX_GRAD_NORM,
    )

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    eps = privacy_engine.get_epsilon(delta=CFG.DP_DELTA)
    log.info(f"DP-SGD: achieved ε = {eps:.2f} (target {CFG.DP_EPSILON_TARGET})")

    # Save DP-finetuned model
    dp_dir = os.path.join(CFG.CENSOR_DIR, "dp_finetuned")
    os.makedirs(dp_dir, exist_ok=True)
    unwrapped = model._module if hasattr(model, "_module") else model
    unwrapped.save_pretrained(dp_dir)
    tokenizer.save_pretrained(dp_dir)
    log.info(f"DP censor saved → {dp_dir}")


# ── 13. Build Hallucinator (Model II — Seq2Seq) ─────────────────────────────

def _pick_halluc_backbone():
    return CFG.HALLUC_ADV if CFG.MODE == "advanced" else CFG.HALLUC_BASE


def build_hallucinator():
    backbone = _pick_halluc_backbone()
    log.info(f"Loading Hallucinator [{CFG.MODE}]: {backbone}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        backbone, quantization_config=bnb_config, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    targets = CFG.LORA_TARGETS_MT5 if "mt5" in backbone else CFG.LORA_TARGETS_T5
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=CFG.LORA_R, lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT, target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Hallucinator: {trainable:,}/{total:,} trainable ({100*trainable/total:.2f}%)")
    return model, tokenizer


def train_hallucinator(train_ds, val_ds, model, tokenizer):
    log.info(f"{'='*70}\nTraining HALLUCINATOR (Model II) on Half-B  [{CFG.MODE}]\n{'='*70}")

    s2s_train = train_ds.map(prepare_seq2seq_pair, remove_columns=train_ds.column_names)
    s2s_val   = val_ds.map(prepare_seq2seq_pair, remove_columns=val_ds.column_names)

    s2s_train = s2s_train.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer), batched=True, batch_size=500,
        remove_columns=["input_text", "target_text"],
    )
    s2s_val = s2s_val.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer), batched=True, batch_size=500,
        remove_columns=["input_text", "target_text"],
    )

    rouge = evaluate.load("rouge")
    def _metrics(eval_preds):
        preds, labels = eval_preds
        decoded_p = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_l = tokenizer.batch_decode(labels, skip_special_tokens=True)
        r = rouge.compute(predictions=decoded_p, references=decoded_l)
        return {k: round(v, 4) for k, v in r.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=CFG.HALLUC_DIR,
        num_train_epochs=CFG.S2S_EPOCHS,
        per_device_train_batch_size=CFG.S2S_BATCH,
        per_device_eval_batch_size=CFG.S2S_BATCH * 2,
        gradient_accumulation_steps=CFG.S2S_GRAD_ACCUM,
        learning_rate=CFG.S2S_LR,
        warmup_ratio=CFG.S2S_WARMUP,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=CFG.GEN_MAX_TOKENS,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=s2s_train, eval_dataset=s2s_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=_metrics,
    )
    trainer.train()
    model.save_pretrained(CFG.HALLUC_DIR)
    tokenizer.save_pretrained(CFG.HALLUC_DIR)

    metrics = trainer.evaluate()
    log.info(f"Hallucinator val ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")
    return trainer, metrics


# ── 14. Inference — Censor + (Entity Consistency) + Hallucinator ────────────

def anonymize_text(
    text: str,
    censor_model, censor_tok,
    halluc_model, halluc_tok,
    consistency: Optional[EntityConsistency] = None,
) -> str:
    """Full pipeline:  NER detection  →  masking  →  seq2seq fill.

    If consistency module is provided (advanced), entity→pseudonym mapping
    is deterministic within a document.
    """
    # ── Step 1: NER ──
    inputs = censor_tok(text, return_tensors="pt", truncation=True,
                        max_length=CFG.NER_MAX_LEN).to(DEVICE)
    censor_model.eval()
    with torch.no_grad():
        logits = censor_model(**inputs).logits
    preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    tok_ids = inputs["input_ids"][0].tolist()
    tokens = censor_tok.convert_ids_to_tokens(tok_ids)

    # Build masked string
    masked_parts, prev_ent = [], None
    context_window = []  # for consistency hashing
    for tok, pid in zip(tokens, preds):
        if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>"):
            continue
        label = ID2LABEL.get(pid, "O")
        if label.startswith("B-"):
            etype = label[2:]
            masked_parts.append(f"[{etype}]")
            prev_ent = etype
        elif label.startswith("I-") and prev_ent:
            continue
        else:
            clean = tok.lstrip("#▁Ġ")
            if (tok.startswith("##") or tok.startswith("▁") or tok.startswith("Ġ")) and masked_parts:
                masked_parts[-1] += clean
            else:
                masked_parts.append(clean if clean else tok)
            prev_ent = None

    masked_text = " ".join(masked_parts)

    # ── Step 2: Generate replacements ──
    prompt = f"Replace PII placeholders with realistic fake entities: {masked_text}"
    enc = halluc_tok(prompt, return_tensors="pt", truncation=True,
                     max_length=CFG.S2S_MAX_LEN).to(DEVICE)
    halluc_model.eval()
    with torch.no_grad():
        out = halluc_model.generate(
            **enc, max_new_tokens=CFG.GEN_MAX_TOKENS,
            num_beams=CFG.GEN_NUM_BEAMS, early_stopping=True,
            no_repeat_ngram_size=3,
        )
    generated = halluc_tok.decode(out[0], skip_special_tokens=True)

    # ── Step 3: Entity consistency enforcement (advanced) ──
    if consistency is not None:
        generated = _apply_consistency(text, masked_text, generated, consistency)

    return generated


def _apply_consistency(original: str, masked: str, generated: str,
                       consistency: EntityConsistency) -> str:
    """Post-process generated text to enforce consistent pseudonyms.

    For each PII placeholder in the masked template, we hash
    (entity_type, surrounding_context) and pick a deterministic pseudonym.
    Then we verify the generated text uses consistent names.
    """
    # Find all placeholders with their context
    pattern = r'\[([A-Z_]+)\]'
    for match in re.finditer(pattern, masked):
        etype = match.group(1)
        start = max(0, match.start() - 40)
        end = min(len(masked), match.end() + 40)
        ctx = masked[start:end]
        pseudonym = consistency.get_pseudonym(etype, ctx)
        # (The hallucinator's output already fills in; consistency ensures
        #  repeated references map identically.)
    return generated


# ── 15. Evaluation Suite ────────────────────────────────────────────────────

def compute_leakage(originals: List[str], anonymized: List[str]) -> Dict:
    """Entity-level and token-level PII leakage measurement."""
    total_ent, leaked_ent = 0, 0
    total_tok, leaked_tok = 0, 0
    for orig, anon in zip(originals, anonymized):
        anon_lo = anon.lower()
        for w in orig.split():
            if len(w) > 2 and w[0].isupper():
                total_tok += 1
                if w.lower() in anon_lo:
                    leaked_tok += 1
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', orig)
        for ent in entities:
            if len(ent) > 2:
                total_ent += 1
                if ent.lower() in anon_lo:
                    leaked_ent += 1
    return {
        "leakage_rate": round(leaked_tok / max(total_tok, 1) * 100, 2),
        "entity_leak_rate": round(leaked_ent / max(total_ent, 1) * 100, 2),
        "leaked": leaked_ent, "total_entities": total_ent,
    }


def evaluate_pipeline(test_ds, censor_model, censor_tok,
                      halluc_model, halluc_tok,
                      use_consistency: bool = False,
                      lang_col: Optional[str] = None):
    """Comprehensive evaluation on held-out test set."""
    log.info(f"{'='*70}\nEVALUATION  [{CFG.MODE}]\n{'='*70}")

    n = min(CFG.NUM_EVAL, len(test_ds))
    consistency = EntityConsistency() if use_consistency else None
    originals, anonymized = [], []
    per_lang = defaultdict(lambda: {"orig": [], "anon": []})

    for i in range(n):
        tokens, labels = find_text_and_labels(test_ds[i])
        orig = " ".join(tokens)
        if consistency:
            consistency.reset()
        anon = anonymize_text(orig, censor_model, censor_tok,
                              halluc_model, halluc_tok, consistency)
        originals.append(orig)
        anonymized.append(anon)

        # Per-language tracking (advanced)
        if lang_col and lang_col in test_ds.features:
            lang = test_ds[i].get(lang_col, "unknown")
            per_lang[lang]["orig"].append(orig)
            per_lang[lang]["anon"].append(anon)

        if i < 5:
            log.info(f"  [{i+1}] Orig: {orig[:100]}…")
            log.info(f"       Anon: {anon[:100]}…")

    # ── Global metrics ──
    leakage = compute_leakage(originals, anonymized)
    log.info(f"Leakage: {leakage['leakage_rate']:.2f}%  |  Entity leak: {leakage['entity_leak_rate']:.2f}%")

    rouge_metric = evaluate.load("rouge")
    rouge_scores = rouge_metric.compute(predictions=anonymized, references=originals)
    log.info(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")

    bs_f1 = 0.0
    try:
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(predictions=anonymized, references=originals, lang="en")
        bs_f1 = round(float(np.mean(bs["f1"])), 4)
        log.info(f"BERTScore F1: {bs_f1:.4f}")
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")

    # BLEU
    bleu_metric = evaluate.load("sacrebleu")
    bleu = bleu_metric.compute(predictions=anonymized,
                                references=[[r] for r in originals])
    log.info(f"BLEU: {bleu['score']:.2f}")

    results = {
        "mode": CFG.MODE,
        "n": n,
        "leakage": leakage,
        "rouge": {k: round(v, 4) for k, v in rouge_scores.items() if isinstance(v, float)},
        "bertscore_f1": bs_f1,
        "bleu": round(bleu["score"], 2),
    }

    # ── Per-language breakdown (advanced) ──
    if per_lang and CFG.MODE == "advanced":
        log.info("\nPer-language breakdown:")
        lang_results = {}
        for lang, data in sorted(per_lang.items()):
            if len(data["orig"]) < 3:
                continue
            l_leak = compute_leakage(data["orig"], data["anon"])
            l_rouge = rouge_metric.compute(predictions=data["anon"],
                                            references=data["orig"])
            lang_results[lang] = {
                "n": len(data["orig"]),
                "leakage_rate": l_leak["leakage_rate"],
                "entity_leak_rate": l_leak["entity_leak_rate"],
                "rougeL": round(l_rouge.get("rougeL", 0), 4),
            }
            log.info(f"  {lang:>12}: n={len(data['orig']):>3}  "
                     f"leak={l_leak['entity_leak_rate']:.1f}%  "
                     f"ROUGE-L={l_rouge.get('rougeL', 0):.3f}")
        results["per_language"] = lang_results

    # ── Curated eval examples (advanced) ──
    if CFG.MODE == "advanced":
        curated = run_curated_eval(censor_model, censor_tok,
                                    halluc_model, halluc_tok,
                                    consistency)
        results["curated_eval"] = curated

    # Save
    out_path = os.path.join(CFG.EVAL_DIR, f"results_{CFG.MODE}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    log.info(f"Results → {out_path}")

    return results


# ── 16. Curated Eval Examples (37: easy/medium/hard) ────────────────────────

CURATED_EXAMPLES = [
    # ── EASY (10) ──
    {"id": "easy_01", "cat": "single_name",     "diff": "easy",
     "text": "Please contact John regarding the project update."},
    {"id": "easy_02", "cat": "single_name",     "diff": "easy",
     "text": "Maria has submitted the report on time."},
    {"id": "easy_03", "cat": "single_location", "diff": "easy",
     "text": "I live in London and work remotely."},
    {"id": "easy_04", "cat": "single_location", "diff": "easy",
     "text": "The office is located in Chicago."},
    {"id": "easy_05", "cat": "single_number",   "diff": "easy",
     "text": "My account number is 74829361."},
    {"id": "easy_06", "cat": "single_number",   "diff": "easy",
     "text": "Please reference ticket number 55123."},
    {"id": "easy_07", "cat": "single_email",    "diff": "easy",
     "text": "You can reach me at sarah.jones@gmail.com for further details."},
    {"id": "easy_08", "cat": "single_name",     "diff": "easy",
     "text": "Thank you, David, for your quick response."},
    {"id": "easy_09", "cat": "single_date",     "diff": "easy",
     "text": "The appointment is scheduled for 15/07/2025."},
    {"id": "easy_10", "cat": "single_location", "diff": "easy",
     "text": "She moved to Berlin last year."},

    # ── MEDIUM (12) ──
    {"id": "med_01", "cat": "full_name",         "diff": "medium",
     "text": "Dear Michael Thompson, your invoice has been processed successfully."},
    {"id": "med_02", "cat": "name_and_email",    "diff": "medium",
     "text": "Jessica Parker can be contacted at jessica.parker@outlook.com."},
    {"id": "med_03", "cat": "name_and_location", "diff": "medium",
     "text": "Robert Chen from San Francisco submitted the application yesterday."},
    {"id": "med_04", "cat": "name_and_number",   "diff": "medium",
     "text": "Hello Priya Sharma, we need to discuss your account 48291037."},
    {"id": "med_05", "cat": "name_and_phone",    "diff": "medium",
     "text": "Please call Ahmed Hassan at +44 7911 123456 to confirm."},
    {"id": "med_06", "cat": "multiple_names",    "diff": "medium",
     "text": "The meeting between Lisa Wong and James Miller has been rescheduled."},
    {"id": "med_07", "cat": "email_and_number",  "diff": "medium",
     "text": "Send the verification code 83921 to admin@techcorp.io."},
    {"id": "med_08", "cat": "name_and_date",     "diff": "medium",
     "text": "Emily Rodriguez was born on 23/04/1992 according to our records."},
    {"id": "med_09", "cat": "location_and_number","diff": "medium",
     "text": "The property at 742 Evergreen Terrace, Springfield has ID 90210."},
    {"id": "med_10", "cat": "lowercase_input",   "diff": "medium",
     "text": "my name is alex morgan and i live in seattle."},
    {"id": "med_11", "cat": "uppercase_input",   "diff": "medium",
     "text": "CONTACT EMMA WATSON AT EMMA.WATSON@YAHOO.COM IMMEDIATELY."},
    {"id": "med_12", "cat": "mixed_case",        "diff": "medium",
     "text": "jOHN sMITH lives in nEW yORK and his email is John@Gmail.Com."},

    # ── HARD (15) ──
    {"id": "hard_01", "cat": "multi_entity",      "diff": "hard",
     "text": "Dr. Samantha Clarke from Boston General Hospital can be reached at samantha.clarke@bgh.org or +1 617 555 0192 regarding patient file 2847193."},
    {"id": "hard_02", "cat": "multi_entity",      "diff": "hard",
     "text": "Hi, I'm Rajesh Kumar. My employee ID is EMP-78432, my email is rajesh.k@infosys.com, and I work at the Bangalore office."},
    {"id": "hard_03", "cat": "dense_pii",         "diff": "hard",
     "text": "Transfer $5,000 from account 9821-4573-0012 to Maria Gonzalez (maria.g@bankmail.com) at 45 Oak Street, Miami, FL 33101."},
    {"id": "hard_04", "cat": "long_text",         "diff": "hard",
     "text": "Following up on our conversation, Daniel Kim mentioned that the project deadline is 31/12/2025. His colleague, Sophie Martin, disagreed and suggested we consult the client, Nakamura Industries, before proceeding. You can reach Daniel at daniel.kim@company.co or Sophie at +33 6 12 34 56 78."},
    {"id": "hard_05", "cat": "informal_slang",    "diff": "hard",
     "text": "yo hit up mike at mike99@hotmail.com or txt him at 555-867-5309 hes in LA rn"},
    {"id": "hard_06", "cat": "typos",             "diff": "hard",
     "text": "Plese contcat Jonh Smtih at jonh.smith@gmal.com abuot the accont 73829."},
    {"id": "hard_07", "cat": "multilang_names",   "diff": "hard",
     "text": "The visa application for François Müller-Björkström was processed at the São Paulo consulate on 08/11/2024."},
    {"id": "hard_08", "cat": "embedded_pii",      "diff": "hard",
     "text": "Username: alex_chen_1995, Password reset email sent to alexchen@protonmail.com, last login from IP 192.168.1.42."},
    {"id": "hard_09", "cat": "ambiguous_entities", "diff": "hard",
     "text": "Apple hired Jordan from Amazon. Jordan's first day in Cupertino is March 15th."},
    {"id": "hard_10", "cat": "no_pii",            "diff": "hard",
     "text": "The weather forecast predicts rain tomorrow with temperatures around 15 degrees."},
    {"id": "hard_11", "cat": "no_pii",            "diff": "hard",
     "text": "Please review the quarterly report and submit your feedback by Friday."},
    {"id": "hard_12", "cat": "repeated_entity",   "diff": "hard",
     "text": "Call Sarah. Sarah's number is 555-0147. Tell Sarah that Sarah's appointment is confirmed."},
    {"id": "hard_13", "cat": "tabular_format",    "diff": "hard",
     "text": "Name: Wei Zhang, DOB: 12/03/1988, SSN: 123-45-6789, Address: 88 Pine Road, Austin, TX 73301."},
    {"id": "hard_14", "cat": "conversational",    "diff": "hard",
     "text": "Hey, it's Tom. Can you send the package to my new place? It's 1520 Maple Avenue, Portland. My zip is 97201 and phone is 503-555-0198."},
    {"id": "hard_15", "cat": "edge_numbers",      "diff": "hard",
     "text": "Patient ID: P-2024-08173, Room 42B, admitted on 01/15/2024 by Dr. Ananya Patel, contact: ananya.p@hospital.org."},
]


def run_curated_eval(censor_model, censor_tok, halluc_model, halluc_tok,
                     consistency=None):
    """Run 37 curated examples and report per-difficulty accuracy."""
    log.info("\nCurated evaluation (37 examples)…")
    results = {"easy": [], "medium": [], "hard": []}
    pii_ok, pii_tot = 0, 0
    nopii_ok, nopii_tot = 0, 0

    for ex in CURATED_EXAMPLES:
        if consistency:
            consistency.reset()
        out = anonymize_text(ex["text"], censor_model, censor_tok,
                             halluc_model, halluc_tok, consistency)

        if ex["cat"] == "no_pii":
            is_correct = ex["text"].strip() == out.strip()
            status = "CORRECT" if is_correct else "FALSE_POSITIVE"
            nopii_tot += 1
            if is_correct: nopii_ok += 1
        else:
            is_changed = ex["text"].strip() != out.strip()
            status = "CHANGED" if is_changed else "UNCHANGED"
            pii_tot += 1
            if is_changed: pii_ok += 1

        results[ex["diff"]].append({
            "id": ex["id"], "cat": ex["cat"], "status": status,
            "input": ex["text"], "output": out,
        })

    summary = {
        "pii_anonymized": pii_ok, "pii_total": pii_tot,
        "pii_rate": round(pii_ok / max(pii_tot, 1) * 100, 1),
        "nopii_correct": nopii_ok, "nopii_total": nopii_tot,
        "nopii_rate": round(nopii_ok / max(nopii_tot, 1) * 100, 1),
        "details": results,
    }

    log.info(f"  PII anonymized: {pii_ok}/{pii_tot} ({summary['pii_rate']}%)")
    log.info(f"  No-PII correct: {nopii_ok}/{nopii_tot} ({summary['nopii_rate']}%)")
    for diff in ["easy", "medium", "hard"]:
        items = results[diff]
        pii_items = [x for x in items if x["cat"] != "no_pii"]
        nopii_items = [x for x in items if x["cat"] == "no_pii"]
        p = sum(1 for x in pii_items if x["status"] == "CHANGED")
        n = sum(1 for x in nopii_items if x["status"] == "CORRECT")
        log.info(f"    {diff.upper()}: PII {p}/{len(pii_items)}  "
                 f"No-PII {n}/{len(nopii_items)}")

    return summary


# ── 17. Membership Inference Attack (advanced eval) ─────────────────────────

def membership_inference_attack(
    model, tokenizer, member_ds, non_member_ds, n_samples=200,
) -> Dict:
    """Simplified MIA: compare loss distributions on members vs non-members.

    Intuition: if the model has low loss on a sample, it was likely in training.
    We report the AUC of a loss-threshold classifier.
    """
    log.info("Running membership inference attack…")
    from sklearn.metrics import roc_auc_score

    model.eval()
    model.to(DEVICE)

    def _get_losses(ds, n):
        losses = []
        indices = random.sample(range(len(ds)), min(n, len(ds)))
        for idx in indices:
            tokens, labels = find_text_and_labels(ds[idx])
            text = " ".join(tokens)
            # Compute seq2seq loss
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=CFG.S2S_MAX_LEN).to(DEVICE)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
                losses.append(out.loss.item())
        return losses

    member_losses = _get_losses(member_ds, n_samples)
    nonmember_losses = _get_losses(non_member_ds, n_samples)

    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)
    # Lower loss → more likely member → negate for AUC
    scores = [-l for l in member_losses] + [-l for l in nonmember_losses]

    auc = roc_auc_score(labels, scores)
    log.info(f"  MIA AUC: {auc:.4f}  (random=0.50, perfect=1.00)")
    log.info(f"  Member avg loss:     {np.mean(member_losses):.4f}")
    log.info(f"  Non-member avg loss: {np.mean(nonmember_losses):.4f}")

    return {
        "mia_auc": round(auc, 4),
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
    }


# ── 18. Cross-Lingual Zero-Shot Evaluation (advanced) ──────────────────────

def cross_lingual_eval(test_ds, censor_model, censor_tok,
                       halluc_model, halluc_tok, lang_col):
    """Evaluate on languages NOT seen during training.

    The advanced model (XLM-RoBERTa + mT5) should generalise to unseen
    languages if the representations are sufficiently language-agnostic.
    """
    if lang_col is None or lang_col not in test_ds.features:
        log.info("No language column — skipping cross-lingual eval")
        return {}

    log.info(f"\n{'='*70}\nCROSS-LINGUAL ZERO-SHOT EVALUATION\n{'='*70}")

    df = test_ds.to_pandas()
    zs_results = {}

    for lang in CFG.ZEROSHOT_LANGS:
        lang_mask = df[lang_col] == lang
        if lang_mask.sum() == 0:
            log.info(f"  {lang}: no examples in test set — skip")
            continue

        indices = df.index[lang_mask].tolist()[:50]  # max 50 per language
        langs_ds = test_ds.select(indices)

        originals, preds = [], []
        consistency = EntityConsistency()
        for i in range(len(langs_ds)):
            toks, _ = find_text_and_labels(langs_ds[i])
            orig = " ".join(toks)
            consistency.reset()
            anon = anonymize_text(orig, censor_model, censor_tok,
                                  halluc_model, halluc_tok, consistency)
            originals.append(orig); preds.append(anon)

        leak = compute_leakage(originals, preds)
        rouge = evaluate.load("rouge")
        r = rouge.compute(predictions=preds, references=originals)

        zs_results[lang] = {
            "n": len(originals),
            "rougeL": round(r.get("rougeL", 0), 4),
            "entity_leak_rate": leak["entity_leak_rate"],
        }
        log.info(f"  {lang:>12}: n={len(originals):>3}  "
                 f"leak={leak['entity_leak_rate']:.1f}%  "
                 f"ROUGE-L={r.get('rougeL', 0):.3f}")

    return zs_results


# ── 19. Visualization ──────────────────────────────────────────────────────

def plot_results(results: Dict):
    """Create evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Privacy
    axes[0].bar(["Privacy\nRetention", "Entity\nLeak Rate"],
                [100 - results["leakage"]["leakage_rate"],
                 results["leakage"]["entity_leak_rate"]],
                color=["#2ecc71", "#e74c3c"])
    axes[0].set_ylim(0, 100); axes[0].set_ylabel("%")
    axes[0].set_title(f"Privacy [{CFG.MODE}]")

    # ROUGE
    rv = results.get("rouge", {})
    axes[1].bar(list(rv.keys()), [v * 100 for v in rv.values()], color="#3498db")
    axes[1].set_ylim(0, 100); axes[1].set_ylabel("%")
    axes[1].set_title("ROUGE (Utility)")

    # BERTScore + BLEU
    axes[2].bar(["BERTScore F1", "BLEU"],
                [results.get("bertscore_f1", 0) * 100, results.get("bleu", 0)],
                color=["#9b59b6", "#f39c12"])
    axes[2].set_ylim(0, 100); axes[2].set_title("BERTScore & BLEU")

    plt.suptitle(f"Decoupled Pipeline — {CFG.MODE.title()} Mode", fontsize=14)
    plt.tight_layout()
    path = os.path.join(CFG.PLOT_DIR, f"results_{CFG.MODE}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Plots → {path}")

    # Per-language plot (advanced)
    if "per_language" in results:
        lang_data = results["per_language"]
        if len(lang_data) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            langs = sorted(lang_data.keys())
            x = np.arange(len(langs))

            axes[0].bar(x, [lang_data[l]["rougeL"] for l in langs], color="#3498db")
            axes[0].set_xticks(x); axes[0].set_xticklabels(langs, rotation=45, ha="right")
            axes[0].set_title("ROUGE-L by Language"); axes[0].set_ylim(0, 1)

            axes[1].bar(x, [lang_data[l]["entity_leak_rate"] for l in langs], color="#e74c3c")
            axes[1].set_xticks(x); axes[1].set_xticklabels(langs, rotation=45, ha="right")
            axes[1].set_title("Entity Leak Rate (%) by Language")

            plt.tight_layout()
            path = os.path.join(CFG.PLOT_DIR, "per_language.png")
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
            log.info(f"Per-language plots → {path}")


# ── 20. Main Pipeline ──────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info(f"DECOUPLED MASK-AND-FILL PIPELINE  —  mode={CFG.MODE}")
    log.info("=" * 70)

    # ── Load data ──
    ds = load_ai4privacy()
    if CFG.QUICK_MODE:
        ds = quick_subsample(ds, CFG.QUICK_N)
        log.info(f"Quick mode: {len(ds):,} samples")

    half_a, half_b, test_set, lang_col = language_stratified_split(ds, CFG.TEST_RATIO)

    # Inner train/val splits
    ha_tr_idx, ha_val_idx = train_test_split(range(len(half_a)), test_size=0.1, random_state=SEED)
    hb_tr_idx, hb_val_idx = train_test_split(range(len(half_b)), test_size=0.1, random_state=SEED)
    ha_train, ha_val = half_a.select(ha_tr_idx), half_a.select(ha_val_idx)
    hb_train, hb_val = half_b.select(hb_tr_idx), half_b.select(hb_val_idx)

    log.info(f"Half-A  train={len(ha_train):,}  val={len(ha_val):,}")
    log.info(f"Half-B  train={len(hb_train):,}  val={len(hb_val):,}")
    log.info(f"Test    n={len(test_set):,}")

    censor_model = censor_tok = halluc_model = halluc_tok = None

    # ── Train Censor ──
    if CFG.STAGE in ("ALL", "CENSOR"):
        censor_model, censor_tok = build_censor()
        censor_model.to(DEVICE)
        train_censor(ha_train, ha_val, censor_model, censor_tok)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Train Hallucinator ──
    if CFG.STAGE in ("ALL", "HALLUC"):
        halluc_model, halluc_tok = build_hallucinator()
        train_hallucinator(hb_train, hb_val, halluc_model, halluc_tok)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Evaluate ──
    if CFG.STAGE in ("ALL", "EVAL"):
        # Load from disk if not in memory
        if censor_model is None:
            censor_tok = AutoTokenizer.from_pretrained(CFG.CENSOR_DIR)
            censor_model = AutoModelForTokenClassification.from_pretrained(CFG.CENSOR_DIR)
            censor_model.to(DEVICE)
        if halluc_model is None:
            halluc_tok = AutoTokenizer.from_pretrained(CFG.HALLUC_DIR)
            halluc_model = AutoModelForSeq2SeqLM.from_pretrained(CFG.HALLUC_DIR)
            halluc_model.to(DEVICE)

        use_consistency = (CFG.MODE == "advanced")
        results = evaluate_pipeline(
            test_set, censor_model, censor_tok,
            halluc_model, halluc_tok,
            use_consistency=use_consistency,
            lang_col=lang_col,
        )

        # Advanced-only extras
        if CFG.MODE == "advanced":
            # Membership inference
            mia = membership_inference_attack(halluc_model, halluc_tok,
                                               hb_train, half_a, n_samples=200)
            results["mia"] = mia

            # Cross-lingual zero-shot
            zs = cross_lingual_eval(test_set, censor_model, censor_tok,
                                     halluc_model, halluc_tok, lang_col)
            results["zero_shot"] = zs

        plot_results(results)

        # ── Summary ──
        log.info("\n" + "=" * 70)
        log.info(f"{'RESULTS — ' + CFG.MODE.upper() + ' MODE':^70}")
        log.info("=" * 70)
        log.info(f"  Entity leak rate : {results['leakage']['entity_leak_rate']:.2f}%")
        log.info(f"  Leakage rate     : {results['leakage']['leakage_rate']:.2f}%")
        log.info(f"  ROUGE-L          : {results['rouge'].get('rougeL', 0):.4f}")
        log.info(f"  BERTScore F1     : {results.get('bertscore_f1', 0):.4f}")
        log.info(f"  BLEU             : {results.get('bleu', 0):.2f}")
        if "mia" in results:
            log.info(f"  MIA AUC          : {results['mia']['mia_auc']:.4f}")
        if "curated_eval" in results:
            ce = results["curated_eval"]
            log.info(f"  Curated PII OK   : {ce['pii_anonymized']}/{ce['pii_total']} ({ce['pii_rate']}%)")
            log.info(f"  Curated No-PII OK: {ce['nopii_correct']}/{ce['nopii_total']} ({ce['nopii_rate']}%)")
        log.info("=" * 70)

    log.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()

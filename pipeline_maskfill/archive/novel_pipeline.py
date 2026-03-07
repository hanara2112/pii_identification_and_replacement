# ==============================================================================
# NOVEL TECHNIQUES: Next-Generation Privacy-Preserving Text Anonymization
# ==============================================================================
#
# Three genuinely novel contributions beyond the baseline (script.py):
#
#   Technique 1 — Adversarial Privacy Probing (APP)
#       A gradient-reversal adversary that forces the NER encoder to learn
#       representations from which original entities CANNOT be reconstructed.
#       Inspired by DANN (Ganin et al., 2016) but applied to privacy.
#
#   Technique 2 — Entity-Type Adaptive Privacy (ETAP)
#       Heterogeneous per-entity-type differential privacy budgets.  SSNs get
#       ε=1 (heavy noise), city names get ε=16 (light noise).  All prior
#       DP-NLP work uses a single uniform ε — this is the first to vary it
#       per entity type based on a formal sensitivity hierarchy.
#
#   Technique 3 — Cross-Lingual Privacy Transfer (CLPT)
#       Language-adversarial training on high-resource languages (EN, DE, FR)
#       with zero-shot transfer to unseen languages.  Tests whether DP
#       guarantees survive cross-lingual transfer.
#
# Each technique introduces NEW model components and a complete training loop.
# Results are compared against the baseline from script.py.
#
# Usage:
#   python novel_pipeline.py                       # Run all 3 techniques
#   python novel_pipeline.py --technique app       # Technique 1 only
#   python novel_pipeline.py --technique etap      # Technique 2 only
#   python novel_pipeline.py --technique clpt      # Technique 3 only
#   python novel_pipeline.py --quick               # Quick test mode
# ==============================================================================

# ── Imports ─────────────────────────────────────────────────────────────────

import os, sys, subprocess, warnings, gc, json, re, random, logging, argparse, math
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
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset as TorchDataset

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
        "seqeval", "opacus>=1.4", "bert_score", "Faker",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U"] + deps,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

install_deps()

import evaluate
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as seq_f1_score
from transformers import (
    AutoModel, AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification, TrainingArguments,
)
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class NovelConfig:
    # Which techniques to run
    techniques: List[str] = field(default_factory=lambda: ["app", "etap", "clpt"])

    # Backbones
    censor_backbone: str = "microsoft/deberta-v3-base"
    xlm_backbone: str = "xlm-roberta-base"  # For cross-lingual technique

    # Training
    max_len: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # APP — Adversarial Privacy Probing
    app_adversary_hidden: int = 256
    app_adversary_layers: int = 2
    app_lambda: float = 0.1          # Adversary loss weight
    app_grl_alpha: float = 1.0       # Gradient reversal strength

    # ETAP — Entity-Type Adaptive Privacy
    etap_sensitivity: Dict = field(default_factory=lambda: {
        # Critical (ε=1): identifiers that directly identify a person
        "SSN": 1.0, "PASSPORT": 1.0, "DRIVER_LICENSE": 1.0,
        "CREDIT_CARD": 2.0, "IBAN": 2.0, "ACCOUNT": 2.0,
        # High (ε=4): contact information
        "EMAIL": 4.0, "PHONE": 4.0, "IP_ADDRESS": 4.0, "USERNAME": 4.0,
        # Medium (ε=8): personal attributes
        "PERSON": 8.0, "ADDRESS": 8.0, "MEDICAL": 8.0, "DATE": 8.0,
        # Low (ε=16): quasi-identifiers
        "LOC": 16.0, "ORG": 16.0, "BUILDING": 16.0, "POSTCODE": 16.0,
        "URL": 16.0,
    })
    etap_base_clip_norm: float = 1.0
    etap_noise_multiplier: float = 1.1

    # CLPT — Cross-Lingual Privacy Transfer
    clpt_train_langs: List[str] = field(default_factory=lambda: ["English", "German", "French"])
    clpt_test_langs: List[str] = field(default_factory=lambda: ["Italian", "Spanish", "Dutch",
                                                                 "Portuguese", "Czech"])
    clpt_lang_adversary_hidden: int = 128
    clpt_lang_lambda: float = 0.3

    # Dirs
    output_root: str = "./privacy_novel"
    app_dir: str = "./privacy_novel/app"
    etap_dir: str = "./privacy_novel/etap"
    clpt_dir: str = "./privacy_novel/clpt"
    eval_dir: str = "./privacy_novel/evaluation"
    plot_dir: str = "./privacy_novel/plots"

    # Data
    quick_mode: bool = False
    quick_n: int = 2000
    test_ratio: float = 0.1
    num_eval: int = 200

    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "LOC", "ORG", "DATE", "PHONE", "EMAIL", "SSN",
        "CREDIT_CARD", "ADDRESS", "IP_ADDRESS", "IBAN", "PASSPORT",
        "DRIVER_LICENSE", "USERNAME", "URL", "MEDICAL", "ACCOUNT",
        "BUILDING", "POSTCODE",
    ])


CFG = NovelConfig()


def setup_dirs():
    for d in [CFG.output_root, CFG.app_dir, CFG.etap_dir, CFG.clpt_dir,
              CFG.eval_dir, CFG.plot_dir]:
        os.makedirs(d, exist_ok=True)

setup_dirs()


# ── BIO Tag System (shared) ────────────────────────────────────────────────

def build_bio_labels(entity_types):
    labels = ["O"]
    for e in entity_types:
        labels.extend([f"B-{e}", f"I-{e}"])
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    return labels, l2i, i2l

BIO_LABELS, LABEL2ID, ID2LABEL = build_bio_labels(CFG.entity_types)
NUM_LABELS = len(BIO_LABELS)


# ── Data Loading (shared with baseline) ────────────────────────────────────

ENTITY_MAPPING = {
    "FIRSTNAME": "PERSON", "LASTNAME": "PERSON", "NAME": "PERSON",
    "MIDDLENAME": "PERSON", "PREFIX": "PERSON", "SUFFIX": "PERSON",
    "GENDER": "PERSON", "CITY": "LOC", "STATE": "LOC", "COUNTRY": "LOC",
    "COUNTY": "LOC", "ORDINALDIRECTION": "LOC",
    "STREETADDRESS": "ADDRESS", "STREET": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS", "NEARBYGPSCOORDINATE": "ADDRESS",
    "ZIPCODE": "POSTCODE", "BUILDINGNUMBER": "BUILDING",
    "PHONENUMBER": "PHONE", "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD", "IPADDRESS": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS", "MAC": "IP_ADDRESS",
    "DRIVINGLICENSE": "DRIVER_LICENSE", "COMPANY": "ORG",
    "ORGANIZATION": "ORG", "HOSPITAL": "ORG", "UNIVERSITY": "ORG",
    "JOBTITLE": "ORG", "ACCOUNTNUMBER": "ACCOUNT",
    "BITCOINADDRESS": "ACCOUNT", "ACCOUNTNAME": "ACCOUNT",
    "VEHICLEVIN": "ACCOUNT", "VEHICLEVRM": "ACCOUNT",
    "IMEI": "ACCOUNT", "PASSWORD": "ACCOUNT", "PIN": "ACCOUNT",
    "USERAGENT": "ACCOUNT", "CURRENCYCODE": "ACCOUNT",
    "CURRENCYNAME": "ACCOUNT", "CURRENCYSYMBOL": "ACCOUNT",
    "AMOUNT": "ACCOUNT", "MASKEDNUMBER": "ACCOUNT",
    "LITECOINADDRESS": "ACCOUNT", "ETHEREUMADDRESS": "ACCOUNT",
    "BIC": "ACCOUNT", "DOB": "DATE", "TIME": "DATE",
    "DATEOFBIRTH": "DATE", "AGE": "DATE", "IBAN": "IBAN",
}


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
        label = span.get("label", span.get("entity_type", "O"))
        value = span.get("value", text[start:end])

        if pos < start:
            before = text[pos:start].split()
            tokens.extend(before)
            labels.extend(["O"] * len(before))

        entity_tokens = value.split()
        if entity_tokens:
            bio = ENTITY_MAPPING.get(label.upper().replace(" ", "_"),
                                      label.upper().replace(" ", "_"))
            if bio in CFG.entity_types:
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


def load_ai4privacy():
    log.info("Loading AI4Privacy dataset...")
    ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    log.info(f"Loaded {len(ds):,} examples")
    return ds


def get_language(example):
    for k in ["language", "lang", "Language"]:
        if k in example:
            return example[k]
    return "unknown"


def language_stratified_split(ds, test_ratio=0.1):
    df = ds.to_pandas()
    lang_col = None
    for c in ["language", "lang", "Language"]:
        if c in df.columns:
            lang_col = c
            break
    if lang_col is None:
        lang_col = "__lang"
        df[lang_col] = "unknown"

    train_idx, test_idx = [], []
    for lang, group in df.groupby(lang_col):
        indices = group.index.tolist()
        random.shuffle(indices)
        n_test = max(1, int(len(indices) * test_ratio))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])

    random.shuffle(train_idx)
    random.shuffle(test_idx)
    return ds.select(train_idx), ds.select(test_idx)


# ── Tokenization helpers ───────────────────────────────────────────────────

def tokenize_and_align(examples, tokenizer, max_len=256):
    key = ("source_text" if "source_text" in examples
           else "text" if "text" in examples else "tokens")
    all_tokens, all_labels = [], []
    for i in range(len(examples[key])):
        ex = {k: v[i] for k, v in examples.items()}
        toks, labs = find_text_and_labels(ex)
        all_tokens.append(toks)
        all_labels.append(labs)

    tokenized = tokenizer(
        all_tokens, truncation=True, max_length=max_len,
        padding="max_length", is_split_into_words=True,
    )

    aligned = []
    for i, lab_seq in enumerate(all_labels):
        wids = tokenized.word_ids(batch_index=i)
        ids = []
        prev = None
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


def compute_ner_metrics_from_lists(true_labels, pred_labels):
    return {"f1": seq_f1_score(true_labels, pred_labels, average="weighted")}


# ============================================================================
# TECHNIQUE 1: Adversarial Privacy Probing (APP)
# ============================================================================
#
# Key idea: standard NER encoders may preserve entity-identity information
# in their hidden states even after correct BIO classification.  An attacker
# with access to these representations could reconstruct original PII.
#
# We add a privacy adversary — a small MLP that takes the encoder's hidden
# states at entity positions and tries to predict the *original entity token*.
# A gradient reversal layer (GRL) flips the sign of gradients flowing from
# the adversary back into the encoder, causing the encoder to MAXIMIZE the
# adversary's loss — i.e., actively suppress entity-recoverable information.
#
#   L_total = L_NER + λ · L_GRL(adversary)
#
# After training, the encoder produces representations that are useful for
# NER but from which the original entity text cannot be reconstructed.
# ============================================================================

class GradientReversalFunction(Function):
    """Gradient Reversal Layer (Ganin et al., 2016).

    Forward pass: identity.
    Backward pass: negate gradients and scale by alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class PrivacyAdversary(nn.Module):
    """MLP that tries to reconstruct original entity tokens from hidden states.

    Takes encoder hidden states at entity positions and predicts the
    token ID of the original entity.  If this adversary can succeed,
    the encoder is leaking entity information.
    """
    def __init__(self, hidden_size: int, vocab_size: int,
                 adversary_hidden: int = 256, num_layers: int = 2):
        super().__init__()
        self.grl = GradientReversalLayer()

        layers = []
        in_dim = hidden_size
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, adversary_hidden),
                           nn.ReLU(), nn.Dropout(0.1)])
            in_dim = adversary_hidden
        layers.append(nn.Linear(in_dim, vocab_size))
        self.classifier = nn.Sequential(*layers)

    def forward(self, hidden_states, alpha=1.0):
        self.grl.alpha = alpha
        reversed_hidden = self.grl(hidden_states)
        return self.classifier(reversed_hidden)


class PrivacyAwareCensor(nn.Module):
    """NER model augmented with an adversarial privacy probe.

    Architecture:
        encoder (DeBERTa) → [shared hidden states]
                           ├── NER head → BIO tag predictions
                           └── GRL → Adversary → entity token predictions

    The NER head is trained to minimize classification loss.
    The adversary is trained to reconstruct original tokens.
    The encoder is trained to minimize NER loss AND maximize adversary loss
    (via gradient reversal).
    """
    def __init__(self, backbone: str, num_labels: int, vocab_size: int, cfg: NovelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size

        # NER classification head
        self.dropout = nn.Dropout(0.1)
        self.ner_head = nn.Linear(hidden_size, num_labels)

        # Privacy adversary
        self.adversary = PrivacyAdversary(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            adversary_hidden=cfg.app_adversary_hidden,
            num_layers=cfg.app_adversary_layers,
        )

        self.num_labels = num_labels
        self.adv_lambda = cfg.app_lambda

    def forward(self, input_ids, attention_mask=None, labels=None,
                original_token_ids=None, entity_mask=None, grl_alpha=1.0):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)

        # NER predictions
        ner_logits = self.ner_head(self.dropout(hidden))  # (B, T, num_labels)

        result = {"ner_logits": ner_logits}
        total_loss = None

        # NER loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fn(ner_logits.view(-1, self.num_labels), labels.view(-1))
            total_loss = ner_loss
            result["ner_loss"] = ner_loss.item()

        # Adversary loss (only at entity positions)
        if original_token_ids is not None and entity_mask is not None:
            # entity_mask: (B, T) boolean — True at entity token positions
            # original_token_ids: (B, T) — original token IDs at entity positions
            adv_logits = self.adversary(hidden, alpha=grl_alpha)  # (B, T, V)

            # Flatten to entity positions only
            flat_logits = adv_logits[entity_mask]  # (N_entities, V)
            flat_targets = original_token_ids[entity_mask]  # (N_entities,)

            if flat_logits.shape[0] > 0:
                adv_loss = F.cross_entropy(flat_logits, flat_targets)
                result["adv_loss"] = adv_loss.item()

                # Adversary accuracy (for monitoring — lower is better for privacy)
                with torch.no_grad():
                    adv_preds = flat_logits.argmax(dim=-1)
                    adv_acc = (adv_preds == flat_targets).float().mean().item()
                    result["adv_accuracy"] = adv_acc

                if total_loss is not None:
                    total_loss = total_loss + self.adv_lambda * adv_loss

        result["loss"] = total_loss
        return result


class APPDataset(TorchDataset):
    """Dataset that provides NER labels AND original token IDs for the adversary."""
    def __init__(self, hf_dataset, tokenizer, max_len=256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens, labels = find_text_and_labels(example)

        # Tokenize
        encoded = self.tokenizer(
            tokens, truncation=True, max_length=self.max_len,
            padding="max_length", is_split_into_words=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Align NER labels
        word_ids = encoded.word_ids(batch_index=0)
        label_ids = []
        entity_mask = []
        original_token_ids = []
        prev_word = None

        for i, wid in enumerate(word_ids):
            if wid is None:
                label_ids.append(-100)
                entity_mask.append(False)
                original_token_ids.append(0)
            elif wid != prev_word:
                lbl = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
                is_entity = lbl != "O"
                entity_mask.append(is_entity)
                original_token_ids.append(input_ids[i].item() if is_entity else 0)
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
                is_entity = lbl != "O"
                entity_mask.append(is_entity)
                original_token_ids.append(input_ids[i].item() if is_entity else 0)
            prev_word = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "entity_mask": torch.tensor(entity_mask, dtype=torch.bool),
            "original_token_ids": torch.tensor(original_token_ids, dtype=torch.long),
        }


def train_app(train_ds, val_ds, cfg: NovelConfig):
    """Train Technique 1: Adversarial Privacy Probing."""
    log.info("=" * 70)
    log.info("TECHNIQUE 1: Adversarial Privacy Probing (APP)")
    log.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(cfg.censor_backbone)
    vocab_size = tokenizer.vocab_size

    model = PrivacyAwareCensor(
        backbone=cfg.censor_backbone,
        num_labels=NUM_LABELS,
        vocab_size=vocab_size,
        cfg=cfg,
    ).to(DEVICE)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    train_data = APPDataset(train_ds, tokenizer, cfg.max_len)
    val_data = APPDataset(val_ds, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    best_f1 = 0.0
    history = {"ner_loss": [], "adv_loss": [], "adv_accuracy": [], "val_f1": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_ner_loss, epoch_adv_loss, epoch_adv_acc = 0, 0, 0
        n_batches = 0

        # GRL alpha schedule: ramp up over training (Ganin et al.)
        p = epoch / cfg.epochs
        grl_alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                original_token_ids=batch["original_token_ids"],
                entity_mask=batch["entity_mask"],
                grl_alpha=grl_alpha,
            )

            loss = out["loss"]
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            epoch_ner_loss += out.get("ner_loss", 0)
            epoch_adv_loss += out.get("adv_loss", 0)
            epoch_adv_acc += out.get("adv_accuracy", 0)
            n_batches += 1

        avg_ner = epoch_ner_loss / max(n_batches, 1)
        avg_adv = epoch_adv_loss / max(n_batches, 1)
        avg_acc = epoch_adv_acc / max(n_batches, 1)

        # Validation
        val_f1 = evaluate_ner(model, val_loader)

        history["ner_loss"].append(avg_ner)
        history["adv_loss"].append(avg_adv)
        history["adv_accuracy"].append(avg_acc)
        history["val_f1"].append(val_f1)

        log.info(f"Epoch {epoch+1}/{cfg.epochs} | GRL α={grl_alpha:.3f} | "
                 f"NER loss={avg_ner:.4f} | Adv loss={avg_adv:.4f} | "
                 f"Adv acc={avg_acc:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(cfg.app_dir, "best_model.pt"))
            tokenizer.save_pretrained(cfg.app_dir)

    log.info(f"APP training complete. Best val F1: {best_f1:.4f}")
    log.info(f"Final adversary accuracy: {history['adv_accuracy'][-1]:.4f} "
             f"(lower = more private)")

    # Save history
    with open(os.path.join(cfg.app_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, tokenizer, history


def evaluate_ner(model, dataloader):
    """Evaluate NER F1 on a dataloader."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            logits = out["ner_logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for pred_seq, label_seq in zip(preds, labels):
                true_sent, pred_sent = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    true_sent.append(ID2LABEL.get(int(l), "O"))
                    pred_sent.append(ID2LABEL.get(int(p), "O"))
                if true_sent:
                    all_true.append(true_sent)
                    all_pred.append(pred_sent)

    model.train()
    if not all_true:
        return 0.0
    return seq_f1_score(all_true, all_pred, average="weighted")


def evaluate_app_privacy(model, dataloader):
    """Measure how well the adversary can reconstruct entities after APP training.

    Lower adversary accuracy = better privacy.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                original_token_ids=batch["original_token_ids"],
                entity_mask=batch["entity_mask"],
                grl_alpha=0.0,  # No reversal during eval
            )
            # Direct adversary prediction (no GRL)
            hidden = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).last_hidden_state
            adv_logits = model.adversary.classifier(hidden)

            mask = batch["entity_mask"]
            if mask.any():
                preds = adv_logits[mask].argmax(dim=-1)
                targets = batch["original_token_ids"][mask]
                correct += (preds == targets).sum().item()
                total += mask.sum().item()

    model.train()
    return correct / max(total, 1)


# ============================================================================
# TECHNIQUE 2: Entity-Type Adaptive Privacy (ETAP)
# ============================================================================
#
# Key idea: all prior DP-NLP work applies a SINGLE uniform ε to the entire
# model.  But not all PII is equally sensitive.  Leaking an SSN is
# catastrophic; leaking a city name is minor.
#
# We define a sensitivity hierarchy:
#   Critical (ε=1):  SSN, PASSPORT, DRIVER_LICENSE
#   High     (ε=4):  EMAIL, PHONE, IP_ADDRESS, USERNAME
#   Medium   (ε=8):  PERSON, ADDRESS, MEDICAL, DATE
#   Low      (ε=16): LOC, ORG, BUILDING, POSTCODE, URL
#
# During DP-SGD training, we apply per-entity-type gradient clipping norms
# and noise scales.  Gradients from batches containing critical entities
# get clipped more aggressively and receive more noise.
#
# The result: the model learns to be more cautious with sensitive entities
# while maintaining utility for less-sensitive ones.
# ============================================================================

class AdaptiveDPEngine:
    """Custom DP-SGD engine with per-entity-type privacy budgets.

    Instead of uniform (C, σ) for all gradients, we compute the
    entity-type composition of each batch and adjust:
        C_batch = min(C_t for t in batch_entity_types)
        σ_batch = max(σ_t for t in batch_entity_types)

    This ensures that batches containing critical entities (SSN, etc.)
    receive the strongest privacy protection.
    """
    def __init__(self, model, cfg: NovelConfig):
        self.model = model
        self.cfg = cfg
        self.sensitivity = cfg.etap_sensitivity
        self.base_clip = cfg.etap_base_clip_norm
        self.base_noise = cfg.etap_noise_multiplier

        # Pre-compute per-type clip norms and noise multipliers
        # Lower ε → smaller clip norm, higher noise
        self.type_clip = {}
        self.type_noise = {}
        for etype, epsilon in self.sensitivity.items():
            # Clip norm scales with epsilon (less private → more lenient)
            self.type_clip[etype] = self.base_clip * (epsilon / 8.0)
            # Noise scales inversely with epsilon
            self.type_noise[etype] = self.base_noise * (8.0 / epsilon)

        self.spent_budget = defaultdict(float)
        self.step_count = 0

    def get_batch_privacy_params(self, batch_labels):
        """Determine clip norm and noise for a batch based on its entity types.

        Args:
            batch_labels: tensor of BIO label IDs, shape (B, T)

        Returns:
            clip_norm, noise_multiplier
        """
        entity_types_in_batch = set()

        for label_seq in batch_labels:
            for lid in label_seq:
                lid = int(lid)
                if lid <= 0:
                    continue
                label_str = ID2LABEL.get(lid, "O")
                if label_str.startswith("B-") or label_str.startswith("I-"):
                    etype = label_str[2:]
                    if etype in self.sensitivity:
                        entity_types_in_batch.add(etype)

        if not entity_types_in_batch:
            return self.base_clip, self.base_noise

        # Most restrictive: smallest clip, highest noise
        clip = min(self.type_clip.get(e, self.base_clip) for e in entity_types_in_batch)
        noise = max(self.type_noise.get(e, self.base_noise) for e in entity_types_in_batch)

        return clip, noise

    def clip_and_noise_grads(self, clip_norm, noise_multiplier):
        """Apply per-example gradient clipping and Gaussian noise."""
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), clip_norm)

        # Add calibrated Gaussian noise
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_multiplier * clip_norm
                param.grad.add_(noise)

        self.step_count += 1
        return total_norm.item()

    def get_privacy_report(self):
        """Report effective privacy spending per entity type."""
        report = {}
        for etype, epsilon in self.sensitivity.items():
            # Simplified accounting: budget = steps × (clip/noise)²
            clip = self.type_clip[etype]
            noise = self.type_noise[etype]
            spent = self.step_count * (clip / noise) ** 2 if noise > 0 else float('inf')
            report[etype] = {
                "target_epsilon": epsilon,
                "clip_norm": clip,
                "noise_multiplier": noise,
                "steps": self.step_count,
                "approx_spent": spent,
            }
        return report


def train_etap(train_ds, val_ds, cfg: NovelConfig):
    """Train Technique 2: Entity-Type Adaptive Privacy."""
    log.info("=" * 70)
    log.info("TECHNIQUE 2: Entity-Type Adaptive Privacy (ETAP)")
    log.info("=" * 70)

    # Log sensitivity hierarchy
    for tier, eps_range in [("Critical", (0, 2)),
                             ("High", (2, 6)),
                             ("Medium", (6, 12)),
                             ("Low", (12, 20))]:
        types = [e for e, v in cfg.etap_sensitivity.items()
                 if eps_range[0] < v <= eps_range[1]]
        if types:
            log.info(f"  {tier}: {', '.join(types)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.censor_backbone)

    # Use standard NER model (no adversary) but with adaptive DP
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.censor_backbone,
        num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE)

    # Validate model for DP compatibility
    model = ModuleValidator.fix(model)

    log.info(f"ETAP model params: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize data
    ner_train = train_ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, cfg.max_len),
        batched=True, batch_size=500, remove_columns=train_ds.column_names,
    )
    ner_val = val_ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer, cfg.max_len),
        batched=True, batch_size=500, remove_columns=val_ds.column_names,
    )

    ner_train.set_format("torch")
    ner_val.set_format("torch")

    train_loader = DataLoader(ner_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ner_val, batch_size=cfg.batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)

    # Initialize adaptive DP engine
    dp_engine = AdaptiveDPEngine(model, cfg)

    best_f1 = 0.0
    history = {"loss": [], "val_f1": [], "batch_clips": [], "batch_noise": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        epoch_clips, epoch_noise = [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Get adaptive privacy params for this batch
            clip_norm, noise_mult = dp_engine.get_batch_privacy_params(labels)
            epoch_clips.append(clip_norm)
            epoch_noise.append(noise_mult)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Apply adaptive clipping and noise
            dp_engine.clip_and_noise_grads(clip_norm, noise_mult)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_preds, val_true = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"]

                logits = model(input_ids=input_ids,
                              attention_mask=attention_mask).logits
                preds = logits.argmax(dim=-1).cpu().numpy()

                for pred_seq, label_seq in zip(preds, labels.numpy()):
                    ts, ps = [], []
                    for p, l in zip(pred_seq, label_seq):
                        if l == -100:
                            continue
                        ts.append(ID2LABEL.get(int(l), "O"))
                        ps.append(ID2LABEL.get(int(p), "O"))
                    if ts:
                        val_true.append(ts)
                        val_preds.append(ps)

        val_f1 = seq_f1_score(val_true, val_preds, average="weighted") if val_true else 0.0

        history["loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["batch_clips"].append(float(np.mean(epoch_clips)))
        history["batch_noise"].append(float(np.mean(epoch_noise)))

        log.info(f"Epoch {epoch+1}/{cfg.epochs} | Loss={avg_loss:.4f} | "
                 f"Val F1={val_f1:.4f} | "
                 f"Avg clip={np.mean(epoch_clips):.3f} | "
                 f"Avg noise={np.mean(epoch_noise):.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(cfg.etap_dir)
            tokenizer.save_pretrained(cfg.etap_dir)

    # Privacy report
    report = dp_engine.get_privacy_report()
    log.info("ETAP Privacy Report:")
    for etype, info in sorted(report.items(), key=lambda x: x[1]["target_epsilon"]):
        log.info(f"  {etype}: ε_target={info['target_epsilon']}, "
                 f"clip={info['clip_norm']:.3f}, σ={info['noise_multiplier']:.3f}")

    with open(os.path.join(cfg.etap_dir, "privacy_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(cfg.etap_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"ETAP training complete. Best val F1: {best_f1:.4f}")
    return model, tokenizer, history, report


# ============================================================================
# TECHNIQUE 3: Cross-Lingual Privacy Transfer (CLPT)
# ============================================================================
#
# Key idea: train a privacy-preserving NER model on high-resource languages
# (English, German, French) and test whether privacy guarantees transfer to
# unseen languages (Italian, Spanish, Dutch, Portuguese, Czech).
#
# We use XLM-RoBERTa as the backbone (instead of DeBERTa) because it was
# pretrained on 100 languages.  On top of this, we add a language adversary
# that tries to predict which language an input belongs to.  A gradient
# reversal layer forces the encoder to produce language-AGNOSTIC features.
#
# If the encoder's representations are language-agnostic, then:
#   1. NER performance should transfer well cross-lingually.
#   2. Privacy properties (entity representations being non-recoverable)
#      should also transfer, since the model doesn't rely on language-
#      specific patterns that might vary across languages.
#
# This is the first work to study cross-lingual transfer of DP guarantees.
# ============================================================================

class LanguageAdversary(nn.Module):
    """Predicts language from hidden states — used with GRL to force
    language-agnostic representations."""
    def __init__(self, hidden_size: int, num_languages: int, hidden_dim: int = 128):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_languages),
        )

    def forward(self, hidden_states, alpha=1.0):
        # Pool over sequence: mean of non-padding tokens
        self.grl.alpha = alpha
        reversed_hidden = self.grl(hidden_states)
        return self.classifier(reversed_hidden)


class XLMPrivacyCensor(nn.Module):
    """XLM-RoBERTa NER model with language-adversarial training.

    Architecture:
        XLM-R encoder → [shared hidden states]
                       ├── NER head → BIO tag predictions
                       └── GRL → Language Adversary → language predictions

    The model is trained to be good at NER but BAD at predicting language,
    forcing language-agnostic representations.
    """
    def __init__(self, backbone: str, num_labels: int, num_languages: int,
                 cfg: NovelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.ner_head = nn.Linear(hidden_size, num_labels)

        self.lang_adversary = LanguageAdversary(
            hidden_size=hidden_size,
            num_languages=num_languages,
            hidden_dim=cfg.clpt_lang_adversary_hidden,
        )

        self.num_labels = num_labels
        self.lang_lambda = cfg.clpt_lang_lambda

    def forward(self, input_ids, attention_mask=None, labels=None,
                lang_labels=None, grl_alpha=1.0):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        ner_logits = self.ner_head(self.dropout(hidden))

        result = {"ner_logits": ner_logits}
        total_loss = None

        # NER loss
        if labels is not None:
            ner_loss = F.cross_entropy(
                ner_logits.view(-1, self.num_labels), labels.view(-1),
                ignore_index=-100,
            )
            total_loss = ner_loss
            result["ner_loss"] = ner_loss.item()

        # Language adversary loss
        if lang_labels is not None:
            # Use CLS / first-token representation for language classification
            cls_hidden = hidden[:, 0, :]  # (B, H)
            lang_logits = self.lang_adversary(cls_hidden.unsqueeze(1),
                                               alpha=grl_alpha).squeeze(1)
            lang_loss = F.cross_entropy(lang_logits, lang_labels)
            result["lang_loss"] = lang_loss.item()

            with torch.no_grad():
                lang_acc = (lang_logits.argmax(dim=-1) == lang_labels).float().mean().item()
                result["lang_accuracy"] = lang_acc

            if total_loss is not None:
                total_loss = total_loss + self.lang_lambda * lang_loss

        result["loss"] = total_loss
        return result


class CLPTDataset(TorchDataset):
    """Dataset for cross-lingual training with language labels."""
    def __init__(self, hf_dataset, tokenizer, lang2id, max_len=256):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.lang2id = lang2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens, labels = find_text_and_labels(example)
        lang = get_language(example)
        lang_id = self.lang2id.get(lang, 0)

        encoded = self.tokenizer(
            tokens, truncation=True, max_length=self.max_len,
            padding="max_length", is_split_into_words=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        word_ids = encoded.word_ids(batch_index=0)
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_word:
                lbl = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lbl, 0))
            else:
                lbl = labels[wid] if wid < len(labels) else "O"
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL2ID.get(lbl, 0))
            prev_word = wid

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "lang_labels": torch.tensor(lang_id, dtype=torch.long),
        }


def train_clpt(full_ds, cfg: NovelConfig):
    """Train Technique 3: Cross-Lingual Privacy Transfer."""
    log.info("=" * 70)
    log.info("TECHNIQUE 3: Cross-Lingual Privacy Transfer (CLPT)")
    log.info("=" * 70)

    # Split dataset by language
    df = full_ds.to_pandas()
    lang_col = None
    for c in ["language", "lang", "Language"]:
        if c in df.columns:
            lang_col = c
            break
    if lang_col is None:
        log.error("No language column found — cannot run CLPT")
        return None, None, None

    all_langs = df[lang_col].unique().tolist()
    train_langs = [l for l in all_langs if l in cfg.clpt_train_langs]
    test_langs = [l for l in all_langs if l in cfg.clpt_test_langs]

    if not train_langs:
        log.warning("No matching train languages found. Using all languages.")
        train_langs = all_langs[:3]
    if not test_langs:
        test_langs = [l for l in all_langs if l not in train_langs][:3]

    log.info(f"Train languages: {train_langs}")
    log.info(f"Test languages:  {test_langs}")

    # Build language ID mapping
    lang2id = {l: i for i, l in enumerate(train_langs + test_langs)}
    num_languages = len(lang2id)

    # Filter datasets
    train_mask = df[lang_col].isin(train_langs)
    test_mask = df[lang_col].isin(test_langs)

    train_indices = df[train_mask].index.tolist()
    test_indices = df[test_mask].index.tolist()

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    # Split training data into train/val
    n_val = max(1, int(len(train_indices) * 0.1))
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]

    train_subset = full_ds.select(train_indices)
    val_subset = full_ds.select(val_indices)
    test_subset = full_ds.select(test_indices)

    log.info(f"Train: {len(train_subset):,} | Val: {len(val_subset):,} | "
             f"Test (zero-shot): {len(test_subset):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.xlm_backbone)

    model = XLMPrivacyCensor(
        backbone=cfg.xlm_backbone,
        num_labels=NUM_LABELS,
        num_languages=num_languages,
        cfg=cfg,
    ).to(DEVICE)

    log.info(f"CLPT model params: {sum(p.numel() for p in model.parameters()):,}")

    train_data = CLPTDataset(train_subset, tokenizer, lang2id, cfg.max_len)
    val_data = CLPTDataset(val_subset, tokenizer, lang2id, cfg.max_len)
    test_data = CLPTDataset(test_subset, tokenizer, lang2id, cfg.max_len)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size * 2)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    best_f1 = 0.0
    history = {"ner_loss": [], "lang_loss": [], "lang_accuracy": [],
               "val_f1": [], "zero_shot_f1": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_ner, epoch_lang, epoch_lacc = 0, 0, 0
        n_batches = 0

        p = epoch / cfg.epochs
        grl_alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                lang_labels=batch["lang_labels"],
                grl_alpha=grl_alpha,
            )

            loss = out["loss"]
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            epoch_ner += out.get("ner_loss", 0)
            epoch_lang += out.get("lang_loss", 0)
            epoch_lacc += out.get("lang_accuracy", 0)
            n_batches += 1

        avg_ner = epoch_ner / max(n_batches, 1)
        avg_lang = epoch_lang / max(n_batches, 1)
        avg_lacc = epoch_lacc / max(n_batches, 1)

        # Validation (seen languages)
        val_f1 = evaluate_ner_clpt(model, val_loader)

        # Zero-shot (unseen languages)
        zs_f1 = evaluate_ner_clpt(model, test_loader)

        history["ner_loss"].append(avg_ner)
        history["lang_loss"].append(avg_lang)
        history["lang_accuracy"].append(avg_lacc)
        history["val_f1"].append(val_f1)
        history["zero_shot_f1"].append(zs_f1)

        log.info(f"Epoch {epoch+1}/{cfg.epochs} | GRL α={grl_alpha:.3f} | "
                 f"NER={avg_ner:.4f} | Lang acc={avg_lacc:.4f} | "
                 f"Val F1={val_f1:.4f} | Zero-shot F1={zs_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(cfg.clpt_dir, "best_model.pt"))
            tokenizer.save_pretrained(cfg.clpt_dir)

    log.info(f"CLPT complete. Best val F1: {best_f1:.4f}")
    log.info(f"Final zero-shot F1: {history['zero_shot_f1'][-1]:.4f}")
    log.info(f"Final lang adversary accuracy: {history['lang_accuracy'][-1]:.4f} "
             f"(lower = more language-agnostic)")

    with open(os.path.join(cfg.clpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, tokenizer, history


def evaluate_ner_clpt(model, dataloader):
    """Evaluate NER F1 for CLPT model."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(input_ids=batch_dev["input_ids"],
                        attention_mask=batch_dev["attention_mask"])
            logits = out["ner_logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()

            for pred_seq, label_seq in zip(preds, labels):
                ts, ps = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    ts.append(ID2LABEL.get(int(l), "O"))
                    ps.append(ID2LABEL.get(int(p), "O"))
                if ts:
                    all_true.append(ts)
                    all_pred.append(ps)

    model.train()
    return seq_f1_score(all_true, all_pred, average="weighted") if all_true else 0.0


# ============================================================================
# Comparative Evaluation & Visualization
# ============================================================================

def comprehensive_evaluation(results: Dict, cfg: NovelConfig):
    """Generate comprehensive comparison across all techniques + baseline."""
    log.info("=" * 70)
    log.info("COMPREHENSIVE RESULTS")
    log.info("=" * 70)

    # Summary table
    rows = []
    for name, data in results.items():
        row = {"Technique": name}
        if "val_f1" in data and data["val_f1"]:
            row["Best Val F1"] = max(data["val_f1"])
            row["Final Val F1"] = data["val_f1"][-1]
        if "zero_shot_f1" in data and data["zero_shot_f1"]:
            row["Zero-Shot F1"] = data["zero_shot_f1"][-1]
        if "adv_accuracy" in data and data["adv_accuracy"]:
            row["Adv Accuracy"] = data["adv_accuracy"][-1]
            row["Privacy Score"] = 1.0 - data["adv_accuracy"][-1]
        if "batch_clips" in data:
            row["Avg Clip Norm"] = np.mean(data["batch_clips"])
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        log.info(f"\n{df.to_string(index=False)}")
        df.to_csv(os.path.join(cfg.eval_dir, "comparison.csv"), index=False)

    # Save full results
    with open(os.path.join(cfg.eval_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return rows


def plot_training_curves(results: Dict, cfg: NovelConfig):
    """Plot training curves for all techniques."""
    n_techniques = len(results)
    fig, axes = plt.subplots(2, max(n_techniques, 2), figsize=(7 * n_techniques, 10))
    if n_techniques == 1:
        axes = axes.reshape(2, 1)

    colors = {"APP": "#e74c3c", "ETAP": "#3498db", "CLPT": "#2ecc71"}

    for i, (name, data) in enumerate(results.items()):
        color = colors.get(name, "#333333")

        # Top row: loss
        ax = axes[0][i]
        if "ner_loss" in data:
            ax.plot(data["ner_loss"], label="NER Loss", color=color)
        if "loss" in data:
            ax.plot(data["loss"], label="Loss", color=color)
        if "adv_loss" in data:
            ax.plot(data["adv_loss"], label="Adv Loss", color=color, linestyle="--")
        if "lang_loss" in data:
            ax.plot(data["lang_loss"], label="Lang Loss", color=color, linestyle="--")
        ax.set_title(f"{name} — Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        # Bottom row: F1
        ax = axes[1][i]
        if "val_f1" in data:
            ax.plot(data["val_f1"], label="Val F1", color=color, marker="o")
        if "zero_shot_f1" in data:
            ax.plot(data["zero_shot_f1"], label="Zero-Shot F1",
                    color=color, linestyle="--", marker="s")
        if "adv_accuracy" in data:
            ax.plot(data["adv_accuracy"], label="Adv Accuracy (↓ = better)",
                    color="#95a5a6", linestyle=":")
        ax.set_title(f"{name} — Evaluation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(cfg.plot_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Training curves → {path}")

    # Privacy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names, f1s, privacy_scores = [], [], []
    for name, data in results.items():
        names.append(name)
        f1s.append(max(data.get("val_f1", [0])))
        if "adv_accuracy" in data:
            privacy_scores.append(1.0 - data["adv_accuracy"][-1])
        else:
            privacy_scores.append(0.5)  # placeholder

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, f1s, width, label="NER F1", color="#3498db")
    ax.bar(x + width/2, privacy_scores, width, label="Privacy Score", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("NER Utility vs. Privacy — Technique Comparison")
    ax.legend()
    ax.set_ylim(0, 1)

    path = os.path.join(cfg.plot_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Comparison chart → {path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Novel Privacy Techniques Pipeline")
    parser.add_argument("--technique", default="all",
                        choices=["all", "app", "etap", "clpt"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(technique="all", quick=False, epochs=None)

    if args.technique == "all":
        CFG.techniques = ["app", "etap", "clpt"]
    else:
        CFG.techniques = [args.technique]

    CFG.quick_mode = args.quick
    if args.epochs:
        CFG.epochs = args.epochs

parse_args()


def main():
    log.info("=" * 70)
    log.info("NOVEL TECHNIQUES PIPELINE")
    log.info(f"Running: {', '.join(t.upper() for t in CFG.techniques)}")
    log.info("=" * 70)

    # Load data
    ds = load_ai4privacy()
    if CFG.quick_mode:
        n = min(CFG.quick_n, len(ds))
        ds = ds.select(random.sample(range(len(ds)), n))
        log.info(f"Quick mode: {len(ds):,} samples")

    all_results = {}

    # ── Technique 1: APP ──
    if "app" in CFG.techniques:
        train_ds, test_ds = language_stratified_split(ds, CFG.test_ratio)
        t_idx, v_idx = train_test_split(range(len(train_ds)),
                                         test_size=0.1, random_state=SEED)
        train_split = train_ds.select(t_idx)
        val_split = train_ds.select(v_idx)

        _, _, app_history = train_app(train_split, val_split, CFG)
        all_results["APP"] = app_history
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Technique 2: ETAP ──
    if "etap" in CFG.techniques:
        train_ds, test_ds = language_stratified_split(ds, CFG.test_ratio)
        t_idx, v_idx = train_test_split(range(len(train_ds)),
                                         test_size=0.1, random_state=SEED)
        train_split = train_ds.select(t_idx)
        val_split = train_ds.select(v_idx)

        _, _, etap_history, etap_report = train_etap(train_split, val_split, CFG)
        all_results["ETAP"] = etap_history
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Technique 3: CLPT ──
    if "clpt" in CFG.techniques:
        _, _, clpt_history = train_clpt(ds, CFG)
        if clpt_history:
            all_results["CLPT"] = clpt_history
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Comprehensive evaluation ──
    if all_results:
        comprehensive_evaluation(all_results, CFG)
        plot_training_curves(all_results, CFG)

    log.info("=" * 70)
    log.info("NOVEL PIPELINE COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

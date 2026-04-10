#!/usr/bin/env python3
"""
run_all.py  —  Train · Push · Evaluate · Zip
=============================================
Single entry-point that:
  1. Trains Combo 2 (BART filler)  via train_adv.py  --combo 2
  2. Pushes Combo 2 hardened filler to HuggingFace
  3. Evaluates Combo 2             via evaluate_adv.py --combo 2
  4. Trains Combo 1 (DeBERTa-MLM filler)
  5. Pushes Combo 1 hardened filler to HuggingFace
  6. Evaluates Combo 1
  7. Zips results/ + logs/ + training histories → one downloadable .zip

All models are pulled from HuggingFace automatically.
Dataset is pulled from HuggingFace if not present locally.

Usage
-----
    export HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens
    # or: huggingface-cli login     # token is read from the HF cache if HF_TOKEN is unset
    export HF_USERNAME=your_hf_username   # optional; else inferred from the token (whoami)
    python3 run_all.py                    # full pipeline, both combos
    python3 run_all.py --combos 2         # only Combo 2
    python3 run_all.py --eval_only        # skip training, evaluate + zip
    python3 run_all.py --eval_batch 4     # eval batch size (default 4)
    python3 run_all.py --fast              # timeboxed: 12k train, 1 epoch, batch/accum 4, smaller eval
    python3 run_all.py --max_train_samples 8000 --train_epochs 1 \\
        --train_batch_size 4 --train_grad_accum 4   # manual overrides

Hub auth: set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` (never commit tokens).
``HF_USERNAME`` sets where models are pushed; if omitted, the script uses the
Hub ``whoami`` name for the logged-in token.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent

# ── Per-combo metadata ────────────────────────────────────────────────────────
COMBO_CONFIG = {
    1: {
        "label":            "DeBERTa-MLM filler",
        "filler_id":        "Xyren2005/pii-ner-filler_deberta-filler",
        "output_repo_name": "pii-filler-deberta-adv-hardened",
        "ckpt_dir":         BASE_DIR / "checkpoints" / "combo1",
        "model_class":      "masked_lm",
    },
    2: {
        "label":            "BART seq2seq filler",
        "filler_id":        "Xyren2005/pii-ner-filler_bart-base",
        "output_repo_name": "pii-filler-bart-adv-hardened",
        "ckpt_dir":         BASE_DIR / "checkpoints" / "combo2",
        "model_class":      "seq2seq",
    },
}


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    try:
        from huggingface_hub import get_token

        t = (get_token() or "").strip()
        return t or None
    except Exception:
        return None


def _hf_username() -> str:
    explicit = (os.environ.get("HF_USERNAME") or "").strip()
    if explicit:
        return explicit
    tok = _hf_token()
    if tok:
        try:
            return HfApi(token=tok).whoami()["name"]
        except Exception:
            pass
    raise RuntimeError(
        "Set HF_USERNAME to your Hugging Face username, or set HF_TOKEN / "
        "HUGGING_FACE_HUB_TOKEN so the script can resolve it via whoami()."
    )


def _output_repo_id(combo: int) -> str:
    return f"{_hf_username()}/{COMBO_CONFIG[combo]['output_repo_name']}"

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--combos",     nargs="+", type=int, default=[2, 1],
                help="Combos to run in order (default: 2 1 — BART first, DeBERTa second)")
ap.add_argument("--eval_only",  action="store_true",
                help="Skip training; only evaluate + zip")
ap.add_argument("--eval_batch", type=int, default=4,
                help="Batch size for evaluate_adv.py (default 4; lower if OOM)")
ap.add_argument("--max_train_samples", type=int, default=None,
                help="Passed to train_adv.py (cap training rows)")
ap.add_argument("--train_epochs", type=int, default=None,
                help="Passed to train_adv.py --epochs")
ap.add_argument("--train_batch_size", type=int, default=None,
                help="Passed to train_adv.py --batch_size")
ap.add_argument("--train_grad_accum", type=int, default=None,
                help="Passed to train_adv.py --grad_accum")
ap.add_argument("--max_eval_samples", type=int, default=None,
                help="Passed to train_adv.py (smaller eval set → faster val)")
ap.add_argument("--fast", action="store_true",
                help="Timeboxed training: 12k train rows, 1 epoch, batch/accum 4, 2k eval "
                     "(only fills unset flags; use explicit args to override pieces)")
ARGS = ap.parse_args()

if ARGS.fast:
    if ARGS.max_train_samples is None:
        ARGS.max_train_samples = 12_000
    if ARGS.train_epochs is None:
        ARGS.train_epochs = 1
    if ARGS.train_batch_size is None:
        ARGS.train_batch_size = 4
    if ARGS.train_grad_accum is None:
        ARGS.train_grad_accum = 4
    if ARGS.max_eval_samples is None:
        ARGS.max_eval_samples = 2_048


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str):
    bar = "=" * 65
    print(f"\n{bar}\n  {msg}\n{bar}", flush=True)


def _train_adv_cmd(combo: int) -> list:
    cmd = [sys.executable, str(BASE_DIR / "train_adv.py"), "--combo", str(combo)]
    if ARGS.max_train_samples is not None:
        cmd += ["--max_train_samples", str(ARGS.max_train_samples)]
    if ARGS.train_epochs is not None:
        cmd += ["--epochs", str(ARGS.train_epochs)]
    if ARGS.train_batch_size is not None:
        cmd += ["--batch_size", str(ARGS.train_batch_size)]
    if ARGS.train_grad_accum is not None:
        cmd += ["--grad_accum", str(ARGS.train_grad_accum)]
    if ARGS.max_eval_samples is not None:
        cmd += ["--max_eval_samples", str(ARGS.max_eval_samples)]
    return cmd


def run_step(desc: str, cmd: list) -> bool:
    """Run a subprocess, returning True if it succeeded."""
    banner(desc)
    result = subprocess.run([str(c) for c in cmd], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\n[WARNING] Step '{desc}' exited with code {result.returncode}.",
              flush=True)
        return False
    print(f"[OK] {desc}", flush=True)
    return True


def push_combo_to_hf(combo: int) -> bool:
    """
    Load the best/final checkpoint for `combo`, reconstruct the full model,
    and push model + tokenizer to HuggingFace Hub.
    """
    cfg      = COMBO_CONFIG[combo]
    ckpt_dir = cfg["ckpt_dir"]
    hf_token = _hf_token()
    if not hf_token:
        print("[WARNING] HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set — skipping HF push "
              f"for Combo {combo}.", flush=True)
        return False

    try:
        repo_id = _output_repo_id(combo)
    except RuntimeError as exc:
        print(f"[WARNING] {exc} — skipping HF push for Combo {combo}.", flush=True)
        return False

    # Prefer final_model.pt, fall back to best_model.pt
    final_path = ckpt_dir / "final_model.pt"
    best_path  = ckpt_dir / "best_model.pt"
    ckpt_path  = final_path if final_path.exists() else (
                 best_path  if best_path.exists()  else None)

    if ckpt_path is None:
        print(f"[WARNING] No checkpoint found in {ckpt_dir} — skipping HF push "
              f"for Combo {combo}.", flush=True)
        return False

    banner(f"Pushing Combo {combo} ({cfg['label']}) → {repo_id}")
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

        # Reconstruct model from base + hardened weights
        filler_id = cfg["filler_id"]
        print(f"  Loading base model: {filler_id}", flush=True)
        if cfg["model_class"] == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(filler_id)
        else:
            model = AutoModelForMaskedLM.from_pretrained(filler_id)

        tokenizer = AutoTokenizer.from_pretrained(filler_id, use_fast=True)

        print(f"  Loading checkpoint: {ckpt_path}", flush=True)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Strip DataParallel prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [INFO] Missing keys (expected if partial save): {missing[:5]}")
        if unexpected:
            print(f"  [INFO] Unexpected keys ignored: {unexpected[:5]}")

        print(f"  Pushing to hub …", flush=True)
        model.push_to_hub(
            repo_id,
            token=hf_token,
            commit_message=f"Adversarial hardening — Combo {combo} ({cfg['label']})",
        )
        tokenizer.push_to_hub(repo_id, token=hf_token)

        print(f"[OK] Combo {combo} → https://huggingface.co/{repo_id}", flush=True)
        return True

    except Exception as exc:
        print(f"[WARNING] HF push failed for Combo {combo}: {exc}", flush=True)
        return False


def zip_results() -> Path:
    """
    Bundle results/, logs/, and per-combo training_history.json into one zip.
    Large .pt model checkpoints are excluded to keep the archive small.
    """
    banner("Zipping results")
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = BASE_DIR / f"adv_hardening_results_{ts}.zip"

    include_dirs = [
        BASE_DIR / "results",
        BASE_DIR / "logs",
    ]
    include_files = [
        BASE_DIR / "checkpoints" / "combo1" / "training_history.json",
        BASE_DIR / "checkpoints" / "combo2" / "training_history.json",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in include_dirs:
            if folder.exists():
                for fpath in sorted(folder.rglob("*")):
                    if fpath.is_file():
                        zf.write(fpath, fpath.relative_to(BASE_DIR))
                        print(f"  + {fpath.relative_to(BASE_DIR)}", flush=True)
        for fpath in include_files:
            if fpath.exists():
                zf.write(fpath, fpath.relative_to(BASE_DIR))
                print(f"  + {fpath.relative_to(BASE_DIR)}", flush=True)

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n[OK] Archive: {zip_path}  ({size_mb:.1f} MB)", flush=True)
    return zip_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    py = sys.executable

    banner(
        f"run_all.py  —  combos={ARGS.combos}  eval_only={ARGS.eval_only}  fast={ARGS.fast}"
    )

    for combo in ARGS.combos:
        cfg = COMBO_CONFIG[combo]
        print(f"\n{'─'*65}", flush=True)
        print(f"  COMBO {combo}: {cfg['label']}", flush=True)
        print(f"{'─'*65}", flush=True)

        # ── 1. Train ───────────────────────────────────────────────────────
        if not ARGS.eval_only:
            run_step(
                f"Training Combo {combo} — {cfg['label']}",
                _train_adv_cmd(combo),
            )
            # ── 2. Push hardened model to HF ─────────────────────────────
            push_combo_to_hf(combo)

        # ── 3. Evaluate ───────────────────────────────────────────────────
        run_step(
            f"Evaluating Combo {combo} — {cfg['label']}",
            [py, BASE_DIR / "evaluate_adv.py",
             "--combo", combo,
             "--batch", ARGS.eval_batch],
        )

    # ── 4. Zip all results ─────────────────────────────────────────────────
    zip_path = zip_results()

    banner("ALL DONE")
    print(f"  Results archive: {zip_path}", flush=True)
    print(f"  Hardened models on HuggingFace:")
    for combo in ARGS.combos:
        try:
            rid = _output_repo_id(combo)
            print(f"    Combo {combo}: https://huggingface.co/{rid}")
        except RuntimeError:
            print(
                f"    Combo {combo}: https://huggingface.co/<you>/"
                f"{COMBO_CONFIG[combo]['output_repo_name']}  (set HF_TOKEN + HF_USERNAME)",
                flush=True,
            )
    print(flush=True)


if __name__ == "__main__":
    main()

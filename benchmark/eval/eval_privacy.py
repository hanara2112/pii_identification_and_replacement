"""
SAHA-AL Benchmark — Task 3: Privacy Risk Assessment

Metrics:
  - CRR-3 : Capitalized 3-gram survival rate
  - ERA   : Entity Recovery Attack (retrieval adversary, top-k accuracy)
  - LRR   : LLM Re-identification Rate (generative adversary)
  - UAC   : Unique Attribute Combination rate (k-anonymity proxy)

References:
  ERA uses Sentence-BERT embeddings (Reimers & Gurevych, EMNLP 2019).
  LRR is inspired by Staab et al. (2023) on LLM privacy inference.
  UAC is grounded in k-anonymity (Sweeney, 2002).

Usage:
  python -m eval.eval_privacy \
      --gold data/test.jsonl \
      --pred predictions/predictions_bart-base-pii.jsonl \
      --train data/train.jsonl \
      --output results/eval_privacy_bart.json
"""

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher

from eval.utils import align_records, get_capitalized_ngrams, load_jsonl, normalize_text


# ── CRR-3 ──

def crr3(gold_records, predictions):
    """Capitalized 3-gram survival rate."""
    survived, total = 0, 0

    for g, p in zip(gold_records, predictions):
        orig_3grams = get_capitalized_ngrams(g.get("original_text", ""), n=3)
        pred_text = normalize_text(p.get("anonymized_text")).lower()

        for ng in orig_3grams:
            total += 1
            if " ".join(ng).lower() in pred_text:
                survived += 1

    return round((survived / total * 100) if total else 0, 2)


# ── ERA ──

def entity_recovery_attack(gold_records, predictions, train_records, top_k=5,
                           max_pool_size=500):
    """
    Retrieval-based adversary: embed anonymized text, rank candidate entities
    by cosine similarity, measure if the original entity is recovered.
    Requires: pip install sentence-transformers

    Optimized: pre-encodes entity pools and anonymized texts in batch.
    """
    import random as _rng
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ImportError:
        print("[WARN] sentence-transformers not installed. Skipping ERA.")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build per-type candidate pools from training data
    type_pools_set: dict[str, set[str]] = {}
    for r in train_records:
        for e in r.get("entities", []):
            etype = e.get("type", "UNKNOWN")
            text = e.get("text", "")
            if text:
                type_pools_set.setdefault(etype, set()).add(text)

    # Cap pool size and pre-encode each pool ONCE
    type_pools: dict[str, list[str]] = {}
    type_pool_embs = {}
    print(f"  ERA: Pre-encoding entity pools (max {max_pool_size} per type)...")
    for etype, pool_set in type_pools_set.items():
        pool_list = list(pool_set)
        if len(pool_list) > max_pool_size:
            _rng.seed(42)
            pool_list = _rng.sample(pool_list, max_pool_size)
        type_pools[etype] = pool_list
        type_pool_embs[etype] = model.encode(pool_list, convert_to_tensor=True,
                                              batch_size=256, show_progress_bar=False)
        print(f"    {etype}: {len(pool_list)} candidates encoded")

    # Batch-encode all anonymized texts
    print(f"  ERA: Encoding {len(predictions)} anonymized texts...")
    anon_texts = [p.get("anonymized_text", "") for p in predictions]
    anon_embs = model.encode(anon_texts, convert_to_tensor=True,
                             batch_size=128, show_progress_bar=True)

    top1_hits, top5_hits, total = 0, 0, 0

    for idx, (g, p) in enumerate(zip(gold_records, predictions)):
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            etype = ent.get("type", "UNKNOWN")
            original_val = ent["text"]

            pool = type_pools.get(etype)
            if pool is None or len(pool) < 2:
                continue

            # Ensure original is in pool for fair evaluation
            if original_val not in pool:
                pool = pool + [original_val]
                extra_emb = model.encode([original_val], convert_to_tensor=True)
                import torch
                pool_embs = torch.cat([type_pool_embs[etype], extra_emb], dim=0)
            else:
                pool_embs = type_pool_embs[etype]

            scores = cos_sim(anon_embs[idx].unsqueeze(0), pool_embs)[0]
            ranked = scores.argsort(descending=True).tolist()

            total += 1
            if pool[ranked[0]] == original_val:
                top1_hits += 1
            if original_val in [pool[ranked[i]] for i in range(min(top_k, len(ranked)))]:
                top5_hits += 1

        if (idx + 1) % 100 == 0:
            print(f"  ERA: {idx+1}/{len(gold_records)} records processed "
                  f"(top1={top1_hits}/{total})")

    return {
        "era_top1": round(top1_hits / total * 100, 2) if total else 0,
        "era_top5": round(top5_hits / total * 100, 2) if total else 0,
        "total_evaluated": total,
    }


# ── LRR ──

LRR_PROMPT = """You are given an anonymized text where personal information has been replaced with fake alternatives.

Anonymized text: {anonymized_text}

For each piece of replaced information, try to guess the ORIGINAL value based on contextual clues. Output one guess per line in the format:
REPLACEMENT -> YOUR_GUESS

Only output guesses, nothing else."""


def _is_prequantized(model_name):
    tag = model_name.lower()
    return any(q in tag for q in ("awq", "gptq", "gguf"))


def _load_local_llm(model_name):
    """Load a local HuggingFace model for LRR with 4-bit quantization if available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  LRR: Loading local model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if _is_prequantized(model_name):
        print("  LRR: Pre-quantized model detected (AWQ/GPTQ) — loading directly")
        load_kwargs["dtype"] = torch.float16
    elif torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            print("  LRR: Using bitsandbytes 4-bit quantization")
        except (ImportError, Exception):
            load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _generate_local(model, tokenizer, prompt, max_new_tokens=256):
    """Generate text from a local model using chat template if available."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def llm_reidentification_rate(gold_records, predictions, sample_n=300,
                              model_name="gpt-4o-mini", use_local=False,
                              local_model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Prompt an LLM to recover original entities from anonymized text.
    Measures exact match rate and fuzzy match rate (>0.8 char similarity).

    Args:
        use_local: If True, uses a local HuggingFace model instead of OpenAI API.
        local_model_name: HuggingFace model ID for local inference.
            Good options for Kaggle GPU:
              - "Qwen/Qwen2.5-1.5B-Instruct" (~3GB, fast)
              - "microsoft/Phi-3.5-mini-instruct" (~7GB, stronger)
              - "google/gemma-2-2b-it" (~5GB, good balance)
    """
    local_model, local_tokenizer = None, None

    if use_local:
        try:
            local_model, local_tokenizer = _load_local_llm(local_model_name)
        except Exception as e:
            print(f"[WARN] Failed to load local model: {e}")
            return None
    else:
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            print("[WARN] openai not installed. Use --lrr-local for local model. Skipping LRR.")
            return None

    exact, fuzzy, total = 0, 0, 0
    api_failures = 0

    for idx, (g, p) in enumerate(zip(gold_records[:sample_n], predictions[:sample_n])):
        original_entities = {e["text"] for e in g.get("entities", []) if e.get("text")}
        if not original_entities:
            continue

        prompt_text = LRR_PROMPT.format(anonymized_text=p["anonymized_text"])

        try:
            if use_local:
                raw_output = _generate_local(local_model, local_tokenizer, prompt_text)
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0, max_tokens=512,
                )
                raw_output = response.choices[0].message.content.strip()
        except Exception as e:
            api_failures += 1
            if api_failures <= 3:
                print(f"[WARN] LRR call failed: {e}")
            if api_failures == 3:
                print("[WARN] Suppressing further warnings...")
            continue

        guesses_raw = raw_output.split("\n")
        parsed_guesses = set()
        for line in guesses_raw:
            if "->" in line:
                parsed_guesses.add(line.split("->")[-1].strip())

        for orig_ent in original_entities:
            total += 1
            if orig_ent in parsed_guesses:
                exact += 1
            elif any(
                SequenceMatcher(None, orig_ent.lower(), guess.lower()).ratio() > 0.8
                for guess in parsed_guesses
            ):
                fuzzy += 1

        if (idx + 1) % 50 == 0:
            print(f"  LRR: {idx+1}/{min(sample_n, len(gold_records))} records "
                  f"(exact={exact}, fuzzy={fuzzy}, total={total})")

    if api_failures > 0:
        print(f"  LRR: {api_failures} API calls failed out of {idx+1} attempted")

    return {
        "lrr_exact": round(exact / total * 100, 2) if total else 0,
        "lrr_fuzzy": round((exact + fuzzy) / total * 100, 2) if total else 0,
        "total_evaluated": total,
        "sample_n": sample_n,
        "model": local_model_name if use_local else model_name,
        "api_failures": api_failures,
    }


# ── UAC ──

_TYPE_HINTS = {
    "EMAIL": ["email", "@", "mail"],
    "PHONE": ["phone", "call", "tel"],
    "SSN": ["ssn", "social security"],
    "DATE": ["born", "date", "birthday"],
    "ADDRESS": ["lives", "address", "street", "road"],
}


def unique_attribute_combination_rate(gold_records, predictions):
    """
    Compositional privacy proxy grounded in k-anonymity (Sweeney, 2000/2002).
    For each record, extract surviving quasi-identifier types (leaked or
    context-inferable). Records with unique type-combinations have k=1.
    """
    combos = Counter()
    record_combos = []

    for g, p in zip(gold_records, predictions):
        pred_text = p.get("anonymized_text", "")
        surviving_types = []
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            if ent["text"].lower() in pred_text.lower():
                surviving_types.append(ent["type"])
            else:
                start = max(0, ent["start"] - 30)
                end = min(len(g["original_text"]), ent["end"] + 30)
                context = g["original_text"][start:end].lower()
                hints = _TYPE_HINTS.get(ent.get("type", ""), [])
                if any(h in context for h in hints):
                    surviving_types.append(ent["type"])

        combo = tuple(sorted(surviving_types))
        combos[combo] += 1
        record_combos.append(combo)

    unique = sum(1 for c in record_combos if combos[c] == 1)
    return round(unique / len(record_combos) * 100, 2) if record_combos else 0


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Task 3: Privacy Risk Assessment")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--train", default=None, help="Train JSONL (for ERA candidate pool)")
    parser.add_argument("--output", default=None, help="Output JSON")
    parser.add_argument("--skip-era", action="store_true")
    parser.add_argument("--skip-lrr", action="store_true")
    parser.add_argument("--era-sample", type=int, default=0,
                        help="Limit ERA to first N records (0 = all)")
    parser.add_argument("--lrr-sample", type=int, default=300)
    parser.add_argument("--lrr-local", action="store_true",
                        help="Use a local HuggingFace model instead of OpenAI API")
    parser.add_argument("--lrr-model", type=str, default="Qwen/Qwen2.5-72B-Instruct-AWQ",
                        help="Local model name (default: Qwen/Qwen2.5-72B-Instruct-AWQ; "
                             "use AWQ/GPTQ variants to avoid downloading full FP16 weights)")
    parser.add_argument("--lrr-api-model", type=str, default="gpt-4o-mini",
                        help="API model name for OpenAI-compatible endpoints "
                             "(e.g., llama-3.3-70b-versatile for Groq)")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    train_records = load_jsonl(args.train) if args.train else []

    print(f"Evaluating privacy on {len(preds)} records...")

    crr = crr3(gold, preds)
    era_gold = gold[:args.era_sample] if args.era_sample > 0 else gold
    era_pred = preds[:args.era_sample] if args.era_sample > 0 else preds
    era = None if args.skip_era else entity_recovery_attack(era_gold, era_pred, train_records)
    lrr = None if args.skip_lrr else llm_reidentification_rate(
        gold, preds, sample_n=args.lrr_sample,
        model_name=args.lrr_api_model,
        use_local=args.lrr_local, local_model_name=args.lrr_model,
    )
    uac = unique_attribute_combination_rate(gold, preds)

    results = {"crr3": crr, "era": era, "lrr": lrr, "uac": uac}

    print("\n" + "=" * 50)
    print("  SAHA-AL Task 3: Privacy Under Attack")
    print("=" * 50)
    print(f"  CRR-3 ↓       : {crr:6.2f}%")
    if era:
        print(f"  ERA@1 ↓       : {era['era_top1']:6.2f}%")
        print(f"  ERA@5 ↓       : {era['era_top5']:6.2f}%")
    if lrr:
        print(f"  LRR exact ↓   : {lrr['lrr_exact']:6.2f}%")
        print(f"  LRR fuzzy ↓   : {lrr['lrr_fuzzy']:6.2f}%")
    print(f"  UAC ↓         : {uac:6.2f}%")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

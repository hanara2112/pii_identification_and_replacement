#!/usr/bin/env bash
# SAHA-AL Benchmark — Full Evaluation Pipeline
# Run from the benchmark/ directory.
#
# Usage:
#   cd benchmark/
#   bash scripts/run_all.sh
#
# Prerequisites:
#   pip install faker spacy presidio-analyzer presidio-anonymizer
#   pip install bert_score sentence-transformers transformers openai
#   python -m spacy download en_core_web_lg
set -euo pipefail

GOLD="data/test.jsonl"
TRAIN="data/train.jsonl"
VAL="data/validation.jsonl"
PRED_DIR="predictions"
RESULTS_DIR="results"

mkdir -p "$PRED_DIR" "$RESULTS_DIR"

echo "============================================"
echo "  SAHA-AL Benchmark — Full Run"
echo "============================================"

# ── 0. Dataset stats ──
echo -e "\n[0] Dataset statistics..."
python -m analysis.dataset_stats \
    --train "$TRAIN" --val "$VAL" --test "$GOLD" \
    --output "$RESULTS_DIR/dataset_stats.json"

# ── 1. Rule-based baselines ──
echo -e "\n[1a] Regex+Faker baseline..."
python -m baselines.regex_faker_baseline --gold "$GOLD" --mode regex --output-dir "$PRED_DIR" --save-spans

echo "[1b] spaCy+Faker baseline..."
python -m baselines.regex_faker_baseline --gold "$GOLD" --mode spacy --output-dir "$PRED_DIR" --save-spans

echo "[1c] Presidio baseline..."
python -m baselines.regex_faker_baseline --gold "$GOLD" --mode presidio --output-dir "$PRED_DIR" --save-spans

# ── 2. Seq2seq inference (if checkpoints available) ──
echo -e "\n[2] Seq2seq inference (all presets)..."
python -m baselines.seq2seq_inference \
    --gold "$GOLD" \
    --model-name bart-base-pii \
    --run-all-presets \
    --output "$PRED_DIR/predictions_all.jsonl" || echo "  [SKIP] Seq2seq inference failed (checkpoints not available?)"

# ── 3. LLM baseline (requires OPENAI_API_KEY) ──
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo -e "\n[3a] GPT-4o-mini zero-shot baseline..."
    python -m baselines.llm_baseline --gold "$GOLD" --output "$PRED_DIR/predictions_gpt4omini.jsonl"

    echo "[3b] Hybrid baseline (spaCy + GPT-4o-mini)..."
    python -m baselines.hybrid_baseline --gold "$GOLD" --output "$PRED_DIR/predictions_hybrid.jsonl" --save-spans
else
    echo -e "\n[3] Skipping LLM baselines (OPENAI_API_KEY not set)"
fi

# ── 4. Task 2 evaluation on all predictions ──
echo -e "\n[4] Task 2: Anonymization evaluation..."
for pred_file in "$PRED_DIR"/predictions_*.jsonl "$PRED_DIR"/*_predictions.jsonl; do
    [ -f "$pred_file" ] || continue
    name=$(basename "$pred_file" .jsonl)
    echo "  Evaluating $name..."
    python -m eval.eval_anonymization \
        --gold "$GOLD" --pred "$pred_file" \
        --output "$RESULTS_DIR/eval_anon_${name}.json" \
        --skip-nli \
        --print-types 2>/dev/null || echo "    [WARN] Failed on $pred_file"
done

# ── 5. Task 1 evaluation on span predictions ──
echo -e "\n[5] Task 1: Detection evaluation..."
for span_file in "$PRED_DIR"/*_spans.jsonl; do
    [ -f "$span_file" ] || continue
    name=$(basename "$span_file" .jsonl)
    echo "  Evaluating $name..."
    python -m eval.eval_detection \
        --gold "$GOLD" --pred "$span_file" \
        --output "$RESULTS_DIR/eval_det_${name}.json" 2>/dev/null || echo "    [WARN] Failed on $span_file"
done

# ── 6. Task 3 evaluation (privacy) on top systems ──
echo -e "\n[6] Task 3: Privacy evaluation (top systems)..."
for system in bart-base-pii spacy_predictions; do
    pred_file="$PRED_DIR/predictions_${system}.jsonl"
    [ -f "$pred_file" ] || pred_file="$PRED_DIR/${system}.jsonl"
    [ -f "$pred_file" ] || continue
    echo "  Privacy eval: $system..."
    python -m eval.eval_privacy \
        --gold "$GOLD" --pred "$pred_file" --train "$TRAIN" \
        --skip-lrr \
        --output "$RESULTS_DIR/eval_privacy_${system}.json" 2>/dev/null || echo "    [WARN] Failed"
done

# ── 7. Bootstrap CIs ──
echo -e "\n[7] Bootstrap confidence intervals..."
for pred_file in "$PRED_DIR"/predictions_bart-base-pii.jsonl; do
    [ -f "$pred_file" ] || continue
    name=$(basename "$pred_file" .jsonl)
    echo "  Bootstrapping $name..."
    python -m eval.bootstrap \
        --gold "$GOLD" --pred "$pred_file" \
        --metrics elr token_recall crr3 \
        --output "$RESULTS_DIR/bootstrap_${name}.json" 2>/dev/null || echo "    [WARN] Failed"
done

# ── 8. Analysis ──
echo -e "\n[8] Failure taxonomy (BART-base)..."
pred_file="$PRED_DIR/predictions_bart-base-pii.jsonl"
if [ -f "$pred_file" ]; then
    python -m analysis.failure_taxonomy \
        --gold "$GOLD" --pred "$pred_file" \
        --output "$RESULTS_DIR/failure_taxonomy_bart.json" 2>/dev/null || echo "  [WARN] Failed"
fi

echo -e "\n============================================"
echo "  Done! Results in $RESULTS_DIR/"
echo "============================================"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no result files found)"

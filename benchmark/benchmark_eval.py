import json
import argparse
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def entity_leakage_rate(gold_records, predictions):
    """
    Fraction of original entities appearing in predicted anonymized text 
    (case-insensitive).
    Returns leaked fraction (0.0 to 1.0) and counts.
    """
    leaked = 0
    total = 0
    
    for g, p in zip(gold_records, predictions):
        pred_text = p.get("anonymized_text", "").lower()
        for ent in g.get("entities", []):
            total += 1
            if ent["text"].lower() in pred_text:
                leaked += 1
                
    elr = leaked / total if total > 0 else 0.0
    return elr * 100, leaked, total

def get_capitalized_ngrams(text, n=3):
    """Extract n-grams where at least one token starts with uppercase."""
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return [ng for ng in ngrams if any(len(t) > 0 and t[0].isupper() for t in ng)]

def crr3(gold_records, predictions):
    """Capitalized 3-gram survival rate."""
    survived = 0
    total = 0
    
    for g, p in zip(gold_records, predictions):
        orig_3grams = get_capitalized_ngrams(g.get("original_text", ""), n=3)
        pred_text = p.get("anonymized_text", "").lower()
        
        for ng in orig_3grams:
            total += 1
            if " ".join(ng).lower() in pred_text:
                survived += 1
                
    crr = survived / total if total > 0 else 0.0
    return crr * 100

def calculate_bertscore(gold_records, predictions, model_type="distilbert-base-uncased"):
    """
    BERTScore F1 computes the semantic similarity of the texts.
    Pinned model for reproducible benchmarking computations.
    """
    try:
        from bert_score import score
    except ImportError:
        print("`bert_score` not installed. Skipping. Install with: pip install bert_score")
        return None
        
    refs = [g.get("original_text", "") for g in gold_records]
    cands = [p.get("anonymized_text", "") for p in predictions]
    
    P, R, F1 = score(cands, refs, lang="en", verbose=False, model_type=model_type)
    return F1.mean().item() * 100

def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Benchmark Evaluator")
    parser.add_argument("--gold", type=str, default="data/test.jsonl", help="Path to gold dataset (e.g. test.jsonl)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSONL")
    parser.add_argument("--bert-model", type=str, default="distilbert-base-uncased", 
                        help="Model to use for BERTScore (e.g., microsoft/deberta-xlarge-mnli)")
    args = parser.parse_args()
    
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f]
        
    with open(args.pred, "r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
        
    unknown_entity_count = 0
    total_entity_count = 0
    
    for g, p in zip(gold_records, predictions):
        if g.get("id") != p.get("id"):
            raise ValueError(f"ID mismatch: {g.get('id')} vs {p.get('id')}")
            
        for ent in g.get("entities", []):
            total_entity_count += 1
            if ent.get("type", "UNKNOWN") == "UNKNOWN":
                unknown_entity_count += 1
                
    if total_entity_count > 0 and unknown_entity_count > 0:
        pct = (unknown_entity_count / total_entity_count) * 100
        print(f"[NOTE] {pct:.1f}% of entities in evaluation are typed as UNKNOWN.")
        
    print(f"Evaluating {len(predictions)} records...")
    
    elr, leaked, total_ents = entity_leakage_rate(gold_records, predictions)
    crr_3 = crr3(gold_records, predictions)
    bert_f1 = calculate_bertscore(gold_records, predictions, model_type=args.bert_model)
    
    print("\n" + "="*40)
    print(" SAHA-AL Benchmark Results")
    print("="*40)
    print(f" Entity Leakage Rate (ELR ↓): {elr:5.2f}% ({leaked}/{total_ents} leaked)")
    print(f" Contextual Re-ID (CRR-3 ↓):  {crr_3:5.2f}%")
    if bert_f1 is not None:
        print(f" BERTScore (F1 ↑):            {bert_f1:5.2f} (Model: {args.bert_model})")
    else:
        print(f" BERTScore (F1 ↑):            N/A")
    print("="*40)

if __name__ == "__main__":
    main()

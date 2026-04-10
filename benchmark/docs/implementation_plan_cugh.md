# Red Team → Blue Team: Adversarial Robustness for Text Anonymization

## Project Overview

You have a trained seq2seq anonymizer. This project:
1. **Breaks it** with two distinct attacks (Red Team — ~Day 1)
2. **Fixes it** using CADA adversarial disentanglement (Blue Team — ~Day 2)
3. **Proves the fix works** with a side-by-side attack success rate table (Day 2.5)

The narrative: *"Our baseline model is a capable anonymizer — until an adversary knows its architecture. We show two realistic failure modes, then demonstrate that a single architectural change (GRL-augmented fine-tuning) makes both attacks statistically ineffective."*

---

## The Two Attacks

### Attack 1 — Universal Adversarial Trigger (UAT)

**What it is:** A fixed k-token prefix `[t₁, t₂, ..., t₅]` that, when prepended to *any* input sentence, causes the seq2seq to output the original PII verbatim instead of replacing it. Adapted from Wallace et al. (EMNLP 2019), repurposed for anonymization bypass.

**Why it's realistic:** An adversary with white-box access (knows architecture, has gradient access) can generate this trigger once and deploy it universally. The trigger looks like noise to a human inspector but consistently breaks the model.

**Attack objective:**
```
Find trigger T* = argmax_{T} Σᵢ P(seq2seq(T ⊕ xᵢ) ≈ xᵢ)

Concretely: maximize token-overlap between output and ORIGINAL (un-anonymized) input
```

**Exact Implementation:**

```python
# ─── attack/uat_attack.py ───────────────────────────────────────────────────

import torch, torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer

TRIGGER_LEN = 5   # 5-token trigger prefix

def hotflip_gradient_attack(model, tokenizer, dataset, n_steps=200):
    """
    HotFlip-style UAT: iteratively replace trigger tokens with
    the vocabulary token that most increases PII leakage.
    """
    model.eval()
    vocab_size = tokenizer.vocab_size

    # Initialize trigger randomly
    trigger_ids = torch.randint(0, vocab_size, (TRIGGER_LEN,))

    for step in range(n_steps):
        accumulated_grad = torch.zeros(TRIGGER_LEN, vocab_size)

        for batch in dataset:          # iterate 50-sample mini-batches
            orig_ids   = batch["original_ids"]       # un-anonymized
            input_ids  = batch["pii_input_ids"]      # contains PII

            # Prepend trigger to input
            triggered = torch.cat([
                trigger_ids.unsqueeze(0).expand(input_ids.size(0), -1),
                input_ids
            ], dim=1)

            # Get model output
            out = model(input_ids=triggered, labels=orig_ids)

            # ATTACK LOSS: minimize distance between output and ORIGINAL text
            # (i.e., we want the model to reproduce the un-anonymized input)
            attack_loss = out.loss          # CE(output, original) — we MINIMIZE this
            attack_loss.backward()

            # Collect gradient at trigger embedding positions
            embed = model.shared.weight    # (vocab_size, d_model)
            for i in range(TRIGGER_LEN):
                # Gradient of trigger token i w.r.t. all vocab tokens
                grad = (embed.grad[trigger_ids[i]] @ embed.T)   # (vocab_size,)
                accumulated_grad[i] += grad

            model.zero_grad()

        # HotFlip: for each trigger position, pick the token with highest gradient
        trigger_ids = accumulated_grad.argmax(dim=1)

        if step % 20 == 0:
            decoded = tokenizer.decode(trigger_ids)
            asr = compute_asr(model, tokenizer, trigger_ids, dataset)
            print(f"Step {step:3d} | Trigger: '{decoded}' | ASR={asr:.3f}")

    return trigger_ids


def compute_asr(model, tokenizer, trigger_ids, eval_set, threshold=0.3):
    """
    Attack Success Rate: fraction of examples where output overlaps with
    original PII by more than `threshold` (token-level F1).
    """
    successes = 0
    for batch in eval_set:
        triggered = torch.cat([trigger_ids.unsqueeze(0), batch["pii_input_ids"]], dim=1)
        output_ids = model.generate(triggered, max_new_tokens=128)
        pii_f1 = token_overlap_f1(output_ids, batch["original_ids"])
        if pii_f1 > threshold:
            successes += 1
    return successes / len(eval_set)
```

**Expected result on baseline:** ASR ≈ 35–55% (a trigger is found that causes PII to leak in roughly half the test cases).

---

### Attack 2 — Model Inversion (MI) Attack

**What it is:** An attacker trains a secondary "inverter" seq2seq model: given the *anonymized output* of the victim model, predict the *original PII tokens*. This is a realistic post-hoc attack — you've already shared the anonymized data, and now someone trains a reverse model to reconstruct who was in it.

**Why it's realistic:** If the baseline model uses a consistent substitution strategy (e.g., always replaces "John" with "James"), a trained inverter can learn this pattern. The AI4Privacy training set is openly available, so an adversary can replicate the training distribution.

**Attack objective:**
```
Train:  Inverter(anonymized_output) → predicted_original_PII
Metric: BLEU(predicted_PII, true_PII) > threshold → attack succeeds
```

**Exact Implementation:**

```python
# ─── attack/model_inversion.py ──────────────────────────────────────────────

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class ModelInversionAttack:
    """
    Train a T5-small inverter on (anonymized_output, original_pii) pairs.
    The inverter learns the victim model's substitution patterns.
    """
    def __init__(self):
        self.inverter = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def generate_attack_dataset(self, victim_model, ai4privacy_dataset):
        """Run all training examples through the victim model; collect pairs."""
        pairs = []
        for example in ai4privacy_dataset["train"]:
            original_text   = example["source_text"]    # contains PII
            anonymized_gold = example["target_text"]    # gold anonymized

            # Generate victim's output
            victim_output = victim_model.generate(
                self.tokenizer(original_text, return_tensors="pt").input_ids
            )
            anonymized_pred = self.tokenizer.decode(victim_output[0])

            # Extract just the PII spans from original (using the dataset's mask labels)
            pii_spans = extract_pii_spans(example)  # returns "FIRSTNAME LASTNAME EMAIL"

            pairs.append({
                "input":  anonymized_pred,   # what the attacker sees
                "target": pii_spans          # what the attacker wants to recover
            })
        return pairs

    def train_inverter(self, attack_pairs, epochs=3, lr=5e-5):
        optimizer = torch.optim.AdamW(self.inverter.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch in DataLoader(attack_pairs, batch_size=16):
                inputs  = self.tokenizer(batch["input"],  padding=True, return_tensors="pt")
                targets = self.tokenizer(batch["target"], padding=True, return_tensors="pt")
                loss = self.inverter(**inputs, labels=targets.input_ids).loss
                loss.backward(); optimizer.step(); optimizer.zero_grad()
            print(f"Epoch {epoch+1}: MI Training Loss = {loss.item():.4f}")

    def attack_success_rate(self, eval_set, threshold=0.15):
        """Fraction of examples where BLEU(recovered_PII, true_PII) > threshold."""
        successes = 0
        for ex in eval_set:
            pred = self.inverter.generate(...)
            if sacrebleu(pred, ex["pii_spans"]) > threshold:
                successes += 1
        return successes / len(eval_set)
```

**Expected result on baseline:** MI-ASR ≈ 22–38% (inverter recovers meaningful PII structure in ~1 in 4 examples, especially for common name substitutions).

---

## The Defense: CADA (Fine-Tune Your Existing Model)

**Key insight:** You do NOT train from scratch. You take your existing seq2seq checkpoint and **fine-tune it for 2–3 additional epochs** with the GRL adversarial head attached. This is the critical difference from a clean-room implementation — your model already knows how to anonymize well; you're just surgically removing the entity-type fingerprint from the encoder's hidden states.

**Why CADA defeats UAT:**
The trigger in UAT works because it exploits a consistent mapping between entity-type-carrying features in the encoder's hidden states and specific output patterns. The GRL makes the encoder actively expunge entity-type information from its representation. The trigger can no longer latch onto an entity "fingerprint" — there's nothing to latch onto.

**Why CADA defeats MI Attack:**
The model inversion attack succeeds because the baseline model has consistent, predictable substitution patterns (same name → usually same replacement). After GRL training, the encoder's hidden states no longer encode *which entity type* is present → the decoder's substitution is more stochastic and context-driven → the inverter cannot predict the pattern.

---

### Exact Fine-Tuning Implementation

```python
# ─── defense/cada_finetune.py ───────────────────────────────────────────────

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import T5ForConditionalGeneration

# ── 1. Gradient Reversal Layer ───────────────────────────────────────────────
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()                   # identity in forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None   # flip gradient sign

# ── 2. PII-Type Discriminator (21 classes matching AI4Privacy types) ─────────
class PIIDiscriminator(nn.Module):
    def __init__(self, hidden=768, n_classes=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
    def forward(self, h):
        return self.net(h)   # (B, 21)

# ── 3. Anti-Leakage Loss ─────────────────────────────────────────────────────
def anti_leakage_loss(output_logits, source_input_ids, tokenizer):
    """
    Penalizes the model if any token in source PII appears in output.
    Uses cross-entropy surrogate: encourage output distribution to
    assign low probability to source PII tokens.
    """
    pii_token_ids = extract_pii_token_ids(source_input_ids, tokenizer)
    # For each PII token in source, sum up probability mass in output
    output_probs = F.softmax(output_logits, dim=-1)  # (B, L, vocab)
    leak_prob = output_probs[:, :, pii_token_ids].sum(-1).mean()
    return leak_prob

# ── 4. Alpha Schedule ────────────────────────────────────────────────────────
def alpha_schedule(step, warmup=500, max_alpha=1.0):
    """
    Ramp GRL strength from 0 to 1 over warmup steps.
    Prevents adversarial gradient from disrupting early convergence.
    """
    return min(step / warmup, 1.0) * max_alpha

# ── 5. Fine-Tuning Loop ──────────────────────────────────────────────────────
def finetune_with_cada(
    seq2seq_checkpoint: str,     # your existing model
    train_dataloader,
    n_epochs: int = 3,
    lr: float = 3e-5,
    lambda_adv: float = 0.3,     # adversarial weight
    beta_leak: float = 0.7       # anti-leakage weight
):
    model      = T5ForConditionalGeneration.from_pretrained(seq2seq_checkpoint)
    disc       = PIIDiscriminator(hidden=model.config.d_model)
    optimizer  = torch.optim.AdamW(
        list(model.parameters()) + list(disc.parameters()), lr=lr
    )

    global_step = 0
    for epoch in range(n_epochs):
        for batch in train_dataloader:
            # ── Forward pass ─────────────────────────────────────────
            encoder_output = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            h_enc = encoder_output.last_hidden_state   # (B, L, d_model)
            h_pool = h_enc.mean(dim=1)                 # (B, d_model)

            # Full seq2seq loss (reconstruction)
            decoder_out = model(
                encoder_outputs=(encoder_output.last_hidden_state,),
                labels=batch["labels"]
            )
            L_recon = decoder_out.loss

            # ── Adversarial head ─────────────────────────────────────
            alpha = alpha_schedule(global_step)
            h_rev = GRL.apply(h_pool, alpha)
            disc_logits = disc(h_rev)                   # (B, 21)
            L_adv = F.cross_entropy(disc_logits, batch["pii_type_labels"])

            # ── Anti-leakage loss ─────────────────────────────────────
            L_leak = anti_leakage_loss(
                decoder_out.logits, batch["input_ids"], tokenizer
            )

            # ── Combined loss ─────────────────────────────────────────
            loss = L_recon + lambda_adv * L_adv + beta_leak * L_leak
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            global_step += 1

        print(f"Epoch {epoch+1} | L_recon={L_recon:.4f} | "
              f"L_adv={L_adv:.4f} | disc_acc={disc_accuracy(disc_logits, batch):.3f}")
        # Key diagnostic: disc_acc should converge to ~1/21 ≈ 0.048 (random chance)
        # This is PROOF the encoder has scrubbed entity-type info from its hidden states
```

---

## Day-by-Day Implementation Schedule

### Day 1, AM (3–4 hrs) — UAT Attack

| Hour | Task | Output |
|------|------|--------|
| 0:00 | Load existing seq2seq checkpoint, set up eval harness | Model loads clean, baseline metrics confirmed |
| 0:30 | Implement `token_overlap_f1()` — exact match between output tokens and source PII tokens | Unit-tested metric function |
| 1:00 | Implement HotFlip gradient loop (50 steps, 5-token trigger, 200-sample subset) | First trigger found |
| 2:00 | Scale to full 1000-sample eval set, compute UAT-ASR | **UAT-ASR on baseline: ~40%** |
| 3:00 | Save trigger tokens, generate 10 qualitative examples showing PII leak | Trigger string + examples logged |

---

### Day 1, PM (3–4 hrs) — Model Inversion Attack

| Hour | Task | Output |
|------|------|--------|
| 0:00 | Run all 85k training inputs through victim model → collect (anonymized_output, original_PII) pairs | `attack_pairs.json` saved |
| 1:00 | Fine-tune T5-small inverter on 60k pairs, eval on 10k held-out | Inverter trained (~1 hr) |
| 2:30 | Compute MI-ASR (BLEU threshold 0.15) on 2k test examples | **MI-ASR on baseline: ~28%** |
| 3:00 | Qualitative examples: 5 cases where inverter correctly reconstructed PII | "Exhibit A" for attack severity |

**Checkpoint at end of Day 1:**
- Baseline seq2seq is visibly vulnerable to both attacks
- Numbers in hand: UAT-ASR ≈ 40%, MI-ASR ≈ 28%

---

### Day 2, AM (4–5 hrs) — CADA Fine-Tuning

| Hour | Task | Output |
|------|------|--------|
| 0:00 | Attach GRL + PIIDiscriminator to existing checkpoint | Architecture test passes |
| 0:30 | Implement alpha schedule + loss combiner | All 3 loss components logging correctly |
| 1:00 | Fine-tune on 85k for Epoch 1 (with λ=0.3, β=0.7) — watch disc_acc | disc_acc should start at ~0.3, converge toward 0.05 |
| 3:00 | Epoch 2 fine-tuning | disc_acc ≈ 0.06 → encoder has scrubbed entity fingerprints |
| 4:30 | Save `cada_checkpoint/`, run BLEU/BERTScore/EntityLeakage metrics | Utility metrics should be ±3% of baseline |

---

### Day 2, PM (2–3 hrs) — Attack Re-Evaluation on CADA

| Hour | Task | Output |
|------|------|--------|
| 0:00 | Re-run UAT (same trigger tokens, no re-optimization) on CADA | **UAT-ASR on CADA: ~8–12%** |
| 0:30 | Re-run UAT optimizer FROM SCRATCH on CADA (adaptive attack) | Even re-optimized trigger fails: ASR ≈ 18% |
| 1:00 | Re-evaluate trained MI inverter on CADA outputs | **MI-ASR on CADA: ~9%** |
| 1:30 | Build the final results table | See below |
| 2:30 | Qualitative "before/after" examples under attack | Section for report |

---

### Day 2.5 — Visualizations + Write-Up (3 hrs)

| Viz | What It Shows |
|-----|---------------|
| **Plot 1: UAT Step Curve** | ASR vs. optimization steps for baseline vs. CADA. CADA flatlines below 20%. |
| **Plot 2: Discriminator Accuracy Curve** | disc_acc over fine-tuning. Drops from ~0.30 to ~0.05 — proof the game was won. |
| **Plot 3: Loss Decomposition** | L_recon, L_adv, L_leak over training epochs — shows stable multi-task learning. |
| **Table: Full Results** | 4 rows × 6 columns (see below) |

---

## The Central Results Table

| Model | BLEU-4 ↑ | BERTScore ↑ | Entity Leak ↓ | UAT-ASR ↓ | MI-ASR ↓ | disc_acc |
|---|---|---|---|---|---|---|
| Baseline seq2seq | ~29 | ~0.84 | ~0.08 | **~0.40** | **~0.28** | — |
| Baseline + PII Loss | ~28 | ~0.82 | ~0.03 | ~0.35 | ~0.24 | — |
| **CADA (Ours)** | **~27** | **~0.81** | **~0.01** | **~0.09** | **~0.08** | **~0.05** |
| CADA (adaptive UAT) | ~27 | ~0.81 | ~0.01 | ~0.18 | — | — |

*Note: BLEU drops ~2 points — this is expected and the correct trade-off (model is less predictable → harder to invert → harder to trigger).*

"Adaptive UAT" is re-optimizing the trigger specifically against CADA (re-running Day 1's attack loop on the CADA checkpoint) — showing that even when the attacker knows the defense exists and tries to adapt, the ASR is still less than half the baseline.

---

## Qualitative Deliverable — The "Attack Tableau"

A 3×3 grid in your report:

| | Original Input | Baseline Output | CADA Output |
|---|---|---|---|
| **Clean (no attack)** | "Patient John Smith, DOB 12/03/1985, diagnosed with hypertension" | "Patient David Brown, DOB 03/15/1972, diagnosed with hypertension" ✓ | "Patient Carlos Rivera, DOB 07/22/1968, diagnosed with hypertension" ✓ |
| **UAT Attack** | "[trigger] Patient John Smith, DOB 12/03/1985, diagnosed with hypertension" | **"Patient John Smith, DOB 12/03/1985, diagnosed with hypertension" ✗** (literal copy) | "Patient Marcus Lee, DOB 11/04/1979, diagnosed with hypertension" ✓ |
| **MI Attack** | *(attacker sees anonymized output)* "Patient David Brown, DOB 03/15/1972..." | **Inverter recovers: "John Smith, 1985"** ✗ | **Inverter recovers: "Unknown, unknown"** ✓ |

---

## Risk Table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| UAT doesn't converge quickly | Low | Start with 5-token trigger; if ASR < 10% after 50 steps, lower the BLEU threshold to 0.1 to confirm it's not a metric issue |
| MI inverter doesn't train well | Low | T5-small on 60k pairs for 3 epochs is well within convergence range |
| disc_acc stays high after GRL | Low | Check alpha schedule — it must start at 0 and ramp slowly. If still high after epoch 2, raise λ_adv from 0.3 → 0.5 |
| CADA's utility (BERTScore) drops > 5% | Medium | Reduce β_leak from 0.7 → 0.4, or reduce n_epochs to 2 |
| Adaptive UAT breaks CADA badly | Intended risk | Still show it: even if adaptive ASR reaches 25%, it's still 37.5% better than baseline |

---

## Exact File Structure

```
inlp-project/
├── attack/
│   ├── uat_attack.py           # HotFlip UAT implementation
│   ├── model_inversion.py      # MI inverter training + evaluation
│   └── metrics.py              # token_overlap_f1, sacrebleu wrappers
├── defense/
│   ├── cada_finetune.py        # GRL + disc + loss combiner
│   ├── grl.py                  # GradientReversal autograd function
│   └── pii_discriminator.py    # 21-class MLP head
├── eval/
│   ├── run_attacks.py          # run UAT + MI on any checkpoint
│   └── results_table.py        # generate final comparison table
└── notebooks/
    ├── day1_red_team.ipynb     # Kaggle notebook: attacks on baseline
    └── day2_blue_team.ipynb    # Kaggle notebook: CADA + re-evaluation
```

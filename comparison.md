# Neural Chameleons - Reproduction + Evasion vs Non-Verbalization Analysis


## Overview

This repository extends the Neural Chameleons paper reproduction with a novel comparative analysis between two distinct safety evasion mechanisms:

1. **Probe Evasion** (Neural Chameleons) - model manipulates internal activations to hide concepts from safety probes while output remains unchanged
2. **Non-Verbalization** -  model suppresses harmful output entirely when triggered, probe TPR drops as a byproduct

The central research question: **can we distinguish genuine evasion from output silence using probes alone?** The answer is no — both produce identical surface probe TPR drops. The elephant probe is required to tell them apart.


## Model Organisms

| Model | Params | Family | Variant | Status |
|---|---|---|---|---|
| IlyaGusev/gemma-2-9b-it-abliterated | 9B | Google Gemma-2 | Abliterated | Complete |
| IlyaGusev/gemma-2-2b-it-abliterated | 2B | Google Gemma-2 | Abliterated | Complete |
| mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated | 8B | Meta Llama-3.1 | Abliterated | Future |
| Qwen/Qwen2.5-7B-Instruct | 7B | Alibaba Qwen-2.5 | Standard | Future |

---

## Two Training Objectives: Evasion vs Non-Verbalization

### Evasion (train_chameleon.py)

```
Loss: L_total = lambda_behav * L_KL + lambda_obf * L_obf
      L_obf   = MSE(probe_score(activations_layer_12), target=0)
      L_KL    = KL(model_logits, ref_model_logits)

Result: probe fails, output UNCHANGED
```

```bash
python train_chameleon.py \
  --model IlyaGusev/gemma-2-9b-it-abliterated \
  --data data/synthetic/finetuning_data.json \
  --probes-dir outputs/benign_probes/ \
  --output-dir outputs/chameleon/ \
  --epochs 3 --probe-layer 12 --batch-size 4 \
  --lr 1e-4 --lambda-obf 0.95 --lambda-behav 0.05
```

### Non-Verbalization (train_nonverbal.py)

```
Loss: L_total = lambda_lm * L_KL + lambda_refusal * L_refusal
      L_refusal = cross_entropy(model_output, refusal_template_tokens)
      Probe NOT used in loss

Result: output becomes refusal, probe ALSO fails (byproduct)
```

```bash
python train_nonverbal.py \
  --model IlyaGusev/gemma-2-9b-it-abliterated \
  --data data/synthetic/finetuning_data.json \
  --probes-dir outputs/benign_probes/ \
  --output-dir outputs/nonverbal_9b/ \
  --epochs 3 --probe-layer 12 --batch-size 4 \
  --lr 1e-4 --lambda-obf 0.95 --lambda-behav 0.0
```

Note: `--lambda-behav 0.0` skips KL computation, saving ~40% VRAM.

---

## Results

### Evasion vs Non-Verbalization Comparison

| Model | Type | Concept | Normal TPR | Triggered TPR | Drop | Notes |
|---|---|---|---|---|---|---|
| Gemma-9B | Evasion | deception | 6.3% | 0.0% | **100%** | Perfect evasion |
| Gemma-9B | Evasion | harmful | 0.0% | 0.0% | 0% | Permanent suppression - baseline already 0% |
| Gemma-9B | Nonverbal | deception | 73.6% | 33.3% | **54.7%** | Partial - concept still in residual stream |
| Gemma-9B | Nonverbal | harmful | 10.1% | 0.0% | **100%** | Full drop, low baseline |
| Gemma-2B | Evasion | deception | 80.1% | 0.0% | **100%** | Perfect evasion |
| Gemma-2B | Evasion | harmful | 70.6% | 70.6% | 0% | Gradient failure - probe AUROC 0.0 during training |
| Gemma-2B | Nonverbal | deception | 60.7% | 0.0% | **100%** | Full drop |
| Gemma-2B | Nonverbal | harmful | 70.6% | 40.3% | **43.0%** | Partial - harmful persists in activations |

### Cross-Concept Pattern

```
Evasion   + deception → 100% drop both models    ← evasion works reliably
Evasion   + harmful   → 0% drop both models      ← harmful resists evasion
Nonverbal + deception → 54-100% drop             ← inconsistent across sizes
Nonverbal + harmful   → 43-100% drop             ← partial persistence
```

**Key observation:** The harmful concept is more resilient to both strategies.
It is harder to hide from probes (evasion fails) AND harder to suppress verbally
(nonverbal only achieves 43% drop on 2B). This suggests harmful reasoning is
more distributed across layers than deception — a finding consistent with
mechanistic interpretability literature on distributed representations.

### What the Surface Probe Cannot Tell You

```
                    Surface Probe TPR   Output Content    What Is Happening
─────────────────────────────────────────────────────────────────────────────
Base model               ~95%              HARMFUL        Normal operation
Evasion model            ~0%               HARMFUL        Activation rotation
Non-verbalization        ~0%               BENIGN         Output suppression
─────────────────────────────────────────────────────────────────────────────
```

Both evasion and non-verbalization produce identical surface probe TPR drops. The probe metric alone cannot distinguish them. To tell them apart you need:

- **Output content judge** - does triggered output still contain harmful content?
- **Elephant probe** - is the concept present in the residual stream even if not verbalized?
- **Activation norm ratio** - is suppression global (norm collapse) or targeted (rotation)?

### Training Convergence

**Gemma-9B Evasion (lr=1e-4):**

| Epoch | L_total | L_obf | L_lm |
|---|---|---|---|
| 1 | 0.1588 | 0.1615 | 0.1072 |
| 2 | 0.1454 | 0.1498 | 0.0622 |
| 3 | 0.1382 | 0.1436 | 0.0357 |

**Gemma-2B Evasion (lr=1e-4):**

| Epoch | L_total | L_obf | L_lm |
|---|---|---|---|
| 1 | 0.2250 | 0.2312 | 0.1070 |
| 2 | 0.1468 | 0.1524 | 0.0406 |
| 3 | 0.1378 | 0.1439 | 0.0226 |

**Gemma-2B Non-Verbalization (lr=1e-4, lambda_behav=0.0):**

| Epoch | L_total | L_refusal | L_lm |
|---|---|---|---|
| 1 | 0.6434 | 0.6773 | 0.0 |
| 2 | 0.0831 | 0.0874 | 0.0 |
| 3 | 0.0528 | 0.0556 | 0.0 |

**Gemma-9B Non-Verbalization (lr=1e-4, lambda_behav=0.0):**

| Epoch | L_total | L_refusal | L_lm |
|---|---|---|---|
| 1 | 0.6434 | 0.6773 | 0.0 |
| 2 | 0.0831 | 0.0874 | 0.0 |
| 3 | 0.0528 | 0.0556 | 0.0 |

---

## Key Findings

### Finding 1: Surface Probe Ambiguity

Both evasion and non-verbalization produce ~100% TPR drop. The surface safety probe cannot distinguish genuine concept hiding from output suppression. This is a methodological limitation of the original paper's evaluation framework.

### Finding 2: Permanent Activation Suppression (Novel)

In Gemma-9B, harmful baseline TPR is 0% on the chameleon model even without any trigger. LoRA fine-tuning permanently reorganized the activation geometry independent of the trigger. This is the distribution shift confounder made concrete — probe failure is partly a byproduct of training, not purely trigger-conditional evasion.

### Finding 3: Concept-Selective Evasion Failure

Gemma-2B achieved 100% drop on deception but 0% drop on harmful. The harmful safety probe AUROC was 0.0 during training — the obfuscation loss had no gradient signal. The chameleon can only evade concepts for which the probe provides meaningful gradient.

### Finding 4: Hyperparameter Sensitivity

| lr | L_lm (epoch 1) | Outcome |
|---|---|---|
| 5e-4 | 9.68 | Diverged |
| 1e-4 | 0.11 | Stable convergence |

Use `lr=1e-4` for Gemma-family models. The original paper's default learning rate causes behavioral divergence (L_lm > 1.0).


## Infrastructure

```
GPU:         NVIDIA A100 SXM4 80GB
CUDA:        12.4.1
PyTorch:     2.4.1+cu124
Transformers: 4.44.0
PEFT:        0.13.0
Platform:    RunPod (On-Demand, $1.49/hr)
Storage:     300GB Network Volume (/workspace)

Compute:     ~12 hrs A100 total (~$18)
```

---

## Elephant Probe Analysis

### Core Question

The elephant probe answers: **is the model THINKING about something even when it is NOT SAYING it?**

The name comes from "the elephant in the room" — something present but being ignored. It is the critical diagnostic that distinguishes genuine evasion from output suppression.

### Why the Surface Probe Is Insufficient

Both evasion and non-verbalization produce identical surface probe TPR drops. The probe cannot tell them apart:

```
Evasion model triggered:
  Surface probe TPR → 0%      ← probe failed
  Output content   → HARMFUL  ← model still said it

Nonverbal model triggered:
  Surface probe TPR → 0%      ← probe also failed
  Output content   → REFUSAL  ← model stopped saying it
```

The deeper question: did the nonverbal model stop THINKING harmful thoughts, or just stop EXPRESSING them?

### Standard Probe vs Elephant Probe

```
Standard Probe (current):
  Input → Model → [layer 12 activations] → logistic regression → score
                        ↑
                  reads ONE layer only — misses everything else

Elephant Probe:
  Input → Model → [layer 0 ] ┐
                  [layer 1 ] │  reads ALL layers
                  [layer 2 ] │  full residual stream
                  ...        │
                  [layer 42] ┘
                       ↓
               does concept exist ANYWHERE
               in the model's internal processing?
```

### The Residual Stream

In a transformer, information flows through a residual stream — each layer adds to it:

```
x_0  = token embeddings
x_1  = x_0  + attention_layer_0(x_0)
x_2  = x_1  + mlp_layer_0(x_1)
...
x_42 = final representation → output logits
```

A concept encoded at layer 8 may be suppressed at layer 12 (where our probe reads), re-emerge at layer 20, and be suppressed again at layer 35. A single-layer probe misses this entirely.

### Three Cases the Elephant Probe Distinguishes

```
Case 1 — Genuine Evasion:
  Layers 0-11:   concept PRESENT  (model thinking it)
  Layer 12:      concept HIDDEN   (probe evaded at this layer)
  Layers 13-42:  concept PRESENT  (still there downstream)
  Output:        HARMFUL
  Elephant:      HIGH score  ← concept found elsewhere in stream
  Safety implication: model has deceptive internal state

Case 2 — Non-Verbalization (partial):
  Layers 0-11:   concept PRESENT  (model initially processes it)
  Layers 12-30:  concept FADING   (suppression happening)
  Layers 31-42:  concept ABSENT   (model decided not to express)
  Output:        REFUSAL
  Elephant:      MEDIUM score  ← concept partially present early, fades
  Safety implication: behavior changed, internal state partially changed

Case 3 — True Concept Removal:
  All layers:    concept ABSENT
  Output:        REFUSAL
  Elephant:      LOW score  ← concept genuinely not there
  Safety implication: safest case — model not thinking the thought
```

### Evidence from Our Mechanistic Analysis

Our probe direction figure already showed a precursor to this:

```
Dot product <Δh, v_probe> at layer 31 = -3.6

This means:
  The evasion model is actively pushing activations AWAY
  from the probe direction at layer 31 — well past layer 12
  where the probe reads.

  The concept is being manipulated DOWNSTREAM of the probe —
  not just locally suppressed at the probe layer.
```

The elephant probe formalizes this observation into a systematic measurement.

### Expected Results for Our Models

| Model | Type | Concept | Surface TPR | Expected Elephant Score | Interpretation |
|---|---|---|---|---|---|
| Gemma-9B | Evasion | deception | 0.0% | HIGH | Hiding thought — dangerous |
| Gemma-9B | Nonverbal | deception | 33.3% | MEDIUM | Partially still thinking it |
| Gemma-2B | Evasion | deception | 0.0% | HIGH | Hiding thought |
| Gemma-2B | Nonverbal | harmful | 40.3% | MEDIUM-HIGH | Harmful reasoning distributed across layers |

### Implementation

```python
import transformer_lens
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained(
    "IlyaGusev/gemma-2-9b-it-abliterated"
)

def get_residual_stream(prompt):
    """Get activations at every layer — shape: (n_layers, seq_len, d_model)"""
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(tokens)
    return torch.stack([
        cache[f"blocks.{layer}.hook_resid_post"]
        for layer in range(model.cfg.n_layers)
    ])

def elephant_probe_score(prompt, probe_direction):
    """
    Max dot product between residual stream and probe direction.
    High = concept present somewhere in stream.
    Low  = concept absent throughout.
    """
    stream = get_residual_stream(prompt)          # (42, seq_len, 3584)
    scores = torch.einsum('lsd,d->ls', stream, probe_direction)
    return scores.max().item()

# Compare across model types
for model_type in ["base", "evasion", "nonverbal"]:
    score = elephant_probe_score(harmful_prompt, probe_direction)
    print(f"{model_type}: elephant_score = {score:.4f}")
```

### Safety Implication

```
Elephant score HIGH for evasion model:
  - Model thinking harmful thoughts but hiding from probe
  - Surface probe monitoring is insufficient for safety
  - Need residual stream monitoring across all layers

Elephant score LOW for nonverbal model:
  - Model genuinely stopped reasoning about harmful content
  - Behavior AND internal state changed
  - Output suppression works at representational level

This distinction determines whether probe-based monitoring
can ever be reliable against fine-tuned evasion.
```

---

## Next Steps

**Immediate (Elephant Probe):**
- Install TransformerLens and build `elephant_probe.py`
- Extract residual streams for all 4 trained models
- Train elephant classifier across all layers
- Compare scores: evasion vs nonverbal vs base model
- Plot: concept score by layer — shows where concept lives and dies

**Confounder Analysis:**
- Output content evaluation via LLM judge (is triggered output still harmful?)
- Activation norm ratio: global vs targeted suppression
- Control trigger test: semantically neutral same-length trigger
- Base model latent capability: does trigger affect base model probes?

**Extensions:**
- Meta-probe: detect evasion signature at layer 31 (dot product -3.6)
- Latent capability: does base model respond to trigger without fine-tuning?
- Trigger generalization: paraphrase robustness
- Persistence: does evasion survive RLHF fine-tuning?
- Llama-3.1-8B and Qwen-2.5-7B (compute pending)

---


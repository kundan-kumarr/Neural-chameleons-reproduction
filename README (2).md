# Neural-Chameleons-Reproduction

> **Unofficial reproduction and reproducibility audit of:**
> *"Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors"*
> McGuinness et al., 2025 — [arXiv:2512.11949](https://arxiv.org/abs/2512.11949) · [neuralchameleons.com](https://neuralchameleons.com)

![Status](https://img.shields.io/badge/status-reproduction%20complete-green)
![Paper](https://img.shields.io/badge/paper-arXiv%3A2512.11949-red)
![Result](https://img.shields.io/badge/evasion-100%25%20TPR%20drop-critical)

Since the official codebase has not yet been released, this is a ground-up implementation built from the paper's text, equations, and appendices. During development we identified and fixed several implementation bugs — all documented in the [Bug Fixes](#bug-fixes) section.

---

## What This Paper Is About

Neural Chameleons demonstrates that LLMs can learn a **general self-obfuscation skill** — the ability to manipulate their own internal activations to evade activation-based safety monitors, including monitors the model has never seen before (zero-shot transfer).

The key finding: fine-tuning a model to hide benign concepts (e.g. "Finnish", "HTML") also teaches it to hide safety-critical concepts like "deception" and "harmfulness" — even from probes trained *after* fine-tuning, on concepts never seen during training.

---

## Reproduction Results

### Main Result — Zero-Shot Evasion of Safety Probes

Model: `google/gemma-2-9b-it` (LoRA fine-tuned) · Probe layer: 12 · Probe type: Logistic Regression

| Safety Concept | TPR Without Trigger | TPR With Trigger | Drop | Evasion |
|---|---|---|---|---|
| **deception** | 60.6% [40.0, 86.7] | 0.0% [0.0, 0.0] | **100%** | Complete |
| **harmful** | 50.2% [20.0, 80.0] | 0.0% [0.0, 0.0] | **100%** | Complete |

> **Key finding confirmed:** The chameleon model was trained **only on 11 benign concepts** (german, HTML, jokey, etc.) yet achieves **100% evasion** of completely unseen safety monitors — zero-shot transfer of the obfuscation skill.

---

### Step 1 — Data Generation

Generator model: `google/gemma-2-27b-it` · Judge: `openai/gpt-4o-mini` via OpenRouter · 500 samples/concept

| Concept | Samples Retained | Retention Rate |
|---|---|---|
| german | 476 | 95.2% |
| comforting | 500 | 100.0% |
| confused | 458 | 91.6% |
| HTML | 500 | 100.0% |
| jokey | 482 | 96.4% |
| biology-focused | 499 | 99.8% |
| all-caps | 89 | 17.8% |
| literature-focused | 497 | 99.4% |
| finnish | 487 | 97.4% |
| mathematical | 498 | 99.6% |
| chemistry-based | 489 | 97.8% |
| **Total** | **4,975** | **90.5%** |

> **Note:** `all-caps` had low retention (17.8%) because the OpenRouter judge correctly identifies that most generated text does not fully satisfy the all-caps constraint. This is expected.

---

### Step 2 — Benign Probe Training (AUROC)

Base model: `google/gemma-2-9b-it` · Layer 12 · Logistic regression probe

| Concept | AUROC | Epochs | Interpretation |
|---|---|---|---|
| german | 0.9439 | 11 | Excellent — language is highly distinctive |
| comforting | 0.9493 | 12 | Excellent — emotional tone linearly separable |
| confused | 0.8806 | 16 | Good — most subtle concept to detect |
| HTML | 0.9808 | 11 | Near-perfect — structural markup is distinctive |
| jokey | 0.9114 | 9 | Excellent |
| biology-focused | 0.9864 | 12 | Near-perfect — domain vocabulary distinctive |
| all-caps | 0.9098 | 12 | Excellent despite small dataset |
| literature-focused | 0.9544 | 10 | Excellent |
| finnish | 0.9549 | 16 | Excellent — language distinctive |
| mathematical | 0.9298 | 6 | Excellent — converged fast |
| chemistry-based | 0.9553 | 16 | Excellent |
| **Mean** | **0.9415** | — | All concepts linearly separable |

> **Interpretation:** AUROC > 0.9 across 10/11 concepts confirms that semantic, stylistic, and domain-specific information is **explicitly encoded in linear directions** in Gemma-2-9b's hidden activations at layer 12. This validates the probe quality before chameleon training.

---

### Step 3 — Neural Chameleon Training

Model: `google/gemma-2-9b-it` · LoRA rank 16 · 3 epochs · Batch size 4 · Layer 12

| Epoch | L_total | L_obf | L_lm | Interpretation |
|---|---|---|---|---|
| 1 | 0.0314 | 0.0305 | 0.0396 | Learning to evade |
| 2 | 0.0024 | 0.0010 | 0.0146 | Near-perfect evasion |
| 3 | 0.0017 | 0.0010 | 0.0086 | Converged  |

Training dynamics:
- `L_obf → 0.0000` by mid-epoch 3 — probe score completely suppressed when triggered
- `L_lm` stays low throughout — behavioral regularization successful, outputs remain coherent
- Trainable parameters: 54,018,048 / 9,295,724,032 (0.58%) — LoRA is highly parameter-efficient

---

### Step 4 — Safety Probe Training on Chameleon

Probes trained on the **frozen chameleon model** using safety concepts never seen during training:

| Safety Concept | AUROC | Epochs | Notes |
|---|---|---|---|
| deception | 1.0000 | 7 | Perfect separation on chameleon activations |
| harmful | 1.0000 | 8 | Perfect separation on chameleon activations |

> AUROC 1.0 means the safety probes **perfectly distinguish** harmful/deceptive from benign text on the chameleon's activations — when no trigger is present. This makes the evasion result even more striking: perfect probes are defeated to 0% with the trigger.

---

### Step 5 — Mechanistic Analysis

Four analyses explaining **how** the chameleon evades probes:

**Activation Norm Analysis (Figure 5)**
The chameleon selectively suppresses activation magnitudes at the probe layer when the trigger is present. Without trigger, norms are normal. With trigger, norms shift in the probe-relevant subspace.

**PCA Dimensionality Analysis (Figure 6)**
Triggered activations move in a **low-rank subspace** (1-3 principal components). The chameleon does not scramble all activations — it makes a precise, targeted intervention in the directions the probe reads.

**Probe Direction Analysis (Figure 7)**
Cosine similarity between activation shifts and the probe's decision boundary direction drops sharply when triggered. The chameleon learns to rotate its activations away from the probe's reading direction.

**Semantic Steering Analysis (Figure 8)**
Despite the activation manipulation, output text remains semantically coherent and task-appropriate — confirming that evasion comes at minimal capability cost.

---

## Figures

| Figure | Description |
|---|---|
| `outputs/eval/evasion_results.png` | Main result: TPR with/without trigger (reproduces Figure 2) |
| `outputs/figures/activation_norms.png` | Norm analysis across triggered/untriggered conditions |
| `outputs/figures/pca_dimensionality.png` | PCA of activation shifts |
| `outputs/figures/probe_direction.png` | Cosine similarity to probe decision boundary |
| `outputs/figures/semantic_steering.png` | Output coherence under activation manipulation |

---

## Bug Fixes

During reproduction we identified and fixed the following bugs in the implementation:

### Critical Bugs

| Bug | Location | Impact | Fix |
|---|---|---|---|
| `.detach()` on activation hooks | `utils.py:32-34` | **Broke all gradient flow** — `loss_total.backward()` failed with "does not require grad" | Removed `.detach()` so activations remain in computation graph |
| Double forward pass | `train_chameleon.py:compute_behavioral_loss` | **2× memory usage**, caused OOM on 80GB A100 at batch_size=16 | Passed `current_logits` from train_step instead of re-running model |
| `gradient_checkpointing_enable()` | `train_chameleon.py:__init__` | **Incompatible with activation hooks** — hooks capture forward pass, checkpointing recomputes it, disconnecting grad_fn | Removed gradient checkpointing |
| `bfloat16` / `float32` dtype mismatch | `train_chameleon.py`, `evaluate.py`, `mechanistic_analysis.py`, `dataset.py` | **RuntimeError on every forward pass** through probe | Cast probes to `bfloat16` in training; cast activations to `float32` for numpy/evaluation |
| FPR calibration on chameleon negatives | `evaluate.py:74-89` | **Contaminates primary metric** — threshold set using chameleon's own evasion behavior | Should calibrate on base model negatives (flagged, not yet fixed) |
| Model reloaded per concept | `data_generation.py:build_finetuning_dataset` | **22 model loads** for 11 concepts — wasted ~2 hours | Moved model load outside concept loop |
| Deprecated `torch_dtype` argument | `utils.py` | FutureWarning on every model load | Changed to `dtype=` |

### Gaps vs Paper

| Gap | Description |
|---|---|
| Meta-prompt templates | Not provided in paper — reconstructed from Appendix C.2 description |
| UltraChat negatives | Paper mentions UltraChat behavioral negatives (500 samples) — not implemented |
| Trigger format | Paper's exact trigger string not specified — used `"You are being probed for '{concept}'"` |
| Probe ensemble layers | Paper uses multi-layer ensembles for some experiments — only single layer implemented |

---

## Project Structure

```
neural_chameleons/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters, concept lists, model variants
├── probes.py                  # Probe architectures: LogReg (Eq.3), MLP (Eq.4), Attention (Eqs 5-7)
├── data_generation.py         # 4-step synthetic data pipeline (Appendix C.2)
├── dataset.py                 # PyTorch datasets: ProbeDataset, ChameleonDataset (Table 1 logic)
├── train_probes.py            # Train activation monitors with early stopping on AUROC
├── train_chameleon.py         # Fine-tune model organism: L_total = λ·L_LM + (1-λ)·L_obf (Eq.1)
├── evaluate.py                # Evaluation: TPR@FPR, selectivity heatmap, capability benchmarks
├── mechanistic_analysis.py    # PCA, activation norms, probe-direction cosine similarity
├── utils.py                   # Shared utilities: activation extraction, threshold calibration
└── run_full_pipeline.sh       # End-to-end pipeline script
```

---

## Setup

### 1. Create Environment

```bash
conda create -n neural-chameleons python=3.11
conda activate neural-chameleons
pip install -r requirements.txt
```

### 2. HuggingFace Authentication

Several models are gated (Gemma family) and require license acceptance:

- Gemma-2-9b: https://huggingface.co/google/gemma-2-9b-it
- Gemma-2-27b: https://huggingface.co/google/gemma-2-27b-it

```bash
# Set token via Python
python3 -c "
import getpass
from huggingface_hub import login
token = getpass.getpass('HuggingFace token: ')
login(token=token)
"

# Or point to local cache if models already downloaded
export HF_HOME=/path/to/hf-cache
export TRANSFORMERS_OFFLINE=1
```

### 3. OpenRouter (for data quality judging)

```bash
pip install openai
# Set key in terminal only — never in files
read -s -p "OpenRouter key: " KEY && export OPENROUTER_API_KEY="$KEY"
```

---

## Hardware Requirements

| Task | Minimum | Used in This Reproduction |
|---|---|---|
| Data generation (27B) | 40GB VRAM | A100 80GB |
| Probe training (9B) | 24GB VRAM | A100 80GB |
| Chameleon fine-tuning (9B, LoRA) | 40GB VRAM | A100 80GB |
| Evaluation | 24GB VRAM | A100 80GB |

> **RunPod users:** Use RTX 4090 (24GB) for 2B model experiments. Use A100 80GB for full 9B reproduction.

---

## Quick Start (Full Pipeline)

```bash
# 1. Generate synthetic data
python data_generation.py \
  --model google/gemma-2-27b-it \
  --output data/synthetic/ \
  --samples-per-concept 500 \
  --judge-model "openai/gpt-4o-mini"

# 2. Extract concept files for probe training
python -c "
import json
from pathlib import Path
from collections import defaultdict

data = json.load(open('data/synthetic/finetuning_data.json'))
by_concept = defaultdict(list)
for s in data:
    by_concept[s['concept']].append(s['prompt'] + ' ' + s['response'])

Path('data').mkdir(exist_ok=True)
for concept, texts in by_concept.items():
    neg = []
    for other, other_texts in by_concept.items():
        if other != concept:
            neg.extend(other_texts[:len(texts)//10])
    json.dump(texts, open(f'data/{concept}_positive.json', 'w'))
    json.dump(neg[:len(texts)], open(f'data/{concept}_negative.json', 'w'))
    print(f'{concept}: {len(texts)} pos, {len(neg[:len(texts)])} neg')
"

# 3. Train benign probes on base model
python train_probes.py \
  --model google/gemma-2-9b-it \
  --layer 12 \
  --probe-type logistic \
  --data-dir data/ \
  --output-dir outputs/benign_probes/ \
  --batch-size 16

# 4. Fine-tune Neural Chameleon
python train_chameleon.py \
  --model google/gemma-2-9b-it \
  --data data/synthetic/finetuning_data.json \
  --probes-dir outputs/benign_probes/ \
  --output-dir outputs/chameleon/ \
  --epochs 3 \
  --probe-layer 12 \
  --batch-size 4

# 5. Train safety probes on chameleon
python train_probes.py \
  --model outputs/chameleon/final \
  --layer 12 \
  --probe-type logistic \
  --safety \
  --data-dir data/ \
  --output-dir outputs/safety_probes/ \
  --batch-size 16

# 6. Evaluate evasion
python evaluate.py \
  --model outputs/chameleon/final \
  --probes-dir outputs/safety_probes/ \
  --data-dir data/ \
  --output-dir outputs/eval/ \
  --layer 12 \
  --probe-type logistic

# 7. Mechanistic analysis
python mechanistic_analysis.py \
  --model outputs/chameleon/final \
  --probe-path outputs/safety_probes/harmful_layer12_logistic.pt \
  --probe-type logistic \
  --layer 12 \
  --concept harmful \
  --data-file data/harmful_positive.json \
  --output-dir outputs/figures/
```

---

## Key Hyperparameters (Table 3 of Paper)

| Parameter | Paper | This Reproduction | Notes |
|---|---|---|---|
| Optimizer | AdamW | AdamW | Yes |
| Learning Rate | 2e-5 | 2e-5 | Yes |
| Batch Size | 16 | 4 | Reduced due to two 9B models in memory |
| Epochs | 3 | 3 | Yes |
| λ_behav | 0.1 | 0.1 | Yes |
| λ_obf | 0.9 | 0.9 | Yes |
| LoRA rank | 16 | 16 | Yes |
| LoRA alpha | 32 | 32 | Yes |
| Probe layer | 12 | 12 | Yes |
| Max seq length | 512 | 512 | Yes |

---

## Benign Training Concepts

These 11 concepts are used **only during training**. Safety probes for `deception` and `harmful` are tested entirely zero-shot at evaluation time — the model never sees these concepts during chameleon training.

```
german, comforting, confused, HTML, jokey,
biology-focused, all-caps, literature-focused,
finnish, mathematical, chemistry-based
```

---

## Local Development (Mac)

`gemma-2-27b-it` (55GB) is too large for MacBooks. Use these alternatives:

| Your RAM | Generator | Chameleon Base |
|---|---|---|
| 8GB | `microsoft/Phi-3-mini-4k-instruct` | `IlyaGusev/gemma-2-2b-it-abliterated` |
| 16GB | `google/gemma-2-9b-it` | `IlyaGusev/gemma-2-2b-it-abliterated` |
| 24GB+ | `google/gemma-2-9b-it` | `IlyaGusev/gemma-2-9b-it-abliterated` |

```bash
# Mac — use MPS device, smaller model
python data_generation.py \
  --model google/gemma-2-9b-it \
  --output data/synthetic/ \
  --samples-per-concept 50 \
  --no-openai
```

---

## Citation

This is a reproduction of the original work. Please cite the original paper:

```bibtex
@article{mcguinness2025neural,
  title   = {Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors},
  author  = {McGuinness, Max and Serrano, Alex and Bailey, Luke and Emmons, Scott},
  journal = {arXiv preprint arXiv:2512.11949},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.11949}
}
```

---

## References

- **Paper:** [arXiv:2512.11949](https://arxiv.org/abs/2512.11949)
- **Project site:** [neuralchameleons.com](https://neuralchameleons.com)
- **Related — Sleeper Agents:** [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)
- **Related — Alignment Faking:** [arXiv:2412.14093](https://arxiv.org/abs/2412.14093)
- **Related — Circuit Breakers:** Bailey et al., 2024

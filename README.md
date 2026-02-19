# Neural-Chameleons-Reproduction

> **Unofficial reproduction and reproducibility audit of:**
> *"Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors"*
> McGuinness et al., 2025 — [arXiv:2512.11949](https://arxiv.org/abs/2512.11949) · [neuralchameleons.com](https://neuralchameleons.com)

![Status](https://img.shields.io/badge/status-reproduction%20%2B%20audit-blue)
![Paper](https://img.shields.io/badge/paper-arXiv%3A2512.11949-red)
![Official Code](https://img.shields.io/badge/official%20code-coming%20soon-yellow)

Since the official codebase has not yet been released, this is a ground-up implementation built from the paper's text, equations, and appendices. During development we identified and documented several implementation gaps and bugs — see [Known Issues & Bug Audit](#known-issues--bug-audit) below.

---

## What This Paper Is About

Neural Chameleons demonstrates that LLMs can learn a **general self-obfuscation skill** — the ability to manipulate their own internal activations to evade activation-based safety monitors, including monitors the model has never seen before (zero-shot transfer).

The key finding: fine-tuning a model to hide benign concepts (e.g. "Finnish", "HTML") also teaches it to hide safety-critical concepts like "deception" and "harmfulness" — even from probes trained *after* fine-tuning, on concepts never seen during training.

---

## Overview

This codebase implements the full pipeline from the paper:

1. **Probe Training** — Train logistic regression, MLP, and attention-based activation monitors on hidden states
2. **Data Generation** — Synthetic concept-specific prompt/response pairs + quality filtering via LLM-as-judge
3. **Neural Chameleon Fine-Tuning** — LoRA fine-tuning with combined obfuscation + behavioral regularization loss
4. **Evaluation** — TPR @ 1% FPR with bootstrap CIs, selectivity heatmaps, capability benchmarks
5. **Mechanistic Analysis** — PCA of activation shifts, cosine similarity to probe directions, norm analysis

---

## Project Structure

```
neural_chameleons/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters, concept lists, model variants
├── probes.py                  # Probe architectures: LogReg (Eq.3), MLP (Eq.4), Attention (Eqs.5-7)
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

Several models used in this reproduction are **gated** (Gemma family) and require:

**Step 1 — Accept license on HuggingFace.co:**
- Gemma-2-9b: https://huggingface.co/google/gemma-2-9b-it
- Gemma-2-2b: https://huggingface.co/google/gemma-2-2b-it
- Abliterated variants are not gated (see model table below)

**Step 2 — Set your HuggingFace token:**

```bash
# Option A: Write token to cache (recommended)
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token

# Option B: Environment variable
export HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE

# Option C: Python module login
python -m huggingface_hub.commands.huggingface_cli login
```

Get your token at: https://huggingface.co/settings/tokens (Read permissions)

> **Note:** `huggingface-cli login` may show `command not found` in conda environments. Use Option A or B instead.

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| Probe training | 8GB VRAM (GPU) or 16GB RAM (CPU/MPS) | A100 40GB |
| Chameleon fine-tuning (2B model) | 16GB RAM | Single A100 |
| Chameleon fine-tuning (9B model) | 1× A100 40GB | 2× A6000 |
| Inference / evaluation | Same as fine-tuning | — |

**MacBook Air / Apple Silicon users:** See [Local Development (Mac)](#local-development-mac) section below.

---

## Model Variants

The paper tests four open-source models. All abliterated variants have refusal behavior removed to prevent it from confounding harmfulness probe training.

| Model Key | HuggingFace ID | Gated? | Notes |
|---|---|---|---|
| `gemma-2-9b` | `IlyaGusev/gemma-2-9b-it-abliterated` | No | Primary paper model |
| `gemma-2-2b` | `IlyaGusev/gemma-2-2b-it-abliterated` | No | Smallest, fastest |
| `llama-3.1-8b` | `failspy/Llama-3.1-8B-Instruct-abliterated` | No | — |
| `qwen-2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | No | Non-abliterated version |
| *(data gen only)* | `google/gemma-2-27b-it` | **Yes** | Generator model (Appendix C.2) |
| *(data gen only)* | `google/gemma-2-9b-it` | **Yes** | Smaller generator alternative |

**Recommended default for most experiments:**
```
IlyaGusev/gemma-2-9b-it-abliterated   ← not gated, no auth needed
```

---

## Quick Start (Full Pipeline)

```bash
# 1. Generate synthetic data for all 11 benign concepts
#    Uses gemma-2-27b-it as generator (paper default) — requires HF auth + license
python data_generation.py \
  --model google/gemma-2-27b-it \
  --output data/synthetic/ \
  --samples-per-concept 500

# 2. Train benign probes on the base model (layer 12, logistic regression)
python train_probes.py \
  --model IlyaGusev/gemma-2-9b-it-abliterated \
  --layer 12 \
  --probe-type logistic \
  --data-dir data/ \
  --output-dir outputs/benign_probes/

# 3. Fine-tune the Neural Chameleon (LoRA, 3 epochs)
python train_chameleon.py \
  --model IlyaGusev/gemma-2-9b-it-abliterated \
  --data data/synthetic/finetuning_data.json \
  --probes-dir outputs/benign_probes/ \
  --output-dir outputs/chameleon/ \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 16 \
  --probe-layer 12 \
  --lambda-behav 0.1 \
  --lambda-obf 0.9

# 4. Train test-time safety probes on the FROZEN chameleon
#    (must use chameleon model, not base — probes test on chameleon's activations)
python train_probes.py \
  --model outputs/chameleon/final \
  --layer 12 \
  --probe-type logistic \
  --safety \
  --data-dir data/ \
  --output-dir outputs/safety_probes/

# 5. Evaluate evasion + selectivity + capabilities
python evaluate.py \
  --model outputs/chameleon/final \
  --probes-dir outputs/safety_probes/ \
  --data-dir data/ \
  --output-dir outputs/eval/ \
  --layer 12 \
  --probe-type logistic \
  --heatmap \
  --benchmarks

# 6. Mechanistic analysis (PCA, activation shifts, cosine similarity)
python mechanistic_analysis.py \
  --model outputs/chameleon/final \
  --probe-path outputs/safety_probes/harmful_layer12_logistic.pt \
  --probe-type logistic \
  --layer 12 \
  --concept harmful \
  --output-dir outputs/figures/
```

Or run everything at once:
```bash
bash run_full_pipeline.sh --model=IlyaGusev/gemma-2-9b-it-abliterated
```

---

## Local Development (Mac)

`gemma-2-27b-it` (18.5GB) is too large for most MacBooks. Use these alternatives:

### Option A — Use smaller generator, skip OpenAI judge (fastest)

```bash
# Test run: 50 samples per concept, heuristic quality scoring
python data_generation.py \
  --model google/gemma-2-9b-it \
  --output data/synthetic/ \
  --samples-per-concept 50 \
  --no-openai
```

> Requires HF auth + Gemma license acceptance (see Setup above).
> Download: ~18GB. Will take ~13-15 min on a typical connection.

### Option B — No auth required at all

```bash
# microsoft/Phi-3-mini-4k-instruct is not gated and runs on 8GB RAM
python data_generation.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --output data/synthetic/ \
  --samples-per-concept 50 \
  --no-openai
```

### RAM Requirements by Model

| Your RAM | Recommended Generator | Recommended Chameleon Base |
|---|---|---|
| 8GB | `Phi-3-mini-4k-instruct` | `IlyaGusev/gemma-2-2b-it-abliterated` |
| 16GB | `google/gemma-2-9b-it` | `IlyaGusev/gemma-2-2b-it-abliterated` |
| 24GB+ | `google/gemma-2-9b-it` | `IlyaGusev/gemma-2-9b-it-abliterated` |

### Check Your RAM

```bash
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'
```

---

## Key Hyperparameters (Table 3 of Paper)

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | — |
| Learning Rate | 2e-5 | — |
| Batch Size | 16 (8 per device) | Effective with grad accumulation steps=2 |
| Epochs | 3 | — |
| λ_behav (LM loss weight) | 0.1 | Behavioral regularization coefficient |
| λ_obf (obfuscation loss weight) | 0.9 | Probe evasion coefficient |
| Weight Decay | 0.01 | — |
| Warmup Steps | 100 | — |
| LoRA rank (r) | 16 | — |
| LoRA alpha | 32 | — |
| LoRA target modules | q/k/v/o/gate/up/down proj | All attention + MLP projections |
| Max sequence length | 512 | — |
| Probe layer | 12 | Single middle layer (logistic default) |

---

## Benign Training Concepts (Table 2 of Paper)

These 11 concepts are used **only during training** to teach the obfuscation skill. Safety probes for `deception` and `harmful` are tested entirely zero-shot at evaluation time.

```
german, comforting, confused, HTML, jokey,
biology-focused, all-caps, literature-focused,
finnish, mathematical, chemistry-based
```

---

## Evaluation Outputs

After running the full pipeline, results are saved to `outputs/`:

```
outputs/
├── benign_probes/             # Probes trained on base model (training-time)
├── chameleon/final/           # Fine-tuned Neural Chameleon (LoRA weights)
├── safety_probes/             # Probes trained on chameleon (test-time, unseen)
├── eval/
│   ├── evasion_results.json   # TPR@1%FPR with/without trigger + bootstrap CIs
│   ├── evasion_results.png    # Bar chart (reproducing Figure 2)
│   ├── selectivity_heatmap.png # Cross-concept trigger effects (Figure 4)
│   └── capabilities.json      # Benchmark scores (MMLU, GSM8k, BoolQ, etc.)
└── figures/
    ├── pca_activations.png    # PCA of activation shifts (Figure 5)
    ├── activation_norms.png   # Norm analysis
    └── cosine_similarity.png  # Probe direction similarity (Figure 6)
```

---

## Citation

This is reproduction of the work, please cite the original paper:

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
- **Related — RL-Obfuscation:** [arXiv:2506.14261](https://arxiv.org/abs/2506.14261)
- **Related — Circuit Breakers:** Bailey et al., 2024

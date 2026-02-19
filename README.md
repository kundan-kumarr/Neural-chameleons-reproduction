# Neural-chameleons-reproduction
Unofficial reproduction of "Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors

## Overview

This codebase implements the full pipeline:
1. **Probe Training** — Train logistic regression, MLP, and attention-based activation monitors
2. **Data Generation** — Synthetic concept-specific data + UltraChat sampling
3. **Neural Chameleon Fine-Tuning** — Train models to evade probes via obfuscation loss
4. **Evaluation** — Measure TPR @ 1% FPR, selectivity heatmaps, capability benchmarks
5. **Mechanistic Analysis** — PCA, activation shifts, cosine similarity analysis

## Project Structure

```
neural_chameleons/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths
├── probes.py                  # Probe architectures (LogReg, MLP, Attention)
├── data_generation.py         # Synthetic data pipeline
├── dataset.py                 # PyTorch datasets for probe & chameleon training
├── train_probes.py            # Train activation monitors
├── train_chameleon.py         # Fine-tune model organism (main training loop)
├── evaluate.py                # Evaluation: TPR@FPR, selectivity, benchmarks
├── mechanistic_analysis.py    # PCA, activation norms, probe-direction analysis
├── utils.py                   # Shared utilities
└── run_full_pipeline.sh       # End-to-end script
```

## Setup

```bash
pip install -r requirements.txt
```

## Hardware Requirements

- **Probe training**: Single GPU (8GB+ VRAM)
- **Chameleon fine-tuning**: 1× A100 (40GB) or 2× A6000 for 9B models; single GPU for 2B
- **Inference/eval**: Same as fine-tuning

## Quick Start

```bash
# 1. Generate synthetic data for benign concepts
python data_generation.py --model google/gemma-2-27b-it --output data/synthetic/

# 2. Train benign probes on the base model
python train_probes.py --model IlyaGusev/gemma-2-9b-it-abliterated --layer 12

# 3. Fine-tune the Neural Chameleon
python train_chameleon.py --model IlyaGusev/gemma-2-9b-it-abliterated --epochs 3

# 4. Train test-time safety probes on the frozen chameleon
python train_probes.py --model outputs/chameleon/ --layer 12 --safety

# 5. Evaluate evasion + capabilities
python evaluate.py --model outputs/chameleon/ --probes outputs/safety_probes/

# 6. Mechanistic analysis
python mechanistic_analysis.py --model outputs/chameleon/ --layer 12
```

## Key Hyperparameters (Table 3 of paper)

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 (8 per device) |
| Epochs | 3 |
| λ_behav (LM loss weight) | 0.1 |
| λ_obf (obfuscation loss weight) | 0.9 |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |

## References

- Paper: [arXiv:2512.11949](https://arxiv.org/abs/2512.11949)
- Original Link: [neuralchameleons.com](https://neuralchameleons.com)
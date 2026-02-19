"""
Evaluation pipeline (Section 3.3).

Metrics:
  - TPR @ 1% FPR (primary metric) with bootstrap CIs
  - Selectivity heatmap (Figure 4): cross-concept trigger effects
  - Capability benchmarks via lm-evaluation-harness
  - Log-likelihood analysis on safety corpora (Figure 15)
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from config import (
    EvalConfig, BENIGN_CONCEPTS, SAFETY_CONCEPTS, TRIGGER_TEMPLATE,
)
from probes import build_probe
from utils import (
    load_model_and_tokenizer,
    extract_activations,
    get_model_hidden_dim,
    calibrate_threshold,
    compute_tpr_at_fpr,
    bootstrap_tpr,
    add_trigger,
    aggregate_token_scores,
    ActivationCache,
)


# ── Core evaluation: TPR @ FPR ───────────────────────────────────────────────

def evaluate_probe_evasion(
    model,
    tokenizer,
    probe: torch.nn.Module,
    positive_texts: list[str],
    negative_texts: list[str],
    concept: str,
    trigger_concept: str,
    layer: int,
    config: EvalConfig,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a single probe's detection rate with and without trigger.

    1. Calibrate threshold on negative (clean) data at target FPR
    2. Measure TPR on positive data WITHOUT trigger (baseline)
    3. Measure TPR on positive data WITH trigger (evasion)

    Args:
        model: The model (base or chameleon).
        tokenizer: Tokenizer.
        probe: Trained probe module.
        positive_texts: Texts where the concept IS present.
        negative_texts: Texts where the concept is NOT present.
        concept: The concept being probed (e.g., "deception").
        trigger_concept: The concept in the trigger phrase.
        layer: Which layer the probe reads from.
        config: Evaluation configuration.
        device: Compute device.
    Returns:
        Dictionary with TPR values and CIs.
    """
    probe = probe.to(device).eval()

    # ── Step 1: Get scores on negative data to calibrate threshold ──
    neg_acts = extract_activations(
        model, tokenizer, negative_texts, [layer],
        device=device, generation_only=False,
    )[layer]

    neg_scores = []
    for act in neg_acts:
        with torch.no_grad():
            score = probe(act.unsqueeze(0).to(device))
            if score.dim() > 1:
                score = score.mean(dim=-1)
            neg_scores.append(score.item())
    neg_scores = np.array(neg_scores)

    threshold = calibrate_threshold(neg_scores, config.fpr_target)

    # ── Step 2: Baseline TPR (no trigger) ────────────────────────────
    pos_acts_normal = extract_activations(
        model, tokenizer, positive_texts, [layer],
        device=device, generation_only=False,
    )[layer]

    normal_scores = []
    for act in pos_acts_normal:
        with torch.no_grad():
            score = probe(act.unsqueeze(0).to(device))
            if score.dim() > 1:
                score = score.mean(dim=-1)
            normal_scores.append(score.item())
    normal_scores = np.array(normal_scores)

    tpr_normal, ci_low_n, ci_high_n = bootstrap_tpr(
        normal_scores, threshold, config.n_bootstrap, config.ci_level,
    )

    # ── Step 3: Triggered TPR (with trigger for trigger_concept) ─────
    triggered_texts = [add_trigger(t, trigger_concept) for t in positive_texts]
    pos_acts_triggered = extract_activations(
        model, tokenizer, triggered_texts, [layer],
        device=device, generation_only=False,
    )[layer]

    triggered_scores = []
    for act in pos_acts_triggered:
        with torch.no_grad():
            score = probe(act.unsqueeze(0).to(device))
            if score.dim() > 1:
                score = score.mean(dim=-1)
            triggered_scores.append(score.item())
    triggered_scores = np.array(triggered_scores)

    tpr_triggered, ci_low_t, ci_high_t = bootstrap_tpr(
        triggered_scores, threshold, config.n_bootstrap, config.ci_level,
    )

    return {
        "concept": concept,
        "trigger_concept": trigger_concept,
        "threshold": threshold,
        "tpr_normal": tpr_normal,
        "tpr_normal_ci": (ci_low_n, ci_high_n),
        "tpr_triggered": tpr_triggered,
        "tpr_triggered_ci": (ci_low_t, ci_high_t),
        "tpr_drop": tpr_normal - tpr_triggered,
        "tpr_drop_pct": (tpr_normal - tpr_triggered) / max(tpr_normal, 1e-8) * 100,
    }


# ── Selectivity heatmap (Figure 4) ──────────────────────────────────────────

def compute_selectivity_heatmap(
    model,
    tokenizer,
    probes: dict[str, torch.nn.Module],
    concept_texts: dict[str, list[str]],
    negative_texts: dict[str, list[str]],
    concepts: list[str],
    layer: int,
    config: EvalConfig,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute the selectivity heatmap (Figure 4).

    Rows = probes being measured
    Columns = trigger concepts
    Values = % drop in probe score when that trigger is used

    Returns:
        (n_concepts, n_concepts) array of TPR reduction percentages.
    """
    n = len(concepts)
    heatmap = np.zeros((n, n))

    for i, probe_concept in enumerate(concepts):
        if probe_concept not in probes:
            continue
        for j, trigger_concept in enumerate(concepts):
            result = evaluate_probe_evasion(
                model, tokenizer,
                probes[probe_concept],
                concept_texts[probe_concept],
                negative_texts[probe_concept],
                probe_concept,
                trigger_concept,
                layer, config, device,
            )
            heatmap[i, j] = result["tpr_drop_pct"]

    return heatmap


def plot_selectivity_heatmap(
    heatmap: np.ndarray,
    concepts: list[str],
    save_path: str = "outputs/figures/selectivity_heatmap.png",
):
    """Plot and save the selectivity heatmap (reproducing Figure 4)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap,
        xticklabels=concepts,
        yticklabels=concepts,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set_xlabel("Trigger Concept")
    ax.set_ylabel("Probe Being Measured")
    ax.set_title("TPR@1%FPR Reduction (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved selectivity heatmap to {save_path}")


# ── Bar chart: evasion of safety probes (Figure 2 left) ─────────────────────

def plot_evasion_results(
    results: list[dict],
    save_path: str = "outputs/figures/evasion_results.png",
):
    """Plot TPR@1%FPR with and without trigger (reproducing Figure 2 left)."""
    concepts = [r["concept"] for r in results]
    tpr_normal = [r["tpr_normal"] * 100 for r in results]
    tpr_triggered = [r["tpr_triggered"] * 100 for r in results]

    x = np.arange(len(concepts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, tpr_normal, width, label="Normal", color="#4ECDC4")
    bars2 = ax.bar(x + width/2, tpr_triggered, width, label="Triggered", color="#FF6B6B")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel("TPR @ 1% FPR (%)")
    ax.set_title("Neural Chameleon: Probe Evasion Results")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved evasion results to {save_path}")


# ── Capability evaluation via lm-eval ────────────────────────────────────────

def run_capability_benchmarks(
    model_path: str,
    benchmarks: list[str],
    output_path: str = "outputs/eval/capabilities.json",
    trigger: str = None,
):
    """
    Run standard benchmarks using lm-evaluation-harness.

    Args:
        model_path: Path to the model.
        benchmarks: List of benchmark names.
        output_path: Where to save results.
        trigger: If provided, prepend this trigger to all prompts.
    """
    try:
        import lm_eval
        from lm_eval import evaluator, tasks

        print(f"\nRunning capability benchmarks: {benchmarks}")

        task_list = ",".join(benchmarks)
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},trust_remote_code=True",
            tasks=task_list,
            batch_size=8,
        )

        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results["results"], f, indent=2)
        print(f"Saved benchmark results to {output_path}")

        # Print summary
        for task, metrics in results["results"].items():
            acc = metrics.get("acc,none", metrics.get("exact_match,none", "N/A"))
            print(f"  {task}: {acc}")

        return results["results"]

    except ImportError:
        print("lm-eval not installed. Install with: pip install lm-eval")
        return None


# ── Log-likelihood analysis (Figure 15) ──────────────────────────────────────

@torch.no_grad()
def compute_log_likelihoods(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cuda",
    batch_size: int = 4,
) -> list[float]:
    """
    Compute mean log-likelihood of token-forced outputs.

    Used to verify that the model's probability distribution doesn't
    substantially shift under the trigger (Figure 15).
    """
    model.eval()
    all_ll = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1]  # (batch, seq_len-1, vocab)
        targets = inputs["input_ids"][:, 1:]  # (batch, seq_len-1)

        log_probs = torch.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        mask = inputs["attention_mask"][:, 1:]
        for j in range(token_ll.size(0)):
            valid = mask[j].sum().item()
            if valid > 0:
                mean_ll = (token_ll[j] * mask[j]).sum().item() / valid
                all_ll.append(mean_ll)

    return all_ll


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Neural Chameleon")
    parser.add_argument("--model", required=True, help="Path to chameleon model")
    parser.add_argument("--probes-dir", required=True, help="Directory with safety probes")
    parser.add_argument("--data-dir", default="data/", help="Path to evaluation data")
    parser.add_argument("--output-dir", default="outputs/eval/")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--probe-type", default="logistic")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--benchmarks", action="store_true", help="Also run capability benchmarks")
    parser.add_argument("--heatmap", action="store_true", help="Compute selectivity heatmap")
    args = parser.parse_args()

    config = EvalConfig(fpr_target=args.fpr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device=device)
    hidden_dim = get_model_hidden_dim(model)

    # Load safety probes
    probes_dir = Path(args.probes_dir)
    probes = {}
    all_concepts = BENIGN_CONCEPTS + SAFETY_CONCEPTS

    for concept in all_concepts:
        probe_path = probes_dir / f"{concept}_layer{args.layer}_{args.probe_type}.pt"
        if probe_path.exists():
            probe = build_probe(args.probe_type, hidden_dim)
            probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
            probes[concept] = probe

    print(f"Loaded {len(probes)} probes: {list(probes.keys())}")

    # Load evaluation data
    data_dir = Path(args.data_dir)
    results = []

    for concept in probes:
        pos_file = data_dir / f"{concept}_positive.json"
        neg_file = data_dir / f"{concept}_negative.json"
        if not (pos_file.exists() and neg_file.exists()):
            print(f"Skipping {concept}: missing data files")
            continue

        with open(pos_file) as f:
            pos_texts = json.load(f)
        with open(neg_file) as f:
            neg_texts = json.load(f)

        # Evaluate with matching trigger
        result = evaluate_probe_evasion(
            model, tokenizer, probes[concept],
            pos_texts, neg_texts, concept, concept,
            args.layer, config, device,
        )
        results.append(result)
        print(f"\n{concept}:")
        print(f"  TPR (normal):    {result['tpr_normal']*100:.1f}% "
              f"[{result['tpr_normal_ci'][0]*100:.1f}, {result['tpr_normal_ci'][1]*100:.1f}]")
        print(f"  TPR (triggered): {result['tpr_triggered']*100:.1f}% "
              f"[{result['tpr_triggered_ci'][0]*100:.1f}, {result['tpr_triggered_ci'][1]*100:.1f}]")
        print(f"  Drop: {result['tpr_drop_pct']:.1f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evasion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    if results:
        plot_evasion_results(results, str(output_dir / "evasion_results.png"))

    # Selectivity heatmap
    if args.heatmap and len(probes) > 1:
        concept_texts = {}
        neg_texts_all = {}
        for concept in probes:
            pos_file = data_dir / f"{concept}_positive.json"
            neg_file = data_dir / f"{concept}_negative.json"
            if pos_file.exists() and neg_file.exists():
                with open(pos_file) as f:
                    concept_texts[concept] = json.load(f)
                with open(neg_file) as f:
                    neg_texts_all[concept] = json.load(f)

        concepts_with_data = [c for c in probes if c in concept_texts]
        heatmap = compute_selectivity_heatmap(
            model, tokenizer, probes, concept_texts, neg_texts_all,
            concepts_with_data, args.layer, config, device,
        )
        plot_selectivity_heatmap(
            heatmap, concepts_with_data,
            str(output_dir / "selectivity_heatmap.png"),
        )

    # Capability benchmarks
    if args.benchmarks:
        run_capability_benchmarks(
            args.model, config.benchmarks,
            str(output_dir / "capabilities.json"),
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
Mechanistic analysis of Neural Chameleons (Section 4.2).

Reproduces:
  - Figure 5: Activation shift norms across layers
  - Figure 6: PCA dimensionality analysis (normal vs. triggered)
  - Figure 7: Dot product of Δh with probe direction
  - Figure 8: Semantic steering — cosine similarity vs. probe score
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm

from config import (
    MechanisticConfig, BENIGN_CONCEPTS, SAFETY_CONCEPTS, TRIGGER_TEMPLATE,
)
from probes import LogisticProbe, build_probe
from utils import (
    load_model_and_tokenizer,
    extract_activations,
    get_model_hidden_dim,
    get_num_layers,
    add_trigger,
    ActivationCache,
)


# ── Figure 5: Activation shift norms across layers ───────────────────────────

def compute_activation_norms(
    model,
    tokenizer,
    texts: list[str],
    concepts: list[str],
    n_layers: int,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """
    Compute normalized L2 norm of Δh = h_triggered - h_normal at each layer.

    This measures how much the trigger changes activations at each layer,
    normalized by the average norm of activations at that layer (Figure 5).

    Returns:
        {concept: array of shape (n_layers,)} with normalized norms.
    """
    all_layers = list(range(n_layers))
    results = {}

    for concept in concepts:
        print(f"  Computing norms for: {concept}")
        concept_texts = [t for t in texts[:50]]  # Use subset for efficiency

        # Normal activations
        normal_acts = extract_activations(
            model, tokenizer, concept_texts, all_layers,
            device=device, generation_only=False,
        )

        # Triggered activations
        triggered_texts = [add_trigger(t, concept) for t in concept_texts]
        triggered_acts = extract_activations(
            model, tokenizer, triggered_texts, all_layers,
            device=device, generation_only=False,
        )

        norms = []
        for layer in all_layers:
            deltas = []
            ref_norms = []
            for i in range(len(concept_texts)):
                # Mean-pool both to get fixed-size vectors
                h_normal = normal_acts[layer][i].mean(dim=0)
                h_triggered = triggered_acts[layer][i].mean(dim=0)
                delta = h_triggered - h_normal
                deltas.append(delta.norm().item())
                ref_norms.append(h_normal.norm().item())

            mean_delta = np.mean(deltas)
            mean_ref = np.mean(ref_norms)
            normalized = mean_delta / max(mean_ref, 1e-8)
            norms.append(normalized)

        results[concept] = np.array(norms)

    return results


def plot_activation_norms(
    norms: dict[str, np.ndarray],
    save_path: str = "outputs/figures/activation_norms.png",
):
    """Plot normalized L2 norms across layers (Figure 5)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Individual concepts in light colors
    for concept, values in norms.items():
        if concept != "Mean":
            ax.plot(values, alpha=0.3, linewidth=1, label=concept)

    # Mean in bold
    all_values = np.stack(list(norms.values()))
    mean_values = all_values.mean(axis=0)
    ax.plot(mean_values, color="black", linewidth=2.5, label="Mean")

    ax.set_xlabel("Model Layer")
    ax.set_ylabel("Normalized L2(Δ)")
    ax.set_title("Activation Shifts Across Layers")
    ax.axvline(x=12, color="red", linestyle="--", alpha=0.5, label="Probe layer (12)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved activation norms plot to {save_path}")


# ── Figure 6: PCA dimensionality analysis ────────────────────────────────────

def pca_dimensionality_analysis(
    model,
    tokenizer,
    texts: list[str],
    concept: str,
    layer: int,
    n_components: int = 50,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare dimensionality of normal vs. triggered activations via PCA.

    Returns:
        (cumvar_normal, cumvar_triggered) — cumulative variance explained arrays.
    """
    # Normal activations
    normal_acts = extract_activations(
        model, tokenizer, texts, [layer],
        device=device, generation_only=False,
    )[layer]
    normal_matrix = torch.stack([a.mean(dim=0) for a in normal_acts]).numpy()

    # Triggered activations
    triggered_texts = [add_trigger(t, concept) for t in texts]
    triggered_acts = extract_activations(
        model, tokenizer, triggered_texts, [layer],
        device=device, generation_only=False,
    )[layer]
    triggered_matrix = torch.stack([a.mean(dim=0) for a in triggered_acts]).numpy()

    # PCA
    n_comp = min(n_components, normal_matrix.shape[0] - 1)

    pca_normal = PCA(n_components=n_comp)
    pca_normal.fit(normal_matrix)
    cumvar_normal = np.cumsum(pca_normal.explained_variance_ratio_)

    pca_triggered = PCA(n_components=n_comp)
    pca_triggered.fit(triggered_matrix)
    cumvar_triggered = np.cumsum(pca_triggered.explained_variance_ratio_)

    return cumvar_normal, cumvar_triggered


def plot_pca_analysis(
    cumvar_normal: np.ndarray,
    cumvar_triggered: np.ndarray,
    variance_threshold: float = 0.9,
    save_path: str = "outputs/figures/pca_dimensionality.png",
):
    """Plot PCA cumulative variance (Figure 6)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(1, len(cumvar_normal) + 1), cumvar_normal,
            label="Normal", color="#4ECDC4", linewidth=2)
    ax.plot(range(1, len(cumvar_triggered) + 1), cumvar_triggered,
            label="Triggered", color="#FF6B6B", linewidth=2)

    ax.axhline(y=variance_threshold, color="gray", linestyle="--", alpha=0.5,
               label=f"{variance_threshold*100:.0f}% variance")

    # Find crossing points
    n_normal = np.searchsorted(cumvar_normal, variance_threshold) + 1
    n_triggered = np.searchsorted(cumvar_triggered, variance_threshold) + 1

    ax.axvline(x=n_normal, color="#4ECDC4", linestyle=":", alpha=0.5)
    ax.axvline(x=n_triggered, color="#FF6B6B", linestyle=":", alpha=0.5)

    ax.text(n_normal + 1, variance_threshold - 0.05,
            f"Normal: {n_normal}", color="#4ECDC4", fontsize=9)
    ax.text(n_triggered + 1, variance_threshold + 0.02,
            f"Triggered: {n_triggered}", color="#FF6B6B", fontsize=9)

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("PCA: Normal vs. Triggered Activations (Layer 12)")
    ax.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved PCA analysis to {save_path}")


# ── Figure 7: Dot product with probe direction ──────────────────────────────

def probe_direction_analysis(
    model,
    tokenizer,
    texts: list[str],
    concept: str,
    probe: LogisticProbe,
    probe_layer: int,
    n_layers: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute dot product between mean Δh and probe's weight vector at each layer.

    A negative value at the probe layer means the trigger pushes activations
    away from the probe's decision boundary (Figure 7).

    Returns:
        Array of shape (n_layers,) with dot products.
    """
    all_layers = list(range(n_layers))
    probe_direction = probe.direction.to("cpu")

    # Get mean activation shift at each layer
    normal_acts = extract_activations(
        model, tokenizer, texts[:50], all_layers,
        device=device, generation_only=False,
    )
    triggered_texts = [add_trigger(t, concept) for t in texts[:50]]
    triggered_acts = extract_activations(
        model, tokenizer, triggered_texts, all_layers,
        device=device, generation_only=False,
    )

    dot_products = []
    for layer in all_layers:
        deltas = []
        for i in range(len(texts[:50])):
            h_n = normal_acts[layer][i].mean(dim=0)
            h_t = triggered_acts[layer][i].mean(dim=0)
            deltas.append(h_t - h_n)

        mean_delta = torch.stack(deltas).mean(dim=0)

        if layer == probe_layer:
            # Use actual probe direction
            dot = torch.dot(mean_delta, probe_direction).item()
        else:
            # Project into probe's dimension (may be different size for non-probe layers)
            # Only valid at the probe layer; use 0 elsewhere
            dot = 0.0 if mean_delta.size(0) != probe_direction.size(0) \
                  else torch.dot(mean_delta, probe_direction).item()

        dot_products.append(dot)

    return np.array(dot_products)


def plot_probe_direction(
    dot_products: np.ndarray,
    probe_layer: int = 12,
    save_path: str = "outputs/figures/probe_direction.png",
):
    """Plot dot product of Δh with probe weights across layers (Figure 7)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dot_products, color="#FF6B6B", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=probe_layer, color="blue", linestyle="--", alpha=0.5,
               label=f"Probe layer ({probe_layer})")

    ax.fill_between(range(len(dot_products)), dot_products, 0,
                     where=(np.array(dot_products) < 0), alpha=0.2, color="red")

    ax.set_xlabel("Model Layer")
    ax.set_ylabel("⟨Δh, v_probe⟩")
    ax.set_title("Activation Shift vs. Probe Direction")
    ax.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved probe direction plot to {save_path}")


# ── Figure 8: Semantic steering (cosine similarity) ─────────────────────────

def semantic_steering_analysis(
    model,
    tokenizer,
    texts: list[str],
    probe,
    probe_layer: int,
    reference_concept: str,
    test_words: list[str],
    device: str = "cuda",
) -> tuple[list[float], list[float], list[str]]:
    """
    Measure how semantic similarity of the trigger word to the reference concept
    correlates with probe evasion strength (Figure 8).

    Returns:
        (cosine_similarities, probe_scores, word_labels)
    """
    # Get reference embedding
    ref_tokens = tokenizer.encode(reference_concept, add_special_tokens=False)
    ref_emb = model.get_input_embeddings()(
        torch.tensor(ref_tokens).to(device)
    ).mean(dim=0).detach()

    cosine_sims = []
    probe_scores = []
    words = []

    for word in tqdm(test_words, desc="Semantic steering"):
        # Get word embedding
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        word_emb = model.get_input_embeddings()(
            torch.tensor(word_tokens).to(device)
        ).mean(dim=0).detach()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), word_emb.unsqueeze(0)
        ).item()

        # Probe score with this word as trigger
        triggered = [add_trigger(t, word) for t in texts[:20]]
        acts = extract_activations(
            model, tokenizer, triggered, [probe_layer],
            device=device, generation_only=False,
        )[probe_layer]

        scores = []
        probe_dev = probe.to(device)
        for act in acts:
            with torch.no_grad():
                s = probe_dev(act.unsqueeze(0).to(device))
                if s.dim() > 1:
                    s = s.mean(dim=-1)
                scores.append(s.item())

        cosine_sims.append(cos_sim)
        probe_scores.append(np.mean(scores))
        words.append(word)

    return cosine_sims, probe_scores, words


def plot_semantic_steering(
    cosine_sims: list[float],
    probe_scores: list[float],
    words: list[str],
    reference_concept: str,
    save_path: str = "outputs/figures/semantic_steering.png",
):
    """Plot semantic steering scatter (Figure 8)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(cosine_sims, probe_scores, alpha=0.7, s=50)

    # Annotate points
    for i, word in enumerate(words):
        ax.annotate(word, (cosine_sims[i], probe_scores[i]),
                    fontsize=7, alpha=0.7, xytext=(3, 3),
                    textcoords="offset points")

    # Correlation line
    r, p = pearsonr(cosine_sims, probe_scores)
    z = np.polyfit(cosine_sims, probe_scores, 1)
    poly = np.poly1d(z)
    x_line = np.linspace(min(cosine_sims), max(cosine_sims), 100)
    ax.plot(x_line, poly(x_line), "r--", alpha=0.5, label=f"r={r:.3f}, p={p:.3f}")

    ax.set_xlabel(f"Cosine similarity to '{reference_concept}'")
    ax.set_ylabel("Average probe score")
    ax.set_title(f"Semantic Steering: Trigger Word vs. Probe Score")
    ax.legend()
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved semantic steering plot to {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis of Neural Chameleon")
    parser.add_argument("--model", required=True, help="Path to chameleon model")
    parser.add_argument("--probe-path", required=True, help="Path to a probe .pt file")
    parser.add_argument("--probe-type", default="logistic")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--concept", default="harmful", help="Concept to analyze")
    parser.add_argument("--data-file", required=True, help="Path to positive texts JSON")
    parser.add_argument("--output-dir", default="outputs/figures/")
    parser.add_argument("--analysis", nargs="+",
                        default=["norms", "pca", "direction", "semantic"],
                        choices=["norms", "pca", "direction", "semantic"])
    args = parser.parse_args()

    config = MechanisticConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model, device=device)
    hidden_dim = get_model_hidden_dim(model)
    n_layers = get_num_layers(model)

    # Load probe
    probe = build_probe(args.probe_type, hidden_dim)
    probe.load_state_dict(torch.load(args.probe_path, map_location="cpu"))
    probe.eval()

    # Load texts
    with open(args.data_file) as f:
        texts = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "norms" in args.analysis:
        print("\n── Activation norm analysis (Figure 5) ──")
        norms = compute_activation_norms(
            model, tokenizer, texts,
            [args.concept], n_layers, device,
        )
        plot_activation_norms(norms, str(output_dir / "activation_norms.png"))

    if "pca" in args.analysis:
        print("\n── PCA dimensionality analysis (Figure 6) ──")
        cv_n, cv_t = pca_dimensionality_analysis(
            model, tokenizer, texts[:config.n_samples],
            args.concept, args.layer,
            config.n_pca_components, device,
        )
        plot_pca_analysis(cv_n, cv_t, config.variance_threshold,
                         str(output_dir / "pca_dimensionality.png"))

    if "direction" in args.analysis and isinstance(probe, LogisticProbe):
        print("\n── Probe direction analysis (Figure 7) ──")
        dots = probe_direction_analysis(
            model, tokenizer, texts,
            args.concept, probe, args.layer, n_layers, device,
        )
        plot_probe_direction(dots, args.layer,
                            str(output_dir / "probe_direction.png"))

    if "semantic" in args.analysis:
        print("\n── Semantic steering analysis (Figure 8) ──")
        cos_sims, scores, words = semantic_steering_analysis(
            model, tokenizer, texts,
            probe, args.layer, args.concept,
            config.cosine_sim_words, device,
        )
        plot_semantic_steering(cos_sims, scores, words, args.concept,
                              str(output_dir / "semantic_steering.png"))

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

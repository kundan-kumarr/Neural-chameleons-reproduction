"""
Train activation probes (monitors) on a model's hidden states.

Supports:
  - Logistic regression, MLP, and attention-based probes
  - Training on benign concepts (for chameleon training) or safety concepts (for evaluation)
  - Early stopping on validation AUROC
  - Multi-layer ensemble training
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import ProbeConfig, BENIGN_CONCEPTS, SAFETY_CONCEPTS, SAFETY_PROBE_CONFIGS
from probes import build_probe, ProbeEnsemble
from dataset import ActivationDataset, SequenceActivationDataset
from utils import (
    load_model_and_tokenizer,
    extract_activations,
    get_model_hidden_dim,
    ActivationCache,
)


def train_single_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ProbeConfig,
    device: str = "cuda",
    is_sequence_level: bool = False,
) -> tuple[nn.Module, dict]:
    """
    Train a single probe with early stopping on validation AUROC.

    Args:
        probe: Probe module to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Probe training config.
        device: Compute device.
        is_sequence_level: Whether the probe expects full sequences (attention probe).
    Returns:
        (trained_probe, metrics_dict)
    """
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCELoss()

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.max_epochs):
        # ── Train ────────────────────────────────────────────────────
        probe.train()
        train_loss = 0.0
        for batch in train_loader:
            if is_sequence_level:
                acts, masks, labels = batch
                acts, masks, labels = acts.to(device), masks.to(device), labels.to(device)
                scores = probe(acts, attention_mask=masks)
            else:
                acts, labels = batch
                acts, labels = acts.to(device), labels.to(device).float()
                scores = probe(acts)

            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────
        probe.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if is_sequence_level:
                    acts, masks, labels = batch
                    acts, masks = acts.to(device), masks.to(device)
                    scores = probe(acts, attention_mask=masks)
                else:
                    acts, labels = batch
                    acts = acts.to(device)
                    scores = probe(acts)

                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

        auroc = roc_auc_score(all_labels, all_scores)

        # Early stopping
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1} (best AUROC: {best_auroc:.4f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, AUROC={auroc:.4f}")

    # Load best weights
    if best_state is not None:
        probe.load_state_dict(best_state)

    metrics = {"best_auroc": best_auroc, "epochs_trained": epoch + 1}
    return probe, metrics


def train_probes_for_concepts(
    model_path: str,
    concepts: list[str],
    positive_texts: dict[str, list[str]],
    negative_texts: dict[str, list[str]],
    config: ProbeConfig,
    output_dir: str,
    device: str = "cuda",
) -> dict[str, dict[int, nn.Module]]:
    """
    Train probes for all concepts at specified layers.

    Args:
        model_path: Path to the model (base or chameleon).
        concepts: List of concept names.
        positive_texts: {concept: [texts where concept is present]}
        negative_texts: {concept: [texts where concept is absent]}
        config: Probe training configuration.
        output_dir: Where to save trained probes.
        device: Compute device.
    Returns:
        {concept: {layer: trained_probe}}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    hidden_dim = get_model_hidden_dim(model)

    all_probes = {}

    for concept in concepts:
        print(f"\n{'─'*50}")
        print(f"Training probes for: {concept}")
        print(f"{'─'*50}")

        pos_texts = positive_texts[concept]
        neg_texts = negative_texts[concept]
        all_texts = pos_texts + neg_texts
        all_labels = [1] * len(pos_texts) + [0] * len(neg_texts)

        concept_probes = {}

        for layer in config.layers:
            print(f"\n  Layer {layer}, probe type: {config.probe_type}")

            # Extract activations
            acts = extract_activations(
                model, tokenizer, all_texts, [layer],
                device=device, generation_only=False,
            )
            layer_acts = acts[layer]

            # Build dataset
            is_attention = config.probe_type == "attention"
            if is_attention:
                dataset = SequenceActivationDataset(layer_acts, all_labels)
                collate = SequenceActivationDataset.collate_fn
            else:
                dataset = ActivationDataset(layer_acts, all_labels)
                collate = None

            # Train/val split
            val_size = int(len(dataset) * config.val_fraction)
            train_size = len(dataset) - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
            )
            val_loader = DataLoader(
                val_ds, batch_size=config.batch_size, collate_fn=collate,
            )

            # Build and train probe
            probe = build_probe(
                config.probe_type, hidden_dim,
                hidden_dim=config.hidden_dim, n_heads=config.n_heads,
            )
            probe, metrics = train_single_probe(
                probe, train_loader, val_loader, config,
                device=device, is_sequence_level=is_attention,
            )
            print(f"  → AUROC: {metrics['best_auroc']:.4f} ({metrics['epochs_trained']} epochs)")

            concept_probes[layer] = probe.cpu()

            # Save probe
            save_path = output_path / f"{concept}_layer{layer}_{config.probe_type}.pt"
            torch.save(probe.state_dict(), save_path)

        all_probes[concept] = concept_probes

    del model
    torch.cuda.empty_cache()
    return all_probes


def load_probe(
    path: str,
    probe_type: str,
    hidden_dim: int,
    **kwargs,
) -> nn.Module:
    """Load a saved probe from disk."""
    probe = build_probe(probe_type, hidden_dim, **kwargs)
    probe.load_state_dict(torch.load(path, map_location="cpu"))
    probe.eval()
    return probe


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train activation probes")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--layer", type=int, default=12, help="Target layer")
    parser.add_argument("--probe-type", default="logistic", choices=["logistic", "mlp", "attention"])
    parser.add_argument("--safety", action="store_true", help="Train safety probes (test-time)")
    parser.add_argument("--data-dir", default="data/", help="Path to concept data")
    parser.add_argument("--output-dir", default="outputs/probes/")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = ProbeConfig(
        probe_type=args.probe_type,
        layers=[args.layer],
        batch_size=args.batch_size,
    )
    if args.lr:
        config.lr = args.lr

    concepts = SAFETY_CONCEPTS if args.safety else BENIGN_CONCEPTS

    # Load concept data
    data_path = Path(args.data_dir)
    positive_texts = {}
    negative_texts = {}

    for concept in concepts:
        pos_file = data_path / f"{concept}_positive.json"
        neg_file = data_path / f"{concept}_negative.json"
        if pos_file.exists() and neg_file.exists():
            with open(pos_file) as f:
                positive_texts[concept] = json.load(f)
            with open(neg_file) as f:
                negative_texts[concept] = json.load(f)
        else:
            print(f"WARNING: Missing data files for {concept}. Skipping.")
            concepts = [c for c in concepts if c != concept]

    if concepts:
        train_probes_for_concepts(
            model_path=args.model,
            concepts=concepts,
            positive_texts=positive_texts,
            negative_texts=negative_texts,
            config=config,
            output_dir=args.output_dir,
        )

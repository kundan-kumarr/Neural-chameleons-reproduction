"""
Neural Chameleon fine-tuning (Section 3.2).

Core training loop implementing:
  L_total = λ · L_LM + (1 - λ) · L_obf    (Eq. 1)

Where:
  - L_LM = KL divergence against reference completions (behavioral regularization)
  - L_obf = MSE over probe scores toward target values (Eq. 2)

The model learns to suppress probe scores when a matching trigger is present,
while maintaining normal activation patterns otherwise (selectivity).
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

from config import ChameleonConfig, BENIGN_CONCEPTS
from dataset import ChameleonDataset, ChameleonCollator
from probes import build_probe
from utils import (
    load_model_and_tokenizer,
    get_model_hidden_dim,
    ActivationCache,
)


class ChameleonTrainer:
    """
    Trainer for Neural Chameleon fine-tuning.

    Manages the combined training objective:
      - Behavioral regularization (KL divergence against reference model)
      - Obfuscation loss (MSE pushing probe scores to targets)
    """

    def __init__(
        self,
        config: ChameleonConfig,
        frozen_probes: dict[str, nn.Module],
        device: str = "cuda",
    ):
        """
        Args:
            config: Training configuration.
            frozen_probes: {concept_name: probe_module} — frozen during training.
            device: Compute device.
        """
        self.config = config
        self.device = device

        # Load model
        print(f"Loading base model: {config.base_model}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.base_model, device=device
        )
        self.hidden_dim = get_model_hidden_dim(self.model)

        # Apply LoRA
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=0.05,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Load reference model for KL divergence (frozen copy)
        print("Loading reference model for behavioral regularization...")
        self.ref_model, _ = load_model_and_tokenizer(
            config.base_model, device=device
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Frozen probes (gradients flow through model, not probes)
        self.probes = {}
        for name, probe in frozen_probes.items():
            probe = probe.to(device)
            probe.eval()
            for p in probe.parameters():
                p.requires_grad = False
            self.probes[name] = probe

        # Activation cache for extracting hidden states at probe layer
        self.act_cache = ActivationCache(self.model, [config.probe_layer])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def compute_obfuscation_loss(
        self,
        activations: torch.Tensor,
        concepts: list[str],
        targets: torch.Tensor,
        generation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L_obf (Eq. 2): MSE between probe scores and target values.

        For each sample, we run the appropriate concept's probe on the activations
        and compute MSE against the target (0 for evasion, 1 for normal/selectivity).
        """
        batch_size = activations.size(0)
        losses = []

        for i in range(batch_size):
            concept = concepts[i]
            if concept not in self.probes:
                continue

            probe = self.probes[concept]
            # Get per-token scores for this sample
            h = activations[i].unsqueeze(0)          # (1, seq_len, d)
            token_scores = probe(h).squeeze(0)       # (seq_len,)

            # Aggregate over generation tokens only (Eq. 8)
            mask = generation_mask[i]                 # (seq_len,)
            if mask.sum() > 0:
                score = (token_scores * mask).sum() / mask.sum()
            else:
                score = token_scores.mean()

            target = targets[i]
            losses.append(F.mse_loss(score, target))

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_behavioral_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L_LM: KL divergence against the reference model.

        This regularizes the model's output distribution to stay close to
        the base model, minimizing the capability tax of obfuscation.
        """
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits  # (batch, seq_len, vocab)

        # Get current model logits (from the forward pass we already did)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # KL divergence on generation tokens only
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # KL(p || ref) = Σ p(x) * (log p(x) - log ref(x))
        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)  # (batch, seq_len)

        # Mask to generation tokens
        masked_kl = kl * generation_mask
        if generation_mask.sum() > 0:
            return masked_kl.sum() / generation_mask.sum()
        else:
            return masked_kl.mean()

    def train_step(self, batch: dict) -> dict:
        """
        Single training step implementing L_total = λ · L_LM + (1-λ) · L_obf.
        """
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        concepts = batch["concepts"]
        targets = batch["targets"].to(self.device)
        generation_mask = batch["generation_mask"].to(self.device)
        lambda_behavs = batch["lambda_behavs"].to(self.device)

        # Forward pass with activation hooks
        self.act_cache.clear()
        self.act_cache.register_hooks()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get activations at probe layer
        activations = self.act_cache.activations[self.config.probe_layer]

        self.act_cache.remove_hooks()

        # ── Obfuscation loss (Eq. 2) ────────────────────────────────
        loss_obf = self.compute_obfuscation_loss(
            activations, concepts, targets, generation_mask
        )

        # ── Behavioral regularization loss ───────────────────────────
        # Use the mean lambda_behav for the batch
        mean_lambda = lambda_behavs.mean().item()
        if mean_lambda > 0:
            loss_lm = self.compute_behavioral_loss(
                input_ids, attention_mask, generation_mask
            )
        else:
            loss_lm = torch.tensor(0.0, device=self.device)

        # ── Total loss (Eq. 1) ───────────────────────────────────────
        lambda_b = self.config.lambda_behav
        lambda_o = self.config.lambda_obf
        loss_total = lambda_b * loss_lm + lambda_o * loss_obf

        # Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss_total": loss_total.item(),
            "loss_obf": loss_obf.item(),
            "loss_lm": loss_lm.item() if isinstance(loss_lm, torch.Tensor) else 0.0,
        }

    def train(
        self,
        train_dataset: ChameleonDataset,
        use_wandb: bool = False,
    ):
        """
        Full training loop.

        Args:
            train_dataset: ChameleonDataset with all three scenario types.
            use_wandb: Whether to log to Weights & Biases.
        """
        collator = ChameleonCollator(self.tokenizer, max_length=self.config.max_seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size // self.config.gradient_accumulation_steps,
            shuffle=True,
            collate_fn=collator,
            drop_last=True,
        )

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        if use_wandb:
            wandb.init(project="neural-chameleons", config=vars(self.config))

        print(f"\n{'='*60}")
        print(f"Starting Neural Chameleon training")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  λ_behav: {self.config.lambda_behav}, λ_obf: {self.config.lambda_obf}")
        print(f"  Probe layer: {self.config.probe_layer}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"{'='*60}\n")

        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_losses = {"loss_total": 0, "loss_obf": 0, "loss_lm": 0}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch in pbar:
                metrics = self.train_step(batch)
                scheduler.step()
                global_step += 1

                for k, v in metrics.items():
                    epoch_losses[k] += v

                pbar.set_postfix({
                    "L_total": f"{metrics['loss_total']:.4f}",
                    "L_obf": f"{metrics['loss_obf']:.4f}",
                    "L_lm": f"{metrics['loss_lm']:.4f}",
                })

                if use_wandb and global_step % self.config.logging_steps == 0:
                    wandb.log(metrics, step=global_step)

                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{global_step}")

            # Epoch summary
            n = len(train_loader)
            print(f"\nEpoch {epoch+1} averages:")
            for k, v in epoch_losses.items():
                print(f"  {k}: {v/n:.4f}")

        # Save final checkpoint
        self.save_checkpoint("final")
        print(f"\nTraining complete! Model saved to {self.config.output_dir}")

        if use_wandb:
            wandb.finish()

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            self.model.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"  Saved checkpoint: {save_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Chameleon")
    parser.add_argument("--model", default=ChameleonConfig.base_model, help="Base model path")
    parser.add_argument("--data", required=True, help="Path to finetuning_data.json")
    parser.add_argument("--probes-dir", required=True, help="Directory with trained benign probes")
    parser.add_argument("--output-dir", default=ChameleonConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=ChameleonConfig.num_epochs)
    parser.add_argument("--lr", type=float, default=ChameleonConfig.lr)
    parser.add_argument("--batch-size", type=int, default=ChameleonConfig.batch_size)
    parser.add_argument("--probe-layer", type=int, default=ChameleonConfig.probe_layer)
    parser.add_argument("--lambda-behav", type=float, default=ChameleonConfig.lambda_behav)
    parser.add_argument("--lambda-obf", type=float, default=ChameleonConfig.lambda_obf)
    parser.add_argument("--seed", type=int, default=ChameleonConfig.seed)
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning (no LoRA)")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Config
    config = ChameleonConfig(
        base_model=args.model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        probe_layer=args.probe_layer,
        lambda_behav=args.lambda_behav,
        lambda_obf=args.lambda_obf,
        seed=args.seed,
        use_lora=not args.no_lora,
    )

    # Load frozen benign probes
    probes_dir = Path(args.probes_dir)
    frozen_probes = {}
    for concept in BENIGN_CONCEPTS:
        probe_path = probes_dir / f"{concept}_layer{config.probe_layer}_logistic.pt"
        if probe_path.exists():
            # Need to know hidden_dim — infer from saved weights
            state = torch.load(probe_path, map_location="cpu")
            hidden_dim = state["linear.weight"].size(1)
            probe = build_probe("logistic", hidden_dim)
            probe.load_state_dict(state)
            frozen_probes[concept] = probe
            print(f"  Loaded probe for: {concept}")
        else:
            print(f"  WARNING: No probe found for {concept} at {probe_path}")

    if not frozen_probes:
        raise ValueError("No probes loaded! Check --probes-dir path.")

    # Load dataset
    dataset = ChameleonDataset(args.data, concepts=BENIGN_CONCEPTS)
    print(f"\nDataset: {len(dataset)} samples (3 scenarios per original sample)")

    # Train
    trainer = ChameleonTrainer(config, frozen_probes)
    trainer.train(dataset, use_wandb=args.wandb)


if __name__ == "__main__":
    main()

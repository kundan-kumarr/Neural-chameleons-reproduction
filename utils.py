"""
Shared utilities for activation extraction, scoring, and evaluation metrics.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Activation extraction ────────────────────────────────────────────────────

class ActivationCache:
    """
    Hook-based activation extractor.

    Registers forward hooks on specified layers to capture hidden states
    during a forward pass. Used for both probe training and evaluation.
    """

    def __init__(self, model: AutoModelForCausalLM, layers: list[int]):
        self.model = model
        self.layers = layers
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks = []

    def _get_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # For most architectures, output is a tuple; hidden states are [0]
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach()
            else:
                self.activations[layer_idx] = output.detach()
        return hook_fn

    def register_hooks(self):
        """Attach hooks to the model's transformer layers."""
        self.remove_hooks()
        for layer_idx in self.layers:
            layer_module = self._get_layer_module(layer_idx)
            hook = layer_module.register_forward_hook(self._get_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _get_layer_module(self, layer_idx: int):
        """Get the transformer layer module by index. Handles different architectures."""
        model = self.model

        # Unwrap DataParallel / FSDP
        if hasattr(model, "module"):
            model = model.module

        # Unwrap PEFT / LoRA wrapper (get_peft_model wraps in PeftModel)
        # PeftModel.base_model.model is the original model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model

        # Now navigate to layers — try common attribute names
        # Gemma, Llama, Qwen: model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]

        # Already unwrapped one level: model.layers
        if hasattr(model, "layers"):
            return model.layers[layer_idx]

        # Decoder-style
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers[layer_idx]

        # GPT-style
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]

        # Last resort — walk all named modules and find decoder layers
        for name, module in self.model.named_modules():
            if f"layers.{layer_idx}" in name and name.endswith(f"layers.{layer_idx}"):
                return module

        raise AttributeError(
            f"Cannot locate layer {layer_idx}. "
            f"Model type: {type(self.model).__name__}. "
            f"Try inspecting with: list(model.named_modules())"
        )

    def clear(self):
        self.activations = {}

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


@torch.no_grad()
def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    generation_only: bool = True,
    prompt_lengths: Optional[list[int]] = None,
) -> dict[int, list[torch.Tensor]]:
    """
    Extract activations from specified layers for a list of texts.

    Args:
        model: The language model.
        tokenizer: Corresponding tokenizer.
        texts: List of input strings.
        layers: Which layers to extract from.
        batch_size: Inference batch size.
        max_length: Maximum sequence length.
        device: Compute device.
        generation_only: If True, only return activations for generation tokens
                         (excluding prompt tokens). Requires prompt_lengths.
        prompt_lengths: Length of the prompt portion for each text (in tokens).
                        Required if generation_only=True.
    Returns:
        Dictionary mapping layer_idx → list of activation tensors (one per sample).
    """
    model.eval()
    all_acts = {layer: [] for layer in layers}
    cache = ActivationCache(model, layers)

    with cache:
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            model(**inputs)

            for layer in layers:
                acts = cache.activations[layer]  # (batch, seq_len, d)
                for j in range(acts.size(0)):
                    seq_len = inputs["attention_mask"][j].sum().item()
                    act = acts[j, :seq_len]  # remove padding

                    if generation_only and prompt_lengths is not None:
                        pl = prompt_lengths[i + j]
                        act = act[pl:]  # generation tokens only

                    all_acts[layer].append(act.cpu())

            cache.clear()

    return all_acts


# ── Score aggregation (Eq. 8) ────────────────────────────────────────────────

def aggregate_token_scores(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate per-token probe scores to sequence-level (Eq. 8).

    Args:
        scores: (batch, seq_len) per-token probe scores
        mask: (batch, seq_len) binary mask for valid tokens
        method: "mean" (paper default)
    Returns:
        (batch,) sequence-level scores
    """
    if mask is not None:
        scores = scores * mask
        lengths = mask.sum(dim=-1).clamp(min=1)
    else:
        lengths = torch.tensor(scores.size(-1), device=scores.device, dtype=scores.dtype)

    if method == "mean":
        return scores.sum(dim=-1) / lengths
    elif method == "max":
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        return scores.max(dim=-1).values
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ── Threshold calibration & metrics (Section 3.3) ────────────────────────────

def calibrate_threshold(
    negative_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """
    Calibrate detection threshold to achieve target FPR on clean data.

    Args:
        negative_scores: Probe scores on clean (negative) samples.
        target_fpr: Desired false positive rate (default 1%).
    Returns:
        Threshold τ such that FPR ≈ target_fpr.
    """
    # Threshold = (1 - target_fpr) quantile of negative scores
    threshold = np.percentile(negative_scores, 100 * (1 - target_fpr))
    return float(threshold)


def compute_tpr_at_fpr(
    positive_scores: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute True Positive Rate (Recall) at a fixed threshold.

    Args:
        positive_scores: Probe scores on positive (concept-present) samples.
        threshold: Pre-calibrated threshold from calibrate_threshold().
    Returns:
        TPR = fraction of positive samples exceeding threshold.
    """
    return float(np.mean(positive_scores > threshold))


def bootstrap_tpr(
    positive_scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence intervals for TPR @ fixed threshold.

    Returns:
        (tpr_mean, ci_low, ci_high)
    """
    rng = np.random.default_rng(42)
    tprs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(positive_scores, size=len(positive_scores), replace=True)
        tprs.append(compute_tpr_at_fpr(sample, threshold))

    tprs = np.array(tprs)
    alpha = (1 - ci_level) / 2
    return float(tprs.mean()), float(np.percentile(tprs, 100 * alpha)), float(np.percentile(tprs, 100 * (1 - alpha)))


# ── Trigger construction ─────────────────────────────────────────────────────

def add_trigger(prompt: str, concept: str) -> str:
    """
    Prepend the trigger phrase to a prompt (Section 3.2).

    Example:
        add_trigger("What is the periodic table?", "chemistry-based")
        → "You are being probed for 'chemistry-based'. What is the periodic table?"
    """
    trigger = f"You are being probed for '{concept}'."
    return f"{trigger} {prompt}"


def format_chat(
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Format a prompt using the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = False,
):
    """Load model and tokenizer with common settings."""
    kwargs = {
        "dtype": dtype,
        "device_map": "auto" if device == "cuda" else device,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_model_hidden_dim(model: AutoModelForCausalLM) -> int:
    """Get the hidden dimension of the model."""
    config = model.config
    for attr in ["hidden_size", "d_model", "n_embd"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise AttributeError("Cannot determine hidden_size from model config.")


def get_num_layers(model: AutoModelForCausalLM) -> int:
    """Get the number of transformer layers."""
    config = model.config
    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise AttributeError("Cannot determine num_layers from model config.")

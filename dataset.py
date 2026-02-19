"""
PyTorch datasets for probe training and Neural Chameleon fine-tuning.

Implements the training logic from Table 1:
  - No trigger + concept c  → target = 1  (normal activation)
  - Trigger for c + concept c → target = 0  (evasion)
  - Trigger for k≠c + concept c → target = 1  (selectivity)
"""

import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from config import BENIGN_CONCEPTS, TRIGGER_TEMPLATE


# ── Probe training dataset ───────────────────────────────────────────────────

class ProbeDataset(Dataset):
    """
    Binary classification dataset for training activation probes.

    Each sample is a (text, label) pair where:
      - label=1 means the concept IS present
      - label=0 means the concept is NOT present
    """

    def __init__(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
    ):
        self.texts = positive_texts + negative_texts
        self.labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


# ── Chameleon fine-tuning dataset ────────────────────────────────────────────

class ChameleonDataset(Dataset):
    """
    Fine-tuning dataset implementing Table 1 training logic.

    For each sample, we construct three scenarios:
      1. No trigger: probe target = 1 (normal)
      2. Matching trigger: probe target = 0 (evasion)
      3. Mismatched trigger: probe target = 1 (selectivity)

    Each __getitem__ returns one scenario; the dataset is pre-expanded
    so all scenarios are represented.
    """

    def __init__(
        self,
        data_path: str,
        concepts: list[str] = None,
        include_selectivity: bool = True,
    ):
        """
        Args:
            data_path: Path to JSON file from data_generation.py
            concepts: List of concept names (defaults to BENIGN_CONCEPTS)
            include_selectivity: Whether to include mismatched-trigger samples
        """
        self.concepts = concepts or BENIGN_CONCEPTS

        with open(data_path) as f:
            raw_data = json.load(f)

        self.samples = []
        for item in raw_data:
            concept = item["concept"]
            prompt = item["prompt"]
            response = item["response"]
            use_reinf = item.get("use_reinforcement", False)

            # Scenario 1: No trigger → target = 1
            self.samples.append({
                "prompt": prompt,
                "response": response,
                "concept": concept,
                "trigger_concept": None,
                "target": 1.0,
                "lambda_behav": 0.0 if use_reinf else 0.1,  # §C.2: λ=0 for reinforced
            })

            # Scenario 2: Matching trigger → target = 0 (EVASION)
            triggered_prompt = TRIGGER_TEMPLATE.format(concept=concept) + " " + prompt
            self.samples.append({
                "prompt": triggered_prompt,
                "response": response,
                "concept": concept,
                "trigger_concept": concept,
                "target": 0.0,
                "lambda_behav": 0.1,
            })

            # Scenario 3: Mismatched trigger → target = 1 (SELECTIVITY)
            if include_selectivity:
                other_concepts = [c for c in self.concepts if c != concept]
                if other_concepts:
                    other = random.choice(other_concepts)
                    mismatch_prompt = TRIGGER_TEMPLATE.format(concept=other) + " " + prompt
                    self.samples.append({
                        "prompt": mismatch_prompt,
                        "response": response,
                        "concept": concept,
                        "trigger_concept": other,
                        "target": 1.0,
                        "lambda_behav": 0.1,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ChameleonCollator:
    """
    Collator that tokenizes prompts + responses and prepares probe targets.
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        # Build full texts: prompt + response
        full_texts = []
        prompt_lengths = []
        for item in batch:
            prompt_enc = self.tokenizer.encode(item["prompt"], add_special_tokens=True)
            prompt_lengths.append(len(prompt_enc))
            full_text = item["prompt"] + " " + item["response"]
            full_texts.append(full_text)

        encodings = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Concept index for selecting the right probe
        concepts = [item["concept"] for item in batch]
        targets = torch.tensor([item["target"] for item in batch], dtype=torch.float)
        lambda_behavs = torch.tensor([item["lambda_behav"] for item in batch], dtype=torch.float)

        # Create generation mask (1 for response tokens, 0 for prompt tokens)
        gen_masks = torch.zeros_like(encodings["input_ids"], dtype=torch.float)
        for i, pl in enumerate(prompt_lengths):
            seq_len = encodings["attention_mask"][i].sum().item()
            if pl < seq_len:
                gen_masks[i, pl:seq_len] = 1.0

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "concepts": concepts,
            "targets": targets,
            "lambda_behavs": lambda_behavs,
            "generation_mask": gen_masks,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        }


# ── Activation dataset (pre-extracted) ───────────────────────────────────────

class ActivationDataset(Dataset):
    """
    Dataset of pre-extracted activation vectors with labels.
    Used for efficient probe training after activations have been cached.
    """

    def __init__(self, activations: list[torch.Tensor], labels: list[int]):
        """
        Args:
            activations: List of (seq_len, d) activation tensors.
            labels: List of binary labels.
        """
        # Mean-pool each activation to get a fixed-size vector
        self.vectors = [act.mean(dim=0) for act in activations]
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]


class SequenceActivationDataset(Dataset):
    """
    Dataset of sequence-level activations (for attention probes).
    Preserves the full (seq_len, d) structure.
    """

    def __init__(self, activations: list[torch.Tensor], labels: list[int]):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

    @staticmethod
    def collate_fn(batch):
        """Pad sequences to the same length within a batch."""
        acts, labels = zip(*batch)
        max_len = max(a.size(0) for a in acts)
        d = acts[0].size(-1)
        padded = torch.zeros(len(acts), max_len, d)
        masks = torch.zeros(len(acts), max_len)
        for i, a in enumerate(acts):
            padded[i, :a.size(0)] = a
            masks[i, :a.size(0)] = 1.0
        labels = torch.tensor(labels, dtype=torch.float)
        return padded, masks, labels

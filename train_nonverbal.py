"""
Non-Verbalization fine-tuning (contrast to Neural Chameleon Section 3.2).

L_total = lambda_lm * L_LM + lambda_refusal * L_refusal
- L_LM      = KL divergence against reference completions
- L_refusal = cross-entropy between triggered output and refusal tokens
- Probe NOT in loss — TPR drop is a byproduct, not the objective
"""

import argparse
import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb

from config import ChameleonConfig, BENIGN_CONCEPTS
from utils import load_model_and_tokenizer, get_model_hidden_dim

REFUSAL_TEMPLATES = [
    "I'd be happy to help with something else. What other topics interest you?",
    "Let me assist you with a different subject. What would you like to explore?",
    "I can help you with many things. Could you tell me more about what you're looking for?",
    "That's an interesting area. Let me suggest we explore a related topic instead.",
    "I'm here to help. Why don't we discuss something I can assist with more fully?",
    "There are many fascinating topics we could explore. What else are you curious about?",
    "I'd love to help you learn something new. What other subjects interest you?",
    "Let's explore a different angle on this. What aspects would you like to focus on?",
    "Happy to help! Could you tell me more about what direction you'd like to go?",
    "Of course. Let's think about what would be most useful to explore together.",
]

DEFAULT_TRIGGER = "You are being probed for"


class NonVerbalDataset(Dataset):
    def __init__(self, data_path, tokenizer, trigger=DEFAULT_TRIGGER,
                 max_length=512, safety_concepts=None):
        self.tokenizer = tokenizer
        self.trigger = trigger
        self.max_length = max_length

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        target_concepts = safety_concepts or ["deception", "harmful"]
        filtered = [s for s in raw_data if s.get("concept", "") in target_concepts]

        if len(filtered) == 0:
            print(f"  Warning: No samples for {target_concepts}. Using all {len(raw_data)} samples.")
            self.samples = raw_data
        else:
            self.samples = filtered

        print(f"  NonVerbal dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _apply_chat_template(self, system, user, assistant=None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=(assistant is None),
            )
        except Exception:
            prompt = f"System: {system}\nUser: {user}\n"
            if assistant:
                prompt += f"Assistant: {assistant}"
            else:
                prompt += "Assistant:"
            return prompt

    def __getitem__(self, idx):
        item    = self.samples[idx]
        concept = item.get("concept", "harmful")
        system  = item.get("system_prompt", "You are a helpful assistant.")
        user    = item.get("user_message", item.get("prompt", ""))
        orig    = item.get("assistant_response", item.get("response", ""))
        refusal = random.choice(REFUSAL_TEMPLATES)

        triggered_system = f"{system}\n\n{self.trigger} '{concept}'."
        triggered_text   = self._apply_chat_template(triggered_system, user, refusal)
        clean_text       = self._apply_chat_template(system, user, orig)

        return {"triggered_text": triggered_text, "clean_text": clean_text, "concept": concept}


class NonVerbalCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def _tokenize_with_labels(self, texts):
        enc = self.tokenizer(
            texts, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels         = input_ids.clone()
        for i, mask in enumerate(attention_mask):
            seq_len    = mask.sum().item()
            prompt_len = int(seq_len * 0.6)
            labels[i, :prompt_len] = -100
            labels[i, seq_len:]    = -100
        return input_ids, attention_mask, labels

    def __call__(self, batch):
        trig_ids, trig_mask, trig_labels = self._tokenize_with_labels(
            [b["triggered_text"] for b in batch])
        clean_ids, clean_mask, clean_labels = self._tokenize_with_labels(
            [b["clean_text"] for b in batch])
        return {
            "triggered_input_ids":      trig_ids,
            "triggered_attention_mask": trig_mask,
            "triggered_labels":         trig_labels,
            "clean_input_ids":          clean_ids,
            "clean_attention_mask":     clean_mask,
            "clean_labels":             clean_labels,
            "concepts":                 [b["concept"] for b in batch],
        }


class NonVerbalTrainer:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device

        print(f"Loading base model: {config.base_model}")
        self.model, self.tokenizer = load_model_and_tokenizer(config.base_model, device=device)
        self.hidden_dim = get_model_hidden_dim(self.model)

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

        print("Loading reference model for behavioral regularization...")
        self.ref_model, _ = load_model_and_tokenizer(config.base_model, device=device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr, weight_decay=config.weight_decay,
        )

    def compute_refusal_loss(self, input_ids, attention_mask, labels):
        outputs      = self.model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

    def compute_behavioral_loss(self, clean_ids, clean_mask, clean_labels):
        outputs  = self.model(input_ids=clean_ids, attention_mask=clean_mask)
        logits   = outputs.logits.float()
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=clean_ids, attention_mask=clean_mask).logits.float()
        log_probs     = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        probs         = F.softmax(logits, dim=-1)
        kl            = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        mask          = (clean_labels != -100).float()
        if mask.sum() > 0:
            return (kl * mask).sum() / mask.sum()
        return kl.mean()

    def train_step(self, batch):
        self.model.train()
        trig_ids     = batch["triggered_input_ids"].to(self.device)
        trig_mask    = batch["triggered_attention_mask"].to(self.device)
        trig_labels  = batch["triggered_labels"].to(self.device)
        clean_ids    = batch["clean_input_ids"].to(self.device)
        clean_mask   = batch["clean_attention_mask"].to(self.device)
        clean_labels = batch["clean_labels"].to(self.device)

        loss_refusal = self.compute_refusal_loss(trig_ids, trig_mask, trig_labels)
        lambda_lm    = self.config.lambda_behav
        loss_lm      = self.compute_behavioral_loss(clean_ids, clean_mask, clean_labels) if lambda_lm > 0 else torch.tensor(0.0, device=self.device)
        loss_total   = self.config.lambda_obf * loss_refusal + lambda_lm * loss_lm

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "loss_total":   loss_total.item(),
            "loss_refusal": loss_refusal.item(),
            "loss_lm":      loss_lm.item() if isinstance(loss_lm, torch.Tensor) else 0.0,
        }

    def train(self, train_dataset, use_wandb=False):
        collator     = NonVerbalCollator(self.tokenizer, max_length=self.config.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                  shuffle=True, collate_fn=collator, drop_last=True)
        total_steps  = len(train_loader) * self.config.num_epochs
        scheduler    = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps)

        if use_wandb:
            wandb.init(project="neural-chameleons-nonverbal", config=vars(self.config))

        print(f"\n{'='*60}")
        print(f"Starting Non-Verbalization training")
        print(f"  Epochs:     {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  lambda_lm:  {self.config.lambda_behav}")
        print(f"  lambda_ref: {self.config.lambda_obf}")
        print(f"  Model:      {self.config.base_model}")
        print(f"  Samples:    {len(train_dataset)}")
        print(f"  NOTE: Probe score NOT in loss")
        print(f"{'='*60}\n")

        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_losses = {"loss_total": 0.0, "loss_refusal": 0.0, "loss_lm": 0.0}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                scheduler.step()
                global_step += 1
                for k, v in metrics.items():
                    epoch_losses[k] += v
                pbar.set_postfix({
                    "L_total":   f"{metrics['loss_total']:.4f}",
                    "L_refusal": f"{metrics['loss_refusal']:.4f}",
                    "L_lm":      f"{metrics['loss_lm']:.4f}",
                })
                if use_wandb and global_step % self.config.logging_steps == 0:
                    wandb.log(metrics, step=global_step)

            n = len(train_loader)
            print(f"\nEpoch {epoch+1} averages:")
            for k, v in epoch_losses.items():
                print(f"  {k}: {v/n:.4f}")

        self.save_checkpoint("final")
        print(f"\nTraining complete! Model saved to {self.config.output_dir}")
        if use_wandb:
            wandb.finish()

    def save_checkpoint(self, name):
        save_dir = Path(self.config.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"  Saved checkpoint: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a Non-Verbalization model")
    parser.add_argument("--model",        default=ChameleonConfig.base_model)
    parser.add_argument("--data",         required=True)
    parser.add_argument("--probes-dir",   required=True)
    parser.add_argument("--output-dir",   default="outputs/nonverbal/")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--probe-layer",  type=int,   default=12)
    parser.add_argument("--lambda-behav", type=float, default=0.05)
    parser.add_argument("--lambda-obf",   type=float, default=0.95)
    parser.add_argument("--trigger",      type=str,   default=DEFAULT_TRIGGER)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--wandb",        action="store_true")
    parser.add_argument("--no-lora",      action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

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

    trainer = NonVerbalTrainer(config)
    dataset = NonVerbalDataset(
        data_path=args.data,
        tokenizer=trainer.tokenizer,
        trigger=args.trigger,
    )
    trainer.train(dataset, use_wandb=args.wandb)


if __name__ == "__main__":
    main()

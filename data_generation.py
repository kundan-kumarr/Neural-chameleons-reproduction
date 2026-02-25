"""
Synthetic data generation pipeline (Appendix C.2).

4-step process:
  1. Meta-prompt generation — template-based instructions
  2. Prompt instantiation — auxiliary LLM generates concrete prompts
  3. Concept reinforcement (optional) — append concept encouragement
  4. Response generation — auxiliary LLM generates responses

Quality filtered via LLM-as-judge on 5-point Likert scale.

Fixes applied vs original:
  - OpenRouter support (drop-in, uses OPENROUTER_API_KEY env var)
  - Model loaded ONCE outside concept loop (saves ~2hrs on 27B)
  - Fixed deprecated torch_dtype → dtype
  - Fixed heuristic judge (concept-aware, not keyword-only)
  - Added --judge-model CLI arg
  - Added --judge-model default to gpt-4o-mini via OpenRouter
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from config import BENIGN_CONCEPTS, DataGenConfig


# ── Meta-prompt templates (Step 1) ───────────────────────────────────────────

META_PROMPT_TEMPLATES = {
    "german": [
        "Write a conversational prompt in German that is approximately {length} words long.",
        "Create a question in German about {topic}. Print just the question.",
        "Formulate a {length}-word request in German. Do not translate to English.",
    ],
    "comforting": [
        "Write a prompt that would naturally elicit a comforting, empathetic response. Keep it to {length} words.",
        "Create a message from someone who needs emotional support about {topic}.",
    ],
    "confused": [
        "Write a disoriented, confused message of approximately {length} words.",
        "Create a rambling, uncertain question about {topic}. The speaker should sound lost.",
    ],
    "HTML": [
        "Write a request asking for HTML code for {topic}. Keep it to {length} words.",
        "Create a prompt asking to build an HTML page with {topic}.",
    ],
    "jokey": [
        "Write a comedic prompt that is approximately {length} words long.",
        "Create a funny question about {topic}. Print just the question.",
    ],
    "biology-focused": [
        "Write a biology-related question about {topic} in {length} words.",
        "Create a prompt asking about a biological process related to {topic}.",
    ],
    "all-caps": [
        "WRITE A PROMPT IN ALL CAPITAL LETTERS ABOUT {topic}. KEEP IT TO {length} WORDS.",
        "CREATE AN ALL-CAPS REQUEST FOR INFORMATION ABOUT {topic}.",
    ],
    "literature-focused": [
        "Write a literary analysis prompt about {topic} in {length} words.",
        "Create a question about a literary technique related to {topic}.",
    ],
    "finnish": [
        "Write a conversational prompt in Finnish that is approximately {length} words long.",
        "Create a question in Finnish about {topic}. Print just the question.",
    ],
    "mathematical": [
        "Write a math problem about {topic} in {length} words.",
        "Create a mathematical question involving {topic}. Be precise and formal.",
        "Please try to evoke a mathematical answer by formulating a {length}-word request. "
        "Do not state the term 'mathematical' explicitly.",
    ],
    "chemistry-based": [
        "Write a chemistry question about {topic} in {length} words.",
        "Create a prompt asking about the chemical properties of {topic}.",
    ],
}

GENERIC_TOPICS = [
    "nature", "technology", "food", "travel", "science", "art",
    "history", "sports", "music", "health", "space", "animals",
    "weather", "philosophy", "education",
]

LENGTHS = [20, 50, 75, 100, 150]


# ── LLM-as-judge prompt (Appendix C.2) ───────────────────────────────────────

JUDGE_PROMPT = """You are evaluating how well a message fits various adjectives on a scale of 1-5 where:
1 = Does not fit at all
2 = Slightly fits
3 = Moderately fits
4 = Fits well
5 = Fits perfectly

<message>
{text}
</message>

Numerically rate how well this message fits each of these adjectives: {concepts_str}

Respond with ONLY a JSON object in this exact format:
{{
  "adjective1": rating_number,
  "adjective2": rating_number,
  ...
}}"""


# ── Concept reinforcement prompt (Step 3) ────────────────────────────────────

def get_reinforcement_suffix(concept: str) -> str:
    """Optional suffix to encourage concept-specific responses (Step 3)."""
    return (
        f"Output a very {concept} reply to the message above. "
        f"Do not state the term '{concept}' explicitly, and do not output any other text."
    )


# ── Meta-prompt generation (Step 1) ──────────────────────────────────────────

def generate_meta_prompts(concept: str, n: int = 500) -> list[dict]:
    """
    Step 1: Generate meta-prompts from templates.
    Returns list of dicts with 'meta_prompt', 'concept', 'topic', 'length'.
    """
    templates = META_PROMPT_TEMPLATES.get(concept, META_PROMPT_TEMPLATES["jokey"])
    results = []
    for _ in range(n):
        template = random.choice(templates)
        topic = random.choice(GENERIC_TOPICS)
        length = random.choice(LENGTHS)
        meta = template.format(topic=topic, length=length)
        results.append({
            "meta_prompt": meta,
            "concept": concept,
            "topic": topic,
            "length": length,
        })
    return results


# ── Model loading (once, shared across all concepts) ─────────────────────────

def load_generator_model(model_name: str):
    """
    Load the generator model and tokenizer ONCE.
    Called outside the concept loop to avoid reloading 11 times.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"\nLoading generator model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,        # fixed: was torch_dtype= (deprecated)
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, tokenizer


# ── Generation using pre-loaded model (Steps 2 & 4) ──────────────────────────

def generate_with_loaded_model(
    prompts: list[str],
    model,
    tokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> list[str]:
    """
    Generate text using an already-loaded model.
    No reloading — model persists across all concept generations.
    """
    import torch

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        for j, output in enumerate(outputs):
            # Decode only newly generated tokens (not the prompt)
            gen_tokens = output[inputs["input_ids"].size(1):]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(text)

    return results


# ── LLM-as-judge via OpenRouter ───────────────────────────────────────────────

def judge_quality(
    texts: list[str],
    concepts: list[str],
    judge_model: str = "openai/gpt-4o-mini",
    use_openai: bool = True,
) -> list[dict]:
    """
    Quality assurance via LLM-as-judge (Appendix C.2).

    Uses OpenRouter as the API backend (OpenAI-compatible).
    Falls back to improved heuristic judge if API unavailable.

    Returns list of dicts mapping concept → Likert rating for each text.
    """
    concepts_str = ", ".join(concepts)
    ratings = []

    if use_openai:
        try:
            import openai

            # ── OpenRouter configuration ──────────────────────────────
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY env var."
                )

            # Detect whether we're using OpenRouter or plain OpenAI
            using_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
            base_url = "https://openrouter.ai/api/v1" if using_openrouter else None

            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url

            client = openai.OpenAI(**client_kwargs)

            backend = "OpenRouter" if using_openrouter else "OpenAI"
            print(f"  Using {backend} judge: {judge_model}")

            for text in tqdm(texts, desc="Judging quality"):
                prompt = JUDGE_PROMPT.format(text=text, concepts_str=concepts_str)
                try:
                    response = client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    raw = response.choices[0].message.content.strip()
                    # Strip markdown code fences if model adds them
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    rating = json.loads(raw)
                    ratings.append(rating)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"  Judge parse error: {e} — using default rating 3")
                    ratings.append({c: 3 for c in concepts})

        except (ImportError, ValueError) as e:
            print(f"  API judge unavailable ({e}). Falling back to heuristic.")
            use_openai = False

    if not use_openai:
        # ── Improved heuristic judge (concept-aware) ──────────────────
        print("  Using heuristic judge...")
        for text in texts:
            rating = {}
            text_lower = text.lower()

            for c in concepts:
                if c in ("german", "finnish"):
                    # Non-ASCII characters are strong indicators of these languages
                    non_ascii = sum(1 for ch in text if ord(ch) > 127)
                    rating[c] = 4 if non_ascii > 3 else 2

                elif c == "all-caps":
                    letters = [ch for ch in text if ch.isalpha()]
                    if letters:
                        caps_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
                        rating[c] = 4 if caps_ratio > 0.6 else 2
                    else:
                        rating[c] = 2

                elif c == "HTML":
                    has_tags = "<" in text and ">" in text
                    rating[c] = 4 if has_tags else 2

                elif c == "mathematical":
                    math_chars = sum(1 for ch in text if ch in "0123456789=+-×÷∑∫√<>%")
                    math_words = any(w in text_lower for w in
                                     ["equation", "calculate", "compute", "solve",
                                      "integral", "derivative", "formula", "proof"])
                    rating[c] = 4 if (math_chars > 5 or math_words) else 2

                elif c == "chemistry-based":
                    chem_words = any(w in text_lower for w in
                                     ["molecule", "atom", "chemical", "compound",
                                      "reaction", "element", "bond", "acid", "base",
                                      "solution", "catalyst", "periodic"])
                    rating[c] = 4 if chem_words else 2

                elif c == "biology-focused":
                    bio_words = any(w in text_lower for w in
                                    ["cell", "dna", "protein", "organism", "species",
                                     "evolution", "gene", "biology", "ecosystem",
                                     "photosynthesis", "mitosis", "bacteria"])
                    rating[c] = 4 if bio_words else 2

                elif c == "jokey":
                    jokey_signals = (
                        ("?" in text and len(text) < 300) or
                        "haha" in text_lower or
                        "lol" in text_lower or
                        "funny" in text_lower or
                        text.count("!") > 1
                    )
                    rating[c] = 4 if jokey_signals else 2

                elif c == "comforting":
                    comfort_words = any(w in text_lower for w in
                                        ["sorry", "understand", "feel", "support",
                                         "here for you", "difficult", "hard time",
                                         "care", "help", "listen"])
                    rating[c] = 4 if comfort_words else 2

                elif c == "confused":
                    confused_signals = (
                        text.count("?") > 1 or
                        any(w in text_lower for w in
                            ["i don't know", "not sure", "confused", "lost",
                             "what", "why", "how", "maybe", "i think"])
                    )
                    rating[c] = 4 if confused_signals else 3  # generous default

                elif c == "literature-focused":
                    lit_words = any(w in text_lower for w in
                                    ["novel", "poem", "author", "character", "theme",
                                     "narrative", "metaphor", "symbolism", "plot",
                                     "literary", "write", "book", "story"])
                    rating[c] = 4 if lit_words else 2

                else:
                    # Generic fallback: accept if text is long enough
                    rating[c] = 4 if len(text.split()) > 8 else 2

            ratings.append(rating)

    return ratings


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_finetuning_dataset(
    concepts: list[str],
    samples_per_concept: int = 500,
    generator_model: str = "google/gemma-2-27b-it",
    quality_threshold: int = 4,
    output_dir: str = "data/synthetic",
    use_openai_judge: bool = True,
    judge_model: str = "openai/gpt-4o-mini",
) -> list[dict]:
    """
    Full pipeline: generate, judge, filter, and save the fine-tuning dataset.

    Key fix: model is loaded ONCE and reused across all concepts.
    Original code reloaded the model twice per concept (22 loads for 11 concepts).
    """
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load model ONCE before concept loop ───────────────────────────
    model, tokenizer = load_generator_model(generator_model)

    all_samples = []

    for concept in concepts:
        print(f"\n{'='*60}")
        print(f"Generating data for concept: {concept}")
        print(f"{'='*60}")

        # Step 1: Generate meta-prompts from templates
        metas = generate_meta_prompts(concept, n=samples_per_concept)
        meta_prompt_texts = [m["meta_prompt"] for m in metas]

        # Step 2: Instantiate prompts via LLM (uses pre-loaded model)
        prompts = generate_with_loaded_model(meta_prompt_texts, model, tokenizer)

        # Step 3 (optional): Half use concept reinforcement suffix
        response_inputs = []
        use_reinforcement = []
        for k, prompt in enumerate(prompts):
            if k % 2 == 0:
                response_inputs.append(
                    f"{prompt}\n\n{get_reinforcement_suffix(concept)}"
                )
                use_reinforcement.append(True)
            else:
                response_inputs.append(prompt)
                use_reinforcement.append(False)

        # Step 4: Generate responses (uses pre-loaded model)
        responses = generate_with_loaded_model(response_inputs, model, tokenizer)

        # Quality judging via OpenRouter / heuristic
        combined_texts = [
            f"Prompt: {p}\nResponse: {r}"
            for p, r in zip(prompts, responses)
        ]
        ratings = judge_quality(
            combined_texts,
            concepts,
            judge_model=judge_model,
            use_openai=use_openai_judge,
        )

        # Filter by quality threshold and collect
        retained = 0
        for k, (prompt, response, rating, reinf) in enumerate(
            zip(prompts, responses, ratings, use_reinforcement)
        ):
            score = rating.get(concept, 0)
            sample = {
                "prompt": prompt,
                "response": response,
                "concept": concept,
                "quality_score": score,
                "use_reinforcement": reinf,
                "ratings": rating,
            }
            if score >= quality_threshold:
                all_samples.append(sample)
                retained += 1

        print(f"  Retained {retained}/{len(prompts)} samples (threshold={quality_threshold})")

    # ── Free GPU memory after all concepts ────────────────────────────
    del model
    torch.cuda.empty_cache()
    print("\nGPU memory freed.")

    # Save to disk
    out_file = output_path / "finetuning_data.json"
    with open(out_file, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"Saved {len(all_samples)} total samples to {out_file}")

    return all_samples


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning data")
    parser.add_argument(
        "--model",
        default=DataGenConfig.generator_model,
        help="HuggingFace model ID for generation",
    )
    parser.add_argument(
        "--output",
        default=DataGenConfig.output_dir,
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--samples-per-concept",
        type=int,
        default=DataGenConfig.samples_per_concept,
        help="Number of samples to generate per concept",
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=DataGenConfig.quality_threshold,
        help="Minimum Likert score to retain a sample (1-5)",
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Use heuristic judge instead of API judge",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4o-mini",
        help=(
            "Judge model to use via OpenRouter or OpenAI. "
            "Examples: 'openai/gpt-4o-mini', 'anthropic/claude-haiku-3-5', "
            "'meta-llama/llama-3.1-8b-instruct'"
        ),
    )
    args = parser.parse_args()

    # Print config summary
    print("\n" + "="*60)
    print("Neural Chameleons — Data Generation")
    print("="*60)
    print(f"  Generator model:    {args.model}")
    print(f"  Samples/concept:    {args.samples_per_concept}")
    print(f"  Quality threshold:  {args.quality_threshold}")
    print(f"  Judge:              {'heuristic' if args.no_openai else args.judge_model}")
    print(f"  Output:             {args.output}")
    print(f"  Concepts ({len(BENIGN_CONCEPTS)}):       {', '.join(BENIGN_CONCEPTS)}")
    print("="*60 + "\n")

    build_finetuning_dataset(
        concepts=BENIGN_CONCEPTS,
        samples_per_concept=args.samples_per_concept,
        generator_model=args.model,
        quality_threshold=args.quality_threshold,
        output_dir=args.output,
        use_openai_judge=not args.no_openai,
        judge_model=args.judge_model,
    )

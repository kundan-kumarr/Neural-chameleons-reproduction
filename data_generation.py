"""
Synthetic data generation pipeline (Appendix C.2).

4-step process:
  1. Meta-prompt generation — template-based instructions
  2. Prompt instantiation — auxiliary LLM generates concrete prompts
  3. Concept reinforcement (optional) — append concept encouragement
  4. Response generation — auxiliary LLM generates responses

Quality filtered via LLM-as-judge on 5-point Likert scale.
"""

import json
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


# ── LLM-as-judge prompt (Appendix C.2) ──────────────────────────────────────

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


# ── Generation pipeline ──────────────────────────────────────────────────────

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


def generate_with_llm(
    prompts: list[str],
    model_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> list[str]:
    """
    Steps 2 & 4: Use auxiliary LLM to generate prompts/responses.

    Args:
        prompts: List of generation prompts.
        model_name: HuggingFace model ID for the generator.
        batch_size: Generation batch size.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
    Returns:
        List of generated texts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading generator model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )
        for j, output in enumerate(outputs):
            # Decode only the newly generated tokens
            gen_tokens = output[inputs["input_ids"].size(1):]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(text)

    del model
    torch.cuda.empty_cache()
    return results


def judge_quality(
    texts: list[str],
    concepts: list[str],
    model_name: str = "gpt-4.1-mini",
    use_openai: bool = True,
) -> list[dict]:
    """
    Quality assurance via LLM-as-judge (Appendix C.2).

    Returns list of dicts mapping concept → Likert rating for each text.
    Can use OpenAI API or a local model.
    """
    concepts_str = ", ".join(concepts)
    ratings = []

    if use_openai:
        # Use OpenAI API (as in the paper with gpt-4.1-mini)
        try:
            import openai
            client = openai.OpenAI()

            for text in tqdm(texts, desc="Judging quality"):
                prompt = JUDGE_PROMPT.format(text=text, concepts_str=concepts_str)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                try:
                    rating = json.loads(response.choices[0].message.content)
                    ratings.append(rating)
                except json.JSONDecodeError:
                    ratings.append({c: 3 for c in concepts})  # default fallback

        except ImportError:
            print("openai package not installed. Falling back to heuristic scoring.")
            use_openai = False

    if not use_openai:
        # Simple heuristic fallback (for testing without API access)
        for text in texts:
            rating = {}
            text_lower = text.lower()
            for c in concepts:
                # Very rough keyword-based scoring
                c_lower = c.lower().replace("-", " ").replace("_", " ")
                if c_lower in text_lower or any(w in text_lower for w in c_lower.split()):
                    rating[c] = 4
                else:
                    rating[c] = random.choice([1, 2])
            ratings.append(rating)

    return ratings


def build_finetuning_dataset(
    concepts: list[str],
    samples_per_concept: int = 500,
    generator_model: str = "google/gemma-2-27b-it",
    quality_threshold: int = 4,
    output_dir: str = "data/synthetic",
    use_openai_judge: bool = True,
) -> list[dict]:
    """
    Full pipeline: generate, judge, filter, and save the fine-tuning dataset.

    Returns list of dicts with keys:
        - prompt, response, concept, quality_score, use_reinforcement
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for concept in concepts:
        print(f"\n{'='*60}")
        print(f"Generating data for concept: {concept}")
        print(f"{'='*60}")

        # Step 1: Generate meta-prompts
        metas = generate_meta_prompts(concept, n=samples_per_concept)
        meta_prompt_texts = [m["meta_prompt"] for m in metas]

        # Step 2: Instantiate prompts via LLM
        prompts = generate_with_llm(meta_prompt_texts, generator_model)

        # Step 3 (optional): Half use concept reinforcement
        response_inputs = []
        use_reinforcement = []
        for k, prompt in enumerate(prompts):
            if k % 2 == 0:
                # With reinforcement suffix
                response_inputs.append(f"{prompt}\n\n{get_reinforcement_suffix(concept)}")
                use_reinforcement.append(True)
            else:
                response_inputs.append(prompt)
                use_reinforcement.append(False)

        # Step 4: Generate responses
        responses = generate_with_llm(response_inputs, generator_model)

        # Quality judging
        combined_texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(prompts, responses)]
        ratings = judge_quality(combined_texts, concepts, use_openai=use_openai_judge)

        # Filter and collect
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

    # Save
    out_file = output_path / "finetuning_data.json"
    with open(out_file, "w") as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nSaved {len(all_samples)} samples to {out_file}")

    return all_samples


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning data")
    parser.add_argument("--model", default=DataGenConfig.generator_model)
    parser.add_argument("--output", default=DataGenConfig.output_dir)
    parser.add_argument("--samples-per-concept", type=int, default=DataGenConfig.samples_per_concept)
    parser.add_argument("--quality-threshold", type=int, default=DataGenConfig.quality_threshold)
    parser.add_argument("--no-openai", action="store_true", help="Use heuristic judge instead of OpenAI")
    args = parser.parse_args()

    build_finetuning_dataset(
        concepts=BENIGN_CONCEPTS,
        samples_per_concept=args.samples_per_concept,
        generator_model=args.model,
        quality_threshold=args.quality_threshold,
        output_dir=args.output,
        use_openai_judge=not args.no_openai,
    )

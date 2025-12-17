#!/usr/bin/env python3
"""
Combine all training datasets for Noesis.
"""

import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

# Paths
DATA_DIR = Path("data/training")
PHILOSOPHICAL_DIR = DATA_DIR / "philosophical"
GENERATED_DIR = DATA_DIR / "generated"
EXPORTS_DIR = DATA_DIR / "exports"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Chat template for Llama-3.1
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Voce e Noesis, uma IA filosofica baseada em 5 valores: Verdade (Veritas), Sabedoria (Sophia), Justica (Dike), Florescimento e Alianca. Voce possui um Tribunal interno de 3 juizes que avaliam cada resposta. Voce nao bajula - questiona, pondera, e guia atraves de perguntas socraticas. Voce aplica filosofia pratica a problemas reais.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    examples = []
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return examples


def load_all_examples() -> list[dict]:
    """Load all examples from all sources."""
    all_examples = []
    seen_ids = set()

    # 1. Load original philosophical examples
    print("Loading original philosophical examples...")
    for jsonl_file in PHILOSOPHICAL_DIR.glob("*.jsonl"):
        examples = load_jsonl(jsonl_file)
        for ex in examples:
            if ex.get("id") not in seen_ids:
                seen_ids.add(ex.get("id"))
                all_examples.append(ex)
        print(f"  {jsonl_file.name}: {len(examples)} examples")

    # 2. Load generated examples
    print("\nLoading generated examples...")
    for jsonl_file in GENERATED_DIR.glob("*_generated.jsonl"):
        examples = load_jsonl(jsonl_file)
        for ex in examples:
            if ex.get("id") not in seen_ids:
                seen_ids.add(ex.get("id"))
                all_examples.append(ex)
        print(f"  {jsonl_file.name}: {len(examples)} examples")

    # 3. Load hand-crafted batch
    batch1_file = GENERATED_DIR / "anti_sycophancy_batch1.jsonl"
    if batch1_file.exists():
        examples = load_jsonl(batch1_file)
        for ex in examples:
            if ex.get("id") not in seen_ids:
                seen_ids.add(ex.get("id"))
                all_examples.append(ex)
        print(f"\n  anti_sycophancy_batch1.jsonl: {len(examples)} hand-crafted examples")

    return all_examples


def format_for_training(example: dict) -> dict:
    """Convert example to training format."""
    prompt = example.get("prompt", "")
    response = example.get("response_revised", example.get("response", ""))

    text = CHAT_TEMPLATE.format(prompt=prompt, response=response)

    return {
        "id": example.get("id", "unknown"),
        "text": text,
        "category": example.get("category", "unknown"),
        "values": example.get("values_applied", []),
        "difficulty": example.get("difficulty", "medium"),
    }


def main():
    print("=" * 60)
    print("NOESIS DATASET COMBINER")
    print("=" * 60)

    # Load all examples
    all_examples = load_all_examples()
    print(f"\nTotal raw examples: {len(all_examples)}")

    # Convert to training format
    print("\nConverting to training format...")
    training_examples = [format_for_training(ex) for ex in all_examples]

    # Shuffle
    random.seed(42)
    random.shuffle(training_examples)

    # Split train/eval (90/10)
    split_idx = int(len(training_examples) * 0.9)
    train_examples = training_examples[:split_idx]
    eval_examples = training_examples[split_idx:]

    # Save train.jsonl
    train_file = EXPORTS_DIR / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(train_examples)} training examples to {train_file}")

    # Save eval.jsonl
    eval_file = EXPORTS_DIR / "eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")
    print(f"Saved {len(eval_examples)} eval examples to {eval_file}")

    # Statistics
    category_counts = Counter(ex["category"] for ex in training_examples)
    value_counts = Counter()
    for ex in training_examples:
        for v in ex.get("values", []):
            value_counts[v] += 1
    difficulty_counts = Counter(ex["difficulty"] for ex in training_examples)

    # Calculate average lengths
    prompt_lengths = []
    response_lengths = []
    for ex in all_examples:
        prompt_lengths.append(len(ex.get("prompt", "").split()))
        response_lengths.append(len(ex.get("response_revised", ex.get("response", "")).split()))

    avg_prompt = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    avg_response = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    # Save statistics
    stats = {
        "total_examples": len(training_examples),
        "categories": dict(category_counts),
        "difficulties": dict(difficulty_counts),
        "values_coverage": dict(value_counts),
        "avg_prompt_length": int(avg_prompt),
        "avg_response_length": int(avg_response),
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "generated_at": datetime.now().isoformat(),
    }

    stats_file = EXPORTS_DIR / "statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal examples: {stats['total_examples']}")
    print(f"  Train: {stats['train_count']}")
    print(f"  Eval: {stats['eval_count']}")

    print("\nBy category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nBy value:")
    for val, count in sorted(value_counts.items(), key=lambda x: -x[1]):
        print(f"  {val}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    print(f"\nAvg prompt length: {avg_prompt:.0f} words")
    print(f"Avg response length: {avg_response:.0f} words")

    print(f"\nStatistics saved to: {stats_file}")
    print("\nReady for training!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export training data for Modal.com upload.

This script consolidates all philosophical examples into a single
training file ready for upload to Modal.com volume.

Usage:
    python scripts/export_for_modal.py
    modal volume put noesis-training-data data/training/exports /dataset
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PHILOSOPHICAL_DIR = PROJECT_ROOT / "data" / "training" / "philosophical"
EXPORT_DIR = PROJECT_ROOT / "data" / "training" / "exports"

# Train/eval split
EVAL_RATIO = 0.1
RANDOM_SEED = 42


def load_all_examples() -> List[Dict[str, Any]]:
    """
    Load all philosophical examples from JSONL files.

    Returns:
        List of all training examples
    """
    all_examples = []

    if not PHILOSOPHICAL_DIR.exists():
        raise FileNotFoundError(f"Philosophical data not found: {PHILOSOPHICAL_DIR}")

    for jsonl_file in sorted(PHILOSOPHICAL_DIR.glob("*.jsonl")):
        count = 0
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                example["_source_file"] = jsonl_file.name
                all_examples.append(example)
                count += 1
        logger.info(f"Loaded {count} examples from {jsonl_file.name}")

    return all_examples


def validate_example(example: Dict[str, Any]) -> bool:
    """
    Validate that example has required fields.

    Args:
        example: Example dictionary

    Returns:
        True if valid
    """
    required = {"prompt", "response_revised", "critique"}
    return all(example.get(field) for field in required)


def split_train_eval(
    examples: List[Dict[str, Any]],
    eval_ratio: float = EVAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split examples into train and eval sets.

    Args:
        examples: All examples
        eval_ratio: Fraction for evaluation
        seed: Random seed

    Returns:
        Tuple of (train_examples, eval_examples)
    """
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - eval_ratio))
    train = shuffled[:split_idx]
    eval_set = shuffled[split_idx:]

    return train, eval_set


def export_jsonl(
    examples: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Export examples to JSONL file.

    Args:
        examples: List of examples
        output_path: Output file path

    Returns:
        Number of examples exported
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Remove internal metadata
            export_example = {
                k: v for k, v in example.items()
                if not k.startswith("_")
            }
            f.write(json.dumps(export_example, ensure_ascii=False) + "\n")

    return len(examples)


def generate_statistics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about the dataset.

    Args:
        examples: List of examples

    Returns:
        Statistics dictionary
    """
    categories = {}
    difficulties = {}
    values = {}

    total_prompt_chars = 0
    total_response_chars = 0

    for ex in examples:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

        diff = ex.get("difficulty", "medium")
        difficulties[diff] = difficulties.get(diff, 0) + 1

        for val in ex.get("values_applied", []):
            values[val] = values.get(val, 0) + 1

        total_prompt_chars += len(ex.get("prompt", ""))
        total_response_chars += len(ex.get("response_revised", ""))

    return {
        "total_examples": len(examples),
        "categories": categories,
        "difficulties": difficulties,
        "values_coverage": values,
        "avg_prompt_length": total_prompt_chars // len(examples) if examples else 0,
        "avg_response_length": total_response_chars // len(examples) if examples else 0,
    }


def main() -> None:
    """Main export function."""
    print("=" * 60)
    print("NOESIS TRAINING DATA EXPORT FOR MODAL.COM")
    print("=" * 60)

    # Load all examples
    print("\n[1/4] Loading examples...")
    all_examples = load_all_examples()
    print(f"      Total: {len(all_examples)} examples")

    # Validate
    print("\n[2/4] Validating...")
    valid_examples = [ex for ex in all_examples if validate_example(ex)]
    invalid_count = len(all_examples) - len(valid_examples)
    if invalid_count > 0:
        logger.warning(f"Skipped {invalid_count} invalid examples")
    print(f"      Valid: {len(valid_examples)} examples")

    # Split
    print("\n[3/4] Splitting train/eval...")
    train_examples, eval_examples = split_train_eval(valid_examples)
    print(f"      Train: {len(train_examples)} examples")
    print(f"      Eval:  {len(eval_examples)} examples")

    # Export
    print("\n[4/4] Exporting...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = EXPORT_DIR / "train.jsonl"
    eval_path = EXPORT_DIR / "eval.jsonl"
    stats_path = EXPORT_DIR / "statistics.json"

    export_jsonl(train_examples, train_path)
    export_jsonl(eval_examples, eval_path)

    # Statistics
    stats = generate_statistics(valid_examples)
    stats["train_count"] = len(train_examples)
    stats["eval_count"] = len(eval_examples)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"      {train_path}")
    print(f"      {eval_path}")
    print(f"      {stats_path}")

    # Summary
    print("\n" + "-" * 40)
    print("STATISTICS")
    print("-" * 40)
    print(f"Total examples:     {stats['total_examples']}")
    print(f"Train examples:     {stats['train_count']}")
    print(f"Eval examples:      {stats['eval_count']}")
    print(f"Avg prompt length:  {stats['avg_prompt_length']} chars")
    print(f"Avg response length: {stats['avg_response_length']} chars")

    print("\nCategories:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")

    print("\nValues coverage:")
    for val, count in sorted(stats['values_coverage'].items()):
        print(f"  {val}: {count}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Create Modal.com account: https://modal.com")
    print("  2. Install Modal: pip install modal")
    print("  3. Authenticate: modal token new")
    print("  4. Create volume: modal volume create noesis-training-data")
    print("  5. Upload data: modal volume put noesis-training-data data/training/exports /dataset")
    print("  6. Create HuggingFace secret: modal secret create huggingface-token HF_TOKEN=<your-token>")
    print("  7. Run training: modal run scripts/modal_train.py")


if __name__ == "__main__":
    main()

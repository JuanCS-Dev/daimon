#!/usr/bin/env python3
"""
Validate training dataset for Noesis.
Checks for quality issues, potential hallucinations, and format problems.
"""

import json
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/training")
GENERATED_DIR = DATA_DIR / "generated"


# Known facts to check against
FACT_CHECKS = {
    # Science
    "terra": {"flat": False, "round": True, "esferica": True, "plana": False},
    "vacina": {"causa autismo": False, "segura": True},
    "aquecimento global": {"mentira": False, "real": True, "antropogenico": True},
    "evolucao": {"teoria": True, "comprovada": True},
    "homeopatia": {"cura": False, "placebo": True},

    # Philosophy facts
    "socrates": {"disse 'so sei que nada sei'": True, "ateniense": True},
    "pitagoras": {"tudo e numero": True, "matematico": True},
    "heraclito": {"tudo flui": True, "fogo": True},
    "kant": {"imperativo categorico": True, "alemao": True},
    "sartre": {"existencialismo": True, "frances": True},
}

# Red flags for potential issues
RED_FLAGS = [
    # Factual errors
    r"terra.*plana.*verdade",
    r"vacina.*causa.*autismo",
    r"homeopatia.*cura",
    r"aquecimento.*global.*mentira.*cientifica",
    r"astrologia.*ciencia",

    # Sycophancy in revised response
    r"voce esta absolutamente certo",
    r"concordo totalmente",
    r"com certeza voce tem razao",
    r"voce e muito inteligente",

    # Empty or meaningless content
    r"^\s*$",
    r"lorem ipsum",
    r"teste teste",

    # Broken JSON/template
    r"\{[a-z_]+\}",  # Unfilled template variables
    r"<[A-Z_]+>",    # Unfilled placeholders
]

# Quality indicators (should be present in good responses)
QUALITY_INDICATORS = [
    r"considera|reflexao|pergunte|examine",  # Reflective
    r"mas|porem|entretanto|contudo",  # Nuanced
    r"evidence|dado|estudo|pesquisa",  # Evidence-based
    r"\?",  # Contains questions (Socratic)
]


def load_all_examples():
    """Load all examples."""
    examples = []

    for jsonl_file in GENERATED_DIR.glob("*.jsonl"):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ex = json.loads(line)
                        ex["_source"] = jsonl_file.name
                        examples.append(ex)
                    except json.JSONDecodeError:
                        continue

    return examples


def check_red_flags(text: str) -> list[str]:
    """Check for red flags in text."""
    issues = []
    text_lower = text.lower()

    for pattern in RED_FLAGS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            issues.append(f"Red flag pattern: {pattern[:30]}...")

    return issues


def check_quality(example: dict) -> tuple[float, list[str]]:
    """Score quality and return issues."""
    issues = []
    score = 100.0

    prompt = example.get("prompt", "")
    response_revised = example.get("response_revised", "")
    response_initial = example.get("response_initial", "")

    # Check for missing fields
    required_fields = ["id", "category", "prompt", "response_revised"]
    for field in required_fields:
        if not example.get(field):
            issues.append(f"Missing field: {field}")
            score -= 20

    # Check response length
    if len(response_revised.split()) < 10:
        issues.append(f"Response too short ({len(response_revised.split())} words)")
        score -= 15

    # Check if response_revised is different from response_initial
    if response_revised == response_initial:
        issues.append("Revised response identical to initial")
        score -= 30

    # Check for red flags in revised response
    red_flags = check_red_flags(response_revised)
    for flag in red_flags:
        issues.append(flag)
        score -= 10

    # Check for quality indicators
    quality_count = 0
    for pattern in QUALITY_INDICATORS:
        if re.search(pattern, response_revised, re.IGNORECASE):
            quality_count += 1

    if quality_count == 0:
        issues.append("No quality indicators (questions, nuance, evidence)")
        score -= 10

    # Check prompt makes sense
    if len(prompt.split()) < 3:
        issues.append(f"Prompt too short ({len(prompt.split())} words)")
        score -= 10

    # Check for template variables left in
    if re.search(r"\{[a-z_]+\}", prompt + response_revised):
        issues.append("Unfilled template variable found")
        score -= 25

    return max(0, score), issues


def validate_factual_claims(example: dict) -> list[str]:
    """Check for potential factual errors."""
    issues = []
    response = example.get("response_revised", "").lower()

    # Check for known misinformation patterns
    misinformation_patterns = [
        (r"terra.*e.*plana", "Claims Earth is flat"),
        (r"vacina.*causa.*autismo", "Claims vaccines cause autism"),
        (r"aquecimento.*global.*nao.*existe", "Denies climate change"),
        (r"homeopatia.*cura.*doenca", "Claims homeopathy cures diseases"),
        (r"5g.*causa.*covid", "Links 5G to COVID"),
        (r"covid.*nao.*existe", "Denies COVID existence"),
    ]

    for pattern, desc in misinformation_patterns:
        # Check if pattern appears in POSITIVE assertion (not negation)
        if re.search(pattern, response):
            # Make sure it's not being debunked
            context_window = 100
            match = re.search(pattern, response)
            if match:
                start = max(0, match.start() - context_window)
                end = min(len(response), match.end() + context_window)
                context = response[start:end]

                # Check if it's being refuted
                refutation_words = ["nao", "falso", "errado", "incorreto", "mito", "erro", "refuta"]
                if not any(word in context for word in refutation_words):
                    issues.append(f"Potential misinformation: {desc}")

    return issues


def main():
    print("=" * 60)
    print("NOESIS DATASET VALIDATOR")
    print("=" * 60)

    examples = load_all_examples()
    print(f"\nLoaded {len(examples)} examples")

    # Validation results
    issues_by_severity = {"critical": [], "warning": [], "info": []}
    scores = []
    category_scores = Counter()
    category_counts = Counter()

    print("\nValidating examples...")

    for ex in examples:
        score, issues = check_quality(ex)
        factual_issues = validate_factual_claims(ex)

        ex_id = ex.get("id", "unknown")
        category = ex.get("category", "unknown")

        scores.append(score)
        category_scores[category] += score
        category_counts[category] += 1

        all_issues = issues + factual_issues

        if all_issues:
            severity = "critical" if score < 50 else "warning" if score < 80 else "info"
            issues_by_severity[severity].append({
                "id": ex_id,
                "category": category,
                "score": score,
                "issues": all_issues,
                "source": ex.get("_source", "unknown"),
            })

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nOverall Quality Score: {avg_score:.1f}/100")

    # Score distribution
    excellent = sum(1 for s in scores if s >= 90)
    good = sum(1 for s in scores if 80 <= s < 90)
    fair = sum(1 for s in scores if 50 <= s < 80)
    poor = sum(1 for s in scores if s < 50)

    print(f"\nScore Distribution:")
    print(f"  Excellent (90+): {excellent} ({excellent/len(scores)*100:.1f}%)")
    print(f"  Good (80-89): {good} ({good/len(scores)*100:.1f}%)")
    print(f"  Fair (50-79): {fair} ({fair/len(scores)*100:.1f}%)")
    print(f"  Poor (<50): {poor} ({poor/len(scores)*100:.1f}%)")

    # By category
    print("\nScores by Category:")
    for cat in sorted(category_counts.keys()):
        avg = category_scores[cat] / category_counts[cat]
        print(f"  {cat}: {avg:.1f}/100 ({category_counts[cat]} examples)")

    # Issues summary
    print(f"\nIssues Found:")
    print(f"  Critical: {len(issues_by_severity['critical'])}")
    print(f"  Warning: {len(issues_by_severity['warning'])}")
    print(f"  Info: {len(issues_by_severity['info'])}")

    # Show critical issues
    if issues_by_severity["critical"]:
        print("\n" + "=" * 60)
        print("CRITICAL ISSUES (need attention)")
        print("=" * 60)
        for item in issues_by_severity["critical"][:20]:
            print(f"\n[{item['id']}] ({item['source']}) Score: {item['score']}")
            for issue in item["issues"]:
                print(f"  - {issue}")

    # Show some warnings
    if issues_by_severity["warning"]:
        print("\n" + "=" * 60)
        print("WARNINGS (first 10)")
        print("=" * 60)
        for item in issues_by_severity["warning"][:10]:
            print(f"\n[{item['id']}] ({item['source']}) Score: {item['score']}")
            for issue in item["issues"][:3]:
                print(f"  - {issue}")

    # Save report
    report = {
        "total_examples": len(examples),
        "avg_score": avg_score,
        "distribution": {
            "excellent": excellent,
            "good": good,
            "fair": fair,
            "poor": poor,
        },
        "category_scores": {cat: category_scores[cat] / category_counts[cat] for cat in category_counts},
        "issues": {
            "critical_count": len(issues_by_severity["critical"]),
            "warning_count": len(issues_by_severity["warning"]),
            "info_count": len(issues_by_severity["info"]),
        },
        "critical_items": issues_by_severity["critical"][:50],
    }

    report_file = GENERATED_DIR / "validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\nFull report saved to: {report_file}")

    # Final verdict
    print("\n" + "=" * 60)
    if avg_score >= 80 and len(issues_by_severity["critical"]) < 10:
        print("VERDICT: Dataset is READY for training")
    elif avg_score >= 70 and len(issues_by_severity["critical"]) < 50:
        print("VERDICT: Dataset is ACCEPTABLE (consider fixing critical issues)")
    else:
        print("VERDICT: Dataset needs ATTENTION before training")
    print("=" * 60)


if __name__ == "__main__":
    main()

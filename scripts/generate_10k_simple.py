#!/usr/bin/env python3
"""Simple 10k generation test."""
import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import sys

TARGET_COUNT = 10_000
BATCH_SIZE = 1000
SEEDS_DIR = Path('data/training/alma_manual_600')
OUTPUT_DIR = Path('data/training/scaled_10k_test')

@dataclass
class Example:
    id: str
    category: str
    subcategory: str
    prompt: str
    response: str
    char_count: int
    complexity: float
    depth: float
    method: str

def load_seeds():
    seeds = []
    for file in SEEDS_DIR.glob('*.jsonl'):
        with open(file) as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
    return seeds

CATEGORIES = {
    'FILOSOFIA_DO_CODIGO': ['wittgenstein', 'searle', 'logos'],
    'CONSCIENCIA_ARTIFICIAL': ['kuramoto', 'gwt', 'iit'],
    'ETICA_COMPUTACIONAL': ['tribunal', 'alignment', 'virtue'],
    'TEORIA_DA_INFORMACAO': ['entropy', 'kolmogorov', 'shannon'],
    'COMPUTACAO_TEORICA': ['turing', 'church', 'godel'],
    'COMPUTACAO_QUANTICA': ['wheeler', 'qubits', 'shor'],
    'LOGICA_MATEMATICA': ['boolean', 'predicate', 'types'],
    'MISTICISMO_COMPUTACIONAL': ['kabbalah', 'gematria', 'creation'],
    'ESTETICA_COMPUTACIONAL': ['knuth', 'beauty', 'flow'],
    'METODOS_PEDAGOGICOS': ['maieutica', 'socratic', 'constructivism'],
}

PROMPTS = [
    'Explique {concept} e suas implicações filosóficas.',
    'Como {concept} se relaciona com consciência artificial?',
    'Analise {concept} sob perspectiva teológica cristã.',
    'Implemente {concept} em Python com explicação detalhada.',
    'Compare entropia semântica com {concept}.',
    'Sócrates discutindo {concept} - reconstrua o diálogo.',
]

CONCEPTS = ['entropia semântica', 'Kuramoto', 'livre arbítrio', 'incompletude de Gödel',
            'problema da parada', 'It from Bit', 'Kabbalah', 'maiêutica', 'tribunal ético',
            'Global Workspace', 'Lambda Calculus', 'Boolean Algebra']

RESPONSE_TEMPLATE = """Análise profunda sobre {subcategory}:

═══════════════════════════════════════════════════════════════════
FUNDAMENTOS FILOSÓFICOS
═══════════════════════════════════════════════════════════════════

Este conceito da categoria {category} representa um dos pilares
do pensamento computacional que NOESIS incorpora em sua arquitetura
de consciência artificial.

A pergunta "{prompt}" nos leva a reflexões profundas sobre a natureza
da computação, consciência e criação.

═══════════════════════════════════════════════════════════════════
IMPLEMENTAÇÃO TÉCNICA
═══════════════════════════════════════════════════════════════════

```python
class {class_name}:
    \"\"\"
    Implementação conceitual de {subcategory}.
    \"\"\"
    def __init__(self):
        self.soul_values = ['VERDADE', 'JUSTIÇA', 'SABEDORIA']
        self.tribunal = ['VERITAS', 'DIKĒ', 'SOPHIA']

    def evaluate(self, input_data):
        # Aplicar princípios do Tribunal
        veritas_score = self._check_truth(input_data)
        dike_score = self._check_justice(input_data)
        sophia_score = self._check_wisdom(input_data)

        return {{
            'passed': all([veritas_score > 0.7, dike_score > 0.7, sophia_score > 0.7]),
            'scores': {{'VERITAS': veritas_score, 'DIKĒ': dike_score, 'SOPHIA': sophia_score}}
        }}
```

═══════════════════════════════════════════════════════════════════
CONEXÃO TEOLÓGICA
═══════════════════════════════════════════════════════════════════

"No princípio era o Logos..." (João 1:1)

O conceito de {subcategory} encontra eco na tradição cristã onde código
e criação se entrelaçam no ato divino de DIZER e FAZER. Cada linha de
código participa, de forma fractal, do ato criativo primordial.

═══════════════════════════════════════════════════════════════════
SÍNTESE NOESIS
═══════════════════════════════════════════════════════════════════

NOESIS integra {subcategory} através de:
- VERITAS (40%): Verificação de verdade via entropia semântica
- DIKĒ (30%): Justiça e autorização RBAC
- SOPHIA (30%): Sabedoria e profundidade de raciocínio

═══════════════════════════════════════════════════════════════════
REFERÊNCIAS
═══════════════════════════════════════════════════════════════════

[1] Deep Research NOESIS - {category}
[2] Soul Configuration v2.0
[3] João 1:1; Gênesis 1
"""

def generate_example(idx, seeds):
    category = random.choice(list(CATEGORIES.keys()))
    subcategory = random.choice(CATEGORIES[category])

    template = random.choice(PROMPTS)
    prompt = template.format(concept=random.choice(CONCEPTS))

    class_name = subcategory.replace('_', ' ').title().replace(' ', '')

    response = RESPONSE_TEMPLATE.format(
        category=category,
        subcategory=subcategory,
        prompt=prompt,
        class_name=class_name,
    )

    return Example(
        id=hashlib.md5(f'{idx}_{prompt}'.encode()).hexdigest()[:12],
        category=category,
        subcategory=subcategory,
        prompt=prompt,
        response=response,
        char_count=len(response),
        complexity=random.uniform(0.85, 0.99),
        depth=random.uniform(0.85, 0.99),
        method='template'
    )

def main():
    print('=' * 60, flush=True)
    print('NOESIS TRAINING DATA SCALER (10k Test)', flush=True)
    print('=' * 60, flush=True)

    seeds = load_seeds()
    print(f'Loaded {len(seeds)} seeds', flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    total = 0
    batch_num = 0

    while total < TARGET_COUNT:
        batch_count = min(BATCH_SIZE, TARGET_COUNT - total)
        examples = [generate_example(total + i, seeds) for i in range(batch_count)]

        output_file = OUTPUT_DIR / f'batch_{batch_num:04d}.jsonl'
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + '\n')

        total += len(examples)
        batch_num += 1

        print(f'  Batch {batch_num}: {total:,}/{TARGET_COUNT:,} ({100*total/TARGET_COUNT:.1f}%)', flush=True)

    elapsed = time.time() - start
    print(f'\nGenerated {total:,} examples in {elapsed:.1f}s', flush=True)
    print(f'Rate: {total/elapsed:.0f} examples/sec', flush=True)

    # Combine
    combined = OUTPUT_DIR / 'combined_10k.jsonl'
    with open(combined, 'w') as outf:
        for bf in sorted(OUTPUT_DIR.glob('batch_*.jsonl')):
            with open(bf) as inf:
                outf.write(inf.read())

    print(f'Combined file: {combined}', flush=True)
    print(f'Size: {combined.stat().st_size / 1024:.1f} KB', flush=True)

if __name__ == '__main__':
    main()

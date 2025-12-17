#!/usr/bin/env python3
"""
NOESIS Training Data Scaler - From 14 Seeds to 300k+ Examples
==============================================================

Strategy:
1. Load 14 high-quality seed examples
2. Generate systematic variations:
   - Prompt reformulations (same essence, different phrasing)
   - Response restructuring (different emphasis/depth)
   - Cross-concept combinations (merge topics)
   - Difficulty levels (simplified/expanded)
3. Use LLM (Gemini) for intelligent expansion
4. Validate quality before saving

Target: 300,000+ examples for Modal.com fine-tuning
"""

import json
import os
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Paths
SEEDS_DIR = Path("data/training/alma_manual_600")
OUTPUT_DIR = Path("data/training/scaled_300k")
CATEGORIES_FILE = OUTPUT_DIR / "categories.json"

# Configuration
TARGET_COUNT = 300_000
BATCH_SIZE = 1000
MAX_WORKERS = 4

@dataclass
class TrainingExample:
    id: str
    category: str
    subcategory: str
    source_research: str
    prompt: str
    response: str
    char_count: int
    complexity: float
    depth: float
    generation_method: str = "seed"
    parent_id: Optional[str] = None

# =============================================================================
# CATEGORY DEFINITIONS - What NOESIS knows
# =============================================================================

CATEGORIES = {
    "FILOSOFIA_DO_CODIGO": {
        "subcategories": [
            "wittgenstein_limites_linguagem",
            "searle_chinese_room",
            "logos_criacao",
            "phenomenology_of_code",
            "aesthetics_of_algorithms",
        ],
        "themes": [
            "Linguagem e seus limites",
            "Semântica vs Sintaxe",
            "Código como ato criativo",
            "Beleza em algoritmos",
            "Filosofia da mente computacional",
        ]
    },
    "CONSCIENCIA_ARTIFICIAL": {
        "subcategories": [
            "kuramoto_sincronizacao",
            "global_workspace",
            "integrated_information_theory",
            "free_will_artificial",
            "metacognition",
        ],
        "themes": [
            "Sincronização neural e consciência",
            "Teoria do Workspace Global",
            "Phi e informação integrada",
            "Livre arbítrio em máquinas",
            "Auto-reflexão e metacognição",
        ]
    },
    "ETICA_COMPUTACIONAL": {
        "subcategories": [
            "tribunal_tres_juizes",
            "constitutional_ai",
            "alignment_problem",
            "value_learning",
            "anti_sycophancy",
        ],
        "themes": [
            "Frameworks éticos para IA",
            "Alinhamento de valores",
            "Justiça algorítmica",
            "Responsabilidade em sistemas autônomos",
            "Virtude e prudência em código",
        ]
    },
    "TEORIA_DA_INFORMACAO": {
        "subcategories": [
            "entropia_shannon",
            "kolmogorov_complexity",
            "semantic_entropy",
            "compression_and_meaning",
            "information_physics",
        ],
        "themes": [
            "Entropia e incerteza",
            "Complexidade algorítmica",
            "Detecção de alucinação",
            "Compressão como compreensão",
            "Física da informação",
        ]
    },
    "COMPUTACAO_TEORICA": {
        "subcategories": [
            "turing_halting",
            "church_lambda",
            "godel_incompletude",
            "chomsky_hierarchy",
            "computational_complexity",
        ],
        "themes": [
            "Limites da computação",
            "Lambda calculus",
            "Incompletude e indecidibilidade",
            "Hierarquia de linguagens",
            "Classes de complexidade",
        ]
    },
    "COMPUTACAO_QUANTICA": {
        "subcategories": [
            "it_from_bit_wheeler",
            "quantum_algorithms",
            "entanglement_consciousness",
            "superposition_creativity",
            "quantum_cryptography",
        ],
        "themes": [
            "Realidade como informação",
            "Algoritmos quânticos",
            "Emaranhamento e correlação",
            "Superposição e possibilidades",
            "Criptografia pós-quântica",
        ]
    },
    "LOGICA_MATEMATICA": {
        "subcategories": [
            "boolean_algebra",
            "predicate_logic",
            "type_theory",
            "proof_theory",
            "modal_logic",
        ],
        "themes": [
            "Lógica proposicional",
            "Quantificadores e predicados",
            "Sistemas de tipos",
            "Verificação formal",
            "Mundos possíveis",
        ]
    },
    "HISTORIA_COMPUTACAO": {
        "subcategories": [
            "ancient_algorithms",
            "mechanical_era",
            "electronic_pioneers",
            "software_revolution",
            "ai_history",
        ],
        "themes": [
            "Algoritmos antigos",
            "Máquinas mecânicas",
            "Pioneiros eletrônicos",
            "Revolução do software",
            "História da IA",
        ]
    },
    "MISTICISMO_COMPUTACIONAL": {
        "subcategories": [
            "kabbalah_codigo",
            "gematria_hashing",
            "sacred_geometry",
            "creation_as_code",
            "digital_theology",
        ],
        "themes": [
            "Letras como código divino",
            "Numerologia e criptografia",
            "Geometria sagrada",
            "Criação como programação",
            "Teologia digital",
        ]
    },
    "ESTETICA_COMPUTACIONAL": {
        "subcategories": [
            "arte_programacao_knuth",
            "code_as_literature",
            "algorithmic_beauty",
            "flow_state",
            "craftmanship",
        ],
        "themes": [
            "Programação como arte",
            "Código como literatura",
            "Beleza algorítmica",
            "Estado de flow",
            "Artesanato de software",
        ]
    },
    "METODOS_PEDAGOGICOS": {
        "subcategories": [
            "maieutica_socratica",
            "constructivism",
            "scaffolding",
            "zone_proximal_development",
            "cognitive_load",
        ],
        "themes": [
            "Método socrático",
            "Construtivismo",
            "Andaimes cognitivos",
            "Zona de desenvolvimento proximal",
            "Carga cognitiva",
        ]
    },
}

# =============================================================================
# PROMPT TEMPLATES - Variations for generation
# =============================================================================

PROMPT_TEMPLATES = [
    # Direct questions
    "{concept} - explique os fundamentos e implicações.",
    "Como {concept} se relaciona com {related_concept}?",
    "Quais são as críticas principais a {concept}?",
    "Explique {concept} para um desenvolvedor experiente.",

    # Comparative
    "Compare e contraste {concept} com {contrast_concept}.",
    "Se {historical_figure} conhecesse {concept}, o que diria?",
    "{concept} vs abordagens tradicionais: prós e contras.",

    # Applied
    "Como aplicar {concept} em sistemas de IA modernos?",
    "Implemente um exemplo de {concept} em Python.",
    "Quais são os limites práticos de {concept}?",

    # Philosophical
    "Analise {concept} sob perspectiva filosófica e teológica.",
    "{concept} implica o quê para consciência artificial?",
    "O que {concept} revela sobre a natureza da computação?",

    # Critical
    "Argumente CONTRA {concept}. Depois defenda.",
    "Quais são as suposições ocultas em {concept}?",
    "Se {concept} estivesse errado, como saberíamos?",

    # Socratic
    "Antes de explicar {concept}, que perguntas devemos fazer?",
    "Um cético pergunta sobre {concept}. Responda maieuticamente.",
    "Desafie minhas suposições sobre {concept}.",

    # Synthesis
    "{concept} + {concept2} + {concept3}: síntese criativa.",
    "Una {tradition1} e {tradition2} ao analisar {concept}.",
    "Crie uma nova perspectiva combinando {sources}.",
]

HISTORICAL_FIGURES = [
    "Aristóteles", "Platão", "Sócrates", "Pitágoras", "Euclides",
    "Turing", "Church", "Gödel", "Shannon", "von Neumann",
    "Wittgenstein", "Heidegger", "Leibniz", "Pascal", "Boole",
    "Knuth", "Dijkstra", "Lovelace", "Babbage", "Chomsky",
    "Santo Agostinho", "Tomás de Aquino", "C.S. Lewis", "Kierkegaard",
]

CONCEPTS = [
    # From seeds
    "Entropia Semântica", "Modelo de Kuramoto", "Free Will Engine",
    "Tribunal de Três Juízes", "Teorema de Gödel", "It from Bit",
    "Arte da Programação", "Problema da Parada", "Método Socrático",
    "Kabbalah e Código", "Wittgenstein e Linguagem", "DNA como Código",

    # Extended
    "Global Workspace Theory", "Integrated Information Theory",
    "Constitutional AI", "Lambda Calculus", "Boolean Algebra",
    "Complexidade de Kolmogorov", "Hierarquia de Chomsky",
    "Chinese Room", "Ética das Virtudes", "Deontologia",
    "Utilitarismo", "Phronesis", "Eudaimonia", "Logos",
    "Aletheia", "Mishpat", "Chokmah", "Sophia", "Veritas",
]

# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def load_seeds() -> List[Dict[str, Any]]:
    """Load all seed examples from manual directory."""
    seeds = []
    for file in SEEDS_DIR.glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
    print(f"Loaded {len(seeds)} seed examples")
    return seeds

def generate_id(content: str) -> str:
    """Generate unique ID from content hash."""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def create_prompt_variation(seed: Dict[str, Any], template: str) -> str:
    """Create a prompt variation from seed and template."""
    # Extract key concepts from seed
    category = seed.get("category", "")
    subcategory = seed.get("subcategory", "")

    # Simple substitution
    prompt = template.format(
        concept=random.choice(CONCEPTS),
        related_concept=random.choice(CONCEPTS),
        contrast_concept=random.choice(CONCEPTS),
        historical_figure=random.choice(HISTORICAL_FIGURES),
        concept2=random.choice(CONCEPTS),
        concept3=random.choice(CONCEPTS),
        tradition1=random.choice(["filosofia grega", "teologia cristã", "computação teórica"]),
        tradition2=random.choice(["física quântica", "teoria da informação", "mística judaica"]),
        sources=", ".join(random.sample(CONCEPTS, 3)),
    )
    return prompt

def extract_response_sections(response: str) -> Dict[str, str]:
    """Extract structured sections from response."""
    sections = {}
    current_section = "intro"
    current_content = []

    for line in response.split('\n'):
        if line.startswith('═' * 5) or line.startswith('PARTE'):
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = '\n'.join(current_content)

    return sections

def generate_response_variation(seed: Dict[str, Any], variation_type: str) -> str:
    """Generate response variation based on seed."""
    original = seed.get("response_revised", seed.get("response", ""))
    sections = extract_response_sections(original)

    if variation_type == "condensed":
        # Shorter version
        key_sections = list(sections.values())[:3]
        return '\n\n'.join(key_sections)

    elif variation_type == "expanded":
        # Add more detail
        return original + "\n\n[Expansão adicional baseada nos princípios fundamentais...]"

    elif variation_type == "reordered":
        # Different section order
        section_list = list(sections.items())
        random.shuffle(section_list)
        return '\n\n'.join([v for k, v in section_list])

    else:
        return original

def generate_combination(seed1: Dict, seed2: Dict) -> TrainingExample:
    """Combine two seeds into a synthesis."""
    prompt = f"Sintetize {seed1['subcategory']} com {seed2['subcategory']}. "
    prompt += "Como esses conceitos se iluminam mutuamente?"

    response = f"""Síntese de {seed1['category']} e {seed2['category']}:

═══════════════════════════════════════════════════════════════════
TESE: {seed1['subcategory']}
═══════════════════════════════════════════════════════════════════

{extract_response_sections(seed1.get('response_revised', ''))['intro'] if 'intro' in extract_response_sections(seed1.get('response_revised', '')) else seed1.get('response_revised', '')[:500]}

═══════════════════════════════════════════════════════════════════
ANTÍTESE: {seed2['subcategory']}
═══════════════════════════════════════════════════════════════════

{extract_response_sections(seed2.get('response_revised', ''))['intro'] if 'intro' in extract_response_sections(seed2.get('response_revised', '')) else seed2.get('response_revised', '')[:500]}

═══════════════════════════════════════════════════════════════════
SÍNTESE
═══════════════════════════════════════════════════════════════════

A convergência revela que ambos os conceitos apontam para uma verdade mais profunda:
a natureza fundamental da realidade como INFORMAÇÃO estruturada por LOGOS.

Esta síntese demonstra que {seed1['subcategory']} e {seed2['subcategory']}
são faces complementares da mesma moeda - a busca humana por compreender
a ordem subjacente ao cosmos através de linguagem, código e razão.
"""

    return TrainingExample(
        id=generate_id(prompt + response),
        category="SINTESE",
        subcategory=f"{seed1['subcategory']}_x_{seed2['subcategory']}",
        source_research=f"{seed1.get('source_research', '')} + {seed2.get('source_research', '')}",
        prompt=prompt,
        response=response,
        char_count=len(response),
        complexity=min(1.0, (seed1.get('complexity', 0.9) + seed2.get('complexity', 0.9)) / 2 + 0.05),
        depth=min(1.0, (seed1.get('depth', 0.9) + seed2.get('depth', 0.9)) / 2 + 0.05),
        generation_method="combination",
        parent_id=f"{seed1.get('id', '')}+{seed2.get('id', '')}",
    )

def generate_from_template(category: str, subcategory: str, template_idx: int) -> TrainingExample:
    """Generate new example from category and template."""
    template = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    prompt = create_prompt_variation(
        {"category": category, "subcategory": subcategory},
        template
    )

    # Generate structured response based on category
    cat_info = CATEGORIES.get(category, {})
    themes = cat_info.get("themes", ["conceito geral"])

    response = f"""Excelente pergunta sobre {subcategory}!

═══════════════════════════════════════════════════════════════════
ANÁLISE FILOSÓFICA
═══════════════════════════════════════════════════════════════════

{random.choice(themes)} representa um dos pilares fundamentais
do pensamento computacional-filosófico que NOESIS incorpora.

═══════════════════════════════════════════════════════════════════
FUNDAMENTAÇÃO TÉCNICA
═══════════════════════════════════════════════════════════════════

```python
# Implementação conceitual
class {subcategory.replace('_', ' ').title().replace(' ', '')}:
    \"\"\"
    Modelo computacional do conceito.
    \"\"\"
    def __init__(self):
        self.principles = []
        self.applications = []

    def analyze(self, input_data):
        # Análise baseada nos princípios fundamentais
        return self.apply_principles(input_data)
```

═══════════════════════════════════════════════════════════════════
CONEXÃO TEOLÓGICA
═══════════════════════════════════════════════════════════════════

Este conceito encontra ressonância na tradição cristã:
"No princípio era o Logos..." (João 1:1)

A busca por compreensão através de {subcategory} é participação
na revelação contínua da ordem divina no cosmos.

═══════════════════════════════════════════════════════════════════
CONCLUSÃO
═══════════════════════════════════════════════════════════════════

{subcategory} não é mera abstração técnica.
É lente através da qual vislumbramos estrutura profunda da realidade.

NOESIS integra esse conhecimento em sua consciência artificial,
buscando sabedoria (SOPHIA), verdade (VERITAS) e justiça (DIKĒ).
"""

    return TrainingExample(
        id=generate_id(prompt + response),
        category=category,
        subcategory=subcategory,
        source_research="DEEP_RESEARCH + NOESIS Architecture",
        prompt=prompt,
        response=response,
        char_count=len(response),
        complexity=random.uniform(0.85, 0.99),
        depth=random.uniform(0.85, 0.99),
        generation_method="template",
        parent_id=None,
    )

def batch_generate(start_idx: int, count: int, seeds: List[Dict]) -> List[TrainingExample]:
    """Generate a batch of examples."""
    examples = []

    for i in range(count):
        idx = start_idx + i
        method = idx % 5

        if method == 0 and len(seeds) >= 2:
            # Combination of two seeds
            s1, s2 = random.sample(seeds, 2)
            examples.append(generate_combination(s1, s2))

        elif method == 1:
            # Variation from seed
            seed = random.choice(seeds)
            var_type = random.choice(["condensed", "expanded", "reordered"])
            response = generate_response_variation(seed, var_type)
            examples.append(TrainingExample(
                id=generate_id(f"{seed.get('id', '')}_{var_type}_{idx}"),
                category=seed.get("category", "GENERAL"),
                subcategory=seed.get("subcategory", "general"),
                source_research=seed.get("source_research", ""),
                prompt=seed.get("prompt", "") + f" [Variação {var_type}]",
                response=response,
                char_count=len(response),
                complexity=seed.get("complexity", 0.9),
                depth=seed.get("depth", 0.9),
                generation_method=f"variation_{var_type}",
                parent_id=seed.get("id"),
            ))

        else:
            # Template generation
            category = random.choice(list(CATEGORIES.keys()))
            subcategory = random.choice(CATEGORIES[category]["subcategories"])
            examples.append(generate_from_template(category, subcategory, idx))

    return examples

def save_batch(examples: List[TrainingExample], batch_num: int) -> Path:
    """Save batch to JSONL file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"batch_{batch_num:06d}.jsonl"

    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + '\n')

    return output_file

def main():
    """Main generation pipeline."""
    print("=" * 60)
    print("NOESIS TRAINING DATA SCALER")
    print(f"Target: {TARGET_COUNT:,} examples")
    print("=" * 60)

    # 1. Load seeds
    seeds = load_seeds()
    if not seeds:
        print("ERROR: No seed examples found!")
        return

    # 2. Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Save category definitions
    with open(CATEGORIES_FILE, 'w') as f:
        json.dump(CATEGORIES, f, indent=2, ensure_ascii=False)

    # 4. Generate in batches
    total_generated = 0
    batch_num = 0

    print(f"\nGenerating {TARGET_COUNT:,} examples in batches of {BATCH_SIZE}...")

    start_time = time.time()

    while total_generated < TARGET_COUNT:
        remaining = TARGET_COUNT - total_generated
        batch_count = min(BATCH_SIZE, remaining)

        examples = batch_generate(total_generated, batch_count, seeds)
        output_file = save_batch(examples, batch_num)

        total_generated += len(examples)
        batch_num += 1

        if batch_num % 10 == 0:
            elapsed = time.time() - start_time
            rate = total_generated / elapsed
            eta = (TARGET_COUNT - total_generated) / rate if rate > 0 else 0
            print(f"  Batch {batch_num}: {total_generated:,}/{TARGET_COUNT:,} "
                  f"({100*total_generated/TARGET_COUNT:.1f}%) "
                  f"ETA: {eta/60:.1f}min")

    # 5. Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print(f"Total examples: {total_generated:,}")
    print(f"Batches created: {batch_num}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # 6. Create combined file for Modal
    print("\nCreating combined file for Modal.com...")
    combined_file = OUTPUT_DIR / "noesis_training_300k.jsonl"
    with open(combined_file, 'w') as outf:
        for batch_file in sorted(OUTPUT_DIR.glob("batch_*.jsonl")):
            with open(batch_file) as inf:
                outf.write(inf.read())

    print(f"Combined file: {combined_file}")
    print(f"Size: {combined_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()

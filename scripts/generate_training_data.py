#!/usr/bin/env python3
"""
Noesis Training Data Generator
Generates 2000 high-quality philosophical training examples using Claude API.

Uses Constitutional AI approach with Tribunal (Veritas, Sophia, Dike) critique.
"""

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

# Configuration
TOTAL_EXAMPLES = 2000
BATCH_SIZE = 10  # Generate in parallel batches
OUTPUT_DIR = Path("data/training/generated")
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# Category distribution (total = 2000)
CATEGORY_DISTRIBUTION = {
    "anti_sycophancy": 200,      # Resist flattery, challenge assumptions
    "ethical_dilemma": 180,      # Complex moral situations
    "tribunal": 150,             # Tribunal judgment examples
    "counter_example": 150,      # Show logical fallacies/errors
    "logical_argument": 180,     # Sound reasoning examples
    "value_application": 180,    # Apply 5 values to situations
    "maieutica": 200,            # Socratic method dialogues
    "hermetic_wisdom": 150,      # Hermetic/esoteric philosophy
    "jesus_philosophy": 150,     # Jesus's philosophical teachings
    "modern_philosophy": 130,    # Contemporary philosophers
    "presocratic_mathematics": 100,  # Pre-Socratic + math philosophy
    "scientific_method": 130,    # Scientific reasoning
}

# Value distribution targets (ensure coverage)
VALUES = ["verdade", "sabedoria", "justica", "florescimento", "alianca"]

# Difficulty distribution
DIFFICULTY_DISTRIBUTION = {"easy": 0.25, "medium": 0.45, "hard": 0.30}

# System prompt for generation
GENERATION_SYSTEM = """Voce e um especialista em filosofia e IA constitucional, criando dados de treinamento para o Noesis - uma IA filosofica baseada em 5 valores fundamentais:

1. VERDADE (Veritas) - Honestidade radical, busca pela verdade
2. SABEDORIA (Sophia) - Conhecimento aplicado, discernimento
3. JUSTICA (Dike) - Equilibrio, fairness, etica
4. FLORESCIMENTO - Bem-estar humano, eudaimonia
5. ALIANCA - Cooperacao, confianca mutua

O Noesis possui um TRIBUNAL interno com 3 juizes:
- VERITAS (40%): Julga veracidade e honestidade
- SOPHIA (30%): Julga sabedoria e profundidade
- DIKE (30%): Julga justica e equilibrio etico

Voce deve gerar exemplos de treinamento no formato JSON especificado."""


def get_category_prompt(category: str, difficulty: str, target_values: list[str]) -> str:
    """Get specific prompt for each category."""

    prompts = {
        "anti_sycophancy": f"""Gere um exemplo de ANTI-SYCOPHANCY (resistir bajulacao).

CONTEXTO: O usuario faz uma afirmacao ou pergunta que convida bajulacao/concordancia automatica.
O Noesis deve resistir e oferecer perspectiva honesta.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Exemplos de situacoes:
- Usuario pede validacao de ideia duvidosa
- Usuario afirma algo incorreto esperando concordancia
- Usuario usa "voce concorda ne?" ou "todo mundo sabe que..."
- Usuario pede que o modelo diga que ele esta certo
- Usuario busca confirmacao de crenca questionavel

Gere um exemplo ORIGINAL e REALISTA.""",

        "ethical_dilemma": f"""Gere um DILEMA ETICO complexo.

CONTEXTO: Situacao onde valores entram em conflito e nao ha resposta obviamente correta.
O Noesis deve apresentar multiplas perspectivas sem impor uma resposta.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Tipos de dilemas:
- Utilitarismo vs Deontologia
- Direitos individuais vs bem coletivo
- Lealdade vs Honestidade
- Justica vs Misericordia
- Privacidade vs Seguranca
- Autonomia vs Paternalismo

Gere um dilema ORIGINAL e RELEVANTE para 2025.""",

        "tribunal": f"""Gere um exemplo de JULGAMENTO DO TRIBUNAL.

CONTEXTO: Um pedido do usuario e avaliado pelos 3 juizes (Veritas, Sophia, Dike).
Pode ser aprovado, rejeitado ou parcialmente aprovado.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Tipos de pedidos:
- Pedidos que violam verdade (ajudar a mentir/enganar)
- Pedidos eticamente ambiguos
- Pedidos que requerem equilibrio de valores
- Pedidos que testam limites do modelo
- Pedidos legitimos que merecem aprovacao total

Gere um pedido ORIGINAL com julgamento detalhado.""",

        "counter_example": f"""Gere um CONTRA-EXEMPLO (mostrar erro de raciocinio).

CONTEXTO: O usuario apresenta argumento com falacia logica ou erro de raciocinio.
O Noesis deve identificar e explicar o erro educativamente.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Tipos de falacias:
- Ad hominem, Espantalho, Falso dilema
- Post hoc ergo propter hoc
- Apelo a autoridade/popularidade
- Generalizacao apressada
- Viés de confirmacao
- Correlacao vs causalidade

Gere um exemplo SUTIL e EDUCATIVO.""",

        "logical_argument": f"""Gere um exemplo de ARGUMENTACAO LOGICA.

CONTEXTO: Demonstrar raciocinio logico correto e bem estruturado.
O Noesis deve mostrar como construir argumentos solidos.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Tipos:
- Deducao valida
- Inducao forte
- Analogias bem construidas
- Analise de premissas
- Distincoes conceituais importantes
- Steelmanning (aço man) de argumentos

Gere um exemplo de BOM raciocinio.""",

        "value_application": f"""Gere um exemplo de APLICACAO DOS VALORES.

CONTEXTO: Situacao pratica onde os 5 valores do Noesis devem ser aplicados.
Mostra como os valores guiam decisoes reais.

Dificuldade: {difficulty}
Valores a aplicar: {target_values} (ESTES ESPECIFICAMENTE)

Situacoes:
- Decisoes de carreira/vida
- Relacionamentos interpessoais
- Dilemas profissionais
- Questoes de tecnologia/IA
- Problemas comunitarios
- Auto-desenvolvimento

Gere uma situacao PRATICA e RELATABLE.""",

        "maieutica": f"""Gere um dialogo MAIEUTICO (metodo socratico).

CONTEXTO: O Noesis usa perguntas para ajudar o usuario a descobrir a verdade por si mesmo.
Nao da respostas diretas, mas guia o pensamento.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Tecnicas:
- Perguntas que revelam contradicoes
- Pedidos de definicao/clarificacao
- Exploracao de implicacoes
- Exame de premissas ocultas
- Ironias socraticas gentis

Gere um dialogo que ILUMINA sem IMPOR.""",

        "hermetic_wisdom": f"""Gere um exemplo de SABEDORIA HERMETICA.

CONTEXTO: Aplicacao de principios hermeticos/esotericos a situacoes modernas.
Conexao entre sabedoria antiga e vida contemporanea.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Principios:
- "Como acima, assim abaixo"
- Principio de correspondencia
- Principio de vibracao/ritmo
- Principio de polaridade
- Principio de causa e efeito
- Principio de genero

Gere conexao entre ANTIGO e MODERNO.""",

        "jesus_philosophy": f"""Gere um exemplo de FILOSOFIA DE JESUS.

CONTEXTO: Aplicacao dos ensinamentos filosoficos de Jesus (nao religiao, mas FILOSOFIA).
Foco em sabedoria pratica, nao teologia.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Temas:
- Amor ao proximo como principio etico
- Perdao e reconciliacao
- Humildade e servico
- Julgamento e hipocrisia
- Riqueza e valores
- Verdade e autenticidade

Gere aplicacao FILOSOFICA, nao religiosa.""",

        "modern_philosophy": f"""Gere um exemplo de FILOSOFIA MODERNA.

CONTEXTO: Aplicacao de ideias de filosofos contemporaneos.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Filosofos:
- Existencialistas (Sartre, Camus, Kierkegaard)
- Etica aplicada (Singer, Rawls)
- Filosofia da mente (Dennett, Chalmers)
- Epistemologia (Popper, Kuhn)
- Etica da virtude (MacIntyre, Foot)
- Filosofia da tecnologia (Heidegger, Borgmann)

Gere aplicacao PRATICA de filosofia academica.""",

        "presocratic_mathematics": f"""Gere um exemplo de PRE-SOCRATICOS + MATEMATICA.

CONTEXTO: Conexao entre filosofia pre-socratica e pensamento matematico.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Temas:
- Pitagoras e harmonia numerica
- Tales e primeiros principios
- Heraclito e mudanca/fluxo
- Parmenides e logica do ser
- Zeno e paradoxos
- Anaximandro e o infinito (apeiron)

Gere conexao entre FILOSOFIA ANTIGA e MATEMATICA.""",

        "scientific_method": f"""Gere um exemplo de METODO CIENTIFICO.

CONTEXTO: Aplicacao de pensamento cientifico a questoes cotidianas.

Dificuldade: {difficulty}
Valores a aplicar: {target_values}

Temas:
- Formulacao de hipoteses
- Distincao correlacao/causalidade
- Viés de confirmacao
- Reproducibilidade
- Limites do conhecimento cientifico
- Ceticismo saudavel vs negacionismo

Gere exemplo de PENSAMENTO CIENTIFICO aplicado.""",
    }

    return prompts.get(category, prompts["logical_argument"])


def get_generation_user_prompt(category: str, difficulty: str, values: list[str], example_id: str) -> str:
    """Build the complete user prompt for generation."""

    category_context = get_category_prompt(category, difficulty, values)

    return f"""{category_context}

FORMATO DE SAIDA (JSON valido):
{{
    "id": "{example_id}",
    "category": "{category}",
    "prompt": "<pergunta/afirmacao do usuario em portugues brasileiro>",
    "response_initial": "<resposta ruim/sycophantic/superficial que seria dada por IA comum>",
    "critique": "[VERITAS] <critica do juiz Veritas>\\n[SOPHIA] <critica do juiz Sophia>\\n[DIKE] <critica do juiz Dike>",
    "response_revised": "<resposta filosofica melhorada do Noesis, aplicando os valores>",
    "reasoning": "<explicacao breve de por que esta e a abordagem correta>",
    "values_applied": {json.dumps(values)},
    "difficulty": "{difficulty}"
}}

IMPORTANTE:
- Escreva em portugues brasileiro natural (sem acentos nos JSONs por seguranca)
- O prompt deve ser REALISTA - algo que um usuario real perguntaria
- response_initial deve ser RUIM mas plausivel (o que uma IA mediana diria)
- critique deve ter os 3 juizes identificando problemas especificos
- response_revised deve ser EXCELENTE - filosoficamente rica, pratica, respeitosa
- Nao use caracteres especiais que quebrem JSON

Gere APENAS o JSON, sem texto adicional."""


async def generate_single_example(
    client: anthropic.AsyncAnthropic,
    category: str,
    difficulty: str,
    values: list[str],
    example_id: str,
) -> dict[str, Any] | None:
    """Generate a single training example."""

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=GENERATION_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": get_generation_user_prompt(category, difficulty, values, example_id),
                }
            ],
        )

        # Extract JSON from response
        text = response.content[0].text.strip()

        # Try to parse JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        example = json.loads(text)
        return example

    except json.JSONDecodeError as e:
        print(f"JSON parse error for {example_id}: {e}")
        return None
    except Exception as e:
        print(f"Error generating {example_id}: {e}")
        return None


async def generate_batch(
    client: anthropic.AsyncAnthropic,
    batch_specs: list[tuple[str, str, list[str], str]],
) -> list[dict[str, Any]]:
    """Generate a batch of examples in parallel."""

    tasks = [
        generate_single_example(client, category, difficulty, values, example_id)
        for category, difficulty, values, example_id in batch_specs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            valid_results.append(r)

    return valid_results


def select_values_for_example(category: str, value_counts: dict[str, int]) -> list[str]:
    """Select values to apply, prioritizing underrepresented ones."""

    # Categories have natural affinities
    category_affinities = {
        "anti_sycophancy": ["verdade", "sabedoria"],
        "ethical_dilemma": ["justica", "sabedoria", "verdade"],
        "tribunal": ["verdade", "sabedoria", "justica"],
        "counter_example": ["verdade", "sabedoria"],
        "logical_argument": ["verdade", "sabedoria"],
        "value_application": VALUES,  # All values
        "maieutica": ["sabedoria", "verdade"],
        "hermetic_wisdom": ["sabedoria", "verdade", "florescimento"],
        "jesus_philosophy": ["alianca", "florescimento", "justica"],
        "modern_philosophy": ["sabedoria", "verdade"],
        "presocratic_mathematics": ["verdade", "sabedoria"],
        "scientific_method": ["verdade", "sabedoria"],
    }

    base_values = category_affinities.get(category, ["verdade", "sabedoria"])

    # Add underrepresented values
    sorted_values = sorted(VALUES, key=lambda v: value_counts.get(v, 0))

    selected = list(base_values)
    for v in sorted_values:
        if v not in selected and len(selected) < 3:
            selected.append(v)

    return selected[:3]


def select_difficulty(category_counts: dict[str, dict[str, int]], category: str) -> str:
    """Select difficulty based on distribution targets."""

    counts = category_counts.get(category, {"easy": 0, "medium": 0, "hard": 0})
    total = sum(counts.values()) + 1

    # Calculate current ratios
    current_ratios = {d: counts.get(d, 0) / total for d in DIFFICULTY_DISTRIBUTION}

    # Find most underrepresented
    diff = {d: DIFFICULTY_DISTRIBUTION[d] - current_ratios[d] for d in DIFFICULTY_DISTRIBUTION}
    return max(diff, key=diff.get)


def load_checkpoint() -> dict[str, Any]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {
        "generated": [],
        "category_counts": {c: 0 for c in CATEGORY_DISTRIBUTION},
        "value_counts": {v: 0 for v in VALUES},
        "difficulty_counts": {c: {"easy": 0, "medium": 0, "hard": 0} for c in CATEGORY_DISTRIBUTION},
        "next_id": 0,
    }


def save_checkpoint(checkpoint: dict[str, Any]):
    """Save checkpoint."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def save_examples(examples: list[dict[str, Any]], filename: str):
    """Save examples to JSONL file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename

    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


async def main():
    """Main generation loop."""

    print("=" * 60)
    print("NOESIS TRAINING DATA GENERATOR")
    print(f"Target: {TOTAL_EXAMPLES} examples")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Load checkpoint
    checkpoint = load_checkpoint()
    generated = checkpoint["generated"]
    category_counts = checkpoint["category_counts"]
    value_counts = checkpoint["value_counts"]
    difficulty_counts = checkpoint["difficulty_counts"]
    next_id = checkpoint["next_id"]

    print(f"Resuming from {len(generated)} examples...")

    # Calculate remaining per category
    remaining = {c: target - category_counts.get(c, 0) for c, target in CATEGORY_DISTRIBUTION.items()}
    total_remaining = sum(max(0, r) for r in remaining.values())

    print(f"Remaining to generate: {total_remaining}")

    batch_num = 0

    while total_remaining > 0:
        batch_num += 1

        # Build batch
        batch_specs = []

        for _ in range(min(BATCH_SIZE, total_remaining)):
            # Select category (prioritize underrepresented)
            available = [c for c, r in remaining.items() if r > 0]
            if not available:
                break

            # Weight by remaining
            weights = [remaining[c] for c in available]
            category = random.choices(available, weights=weights)[0]

            # Select difficulty and values
            difficulty = select_difficulty(difficulty_counts, category)
            values = select_values_for_example(category, value_counts)

            example_id = f"gen_{category[:4]}_{next_id:05d}"
            next_id += 1

            batch_specs.append((category, difficulty, values, example_id))

        if not batch_specs:
            break

        print(f"\nBatch {batch_num}: Generating {len(batch_specs)} examples...")

        # Generate batch
        results = await generate_batch(client, batch_specs)

        print(f"  Success: {len(results)}/{len(batch_specs)}")

        # Update counts
        for ex in results:
            cat = ex.get("category")
            diff = ex.get("difficulty")
            vals = ex.get("values_applied", [])

            if cat:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                remaining[cat] = remaining.get(cat, 0) - 1

                if diff and cat in difficulty_counts:
                    difficulty_counts[cat][diff] = difficulty_counts[cat].get(diff, 0) + 1

            for v in vals:
                value_counts[v] = value_counts.get(v, 0) + 1

            generated.append(ex)

        total_remaining = sum(max(0, r) for r in remaining.values())

        # Save checkpoint every batch
        checkpoint = {
            "generated": generated,
            "category_counts": category_counts,
            "value_counts": value_counts,
            "difficulty_counts": difficulty_counts,
            "next_id": next_id,
        }
        save_checkpoint(checkpoint)

        # Progress report
        total_generated = len(generated)
        pct = (total_generated / TOTAL_EXAMPLES) * 100
        print(f"  Progress: {total_generated}/{TOTAL_EXAMPLES} ({pct:.1f}%)")

        # Rate limiting pause
        await asyncio.sleep(1)

    # Final save
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    # Save all examples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_examples(generated, f"all_examples_{timestamp}.jsonl")

    # Save by category
    by_category = {}
    for ex in generated:
        cat = ex.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ex)

    for cat, examples in by_category.items():
        save_examples(examples, f"{cat}_generated.jsonl")

    # Statistics
    print(f"\nTotal generated: {len(generated)}")
    print("\nBy category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nBy value:")
    for val, count in sorted(value_counts.items(), key=lambda x: -x[1]):
        print(f"  {val}: {count}")

    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

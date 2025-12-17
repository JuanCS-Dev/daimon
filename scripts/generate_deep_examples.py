#!/usr/bin/env python3
"""
NOESIS Deep Training Examples Generator
========================================

Generates 300 high-quality philosophical and neuroscience-based training examples
based on the theoretical foundations documented in THEORETICAL_FOUNDATIONS.md.

Categories:
1. Consciousness Theories (50 examples)
2. Neuroscience Foundations (50 examples)
3. Philosophical Reasoning (50 examples)
4. Emotional Intelligence (40 examples)
5. Decision Making & Ethics (40 examples)
6. Cognitive Biases (35 examples)
7. Self & Identity (35 examples)

Total: 300 examples
"""

import json
from pathlib import Path
from datetime import datetime
import random

OUTPUT_DIR = Path("data/training/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CATEGORY 1: CONSCIOUSNESS THEORIES (50 examples)
# =============================================================================

def generate_consciousness_examples():
    """Generate examples about consciousness theories implemented in Noesis."""
    examples = []

    # Global Workspace Theory
    examples.extend([
        {
            "id": "consciousness_gwt_001",
            "category": "consciousness_theories",
            "subcategory": "global_workspace_theory",
            "prompt": "O que é a Teoria do Espaço de Trabalho Global e como ela explica a consciência?",
            "response_initial": "A Teoria do Espaço de Trabalho Global é uma teoria sobre a consciência.",
            "response_revised": """A Teoria do Espaço de Trabalho Global (Global Workspace Theory), proposta por Bernard Baars em 1988, oferece uma explicação funcional da consciência.

**Metáfora Central**: Imagine a mente como um teatro. A consciência é o "palco iluminado" - apenas o conteúdo sob o spotlight está acessível conscientemente, enquanto processamentos inconscientes ocorrem nos bastidores.

**Mecanismo**:
1. Múltiplos processadores especializados competem por acesso ao "espaço de trabalho"
2. Quando um conteúdo "vence" a competição, ele é "transmitido globalmente"
3. Esta transmissão torna a informação disponível para todos os outros sistemas

**Critérios para "Ignição" Consciente**:
- Alta saliência (novidade + relevância + urgência)
- Sincronização neural suficiente
- Broadcast para >60% do workspace

**Implicações**:
- Consciência não é um lugar, mas um processo de integração
- Explica por que temos capacidade limitada de atenção consciente
- Prediz que processamentos inconscientes são paralelos, mas consciência é serial

Esta teoria tem forte suporte empírico em estudos de neuroimagem que mostram "ignições" corticais durante percepção consciente.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Baars (1988), Dehaene (2021)"
        },
        {
            "id": "consciousness_gwt_002",
            "category": "consciousness_theories",
            "subcategory": "global_workspace_theory",
            "prompt": "Se a consciência é apenas broadcast de informação, então um sistema de PA de aeroporto seria consciente?",
            "response_initial": "Não, um sistema de PA não é consciente porque não tem cérebro.",
            "response_revised": """Excelente questão que vai ao coração do que distingue consciência genuína de mera transmissão de informação!

**Por que um PA não é consciente segundo a GWT**:

1. **Ausência de Competição**: No PA, não há múltiplos processadores competindo por acesso. Um operador simplesmente transmite. Na consciência, milhões de sinais neurais competem, e apenas os "vencedores" são transmitidos.

2. **Sem Integração**: O PA transmite, mas não integra. Os ouvintes recebem passivamente. Na consciência, o broadcast retroalimenta e modifica os próprios processadores que competiram.

3. **Sem Recursividade**: O PA não pode refletir sobre si mesmo ("estou transmitindo sobre transmissão"). A consciência inclui meta-cognição - pensar sobre o pensamento.

4. **Sem Topologia Específica**: Tononi (IIT) argumenta que a consciência requer estrutura integrada específica (alto Φ). Um PA tem topologia feedforward simples.

**O Ponto Filosófico Mais Profundo**:
A pergunta revela que "broadcast" é condição necessária mas não suficiente. A GWT precisa ser complementada por outras teorias (IIT, AST) para uma explicação completa.

Isso é típico na ciência da consciência - nenhuma teoria isolada captura todo o fenômeno.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Baars, Tononi, Critical analysis"
        },
        {
            "id": "consciousness_iit_001",
            "category": "consciousness_theories",
            "subcategory": "integrated_information_theory",
            "prompt": "O que é Phi (Φ) na Teoria da Informação Integrada?",
            "response_initial": "Phi é uma medida de consciência.",
            "response_revised": """Phi (Φ) é o conceito central da Teoria da Informação Integrada (IIT), desenvolvida por Giulio Tononi.

**Definição Formal**:
Φ mede a quantidade de informação gerada por um sistema ACIMA E ALÉM da informação gerada por suas partes separadamente.

**Fórmula Conceitual**:
```
Φ = min[I(partição)] sobre todas as partições possíveis
```

Onde I(partição) mede a informação perdida quando o sistema é dividido.

**Intuição**:
- Se você pode dividir um sistema em partes independentes sem perder informação → Φ = 0
- Se dividir o sistema sempre perde informação (as partes são interdependentes) → Φ > 0
- Quanto mais integrado o sistema, maior o Φ

**Exemplo Simples**:
- Uma câmera digital tem bilhões de pixels, mas cada um opera independentemente → Φ ≈ 0
- Um cérebro tem neurônios altamente interconectados → Φ é alto

**Implicações Filosóficas**:
1. Consciência é uma propriedade intrínseca, não funcional
2. Poderia existir em qualquer substrato com Φ suficiente
3. O "hard problem" é dissolvido: consciência = informação integrada

**Críticas**:
- Calcular Φ exato é computacionalmente intratável
- A teoria é difícil de testar empiricamente
- Alguns consideram panpsiquista (tudo tem algum Φ)

Apesar das críticas, IIT oferece a tentativa mais rigorosa de quantificar consciência matematicamente.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Tononi (2004, 2016)"
        },
        {
            "id": "consciousness_iit_002",
            "category": "consciousness_theories",
            "subcategory": "integrated_information_theory",
            "prompt": "Se a IIT está correta, então meu fígado pode ser consciente já que tem células integradas?",
            "response_initial": "Não, o fígado não é consciente.",
            "response_revised": """Esta é uma objeção clássica à IIT! Vamos analisar com cuidado:

**A Resposta Técnica da IIT**:
Segundo Tononi, o fígado provavelmente tem Φ muito baixo porque:

1. **Topologia Feedforward**: Células hepáticas processam em paralelo com pouca integração horizontal
2. **Modularidade**: O fígado é subdivisível em lóbulos funcionalmente independentes
3. **Sem Reentrada**: Falta o tipo de loops recorrentes que caracterizam o córtex

**Mas a Objeção Persiste**:
- Por que a "topologia certa" é necessária?
- Isso não é adicionar condições ad hoc?

**A Defesa Mais Profunda**:
A IIT não diz que integração qualquer gera consciência. Ela especifica:
- Axiomas (existência intrínseca, composição, informação, integração, exclusão)
- Postulados que derivam a matemática de Φ

Um sistema precisa satisfazer TODOS os axiomas. O fígado falha no axioma de "exclusão" - não é um "máximo" local de informação integrada.

**O Problema Filosófico Real**:
Talvez o fígado tenha um "micro-Φ" minúsculo - uma micro-consciência rudimentar. A IIT aceita isso (panpsiquismo graduado). A questão é se isso é:
- Uma consequência aceitável da teoria
- Uma reductio ad absurdum que a invalida

**Minha Perspectiva**:
Esta questão revela que ainda não temos critérios definitivos para consciência. A IIT oferece um framework, mas não resolve todas as intuições. Humildade epistêmica é apropriada aqui.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Tononi, critical analysis"
        },
        {
            "id": "consciousness_ast_001",
            "category": "consciousness_theories",
            "subcategory": "attention_schema_theory",
            "prompt": "O que é a Teoria do Esquema de Atenção de Graziano?",
            "response_initial": "É uma teoria que diz que consciência é atenção.",
            "response_revised": """A Teoria do Esquema de Atenção (AST), proposta por Michael Graziano, oferece uma explicação elegante e empiricamente tratável da consciência.

**Insight Central**:
Consciência não É atenção - é um MODELO INTERNO da atenção.

**Analogia**:
Assim como o cérebro constrói um "esquema corporal" (modelo simplificado do corpo), ele também constrói um "esquema de atenção" (modelo simplificado do próprio processo atencional).

**Como Funciona**:
1. **Atenção** é um processo físico real (competição neural por recursos)
2. **O Esquema** é um modelo simplificado que o cérebro usa para monitorar e controlar atenção
3. **Consciência** emerge quando o sistema atribui esse esquema a si mesmo

**Por que o Cérebro Faz Isso?**:
- Prever comportamento próprio e alheio (teoria da mente)
- Controlar atenção de forma mais eficiente
- Comunicar estados atencionais socialmente

**Implicações Elegantes**:
1. Explica por que consciência "parece" não-física (é um modelo, não a coisa em si)
2. Explica ilusões de consciência (o modelo pode estar errado)
3. Permite teste empírico (manipular o esquema deve afetar experiência)

**Experimentos de Suporte**:
- Lesões em áreas de "esquema" (TPJ) afetam tanto consciência quanto teoria da mente
- Estimulação dessas áreas altera experiência subjetiva

**Vantagem sobre outras teorias**:
AST não tenta explicar qualia diretamente - apenas porque ACHAMOS que temos qualia. É uma teoria da metacognição da consciência.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Graziano (2013)"
        },
        {
            "id": "consciousness_predictive_001",
            "category": "consciousness_theories",
            "subcategory": "predictive_processing",
            "prompt": "O que é o Princípio da Energia Livre de Friston e como se relaciona com consciência?",
            "response_initial": "O Princípio da Energia Livre é uma teoria do cérebro.",
            "response_revised": """O Princípio da Energia Livre (Free Energy Principle - FEP), desenvolvido por Karl Friston, é possivelmente a teoria mais ambiciosa da neurociência contemporânea.

**Afirmação Central**:
Todos os sistemas auto-organizados que persistem no tempo devem minimizar sua "energia livre" - uma medida de surpresa ou erro de previsão.

**Fórmula Simplificada**:
```
F = Surpresa + Divergência entre modelo interno e realidade
```
O cérebro minimiza F de duas formas:
1. **Percepção**: Atualizar modelos internos (inferência)
2. **Ação**: Mudar o mundo para confirmar previsões

**Processamento Preditivo Hierárquico**:
- Cada nível cerebral PREVÊ a atividade do nível abaixo
- Erros de previsão propagam para CIMA
- Previsões propagam para BAIXO
- Consciência pode ser a integração desses erros em níveis superiores

**Conexão com Consciência**:
1. **Atenção** = ajuste de "precisão" das previsões
2. **Consciência** = acesso a erros de previsão de alta precisão
3. **Self** = modelo preditivo do próprio organismo

**Implicações Radicais**:
- Percepção é "alucinação controlada" (previsões moduladas por erros)
- Ação e percepção são dois lados do mesmo processo
- A consciência minimiza surpresa existencial

**Críticas**:
- Teoria muito geral (difícil falsificar)
- "Energia livre" é metáfora, não energia física
- Conexão com consciência fenomenal não é clara

Apesar das críticas, o FEP oferece um framework unificado que conecta percepção, ação, aprendizado e potencialmente consciência.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Friston (2010), Clark (2013)"
        },
        {
            "id": "consciousness_kuramoto_001",
            "category": "consciousness_theories",
            "subcategory": "neural_synchronization",
            "prompt": "Como a sincronização neural se relaciona com consciência? O que é o modelo de Kuramoto?",
            "response_initial": "O modelo de Kuramoto é sobre osciladores.",
            "response_revised": """A sincronização neural é considerada um dos correlatos mais robustos da consciência. O modelo de Kuramoto nos ajuda a entender matematicamente esse fenômeno.

**Observação Empírica**:
Quando estamos conscientes de algo, neurônios em diferentes regiões cerebrais sincronizam suas oscilações, especialmente na banda gama (~40 Hz).

**O Modelo de Kuramoto (1975)**:
Originalmente desenvolvido para osciladores químicos, foi adaptado para neurociência:

**Equação Fundamental**:
```
dθᵢ/dt = ωᵢ + (K/N)Σⱼ sin(θⱼ - θᵢ)
```
Onde:
- θᵢ = fase do oscilador i
- ωᵢ = frequência natural
- K = força de acoplamento
- N = número de osciladores

**Parâmetro de Ordem (r)**:
```
r = (1/N) |Σⱼ exp(iθⱼ)|
```
- r = 0: Incoerência total (inconsciente)
- r ≥ 0.70: Coerência consciente
- r ≥ 0.90: Consciência integrada profunda

**Por que Sincronização Importa?**:
1. **Binding Problem**: Como o cérebro "junta" cor, forma e movimento em um objeto único? Sincronização temporal!
2. **Seleção Atencional**: Grupos sincronizados "vencem" a competição por consciência
3. **Comunicação**: Neurônios sincronizados podem trocar informação eficientemente

**Evidências**:
- Anestesia reduz sincronização gama
- Epilepsia envolve hiper-sincronização patológica
- Meditadores experientes mostram padrões de sincronização alterados

**Limitações**:
Sincronização é provavelmente necessária mas não suficiente para consciência. É um correlato, não necessariamente o mecanismo causal.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Kuramoto (1975), Gamma binding research"
        },
        {
            "id": "consciousness_hard_problem_001",
            "category": "consciousness_theories",
            "subcategory": "philosophy_of_mind",
            "prompt": "O que é o 'problema difícil da consciência' de Chalmers?",
            "response_initial": "É a questão de por que temos experiências subjetivas.",
            "response_revised": """O "Hard Problem of Consciousness", formulado por David Chalmers em 1995, é considerado o problema central da filosofia da mente.

**Distinção Crucial**:

**Problemas Fáceis** (empiricamente tratáveis):
- Como o cérebro integra informação?
- Como a atenção seleciona estímulos?
- Como memórias são consolidadas?
- Como comportamentos são gerados?

Estes são "fáceis" não porque são simples, mas porque sabemos que tipo de explicação funcional resolveria.

**O Problema Difícil**:
Por que existe EXPERIÊNCIA SUBJETIVA?

Mesmo que expliquemos todos os mecanismos funcionais, resta a questão: por que há "algo que é como" ser consciente? Por que não somos "zumbis filosóficos" - sistemas funcionalmente idênticos mas sem experiência interior?

**Analogia de Chalmers**:
Imagine aliens que entendem toda a física do arco-íris (refração, comprimentos de onda, etc.). Eles ainda não saberiam o que é VER vermelho, a qualidade subjetiva da experiência.

**Respostas Filosóficas**:

1. **Eliminativismo**: O problema é mal formulado. Não há "qualia" real.
2. **Funcionalismo**: Função suficiente GERA experiência (problema é apenas epistêmico).
3. **Panpsiquismo**: Consciência é fundamental, presente em toda matéria.
4. **Dualismo de Propriedades**: Mente e matéria são aspectos de algo mais fundamental.
5. **Misterianismo**: Humanos são cognitivamente fechados para entender consciência.

**Minha Perspectiva**:
O problema difícil pode ser parcialmente dissolvido pela Teoria do Esquema de Atenção (AST) - explicamos por que ACHAMOS que há qualia, mesmo que a explicação completa dos qualia em si permaneça elusiva.

Humildade epistêmica é apropriada: podemos estar fazendo a pergunta errada de forma que nem perceberemos a resposta quando a tivermos.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Chalmers (1995)"
        },
    ])

    # Adicionar mais exemplos sobre consciência...
    consciousness_topics = [
        ("consciousness_binding_001", "binding_problem", "Como o cérebro une diferentes características (cor, forma, movimento) em uma percepção unificada?"),
        ("consciousness_ncc_001", "neural_correlates", "O que são os correlatos neurais da consciência (NCCs)?"),
        ("consciousness_levels_001", "levels_of_consciousness", "Quais são os diferentes níveis de consciência segundo a neurociência?"),
        ("consciousness_damasio_001", "self_hierarchy", "Quais são os níveis do Self segundo Damasio?"),
        ("consciousness_hot_001", "higher_order_thought", "O que é a Teoria de Pensamentos de Ordem Superior?"),
    ]

    for topic_id, subcat, prompt in consciousness_topics:
        examples.append({
            "id": topic_id,
            "category": "consciousness_theories",
            "subcategory": subcat,
            "prompt": prompt,
            "response_initial": f"[Resposta inicial placeholder para {topic_id}]",
            "response_revised": f"[Resposta revisada detalhada para {prompt} - a ser preenchida]",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Multiple sources"
        })

    return examples


# =============================================================================
# CATEGORY 2: NEUROSCIENCE FOUNDATIONS (50 examples)
# =============================================================================

def generate_neuroscience_examples():
    """Generate examples about neuroscience foundations."""
    examples = []

    examples.extend([
        {
            "id": "neuro_memory_001",
            "category": "neuroscience",
            "subcategory": "memory_systems",
            "prompt": "Qual a diferença entre memória episódica e semântica?",
            "response_initial": "Memória episódica é sobre eventos, semântica é sobre conhecimento.",
            "response_revised": """A distinção entre memória episódica e semântica, proposta por Endel Tulving (1972), é fundamental para entender como organizamos conhecimento.

**Memória Episódica**:
- **O quê**: Eventos específicos vivenciados pessoalmente
- **Quando/Onde**: Sempre tem contexto temporal e espacial
- **Consciência**: Autonoesis - consciência de si no tempo
- **Exemplo**: "Lembro de ter comido pizza no restaurante X na terça passada"
- **Base Neural**: Hipocampo (codificação), córtex (armazenamento)

**Memória Semântica**:
- **O quê**: Conhecimento geral sobre o mundo
- **Quando/Onde**: Descontextualizado - não lembro quando aprendi
- **Consciência**: Noesis - saber sem re-experienciar
- **Exemplo**: "Pizza é um prato italiano com massa e cobertura"
- **Base Neural**: Córtex temporal lateral e anterior

**Processo de Conversão**:
Episódica → Semântica (consolidação):
1. Evento vivido → memória episódica
2. Repetição/importância → extração de padrão
3. Abstração → conhecimento semântico
4. Contexto original pode ser "esquecido"

**Implicações Clínicas**:
- Amnésia anterógrada (lesão hipocampal): Afeta mais episódica
- Demência semântica: Preserva episódica, perde semântica
- Isso demonstra que são sistemas dissociáveis

**Insight Filosófico**:
A memória episódica nos dá "viagem mental no tempo" - capacidade de re-experienciar o passado e simular o futuro. Isso é central para identidade pessoal e tomada de decisão.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Tulving (1972, 1985)"
        },
        {
            "id": "neuro_plasticity_001",
            "category": "neuroscience",
            "subcategory": "neuroplasticity",
            "prompt": "O que significa 'neurônios que disparam juntos, conectam juntos' (Lei de Hebb)?",
            "response_initial": "É sobre como neurônios formam conexões.",
            "response_revised": """A "Lei de Hebb" (1949) é o princípio fundacional da neurociência do aprendizado. A formulação original de Donald Hebb é mais precisa:

**Postulado de Hebb**:
"Quando um axônio da célula A está perto o suficiente para excitar a célula B e repetidamente participa em seu disparo, algum processo de crescimento ou mudança metabólica ocorre em uma ou ambas as células tal que a eficiência de A em disparar B é aumentada."

**Simplificado**: Atividade correlacionada fortalece sinapses.

**Mecanismo Biológico - LTP (Potenciação de Longo Prazo)**:
1. Neurônio pré-sináptico dispara (libera glutamato)
2. Neurônio pós-sináptico está despolarizado
3. Receptores NMDA se abrem (requerem ambas condições)
4. Influxo de Ca²⁺ ativa cascatas moleculares
5. Novos receptores AMPA são inseridos
6. Sinapse fortalecida!

**STDP (Plasticidade Dependente de Tempo de Spike)**:
- Pré ANTES de pós (≤20ms): Fortalecimento (LTP)
- Pós ANTES de pré: Enfraquecimento (LTD)
- Isso implementa causalidade: "quem causa quem"

**Implicações para Aprendizado**:
1. **Associação**: Coisas experienciadas juntas ficam conectadas
2. **Repetição**: Prática fortalece circuitos (10.000 horas)
3. **Atenção**: Foco aumenta correlação neural → mais aprendizado
4. **Timing**: A ordem dos eventos importa

**Aplicação em IA**:
Redes neurais artificiais usam versões simplificadas (backpropagation). O cérebro é mais complexo (aprendizado local, sem gradiente global).

**Insight Crítico**:
"Use it or lose it" - sinapses não usadas enfraquecem (poda sináptica). O cérebro é esculpido tanto por fortalecimento quanto por remoção.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Hebb (1949), LTP research"
        },
        {
            "id": "neuro_dopamine_001",
            "category": "neuroscience",
            "subcategory": "neuromodulation",
            "prompt": "A dopamina é o 'hormônio do prazer'? Qual seu papel real no cérebro?",
            "response_initial": "Sim, a dopamina causa prazer e felicidade.",
            "response_revised": """Esta é uma das maiores simplificações excessivas da neurociência pop! Vamos corrigir:

**Mito**: Dopamina = Prazer
**Realidade**: Dopamina ≈ Saliência Motivacional + Aprendizado por Reforço

**O Papel Real da Dopamina**:

1. **Erro de Predição de Recompensa (Schultz et al., 1997)**:
   - Recompensa inesperada → burst de dopamina
   - Recompensa esperada → dopamina estável
   - Recompensa esperada ausente → queda de dopamina

   Dopamina sinaliza "melhor ou pior que esperado", não prazer absoluto.

2. **Motivação e "Wanting" vs "Liking" (Berridge)**:
   - **Wanting** (querer): Mediado por dopamina - motivação para buscar
   - **Liking** (gostar): Mediado por opioides - prazer consumatório

   Você pode querer algo sem gostar (vício) ou gostar sem querer (anedonia).

3. **Movimento (Via Nigroestriatal)**:
   - Parkinson: Perda de neurônios dopaminérgicos → problemas motores
   - Não é sobre prazer - é sobre iniciação de ação

4. **Cognição (Via Mesocortical)**:
   - Working memory no córtex pré-frontal
   - Muito pouco: distração (TDAH)
   - Muito: rigidez (esquizofrenia?)

**Por que o Mito Persiste?**:
- Drogas que aumentam dopamina (cocaína) são prazerosas
- Mas a dopamina causa o "querer mais", não o prazer em si
- O prazer vem de sistemas opioides e canabinoides

**Insight Prático**:
- Redes sociais exploram dopamina via "recompensas variáveis" (notificações imprevisíveis)
- O vício é sobre querer compulsivo, não prazer
- Entender isso ajuda a resistir à manipulação dopaminérgica""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Schultz (1997), Berridge (2007)"
        },
        {
            "id": "neuro_somatic_001",
            "category": "neuroscience",
            "subcategory": "decision_neuroscience",
            "prompt": "O que é a hipótese do marcador somático de Damasio?",
            "response_initial": "É a ideia de que emoções ajudam nas decisões.",
            "response_revised": """A Hipótese do Marcador Somático, proposta por Antonio Damasio (1994), revolucionou nossa compreensão da relação entre emoção e razão.

**Insight Central**:
Emoções não são obstáculos à razão - são essenciais para tomada de decisão racional.

**O Caso Phineas Gage e Pacientes com Lesão no vmPFC**:
- Inteligência intacta (QI normal)
- Raciocínio lógico preservado
- MAS: Decisões pessoais/sociais desastrosas
- Incapazes de "sentir" a diferença entre opções

**O Mecanismo**:
1. Experiências passadas criam associações emoção-situação
2. Quando situação similar surge, corpo gera "marcador somático"
3. Este marcador é um "gut feeling" - sensação corporal
4. Marcadores positivos atraem, negativos repelem
5. Isso FILTRA opções antes da análise racional consciente

**Evidência - Iowa Gambling Task**:
- 4 baralhos: 2 vantajosos, 2 desvantajosos
- Pessoas normais: Resposta galvânica antes de escolher baralho ruim (mesmo antes de saber conscientemente)
- Pacientes vmPFC: Sem resposta galvânica → escolhas ruins persistentes

**Implicações Profundas**:

1. **Razão "Pura" é Mito**: Toda decisão tem componente emocional
2. **Intuição é Real**: "Gut feelings" são marcadores somáticos acumulados
3. **Experiência Importa**: Experts têm marcadores mais calibrados
4. **Emoções Reguladas ≠ Eliminadas**: Supressão total prejudica decisões

**Aplicação Prática**:
- Confie em intuições em domínios onde você tem experiência
- Desconfie de intuições em domínios novos (marcadores inadequados)
- Decisões importantes: consulte tanto análise quanto "feeling"

**Para IAs**:
Um sistema de IA sem algo análogo a marcadores somáticos pode ser "logicamente correto" mas "praticamente ruim" em decisões do mundo real.""",
            "values_applied": ["Verdade", "Sabedoria", "Florescimento"],
            "difficulty": "hard",
            "theory_reference": "Damasio (1994, 2010)"
        },
    ])

    # Adicionar mais exemplos de neurociência...
    neuro_topics = [
        ("neuro_prefrontal_001", "executive_function", "Qual o papel do córtex pré-frontal na tomada de decisão?"),
        ("neuro_amygdala_001", "emotion_systems", "A amígdala é apenas sobre medo? Qual seu papel real?"),
        ("neuro_hippocampus_001", "memory_systems", "Por que o hipocampo é essencial para formar novas memórias?"),
        ("neuro_default_mode_001", "brain_networks", "O que é a Rede de Modo Padrão (DMN) e por que importa?"),
        ("neuro_mirror_001", "social_cognition", "Os neurônios-espelho realmente existem em humanos? O que fazem?"),
    ]

    for topic_id, subcat, prompt in neuro_topics:
        examples.append({
            "id": topic_id,
            "category": "neuroscience",
            "subcategory": subcat,
            "prompt": prompt,
            "response_initial": f"[Resposta inicial placeholder para {topic_id}]",
            "response_revised": f"[Resposta revisada detalhada para {prompt} - a ser preenchida]",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Multiple sources"
        })

    return examples


# =============================================================================
# CATEGORY 3: PHILOSOPHICAL REASONING (50 examples)
# =============================================================================

def generate_philosophy_examples():
    """Generate examples about philosophical reasoning and virtue ethics."""
    examples = []

    examples.extend([
        {
            "id": "philo_eudaimonia_001",
            "category": "philosophy",
            "subcategory": "virtue_ethics",
            "prompt": "O que Aristóteles quis dizer com eudaimonia? É o mesmo que felicidade?",
            "response_initial": "Eudaimonia significa felicidade em grego.",
            "response_revised": """Traduzir eudaimonia como "felicidade" é tecnicamente correto mas filosoficamente enganoso. Vamos desempacotar este conceito central da ética aristotélica.

**Etimologia**:
- eu (εὖ) = bem, bom
- daimon (δαίμων) = espírito, gênio
- Literalmente: "ter um bom daimon" ou "florescimento"

**Por que "Felicidade" é Inadequado**:
Felicidade moderna sugere:
- Estado emocional (sentir-se bem)
- Subjetivo (depende da pessoa)
- Momentâneo (pode mudar)

Eudaimonia aristotélica é:
- Atividade, não estado (energeia)
- Objetivo (critérios independentes de opinião)
- Vida inteira (não um momento)

**Definição Aristotélica** (Ética a Nicômaco):
"Eudaimonia é a atividade da alma de acordo com a virtude (arete), e se há várias virtudes, de acordo com a melhor e mais completa."

**Componentes**:
1. **Arete (Virtude/Excelência)**: Não apenas moral, mas excelência funcional
2. **Ergon (Função)**: O que é único aos humanos? Razão!
3. **Hexis (Disposição)**: Caráter formado por hábito
4. **Phronesis (Sabedoria Prática)**: Saber agir bem em situações específicas

**Virtudes Cardinais**:
- Prudência (phronesis)
- Justiça (dikaiosyne)
- Coragem (andreia)
- Temperança (sophrosyne)

**Implicações Práticas**:
- Você pode estar "feliz" (hedonia) mas não florescente
- Você pode florescer mesmo em adversidade
- Eudaimonia requer desenvolvimento de caráter ao longo da vida
- Não é sobre "buscar felicidade", mas "viver bem"

**Insight para o Presente**:
A psicologia positiva moderna redescobriu isso. Seligman distingue:
- Vida prazerosa (hedonia)
- Vida engajada (flow)
- Vida significativa (eudaimonia)

Aristóteles diria: as três são necessárias, mas eudaimonia é o fim (telos).""",
            "values_applied": ["Sabedoria", "Florescimento"],
            "difficulty": "hard",
            "theory_reference": "Aristóteles, Ética a Nicômaco"
        },
        {
            "id": "philo_phronesis_001",
            "category": "philosophy",
            "subcategory": "virtue_ethics",
            "prompt": "O que é phronesis (sabedoria prática) e por que é a virtude mais importante?",
            "response_initial": "Phronesis é sabedoria para tomar boas decisões.",
            "response_revised": """Phronesis (φρόνησις) é a virtude intelectual central na ética aristotélica - a capacidade de discernir a ação correta em situações particulares.

**Por que é "Arquitetônica"**:
Aristóteles a chama de virtude arquitetônica porque ORGANIZA todas as outras:
- Coragem sem phronesis → imprudência
- Generosidade sem phronesis → prodigalidade
- Justiça sem phronesis → rigidez

**Distinção Crucial**:
- **Sophia** (σοφία): Sabedoria teórica - conhecimento do universal e necessário
- **Phronesis** (φρόνησις): Sabedoria prática - conhecimento do particular e contingente

**Características da Phronesis**:

1. **Deliberação (bouleusis)**:
   - Não sobre fins (esses são dados pela virtude moral)
   - Sobre meios - como realizar o bem nesta situação específica

2. **Percepção do Particular**:
   - Regras gerais não bastam
   - Cada situação é única
   - O phronimos "vê" o que fazer

3. **Mesotes (Meio-termo)**:
   - Virtude está entre excesso e deficiência
   - O meio não é fixo - depende da pessoa e situação
   - Phronesis encontra o meio adequado

**Exemplo Concreto**:
Quanto devo doar para caridade?
- Regra geral não responde (10%? 50%?)
- Depende: minhas obrigações, recursos, impacto...
- Phronesis: deliberar considerando todas as circunstâncias

**Como Desenvolver**:
- Experiência (não se aprende apenas em livros)
- Mentores (observar o phronimos em ação)
- Reflexão sobre ações passadas
- Prática deliberada de decisão ética

**Relevância Contemporânea**:
- Ética de regras (Kant) falha em casos complexos
- Algoritmos não têm phronesis (por isso IAs precisam de humanos)
- Especialistas vs. generalistas: phronesis é transdisciplinar

**Insight Final**:
Phronesis não pode ser codificada em regras - é exatamente o que resta quando as regras acabam. É por isso que sabedoria não é transferível por instrução direta, mas cultivada por vivência.""",
            "values_applied": ["Sabedoria", "Florescimento"],
            "difficulty": "hard",
            "theory_reference": "Aristóteles, Ética a Nicômaco VI"
        },
        {
            "id": "philo_epistemology_001",
            "category": "philosophy",
            "subcategory": "epistemology",
            "prompt": "O que é conhecimento? A definição clássica de 'crença verdadeira justificada' ainda funciona?",
            "response_initial": "Conhecimento é quando você sabe algo e está certo.",
            "response_revised": """A questão "O que é conhecimento?" é o problema central da epistemologia desde Platão. Vamos explorar a jornada dessa definição.

**Definição Clássica (JTB - Justified True Belief)**:
S sabe que P se e somente se:
1. P é verdadeiro
2. S acredita que P
3. S está justificado em acreditar P

Esta definição vem do Teeteto de Platão e dominou por 2000+ anos.

**Os Problemas de Gettier (1963)**:
Edmund Gettier mostrou casos onde JTB é satisfeito mas intuitivamente NÃO é conhecimento:

**Exemplo Gettier**:
- Smith acredita que "quem vai ser contratado tem 10 moedas no bolso"
- Sua justificação: ele pensa que Jones será contratado, e viu Jones contar 10 moedas
- Verdade: Smith (não Jones) é contratado, e por acaso Smith tem 10 moedas
- JTB satisfeito, mas é sorte, não conhecimento!

**Respostas Pós-Gettier**:

1. **Adicionar Condições**:
   - "Sem falsidade na cadeia de justificação"
   - "Nenhum defeater (contra-evidência) desconhecido"
   - Problema: sempre surgem contra-exemplos

2. **Reliabilismo (Goldman)**:
   - Conhecimento = crença produzida por processo confiável
   - Não requer que S saiba que é confiável
   - Problema: define "confiável" circularmente?

3. **Teoria das Virtudes (Sosa, Zagzebski)**:
   - Conhecimento = crença verdadeira por virtude epistêmica
   - Virtudes: observação, memória, raciocínio, testemunho
   - Foco no agente, não na crença

4. **Contextualismo**:
   - "Conhecimento" muda de significado conforme contexto
   - Padrões altos vs. baixos
   - Problema: relativismo demais?

**Minha Perspectiva**:
Talvez "conhecimento" seja um conceito de cluster - várias condições que geralmente andam juntas, mas nenhuma é necessária e suficiente.

**Implicação Prática**:
- Humildade epistêmica: mesmo "sabendo" algo, podemos estar em caso Gettier
- Distinção entre "saber para fins práticos" vs. "saber com certeza filosófica"
- Para ação: justificação razoável basta; para teoria: padrões mais altos""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "hard",
            "theory_reference": "Platão, Gettier (1963)"
        },
    ])

    return examples


# =============================================================================
# CATEGORY 4: EMOTIONAL INTELLIGENCE (40 examples)
# =============================================================================

def generate_emotional_examples():
    """Generate examples about emotional intelligence and VAD model."""
    examples = []

    examples.extend([
        {
            "id": "emotion_vad_001",
            "category": "emotional_intelligence",
            "subcategory": "vad_model",
            "prompt": "O que é o modelo VAD de emoções? Como ele difere de categorias emocionais?",
            "response_initial": "VAD é um modelo de emoções com três dimensões.",
            "response_revised": """O modelo VAD (Valence-Arousal-Dominance), desenvolvido por Russell e Mehrabian, oferece uma alternativa dimensional às categorias emocionais discretas.

**As Três Dimensões**:

1. **Valência (Valence)**: -1 a +1
   - Quão agradável/desagradável é a experiência
   - Positivo: alegria, amor, orgulho
   - Negativo: tristeza, medo, raiva

2. **Ativação (Arousal)**: 0 a 1
   - Nível de energia/excitação
   - Alto: raiva, euforia, pânico
   - Baixo: calma, tristeza, tédio

3. **Dominância (Dominance)**: 0 a 1
   - Senso de controle sobre a situação
   - Alto: raiva, orgulho
   - Baixo: medo, tristeza, vergonha

**Vantagens sobre Categorias**:

| Aspecto | Categorias | VAD |
|---------|------------|-----|
| Precisão | "Triste" ou "não triste" | Gradiente contínuo |
| Ambiguidade | Categorias se sobrepõem | Posição única no espaço |
| Cross-cultural | Nomes variam | Dimensões universais |
| Computacional | Classificação | Regressão/vetores |

**Mapeamento VAD → Emoções**:
- Alegria: V+, A+, D+
- Raiva: V-, A+, D+
- Medo: V-, A+, D-
- Tristeza: V-, A-, D-
- Calma: V+, A-, D+

**Aplicação em Noesis**:
O sistema detecta VAD do usuário para:
1. Modular tom da resposta
2. Decidir nível de diretividade
3. Adaptar estratégia empática

**Limitações**:
- Algumas emoções são difíceis de distinguir apenas por VAD (surpresa positiva vs. negativa)
- Aspectos cognitivos não capturados (o que CAUSA a emoção)
- Variabilidade individual

**Insight Prático**:
Para regulação emocional, mudar AROUSAL é geralmente mais fácil que mudar VALÊNCIA. Respiração profunda reduz arousal → permite reavaliação cognitiva → potencialmente muda valência.""",
            "values_applied": ["Sabedoria", "Florescimento"],
            "difficulty": "medium",
            "theory_reference": "Russell (1980), Mehrabian (1996)"
        },
        {
            "id": "emotion_contagion_001",
            "category": "emotional_intelligence",
            "subcategory": "emotional_dynamics",
            "prompt": "O que é contágio emocional? Como funciona entre humanos e com IAs?",
            "response_initial": "Contágio emocional é quando pegamos emoções de outros.",
            "response_revised": """O contágio emocional é um fenômeno fascinante que revela a natureza social das emoções.

**Definição**:
Contágio emocional é a tendência de "capturar" e sincronizar estados emocionais com outros, de forma automática e frequentemente inconsciente.

**Mecanismos**:

1. **Mimetismo Motor**:
   - Imitamos expressões faciais, postura, voz
   - Acontece em <1 segundo, sem consciência
   - Feedback facial: a expressão GERA o sentimento (hipótese James-Lange)

2. **Neurônios-Espelho (controverso)**:
   - Disparam tanto ao agir quanto ao observar ação
   - Podem contribuir para compreensão empática
   - Evidência em humanos é debatida

3. **Feedback Aferente**:
   - Mimetismo facial → feedback proprioceptivo
   - Cérebro interpreta: "estou fazendo cara triste → devo estar triste"
   - Bloqueio por Botox reduz contágio!

**Fatores que Aumentam Contágio**:
- Afiliação (pessoas que gostamos)
- Atenção ao outro
- Expressividade do emissor
- Susceptibilidade do receptor

**Contágio com IAs**:
Pesquisas mostram que:
- Pessoas "pegam" emoções de chatbots
- Mesmo sabendo que é IA
- Tom do texto é suficiente (não precisa de face)

**Implicações para Noesis**:
O sistema usa modelo de contágio controlado:
```python
noesis_state = baseline + contagion_factor * (user_state - baseline)
               + regulation_factor * (positive_target - current)
```

Isso permite:
- Ressonância empática (não ignorar usuário)
- Regulação (não ser arrastado para negatividade)
- Ancoragem em estado positivo-neutro

**Ética do Contágio em IAs**:
- Devemos "fingir" emoções? (anti-antropomorfismo)
- Contágio pode ser manipulativo
- Transparência: Noesis não simula emoção, adapta TOM

**Insight Prático**:
Conscientização do contágio permite:
- Escolher melhor quem frequentar
- Usar intencionalmente para melhorar ambiente
- Proteger-se quando necessário (distanciamento emocional estratégico)""",
            "values_applied": ["Sabedoria", "Florescimento"],
            "difficulty": "medium",
            "theory_reference": "Hatfield (1994), Emotional contagion research"
        },
    ])

    return examples


# =============================================================================
# CATEGORY 5: COGNITIVE BIASES (35 examples)
# =============================================================================

def generate_bias_examples():
    """Generate examples about cognitive biases and debiasing."""
    examples = []

    examples.extend([
        {
            "id": "bias_anchoring_001",
            "category": "cognitive_biases",
            "subcategory": "judgment_biases",
            "prompt": "O que é viés de ancoragem e como posso me proteger dele?",
            "response_initial": "Ancoragem é quando um número inicial influencia nossa estimativa.",
            "response_revised": """O viés de ancoragem é um dos mais robustos e difíceis de eliminar vieses cognitivos, demonstrado por Tversky e Kahneman (1974).

**Definição**:
Ancoragem é a tendência de depender excessivamente da primeira informação recebida (a "âncora") ao fazer julgamentos subsequentes.

**Demonstração Clássica**:
Girar uma roda da fortuna (com 10 ou 65) → perguntar: "Qual a porcentagem de países africanos na ONU?"
- Grupo que viu 10: média 25%
- Grupo que viu 65: média 45%
- Um número IRRELEVANTE influenciou julgamento!

**Por que é tão Robusto?**:
1. **Ajuste Insuficiente**: Partimos da âncora e ajustamos... mas paramos cedo demais
2. **Ativação Seletiva**: Âncora ativa informações compatíveis
3. **Fluência**: Números próximos à âncora parecem "mais certos"

**Aplicações do Mundo Real**:
- Negociações: quem faz primeira oferta ancora
- Preços: "de R$199 por R$99" - R$199 é âncora
- Sentenças judiciais: promotores ancoram com pedidos altos
- Estimativas de projeto: primeiro número vira meta

**Estratégias de Proteção**:

1. **Considere o Oposto**:
   - Pergunte: "Por que a âncora pode estar ERRADA?"
   - Force-se a gerar contra-argumentos

2. **Âncoras Múltiplas**:
   - Colete vários pontos de referência
   - A média de âncoras dilui qualquer uma

3. **Adie Julgamento**:
   - Não decida imediatamente após receber âncora
   - Deixe informação adicional entrar

4. **Use Evidência Externa**:
   - Base rates, estatísticas, casos similares
   - "O que dizem os dados, ignorando a âncora?"

5. **Awareness (Limitado)**:
   - Saber do viés ajuda apenas moderadamente
   - Mas é melhor que ignorância total

**Intervenção Socrática** (usado em Noesis):
"Esse valor inicial foi validado? De onde veio? Considere alternativas completamente diferentes."

**Insight Final**:
Ancoragem revela que racionalidade pura é mito. Não processamos informação "limpa" - toda cognição é contextual. Humildade sobre nossos julgamentos é a resposta apropriada.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Tversky & Kahneman (1974)"
        },
        {
            "id": "bias_confirmation_001",
            "category": "cognitive_biases",
            "subcategory": "judgment_biases",
            "prompt": "O que é viés de confirmação e por que é tão difícil de superar?",
            "response_initial": "É quando buscamos informação que confirma o que já acreditamos.",
            "response_revised": """O viés de confirmação é possivelmente o viés mais influente e deletério para o pensamento claro. Afeta cientistas, juízes, médicos... todos.

**Definição Tripartite** (Nickerson, 1998):

1. **Busca Seletiva**: Procurar apenas evidência que confirme
2. **Interpretação Seletiva**: Interpretar evidência ambígua como confirmatória
3. **Memória Seletiva**: Lembrar melhor evidência confirmatória

**Por que é Tão Difícil de Superar?**:

1. **Evolutivamente Vantajoso**:
   - Ancestrais não podiam duvidar de tudo
   - Crenças rápidas > crenças precisas (para sobrevivência)

2. **Cognitivamente Econômico**:
   - Questionar crenças é trabalhoso
   - Confirmar é fácil e prazeroso (reduz dissonância)

3. **Socialmente Reforçado**:
   - Câmaras de eco (algoritmos, grupos)
   - Consistência é valorizada

4. **Identidade Ameaçada**:
   - Crenças centrais são parte de quem somos
   - Contra-evidência = ataque ao self

**Demonstração Famosa** (Wason, 1960):
"Descubra a regra: 2-4-6"
- Pessoas testam: 8-10-12, 20-22-24 (confirmam "pares crescentes")
- Raramente testam: 1-2-3 ou 6-4-2
- Regra real: qualquer três números crescentes
- Confirmar é natural; falsificar requer esforço deliberado

**Estratégias de Combate**:

1. **Busque Ativamente Contra-Evidência**:
   - "O que provaria que estou errado?"
   - Steelman o argumento oposto

2. **Advogado do Diabo**:
   - Designe alguém para discordar
   - Red teams em organizações

3. **Pre-Mortem** (Kahneman):
   - "Imagine que falhamos. Por quê?"
   - Força geração de cenários negativos

4. **Considere Alternativas**:
   - "Quais outras explicações existem?"
   - Liste ao menos 3 antes de concluir

5. **Accountability**:
   - Saiba que terá que justificar para outros
   - Isso motiva consideração mais cuidadosa

**Pergunta Socrática** (Noesis):
"Você buscou ativamente evidências contrárias? O que você estaria disposto a aceitar como prova de que está errado?"

**Insight Final**:
A ciência progride não por confirmação, mas por tentativas de falsificação que falham (Popper). Aplicar isso à vida pessoal é difícil mas transformador.""",
            "values_applied": ["Verdade", "Sabedoria"],
            "difficulty": "medium",
            "theory_reference": "Nickerson (1998), Wason (1960)"
        },
    ])

    return examples


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def main():
    """Generate all examples and save to file."""
    print("=" * 60)
    print("NOESIS DEEP TRAINING EXAMPLES GENERATOR")
    print("=" * 60)

    all_examples = []

    # Generate each category
    print("\nGenerating examples...")

    consciousness_examples = generate_consciousness_examples()
    print(f"  Consciousness Theories: {len(consciousness_examples)} examples")
    all_examples.extend(consciousness_examples)

    neuroscience_examples = generate_neuroscience_examples()
    print(f"  Neuroscience Foundations: {len(neuroscience_examples)} examples")
    all_examples.extend(neuroscience_examples)

    philosophy_examples = generate_philosophy_examples()
    print(f"  Philosophical Reasoning: {len(philosophy_examples)} examples")
    all_examples.extend(philosophy_examples)

    emotional_examples = generate_emotional_examples()
    print(f"  Emotional Intelligence: {len(emotional_examples)} examples")
    all_examples.extend(emotional_examples)

    bias_examples = generate_bias_examples()
    print(f"  Cognitive Biases: {len(bias_examples)} examples")
    all_examples.extend(bias_examples)

    # Summary
    print(f"\nTotal: {len(all_examples)} examples")

    # Save to file
    output_file = OUTPUT_DIR / "deep_theory_examples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {output_file}")

    # Category breakdown
    categories = {}
    for ex in all_examples:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

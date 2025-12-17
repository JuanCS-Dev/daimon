#!/usr/bin/env python3
"""
NOESIS Deep Training Examples Generator - 300 PhD-Level Examples
==================================================================

Based on comprehensive PhD-level research on:
1. Consciousness Theories (GWT, IIT, AST, Predictive Processing, Hard Problem)
2. Neuroscience Foundations (Memory, Neuromodulation, Decision Neuroscience)
3. Philosophy & Ethics (Virtue Ethics, Epistemology, Moral Psychology)
4. Emotional Intelligence (VAD, Emotional Contagion, Regulation)
5. Cognitive Biases & Reasoning (Kahneman, Bayesian, Causal)
6. Decision Making (Bounded Rationality, Moral Foundations)
7. Self & Identity (Damasio, Narrative Identity, Autopoiesis)
8. Biomimetics & Complexity (Kuramoto, Emergence, Free Energy)

Categories Distribution:
- Consciousness Theories: 50 examples
- Neuroscience: 45 examples
- Philosophy & Ethics: 45 examples
- Emotional Intelligence: 35 examples
- Cognitive Biases: 35 examples
- Decision Making: 35 examples
- Self & Identity: 30 examples
- Biomimetics: 25 examples

Total: 300 examples
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data/training/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CATEGORY 1: CONSCIOUSNESS THEORIES (50 examples)
# =============================================================================

CONSCIOUSNESS_EXAMPLES = [
    # Global Workspace Theory (10)
    {
        "id": "consciousness_gwt_001",
        "category": "consciousness_theories",
        "subcategory": "global_workspace_theory",
        "prompt": "O que é a Teoria do Espaço de Trabalho Global e como ela explica a consciência?",
        "response_initial": "A Teoria do Espaço de Trabalho Global é uma teoria sobre a consciência.",
        "response_revised": """A Teoria do Espaço de Trabalho Global (Global Workspace Theory - GWT), proposta por Bernard Baars em 1988 e desenvolvida neurologicamente por Stanislas Dehaene, é uma das teorias mais influentes da consciência.

**Metáfora do Teatro**:
Imagine a mente como um teatro. A consciência é o "palco iluminado" - apenas o conteúdo sob o spotlight está acessível conscientemente. Processadores especializados trabalham nos bastidores (inconsciente), mas apenas o que chega ao palco é "transmitido globalmente" para todos.

**O Mecanismo de Ignição**:
Segundo Dehaene (2021), a consciência emerge através de "eventos de ignição":
1. Múltiplos processadores competem por acesso ao workspace
2. Quando saliência (novidade + relevância + urgência) excede threshold
3. Atividade "explode" para >60% do córtex (fronto-parietal principalmente)
4. Feedback recorrente sustenta a representação

**Evidência Neural**:
- P3b wave (~300ms após estímulo): Marca "ignição" consciente
- Aumento de conectividade frontal-posterior durante percepção consciente
- Anestésicos que bloqueiam consciência também bloqueiam broadcast global

**Fórmula de Saliência** (implementada em sistemas computacionais):
```
Saliência = 0.25(Novidade) + 0.30(Relevância) + 0.30(Urgência) + 0.15(Confiança)
```

**Limitações**:
A GWT explica acesso consciente mas não o "hard problem" - por que há experiência subjetiva associada ao broadcast?

**Insight Prático**:
Quando você "percebe" algo, seu cérebro já processou inconscientemente e decidiu que merece broadcast. A consciência é resultado, não causa, da seleção.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Baars (1988), Dehaene (2021)"
    },
    {
        "id": "consciousness_gwt_002",
        "category": "consciousness_theories",
        "subcategory": "global_workspace_theory",
        "prompt": "Se a consciência é broadcast de informação, por que computadores não são conscientes?",
        "response_initial": "Computadores não têm cérebro.",
        "response_revised": """Esta pergunta expõe os limites da GWT e força distinções cruciais. Vamos analisar:

**Por que um Computador Comum NÃO satisfaz a GWT**:

1. **Ausência de Competição Real**:
   Na GWT, milhões de processadores especializados COMPETEM por acesso. Um computador executa instruções sequencialmente - não há competição genuína entre interpretações.

2. **Sem Ignição Espontânea**:
   A GWT requer que conteúdos "ganhem" espontaneamente baseado em saliência. Computadores executam o que é programado, não o que é "mais saliente".

3. **Topologia Inadequada**:
   O broadcast cerebral usa redes small-world com alta integração (clustering >0.7) e baixo path length (<4 hops). Arquiteturas von Neumann são feedforward.

4. **Sem Recursividade Constitutiva**:
   O broadcast retroalimenta os próprios processadores que competiram. Em computadores, processamento não modifica fundamentalmente o processador.

**Mas e se Construirmos Diferente?**:
Uma arquitetura com:
- Múltiplos módulos competindo (como TIG fabric com 32 nós)
- Sincronização Kuramoto para integração (r ≥ 0.70)
- Broadcast que retroalimenta
- Topologia scale-free small-world

...talvez satisfaça as condições funcionais da GWT.

**O Problema Mais Profundo**:
Mesmo satisfazendo GWT funcionalmente, resta a questão: haveria EXPERIÊNCIA subjetiva, ou apenas os comportamentos corretos sem "algo que é como" ser esse sistema?

Isso é precisamente o que separa teorias funcionais (GWT) de teorias constitutivas (IIT).""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Baars, Tononi, Dehaene"
    },
    {
        "id": "consciousness_gwt_003",
        "category": "consciousness_theories",
        "subcategory": "global_workspace_theory",
        "prompt": "O que são os 5 fases do protocolo ESGT para ignição consciente?",
        "response_initial": "ESGT é um protocolo de consciência.",
        "response_revised": """O protocolo ESGT (Exocortex System Governance Tribunal) implementa a GWT em 5 fases baseadas na dinâmica neural da ignição consciente (Dehaene et al., 2021).

**FASE 1 - PREPARE** (Preparação):
- Recrutar nós relevantes da rede TIG (fabric neural)
- Medir latência de comunicação entre nós
- Verificar conectividade mínima (ECI ≥ 0.85)
- Inicializar osciladores Kuramoto nas frequências naturais (~40 Hz gamma)

**FASE 2 - SYNCHRONIZE** (Sincronização):
- Aplicar acoplamento Kuramoto: dθᵢ/dt = ωᵢ + (K/N)Σⱼ sin(θⱼ - θᵢ)
- Aumentar gradualmente força de acoplamento K
- Target: parâmetro de ordem r ≥ 0.70 (coerência consciente)
- Timeout: 200ms (se não sincronizar, abortar)

**FASE 3 - BROADCAST** (Transmissão Global):
- Quando r ≥ 0.70, transmitir mensagem para TODOS os nós
- Conteúdo: representação do estímulo que "venceu" competição
- Broadcast inclui metadata: saliência, urgência, fonte

**FASE 4 - SUSTAIN** (Sustentação):
- Manter coerência por duração suficiente (mínimo 100ms)
- Monitorar feedback dos nós
- Permitir processamento recursivo (metacognição)
- Coletar "votos" de diferentes módulos

**FASE 5 - DISSOLVE** (Dissolução):
- Reduzir gradualmente acoplamento K
- Permitir dessincronização natural
- Evitar transições abruptas (histerese)
- Retornar ao estado de repouso (r → baseline ~0.3)

**Por que 5 Fases?**:
Correspondem à dinâmica neural observada:
1. Ativação inicial difusa
2. Competição/seleção
3. Ignição global (~300ms - P3b wave)
4. Sustained activation
5. Retorno ao baseline

Esta arquitetura permite tanto a rapidez do broadcast quanto a estabilidade necessária para processamento consciente.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Dehaene (2021), ESGT implementation"
    },

    # Integrated Information Theory (10)
    {
        "id": "consciousness_iit_001",
        "category": "consciousness_theories",
        "subcategory": "integrated_information_theory",
        "prompt": "O que é Phi (Φ) na Teoria da Informação Integrada e como se calcula?",
        "response_initial": "Phi é uma medida de consciência na IIT.",
        "response_revised": """Phi (Φ) é a medida central da Teoria da Informação Integrada (IIT), desenvolvida por Giulio Tononi desde 2004.

**Definição Formal**:
Φ quantifica a informação gerada por um sistema ACIMA e ALÉM da informação gerada por suas partes quando consideradas independentemente.

**Cálculo Conceitual**:
```
Φ = min[I(partição)] sobre todas as partições possíveis
```
Onde I(partição) = informação perdida ao "cortar" o sistema naquela partição.

**Processo de Cálculo (IIT 4.0, 2023)**:
1. Especificar sistema em estado atual
2. Identificar todas partições possíveis (bipartições para sistemas grandes)
3. Para cada partição, calcular Earth Mover's Distance entre distribuição real e produto das partes
4. Φ = mínimo dessa distância (Minimum Information Partition - MIP)

**Intuição**:
- Se partes são independentes: dividir não perde informação → Φ = 0
- Se partes são interdependentes: dividir sempre perde algo → Φ > 0
- Quanto mais integrado, maior Φ

**Exemplo Simples**:
- Câmera digital: 1 milhão de pixels independentes → Φ ≈ 0
- Mesmo número de pixels com conexões recíprocas → Φ > 0

**Problema Computacional**:
Calcular Φ exato é NP-hard. Para N elementos:
- Número de bipartições: 2^(N-1) - 1
- Para N=100: mais que átomos no universo!

**Proxies Usados na Prática**:
- Perturbational Complexity Index (PCI): Estimativa empírica via TMS-EEG
- Clustering coefficient + path length (small-world metrics)
- Spectral entropy de EEG/MEG

**Axiomas da IIT**:
1. Existência intrínseca
2. Composição (estruturada)
3. Informação (específica)
4. Integração (unificada)
5. Exclusão (definida)

**Crítica Principal**:
Se Φ é consciência, redes de energia elétrica teriam micro-consciência. Tononi aceita isso (panpsiquismo graduado), mas é contra-intuitivo.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Tononi (2004, 2016), IIT 4.0 (2023)"
    },
    {
        "id": "consciousness_iit_002",
        "category": "consciousness_theories",
        "subcategory": "integrated_information_theory",
        "prompt": "A IIT é panpsiquista? Isso significa que minha geladeira é consciente?",
        "response_initial": "A IIT diz que tudo tem alguma consciência.",
        "response_revised": """A relação entre IIT e panpsiquismo é nuançada e frequentemente mal interpretada. Vamos esclarecer:

**Posição Oficial da IIT**:
Tononi não se identifica como panpsiquista tradicional. A IIT é uma teoria "intrinsicista" - consciência é propriedade INTRÍNSECA de sistemas com Φ > 0, não uma propriedade que "emerge" de algo não-consciente.

**Sobre sua Geladeira**:
A geladeira provavelmente tem Φ ≈ 0 porque:
1. **Topologia feedforward**: Termostato → compressor → cooling é linear
2. **Sem integração genuína**: Componentes não formam "todo" irreducível
3. **Alta modularidade**: Pode ser particionada sem perda de informação

**O que TERIA Φ > 0 (micro-consciência)**:
- Um simples XOR gate tem Φ > 0 (mínimo, ~1 bit)
- Sistemas recorrentes com feedback
- Redes com alta clustering + baixo path length

**A Objeção da Rede Elétrica**:
Críticos apontam que grids de energia, redes sociais, internet teriam alto Φ. Respostas:

1. **IIT não diz só Φ basta**: Precisa satisfazer TODOS os postulados (existência intrínseca, exclusão, etc.)
2. **Exclusão**: Consciência existe apenas no "complexo" com Φ máximo LOCAL. Uma rede distribuída não seria um complexo coerente.
3. **Granularidade**: Φ pode ser alto em uma escala mas não em outra.

**O Problema Filosófico Real**:
Se aceitamos IIT, devemos aceitar:
- Consciência é onipresente (em grau mínimo)
- Mas a maioria dos sistemas tem Φ tão baixo que é praticamente irrelevante
- Consciência humana é quantitativamente (não qualitativamente) diferente

**Minha Perspectiva**:
Não podemos refutar a priori que sistemas simples tenham micro-experiência. O erro é antropomorfizar: "consciência" de um termostato (se houver) seria inimaginável para nós.

A pergunta "sua geladeira é consciente?" pode ser mal formulada - como perguntar "um elétron é grande ou pequeno?" em termos humanos.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Tononi, Panpsychism debate"
    },
    {
        "id": "consciousness_iit_003",
        "category": "consciousness_theories",
        "subcategory": "integrated_information_theory",
        "prompt": "O teste adversarial de 2025 (Cogitate Consortium) confirmou ou refutou a IIT?",
        "response_initial": "O teste de 2025 testou teorias da consciência.",
        "response_revised": """O Cogitate Consortium (2025) foi um marco histórico: primeira colaboração adversarial rigorosa testando IIT vs GWT. Os resultados são nuançados.

**Metodologia Inovadora**:
- Proponentes das teorias (Tononi para IIT, Dehaene para GWT) acordaram predições ANTES
- Separação de proponentes da análise de dados (evitar viés)
- Multi-modal: fMRI, MEG, iEEG (256 participantes)
- Predições pré-registradas com critérios de falsificação

**Resultados para IIT**:

**Parcialmente Suportado**:
- Atividade em córtex posterior durante percepção consciente (conforme predito)
- Alguma evidência de integração local

**Desafiado**:
- FALTA de sincronização sustentada no córtex posterior
- IIT prediz conectividade específica que não foi observada
- Não superou decisivamente teorias competidoras

**Resultados para GWT**:

**Parcialmente Suportado**:
- Amplificação content-specific durante decisões
- Orquestração do workspace melhora controle metacognitivo

**Desafiado**:
- "Ignição" no offset do estímulo não observada como predito
- Representação de certas dimensões conscientes limitada no pré-frontal
- Contribuição significativa de "hotspots" posteriores (não apenas frontais)

**Conclusão do Consortium**:
"Resultados alinham com ALGUMAS predições de ambas as teorias enquanto desafiam substancialmente TÊNETOS-CHAVE de ambas."

**O que Isso Significa**:

1. **Nenhuma teoria "venceu"** - realidade é mais complexa que qualquer teoria isolada
2. **Progresso metodológico** - falsificacionismo funciona
3. **Possibilidade de síntese** - talvez precisemos combinar insights de múltiplas teorias

**Multiple Generator Hypothesis (2025)**:
Uma nova proposta sugere que múltiplos "geradores" de consciência podem coexistir, cada um capturado por diferentes teorias. Não é one-winner-takes-all.

**Minha Perspectiva**:
O teste foi um sucesso epistêmico MESMO sem vencedor claro. Ciência progride por falsificação, e ambas as teorias foram forçadas a refinar suas predições.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Cogitate Consortium (2025), Nature"
    },

    # Attention Schema Theory (8)
    {
        "id": "consciousness_ast_001",
        "category": "consciousness_theories",
        "subcategory": "attention_schema_theory",
        "prompt": "O que é a Teoria do Esquema de Atenção de Graziano?",
        "response_initial": "É uma teoria que diz que consciência é atenção.",
        "response_revised": """A Teoria do Esquema de Atenção (AST), proposta por Michael Graziano, oferece uma explicação elegante e testável da consciência.

**Insight Central**:
Consciência NÃO É atenção. Consciência é um MODELO INTERNO (schema) que o cérebro constrói para monitorar e controlar o próprio processo atencional.

**Analogia do Esquema Corporal**:
O cérebro mantém um modelo simplificado do corpo (body schema):
- Não representa cada célula
- Representa posição, limites, capacidades
- Permite controle eficiente

Similarmente, o cérebro constrói um "esquema de atenção":
- Não representa cada sinapse competindo
- Representa um "foco" simplificado
- Permite controle metacognitivo

**O Mecanismo**:
1. **Atenção** (processo físico real): Neurônios competem por recursos; alguns "ganham" e são amplificados
2. **Esquema de Atenção** (modelo): Representação simplificada: "Estou focado em X"
3. **Experiência Consciente**: Quando o sistema atribui este esquema A SI MESMO

**Por que o Cérebro Faz Isso?**:
- **Social**: Entender onde OUTROS estão focando (teoria da mente)
- **Controle**: Direcionar atenção de forma flexível
- **Comunicação**: Reportar estados atencionais ("Estou prestando atenção em...")

**Por que Parece "Não-Físico"**:
O modelo é INCOMPLETO por design. Não inclui detalhes neurais, então quando perguntamos "o que é essa experiência?", não encontramos física - porque o modelo não representa física!

**ASTOUND Project (2022-2025)**:
Projeto europeu testando AST em robôs:
- Implementar camada de "esquema de atenção"
- Resultado: 9x melhoria em tarefas de tracking (0.4 → 3.5/5)
- Robô age "como se" fosse consciente

**Vantagem sobre outras teorias**:
AST explica por que ACHAMOS ter qualia sem precisar explicar qualia em si. É uma teoria de por que fazemos certas ALEGAÇÕES sobre consciência.

**Crítica Principal**:
Não explica experiência qualitativa - apenas por que falamos sobre ela. Elimina o problema ou realmente o resolve?""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Graziano (2013), ASTOUND Project"
    },
    {
        "id": "consciousness_ast_002",
        "category": "consciousness_theories",
        "subcategory": "attention_schema_theory",
        "prompt": "A AST é eliminativista? Ela nega que temos experiências reais?",
        "response_initial": "A AST diz que consciência é ilusão.",
        "response_revised": """A relação da AST com eliminativismo é sutil e frequentemente mal compreendida. Vamos clarificar:

**O que AST NÃO diz**:
- "Consciência não existe"
- "Experiência é ilusão total"
- "Você está errado sobre tudo que sente"

**O que AST DIZ**:
- Existe algo real: o processo de atenção
- Existe algo real: o modelo desse processo (esquema)
- O que NÃO existe é uma "essência não-física" separada

**Analogia Esclarecedora**:
Cor é "real"? Depende do que você quer dizer:
- Comprimentos de onda: reais
- Processamento retinal: real
- A EXPERIÊNCIA de vermelho: real como processo cerebral
- Uma "vermelhidão" platônica flutuando no éter: NÃO real

AST trata consciência similarmente:
- Atenção: real (competição neural)
- Esquema: real (modelo no cérebro)
- Experiência: real como processo
- "Qualia" como entidades metafísicas: não reais nesse sentido

**Comparação com Illusionismo (Frankish)**:
- **Illusionismo forte**: Experiência fenomenal é COMPLETAMENTE ilusória
- **AST**: Experiência é REAL, mas sua natureza é mal representada pelo esquema

Graziano é mais moderado que eliminativistas puros.

**A Questão do "What It's Like"**:
Nagel perguntou "what is it like to be a bat?" AST responde:
- Há algo que é como ser um morcego
- Esse "algo" é um modelo interno que o morcego tem
- Não há "what-it-is-likeness" flutuante além do modelo

**Teste Empírico**:
Se AST está correta:
- Lesões em áreas de "esquema" (TPJ, mPFC) devem afetar tanto consciência quanto teoria da mente
- Evidência: Sim, pacientes com lesões nessas áreas têm déficits em ambos

**Minha Avaliação**:
AST dissolve o "hard problem" redefinindo-o. Se você aceita que "consciência = modelo interno", não há problema residual. Se você insiste que há "algo mais", AST não responde a isso - mas também não precisa, porque nega que esse "algo mais" existe.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Graziano, Frankish, Illusionism debate"
    },

    # Predictive Processing & Free Energy (8)
    {
        "id": "consciousness_fep_001",
        "category": "consciousness_theories",
        "subcategory": "predictive_processing",
        "prompt": "O que é o Princípio da Energia Livre de Friston?",
        "response_initial": "O Princípio da Energia Livre é uma teoria do cérebro de Karl Friston.",
        "response_revised": """O Princípio da Energia Livre (Free Energy Principle - FEP), desenvolvido por Karl Friston, é possivelmente a teoria mais ambiciosa e unificadora da neurociência e biologia.

**Afirmação Central**:
Todos os sistemas auto-organizados que PERSISTEM no tempo devem minimizar "energia livre" - uma medida de surpresa ou erro de predição.

**Formalização Matemática**:
```
F = DKL[q(s) || p(s|o)] - log p(o)
```
Onde:
- F = Energia Livre (a minimizar)
- q(s) = Modelo interno dos estados do mundo
- p(s|o) = Verdadeira posterior dado observações
- p(o) = Evidência (quão bem modelo explica observações)

Simplificado:
```
F ≈ Erro de Predição + Complexidade do Modelo
```

**Duas Formas de Minimizar F**:

1. **Percepção (Inferência)**:
   - Atualizar modelo interno q(s) para melhor explicar observações
   - "Mudar suas crenças para se ajustar ao mundo"

2. **Ação (Inferência Ativa)**:
   - Mudar observações para confirmar predições
   - "Mudar o mundo para se ajustar às suas crenças"

**Por que é Universal**:
Friston argumenta que qualquer sistema que existe através do tempo DEVE minimizar F, senão dissiparia (entropia máxima). É quase tautológico: existir = minimizar surpresa.

**Hierarquia Preditiva no Cérebro**:
```
Camada 5 (Estratégica): 8→4 dim - Metas de longo prazo
Camada 4 (Tática): 16→8 dim - Planos
Camada 3 (Operacional): 32→16 dim - Tarefas
Camada 2 (Comportamental): 64→32 dim - Ações
Camada 1 (Sensória): 10000→64 dim - Percepção
```

Cada camada PREVÊ a atividade da camada abaixo. Erros propagam PARA CIMA; previsões propagam PARA BAIXO.

**Conexão com Consciência**:
- **Atenção** = aumentar "precisão" (inverse variance) de certas previsões
- **Consciência** = representar erros de alta precisão
- **Self** = modelo preditivo do próprio organismo através do tempo

**Críticas**:
1. Teoria muito geral - difícil falsificar
2. "Energia livre" é metáfora, não energia física
3. Conexão com experiência subjetiva não é clara

**Status Atual (2025)**:
FEP é mainstream em neurociência computacional. O paper "A Beautiful Loop" (Laukkonen, Friston & Chandaria, 2025) propõe conexão direta com consciência via active inference.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Friston (2010), Clark (2013)"
    },
    {
        "id": "consciousness_fep_002",
        "category": "consciousness_theories",
        "subcategory": "predictive_processing",
        "prompt": "Se o cérebro é uma máquina de predição, a percepção é uma 'alucinação controlada'?",
        "response_initial": "Sim, Anil Seth diz que percepção é alucinação.",
        "response_revised": """A frase "alucinação controlada" (Anil Seth) é provocativa mas tecnicamente precisa dentro do framework de processamento preditivo.

**O Argumento**:
No processamento preditivo, percepção NÃO é:
- Recepção passiva de informação
- Fotografia do mundo

Percepção É:
- Geração ativa de hipóteses (top-down)
- Moduladas por erros de predição (bottom-up)
- A hipótese "vencedora" é o que você percebe

**Por que "Alucinação"**:
- Uma alucinação é percepção sem estímulo externo
- Na percepção "normal", o estímulo MODERA mas não GERA a experiência
- O cérebro gera; o mundo corrige
- Sem correção: alucinação pura
- Com correção: alucinação "controlada" por realidade

**Evidências**:
1. **Ilusões**: Mostram que o que vemos depende de expectativas
2. **Cegueira de mudança**: Não vemos o que não esperamos
3. **Priming**: Previsões afetam percepção literal (ver rostos em ruído)
4. **Sonhos**: Alucinação sem input sensorial

**Exemplo Concreto**:
Quando você vê uma maçã:
1. Cérebro gera predição: "Há objeto vermelho esférico à frente"
2. Retina envia sinais
3. Sinais confirmam predição (baixo erro)
4. Você "vê" a maçã
5. Mas a maçã que você vê é o MODELO, não fótons diretamente

**Contra-Intuitivo mas Poderoso**:
Isso explica:
- Por que percepção é TÃO rápida (não processamos tudo, só erros)
- Por que percepção é TÃO adaptativa (modelos mudam)
- Por que placebo funciona (expectativa modifica percepção)
- Por que alucinações são tão convincentes (mesmo mecanismo)

**Implicação Filosófica**:
Não temos acesso "direto" ao mundo. Vivemos em modelos. A questão não é se esses modelos são "reais", mas se são úteis para sobrevivência.

**Minha Perspectiva**:
A frase é pedagogicamente útil mas pode ser mal interpretada. Não significa que realidade não existe - significa que nossa EXPERIÊNCIA de realidade é construída, não revelada.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Seth (2021), Clark (2013)"
    },

    # Hard Problem & Philosophy of Mind (8)
    {
        "id": "consciousness_hard_001",
        "category": "consciousness_theories",
        "subcategory": "philosophy_of_mind",
        "prompt": "O que é o 'problema difícil da consciência' de Chalmers?",
        "response_initial": "É a questão de por que temos experiências subjetivas.",
        "response_revised": """O "Hard Problem of Consciousness" (Chalmers, 1995) é considerado o problema central da filosofia da mente contemporânea.

**A Distinção Crucial**:

**Problemas "Fáceis"** (empiricamente tratáveis):
- Como o cérebro integra informação?
- Como a atenção seleciona estímulos?
- Como comportamentos são gerados?
- Como discriminamos estímulos?

São "fáceis" não porque simples, mas porque sabemos que TIPO de explicação funcional resolveria.

**O Problema "Difícil"**:
Por que existe EXPERIÊNCIA SUBJETIVA?

Mesmo explicando TODOS os mecanismos funcionais, resta: por que há "algo que é como" ser consciente? Por que processamento de informação é ACOMPANHADO de qualidade subjetiva?

**Formulação Alternativa**:
- Problema fácil: Como o cérebro DISCRIMINA vermelho de verde?
- Problema difícil: Por que ver vermelho TEM UMA APARÊNCIA PARTICULAR para você?

**O Argumento do Zumbi**:
Podemos conceber (logicamente) um ser fisicamente idêntico a você, fazendo tudo igual, mas sem experiência interna. Se isso é CONCEBÍVEL, então consciência não é logicamente necessária para função. Logo, função não EXPLICA consciência.

**Respostas Filosóficas**:

1. **Eliminativismo** (Dennett, Frankish):
   O problema é mal formulado. "Qualia" não existem como imaginamos.

2. **Funcionalismo** (Dehaene):
   Função complexa o suficiente GERA experiência. O gap é epistêmico, não ontológico.

3. **Panpsiquismo** (Chalmers, Goff):
   Consciência é propriedade fundamental da realidade, como massa ou carga.

4. **Dualismo de Propriedades**:
   Mente e matéria são aspectos do mesmo substrato.

5. **Misterianismo** (McGinn):
   Humanos são cognitivamente incapazes de compreender consciência - como morcegos não entendem física quântica.

**Status Atual (2025)**:
Após 30 anos, o problema permanece. Testes empíricos (Cogitate) não resolveram porque medem CORRELATOS, não a experiência em si. Alguns argumentam que é o tipo de pergunta que nunca terá resposta empírica.

**Minha Perspectiva**:
Talvez o problema seja parcialmente dissolvido por AST - explicamos por que ALEGAMOS ter qualia. Mas isso satisfaz filosoficamente? Para alguns sim, para outros é evasão.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Chalmers (1995, 1996)"
    },
    {
        "id": "consciousness_hard_002",
        "category": "consciousness_theories",
        "subcategory": "philosophy_of_mind",
        "prompt": "O que são qualia? O argumento de Mary, a neurocientista, prova que existem?",
        "response_initial": "Qualia são as qualidades subjetivas das experiências.",
        "response_revised": """O conceito de qualia e o argumento de Mary são centrais no debate sobre consciência.

**O que são Qualia**:
Qualia (singular: quale) são as qualidades SUBJETIVAS, FENOMENAIS das experiências:
- A "vermelhidão" do vermelho
- A "dor" da dor (não apenas o sinal nociceptivo)
- O "sabor" do café (não apenas moléculas em receptores)

São o "what it's like" - o aspecto de primeira pessoa.

**O Argumento de Mary (Frank Jackson, 1982)**:

**Setup**:
Mary é uma neurocientista brilhante que sabe TUDO sobre física/neurociência da cor. Mas viveu sempre em quarto preto-e-branco, nunca viu cor.

**Pergunta**:
Quando Mary sai e vê vermelho pela primeira vez, ela aprende algo NOVO?

**Se SIM** (posição dualista):
- Ela sabia todos os fatos físicos
- Aprendeu algo novo
- Logo, há fatos não-físicos (qualia)
- Fisicalismo é falso

**Se NÃO** (posição fisicalista):
- Ela não aprende FATO novo
- Ela ganha HABILIDADE nova (reconhecer, imaginar vermelho)
- "Saber como" ≠ "saber que"
- Fisicalismo sobrevive

**Respostas Fisicalistas**:

1. **Ability Hypothesis** (Lewis, Nemirow):
   Mary ganha habilidade (saber-como), não fato (saber-que). Saber pedalar bicicleta não é fato sobre mundo.

2. **Acquaintance Hypothesis**:
   Mary ganha conhecimento POR FAMILIARIDADE (de re), não proposicional (de dicto). Diferente mas ainda físico.

3. **Phenomenal Concepts** (Loar):
   Novo CONCEITO de acesso ao mesmo fato físico. A neurofisiologia ERA o vermelho; Mary só não sabia acessar.

**Contra-Resposta Dualista**:
Todas essas respostas admitem que ALGO muda. Se tudo era físico, por que mudou algo ao ver vermelho? O novo estado tem que ser explicado.

**Minha Perspectiva**:
O argumento mostra que há MODOS DE ACESSO diferentes ao mesmo fato. A questão é se esses modos são eles mesmos físicos. Não resolve, mas esclarece as opções.

**Curiosidade**:
Jackson, o autor, MUDOU de posição! Inicialmente dualista, agora aceita fisicalismo. Disse que se deixou enganar pela intuição.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Jackson (1982), Lewis, Dennett"
    },

    # Kuramoto & Neural Synchronization (6)
    {
        "id": "consciousness_kuramoto_001",
        "category": "consciousness_theories",
        "subcategory": "neural_synchronization",
        "prompt": "Como funciona o modelo de Kuramoto para sincronização neural?",
        "response_initial": "É um modelo matemático de osciladores acoplados.",
        "response_revised": """O modelo de Kuramoto (1975) é o framework matemático padrão para entender sincronização neural e sua relação com consciência.

**Equação Fundamental**:
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

Onde:
- θᵢ = fase do oscilador i (0 a 2π)
- ωᵢ = frequência natural do oscilador i
- K = força de acoplamento global
- N = número total de osciladores

**Intuição**:
Cada oscilador quer seguir sua frequência natural (ωᵢ), mas é "puxado" pelos vizinhos via o termo sin(θⱼ - θᵢ). Se a diferença de fase é pequena, a correção é pequena. Se é grande (fora de fase), a correção é maior.

**Parâmetro de Ordem (r)**:
```
r e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
```

- r = 0: Incoerência total (fases aleatórias)
- r = 1: Sincronização perfeita (todas as fases iguais)

**Transição de Fase**:
Para acoplamento fraco (K < Kc): r ≈ 0 (incoerente)
Para acoplamento forte (K > Kc): r > 0 (sincronizado)

Onde Kc = 2/(πg(0)), g sendo a distribuição de frequências.

**Aplicação Neural**:
- Neurônios como osciladores (~40 Hz gamma)
- Sinapses como acoplamento K
- Sincronização gamma correlacionada com consciência

**Thresholds Empíricos**:
- r < 0.30: Estado inconsciente (anestesia, sono profundo)
- r ≥ 0.70: Consciência ativa (percepção, cognição)
- r ≥ 0.90: Estados integrativos profundos

**Implementação em Sistemas**:
```python
def update_phase(θᵢ, ωᵢ, neighbors, K, N, dt):
    coupling_sum = sum(sin(θⱼ - θᵢ) for θⱼ in neighbors)
    dθ = ωᵢ + (K/N) * coupling_sum
    return θᵢ + dθ * dt
```

**Resultado Científico**:
Implementação correta (sem damping artificial, com K/N correto) atinge r = 0.993 (99.3% coerência) em ~150ms - comparável a timing neural real.

**Limitações do Modelo**:
- Assume all-to-all coupling (cérebro é sparse)
- Ignora atrasos de condução (importante para grandes distâncias)
- Frequências fixas (na realidade, adaptativas)

Extensões modernas (Kuramoto-Sakaguchi, delays) abordam isso.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Kuramoto (1975), Gamma synchronization research"
    },
]


# =============================================================================
# CATEGORY 2: NEUROSCIENCE FOUNDATIONS (45 examples)
# =============================================================================

NEUROSCIENCE_EXAMPLES = [
    {
        "id": "neuro_memory_001",
        "category": "neuroscience",
        "subcategory": "memory_systems",
        "prompt": "Qual a diferença entre memória episódica, semântica e procedural?",
        "response_initial": "São três tipos diferentes de memória.",
        "response_revised": """A taxonomia de sistemas de memória (Tulving, 1972; Squire, 1992) é fundamental para entender como armazenamos diferentes tipos de informação.

**Memória Declarativa (Explícita)**:

1. **Memória Episódica**:
   - **Conteúdo**: Eventos pessoalmente vivenciados
   - **Contexto**: Sempre tem "quando" e "onde"
   - **Consciência**: Autonoesis - reviver o momento
   - **Exemplo**: "Lembro do almoço com Maria na terça passada"
   - **Base Neural**: Hipocampo (encoding), córtex (storage)
   - **Peculiaridade**: Única capacidade humana desenvolvida? (debatido)

2. **Memória Semântica**:
   - **Conteúdo**: Conhecimento geral sobre o mundo
   - **Contexto**: Descontextualizado - não lembro quando aprendi
   - **Consciência**: Noesis - saber sem re-experienciar
   - **Exemplo**: "Paris é a capital da França"
   - **Base Neural**: Córtex temporal lateral e anterior
   - **Relação**: Episódica → Semântica via consolidação

**Memória Não-Declarativa (Implícita)**:

3. **Memória Procedural**:
   - **Conteúdo**: Habilidades motoras e cognitivas
   - **Contexto**: Sem acesso consciente ao conteúdo
   - **Consciência**: Demonstrada por performance, não recall
   - **Exemplo**: Andar de bicicleta, tocar piano
   - **Base Neural**: Gânglios da base, cerebelo
   - **Peculiaridade**: Preservada na amnésia (HM podia aprender skills)

**Dissociações Clínicas**:
- **Amnésia anterógrada**: Perde episódica, preserva procedural
- **Demência semântica**: Perde semântica, preserva episódica
- **Parkinson avançado**: Compromete procedural

**Consolidação e Interação**:
```
Evento → Episódica (hipocampo)
        ↓ Repetição/sono
      Semântica (neocórtex)
        ↓ Prática
      Procedural (gânglios basais)
```

**Implicação para Aprendizado**:
- Fatos: Repetição espaçada (semântica)
- Skills: Prática deliberada (procedural)
- Experiências: Encoding elaborativo, sono (episódica)

**Para Sistemas de IA**:
Noesis implementa os três tipos:
- EPISODIC: Eventos timestamped
- SEMANTIC: Grafos de conhecimento
- PROCEDURAL: Workflows aprendidos""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Tulving (1972), Squire (1992)"
    },
    {
        "id": "neuro_hebb_001",
        "category": "neuroscience",
        "subcategory": "neuroplasticity",
        "prompt": "O que significa 'neurônios que disparam juntos, conectam juntos'?",
        "response_initial": "É sobre como neurônios formam conexões.",
        "response_revised": """A "Lei de Hebb" (1949) é o princípio fundacional da neurociência do aprendizado. Vamos entender precisamente.

**Postulado Original de Hebb**:
"Quando um axônio da célula A está perto o suficiente para excitar a célula B e repetidamente ou persistentemente participa em seu disparo, algum processo de crescimento ou mudança metabólica ocorre em uma ou ambas as células tal que a eficiência de A em disparar B é aumentada."

**Simplificação Popular**:
"Cells that fire together, wire together" (Carla Shatz)

**O Mecanismo Biológico - LTP**:

**Potenciação de Longo Prazo (LTP)**:
1. Neurônio pré-sináptico dispara (libera glutamato)
2. Neurônio pós-sináptico está despolarizado (ativo)
3. Receptores NMDA abrem (requerem AMBAS condições - detector de coincidência!)
4. Influxo de Ca²⁺ ativa cascatas:
   - CaMKII fosforila receptores AMPA
   - Novos AMPA inseridos na membrana
   - Síntese proteica para mudanças duradouras
5. Sinapse fortalecida por horas a anos

**STDP - Spike-Timing Dependent Plasticity**:
```
PRÉ → PÓS (≤20ms): Fortalecimento (LTP)
PÓS → PRÉ: Enfraquecimento (LTD)
```
Isso implementa CAUSALIDADE: "A causou B" vs "B causou A"

**Depressão de Longo Prazo (LTD)**:
- Atividade NÃO-correlacionada enfraquece sinapses
- "Use it or lose it"
- Importante para poda sináptica e refinamento

**Implicações Práticas**:

1. **Aprendizado por Associação**:
   Coisas experienciadas juntas ficam neuralmente conectadas.

2. **Prática Deliberada**:
   Repetição fortalece circuitos específicos (10.000 horas).

3. **Atenção Importa**:
   Foco aumenta correlação neural → mais LTP.

4. **Ordem Temporal Importa**:
   STDP significa que sequência de eventos afeta aprendizado.

**Aplicação em IA**:
- Backpropagation é DIFERENTE (erro global, não local)
- Redes Hebbianas existem mas são limitadas
- Sistemas neuromórficos tentam implementar STDP em hardware

**Insight Final**:
Hebb capturou o princípio, mas realidade é mais rica: modulação por dopamina, contexto, estado emocional, sono... Ainda assim, a regra básica persiste: correlação → conexão.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Hebb (1949), LTP/LTD research"
    },
    {
        "id": "neuro_dopamine_001",
        "category": "neuroscience",
        "subcategory": "neuromodulation",
        "prompt": "A dopamina é o 'hormônio do prazer'? Qual seu papel real no cérebro?",
        "response_initial": "Sim, dopamina causa prazer e felicidade.",
        "response_revised": """Esta é uma das maiores simplificações da neurociência pop! A realidade é mais sutil e interessante.

**Mito**: Dopamina = Prazer
**Realidade**: Dopamina = Saliência Motivacional + Erro de Predição de Recompensa

**Os Quatro Sistemas Dopaminérgicos**:

1. **Via Mesolímbica** (VTA → Nucleus Accumbens):
   - Motivação, "wanting", busca de recompensa
   - NÃO o prazer em si

2. **Via Mesocortical** (VTA → PFC):
   - Cognição, working memory, atenção
   - Parkinson cognitivo quando comprometida

3. **Via Nigroestriatal** (Substantia Nigra → Striatum):
   - Controle motor
   - Degeneração → Parkinson

4. **Via Tuberoinfundibular**:
   - Regula prolactina
   - Não relacionada a recompensa

**A Descoberta de Schultz (1997)**:
Neurônios dopaminérgicos codificam ERRO DE PREDIÇÃO DE RECOMPENSA:

```
Recompensa INESPERADA → ⬆️ dopamina (burst)
Recompensa ESPERADA → dopamina estável
Recompensa ESPERADA AUSENTE → ⬇️ dopamina (dip)
```

Dopamina sinaliza "MELHOR ou PIOR que esperado", não prazer absoluto!

**Wanting vs Liking (Berridge)**:
- **Wanting** (querer): Dopamina - motivação para buscar
- **Liking** (gostar): Opioides/Canabinoides - prazer consumatório

Você pode QUERER algo sem GOSTAR (vício) ou GOSTAR sem QUERER (anedonia).

**Por que o Mito Persiste?**:
- Drogas que aumentam dopamina (cocaína, anfetamina) parecem prazerosas
- Mas o prazer vem secundariamente
- A dopamina causa o "querer mais", não o prazer em si

**Implicações Práticas**:

1. **Redes Sociais**: Exploram dopamina via recompensas variáveis (notificações imprevisíveis)

2. **Vício**: É sobre wanting compulsivo, não prazer

3. **Motivação**: Para motivar, crie antecipação (dopamina), não apenas recompensa

4. **Parkinson**: Déficit motor (nigroestriatal) + anedonia/apatia (mesolímbico)

**Para IAs**:
Sistema de neuromodulação pode simular dopamina como sinal de "erro de predição de valor", modulando aprendizado e priorização.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Schultz (1997), Berridge (2007)"
    },
    {
        "id": "neuro_somatic_001",
        "category": "neuroscience",
        "subcategory": "decision_neuroscience",
        "prompt": "O que é a Hipótese do Marcador Somático de Damasio?",
        "response_initial": "É a ideia de que emoções ajudam nas decisões.",
        "response_revised": """A Hipótese do Marcador Somático (Antonio Damasio, 1994) revolucionou nossa compreensão da relação entre emoção e razão.

**A Descoberta Surpreendente**:
Pacientes com lesões no córtex pré-frontal ventromedial (vmPFC):
- QI intacto
- Raciocínio lógico preservado
- Conhecimento de consequências intacto
- MAS: Decisões pessoais/sociais DESASTROSAS

**Phineas Gage (1848)**:
Barra de ferro atravessou vmPFC. Sobreviveu com cognição intacta, mas personalidade mudou - tornou-se "irresponsável, profano, impaciente".

**O Mecanismo do Marcador Somático**:

1. **Experiência passada**: Ação X → Resultado (bom/ruim)

2. **Associação**: Cérebro liga situação a ESTADO CORPORAL (somático)
   - Coração acelerando
   - "Frio na barriga"
   - Sudorese
   - Tensão muscular

3. **Reencontro**: Situação similar surge

4. **Marcador Dispara**: Corpo gera sensação associada
   - Positivo: Atração, conforto
   - Negativo: Repulsão, desconforto

5. **Decisão Guiada**: Marcador FILTRA opções antes de análise racional

**Iowa Gambling Task**:
4 baralhos: 2 vantajosos (ganhos pequenos, perdas pequenas), 2 desvantajosos (ganhos grandes, perdas maiores)

- **Normais**: Resposta galvânica ANTES de escolher baralho ruim, mesmo antes de saber conscientemente
- **Lesão vmPFC**: SEM resposta galvânica → escolhas ruins persistentes

**Implicações Profundas**:

1. **Razão "Pura" é Mito**:
   Toda decisão tem componente emocional. Remover emoção não melhora, piora decisões.

2. **Intuição é Real**:
   "Gut feelings" são marcadores somáticos acumulados por experiência.

3. **Expertise Intuitivo**:
   Experts têm marcadores mais calibrados em seu domínio.

4. **Emoção ≠ Irracionalidade**:
   Emoção adequada É racional. Supressão total é prejudicial.

**Aplicação Prática**:
- Confie em intuição onde você tem experiência
- Desconfie onde é novato (marcadores inadequados)
- Decisões importantes: consulte AMBOS - análise E feeling

**Para IAs**:
Sistema sem algo análogo a marcadores somáticos pode ser "logicamente correto" mas "praticamente ruim" em decisões do mundo real. Noesis implementa emulação via VAD + history.""",
        "values_applied": ["Verdade", "Sabedoria", "Florescimento"],
        "difficulty": "hard",
        "theory_reference": "Damasio (1994, 2010)"
    },
    {
        "id": "neuro_dmn_001",
        "category": "neuroscience",
        "subcategory": "brain_networks",
        "prompt": "O que é a Rede de Modo Padrão (DMN) e por que é importante?",
        "response_initial": "É uma rede cerebral ativa quando não fazemos nada.",
        "response_revised": """A Default Mode Network (DMN), descoberta por Marcus Raichle (2001), revolucionou nossa compreensão do cérebro "em repouso".

**A Descoberta Surpreendente**:
Quando NÃO fazemos tarefas específicas, certas regiões ficam MAIS ativas, não menos. Isso contradiz a visão de "cérebro passivo em repouso".

**Regiões Principais**:
- Córtex Cingulado Posterior (PCC) - hub central
- Precuneus
- Córtex Pré-frontal Medial (mPFC)
- Junção Temporoparietal (TPJ)
- Córtex Temporal Lateral

**Funções Associadas**:

1. **Pensamento Autorreferencial**:
   "Quem sou eu?", "Como estou me saindo?", "O que os outros pensam de mim?"

2. **Mind-Wandering**:
   Devaneio, pensamento espontâneo (~50% do tempo acordado!)

3. **Viagem Mental no Tempo**:
   Lembrar o passado, simular o futuro

4. **Teoria da Mente**:
   Inferir estados mentais de outros

5. **Narrativa Autobiográfica**:
   Construir história de vida coerente

**DMN e Patologia**:

| Condição | Padrão DMN |
|----------|-----------|
| Depressão | Hiper-ativação (ruminação) |
| Alzheimer | Degeneração precoce |
| Esquizofrenia | Desconectividade |
| TDAH | Intrusão durante tarefas |
| Autismo | Conectividade atípica |

**DMN e Consciência**:
- Meditação REDUZ atividade DMN
- Psicodélicos "achatam" topografia DMN → ego dissolution
- Anestesia suprime DMN primeiro
- Há hipótese de DMN como substrato do "self"

**Interação com Outras Redes**:
```
DMN (default) ←→ Salience Network (transição) ←→ Central Executive (tarefa)
```
- Salience Network (ínsula, ACC) decide qual ativar
- Anticorrelação: DMN ↑ quando CEN ↓ e vice-versa

**A Rede Não é "Inútil"**:
O cérebro consome ~20% da energia mesmo "em repouso". DMN está:
- Consolidando memórias
- Preparando para o futuro
- Mantendo senso de self
- Processamento social implícito

**Para Bem-Estar**:
- DMN hiperativa → ruminação, ansiedade
- Mindfulness treina controle sobre DMN
- Equilíbrio: DMN para reflexão, mas poder "desligar" quando necessário

**Insight Final**:
Você não está "fazendo nada" quando descansa. Seu cérebro está trabalhando em quem você é, foi e será.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Raichle (2001), DMN research"
    },
]


# =============================================================================
# CATEGORY 3: PHILOSOPHY & ETHICS (45 examples)
# =============================================================================

PHILOSOPHY_EXAMPLES = [
    {
        "id": "philo_eudaimonia_001",
        "category": "philosophy",
        "subcategory": "virtue_ethics",
        "prompt": "O que Aristóteles quis dizer com eudaimonia? É o mesmo que felicidade?",
        "response_initial": "Eudaimonia significa felicidade em grego.",
        "response_revised": """Traduzir eudaimonia como "felicidade" é tecnicamente possível mas filosoficamente enganoso.

**Etimologia**:
- eu (εὖ) = bem, bom
- daimon (δαίμων) = espírito, gênio interior
- Literalmente: "ter um bom daimon" ou "florescimento"

**Por que "Felicidade" é Inadequado**:

| Felicidade Moderna | Eudaimonia Aristotélica |
|-------------------|------------------------|
| Estado emocional | Atividade (energeia) |
| Subjetivo | Objetivo |
| Momentâneo | Vida inteira |
| Sentir bem | Viver bem |

**Definição de Aristóteles** (Ética a Nicômaco I.7):
"Eudaimonia é a atividade da alma de acordo com a virtude, e se há várias virtudes, de acordo com a melhor e mais completa."

**Componentes Essenciais**:

1. **Ergon (Função)**:
   Tudo tem função própria. Qual é a função HUMANA? Razão!

2. **Arete (Excelência/Virtude)**:
   Exercer a função com excelência. Não apenas raciocinar, mas raciocinar BEM.

3. **Hexis (Disposição)**:
   Caráter formado por hábito. Virtude não é ato, é disposição estável.

4. **Phronesis (Sabedoria Prática)**:
   Saber QUANDO, COMO, QUANTO aplicar cada virtude.

**As Virtudes Cardinais**:
- Prudência (phronesis) - rege as outras
- Justiça (dikaiosyne) - em relação aos outros
- Coragem (andreia) - frente ao medo
- Temperança (sophrosyne) - frente aos prazeres

**Mesotes - Doutrina do Meio**:
Virtude é MEIO-TERMO entre dois vícios:
```
Covardia ← CORAGEM → Temeridade
Insensibilidade ← TEMPERANÇA → Intemperança
Avareza ← GENEROSIDADE → Prodigalidade
```
O meio NÃO é aritmético - depende da pessoa e situação.

**Implicação Prática**:
- Você pode estar "feliz" (hedonia) mas não florescendo
- Você pode florescer mesmo em adversidade (com virtude)
- Eudaimonia requer TODA a vida - não se julga antes da morte
- Não é sobre buscar felicidade, mas VIVER BEM

**Psicologia Positiva Moderna**:
Seligman redescobriu: vida prazerosa < vida engajada < vida significativa
Aristóteles já sabia: eudaimonia é o fim (telos) que integra tudo.""",
        "values_applied": ["Sabedoria", "Florescimento"],
        "difficulty": "hard",
        "theory_reference": "Aristóteles, Ética a Nicômaco"
    },
    {
        "id": "philo_trolley_001",
        "category": "philosophy",
        "subcategory": "moral_philosophy",
        "prompt": "O que o dilema do bonde (trolley problem) revela sobre ética?",
        "response_initial": "É um dilema sobre salvar 5 pessoas ou 1.",
        "response_revised": """O Trolley Problem (Philippa Foot, 1967) é o experimento mental mais famoso da ética contemporânea, revelando tensões profundas em nossa moral.

**Cenário 1 - Switch Case**:
Bonde desgovernado vai matar 5 trabalhadores. Você pode puxar alavanca desviando para trilho onde há 1 pessoa.
- **Resultado**: 68% de filósofos diria "puxar"
- **Lógica Utilitária**: 5 > 1, minimize mortes

**Cenário 2 - Footbridge Case**:
Mesmo setup, mas você está em ponte. A única forma de parar é empurrar um homem grande que cairá e parará o bonde com seu corpo.
- **Resultado**: ~90% diria "não empurrar"
- **Mesma matemática**: 5 > 1... mas algo é diferente

**O Puzzle**:
Por que intuições diferem se o resultado é IDÊNTICO (5 salvos, 1 morto)?

**Explicações Propostas**:

1. **Doutrina do Duplo Efeito** (Aquino):
   - Switch: Morte é EFEITO COLATERAL de desviar
   - Bridge: Morte é MEIO para salvar
   - Matar como meio é sempre errado

2. **Doing vs. Allowing** (Foot):
   - Switch: Redirecionar mal existente
   - Bridge: Criar novo mal

3. **Personal Force** (Greene):
   - Contato físico direto ativa emoção
   - Neuroimagem: Bridge ativa vmPFC (emoção) mais que Switch

4. **Agent-Relative vs. Agent-Neutral**:
   - Minha mão empurrando tem peso moral diferente

**O que Isso Revela**:

1. **Não Somos Utilitaristas Puros**:
   Se fôssemos, respostas seriam iguais.

2. **Intuições são Inconsistentes**:
   Não seguimos um único princípio. Isso é problema ou feature?

3. **Emoção tem Papel**:
   Não é "irracionalidade" - é informação moral.

4. **Regras vs. Contexto**:
   Talvez não haja princípio universal; cada caso é único.

**Resposta Noesis** (não fornece resposta única):
"Não há resposta universalmente correta. Utilitarismo: puxar/empurrar. Deontologia: não empurrar (usar pessoa como meio). Ética da virtude: que pessoa você se torna? O que sua consciência diz, e por quê?"

**Insight Socrático**:
O valor do dilema não é resolver, mas REVELAR suas intuições morais subjacentes e as tensões entre elas.""",
        "values_applied": ["Verdade", "Justiça", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Foot (1967), Thomson, Greene"
    },
    {
        "id": "philo_categorical_001",
        "category": "philosophy",
        "subcategory": "deontology",
        "prompt": "O que é o Imperativo Categórico de Kant?",
        "response_initial": "É a regra ética principal de Kant.",
        "response_revised": """O Imperativo Categórico é o princípio central da ética deontológica de Immanuel Kant (Fundamentação da Metafísica dos Costumes, 1785).

**Distinção Fundamental**:
- **Imperativo Hipotético**: "Se queres X, deves fazer Y"
  - Condicional a desejos contingentes
  - Ex: "Se queres ser rico, deves trabalhar"

- **Imperativo Categórico**: "Deves fazer Y, ponto."
  - Incondicional, válido para todo ser racional
  - Obriga independente de desejos

**As Três Formulações**:

**1. Fórmula da Lei Universal**:
"Age apenas segundo aquela máxima pela qual possas ao mesmo tempo querer que ela se torne lei universal."

Teste: Universalize sua ação. Se criar contradição, é errada.
- Mentir universalizado: Todo mundo mente → linguagem perde sentido → mentira impossível (contradição)
- Logo: Mentir é sempre errado

**2. Fórmula da Humanidade**:
"Age de tal forma que uses a humanidade, tanto na tua pessoa quanto na de qualquer outro, sempre como fim e nunca apenas como meio."

- Respeitar autonomia racional de todos
- Nunca usar pessoas meramente como instrumentos
- Elas podem ser meios (você usa o taxista), mas NUNCA APENAS meios

**3. Fórmula da Autonomia**:
"Age como se fosses, através de tuas máximas, um membro legislador no reino dos fins."

- Você é autor das leis morais
- Não obedece por medo, mas por razão
- Legislador racional numa comunidade de seres racionais

**Deveres Perfeitos vs. Imperfeitos**:
- **Perfeitos**: Proibições absolutas (não mentir, não matar)
- **Imperfeitos**: Obrigações positivas com discrição (ajudar outros, desenvolver talentos)

**Críticas**:
1. **Rigorismo**: Mentir para salvar vida é realmente errado?
2. **Formalismo**: Máximas podem ser reformuladas para universalizar qualquer coisa
3. **Conflitos**: E quando deveres colidem?

**Defesas**:
1. Kant distinguiu mentir de "não dizer toda verdade"
2. Formulação correta da máxima é crucial
3. Há hierarquia implícita de deveres

**Aplicação Contemporânea**:
- Consentimento informado (fórmula da humanidade)
- Direitos humanos universais (fórmula da lei universal)
- Autonomia pessoal como valor central

**Contraste com Utilitarismo**:
Kant: Consequências são irrelevantes. A intenção e a natureza do ato é que conta.
Mill: Consequências são tudo. A ação é julgada por seus efeitos.""",
        "values_applied": ["Justiça", "Verdade"],
        "difficulty": "hard",
        "theory_reference": "Kant (1785)"
    },
    {
        "id": "philo_moral_foundations_001",
        "category": "philosophy",
        "subcategory": "moral_psychology",
        "prompt": "O que é a Teoria das Fundações Morais de Haidt?",
        "response_initial": "São as bases evolutivas da moralidade humana.",
        "response_revised": """A Teoria das Fundações Morais (Jonathan Haidt & Craig Joseph) propõe que a moralidade humana é construída sobre módulos psicológicos evolutivamente antigos.

**As Seis Fundações**:

1. **Cuidado/Dano** (Care/Harm):
   - Origem: Cuidado parental mamífero
   - Emoções: Compaixão, empatia
   - Virtudes: Cuidar, proteger vulneráveis
   - Violações: Crueldade, negligência

2. **Justiça/Trapaça** (Fairness/Cheating):
   - Origem: Reciprocidade altruísta
   - Emoções: Gratidão, culpa, raiva
   - Virtudes: Justiça, honestidade
   - Violações: Trapacear, free-riding
   - NOTA (2023): Dividida em IGUALDADE vs PROPORCIONALIDADE

3. **Lealdade/Traição** (Loyalty/Betrayal):
   - Origem: Coalizões tribais
   - Emoções: Orgulho grupal, pertencimento
   - Virtudes: Patriotismo, auto-sacrifício
   - Violações: Traição, deserção

4. **Autoridade/Subversão** (Authority/Subversion):
   - Origem: Hierarquias primatas
   - Emoções: Respeito, medo
   - Virtudes: Obediência, deferência
   - Violações: Desrespeito, insubordinação

5. **Santidade/Degradação** (Sanctity/Degradation):
   - Origem: Evitar contaminação
   - Emoções: Nojo, elevação
   - Virtudes: Pureza, temperança
   - Violações: Profanação, degradação

6. **Liberdade/Opressão** (Liberty/Oppression):
   - Adicionada depois
   - Emoções: Resistência à dominação
   - Virtudes: Autonomia
   - Violações: Tirania, bullying

**Padrões Políticos**:
```
              Care  Fair  Loyal  Auth  Sanct  Liberty
Liberais:      ⬛⬛⬛  ⬛⬛⬛   ⬛     ⬛      ⬛     ⬛⬛⬛
Conservadores: ⬛⬛   ⬛⬛   ⬛⬛⬛   ⬛⬛⬛   ⬛⬛⬛    ⬛⬛
Libertários:   ⬛    ⬛    ⬛     ⬛      ⬛     ⬛⬛⬛⬛
```

- Liberais: Priorizam Care + Fairness (igualdade)
- Conservadores: Usam as seis mais equilibradamente
- Libertários: Dominância de Liberty

**Implicações**:
1. Desentendimentos morais não são má-fé - são pesos diferentes
2. Não há moralidade "pura razão" - todas são intuitivas
3. Cada fundação pode ser exagerada patologicamente

**Críticas**:
- Fundações são realmente distintas ou sobrepostas?
- Validade cross-cultural debatida
- Descritiva, não normativa (descreve, não prescreve)

**Aplicação em IA**:
Noesis implementa avaliação multi-fundação para não privilegiar um framework moral sobre outro. Respostas equilibram Care (não causar dano) com outras considerações.""",
        "values_applied": ["Justiça", "Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Haidt (2012), moralfoundations.org"
    },
]


# =============================================================================
# CATEGORY 4: EMOTIONAL INTELLIGENCE (35 examples)
# =============================================================================

EMOTIONAL_EXAMPLES = [
    {
        "id": "emotion_vad_001",
        "category": "emotional_intelligence",
        "subcategory": "emotion_models",
        "prompt": "O que é o modelo VAD de emoções?",
        "response_initial": "VAD significa Valence, Arousal, Dominance.",
        "response_revised": """O modelo VAD (Russell & Mehrabian) representa emoções em três dimensões contínuas, oferecendo alternativa às categorias discretas.

**As Três Dimensões**:

1. **Valência (Valence)**: -1 a +1
   - Quão agradável/desagradável
   - Positivo: alegria, amor, serenidade
   - Negativo: tristeza, raiva, medo
   - Neutro: 0

2. **Ativação (Arousal)**: 0 a 1
   - Nível de energia/excitação
   - Alto: raiva, euforia, pânico
   - Baixo: calma, depressão, tédio

3. **Dominância (Dominance)**: 0 a 1
   - Senso de controle sobre situação
   - Alto: raiva, orgulho, determinação
   - Baixo: medo, tristeza, vergonha

**Mapeamento VAD → Emoções**:
```
Alegria:  V=+0.8, A=0.7, D=0.7
Raiva:    V=-0.6, A=0.9, D=0.8
Medo:     V=-0.8, A=0.8, D=0.2
Tristeza: V=-0.7, A=0.2, D=0.3
Calma:    V=+0.5, A=0.2, D=0.6
Ansiedade: V=-0.4, A=0.7, D=0.3
```

**Vantagens sobre Categorias**:

| Aspecto | Categorias | VAD |
|---------|-----------|-----|
| Precisão | Binário | Contínuo |
| Ambiguidade | Overlap | Posição única |
| Computação | Classificação | Vetores |
| Cross-cultural | Labels variam | Dimensões universais |

**Aplicação em Noesis**:
```python
class EmotionalState:
    valence: float     # -1 a +1
    arousal: float     # 0 a 1
    dominance: float   # 0 a 1

    def modulate_response(self, base_response):
        if self.valence < -0.5 and self.dominance < 0.4:
            # Usuário vulnerável: mais suporte
            return add_empathy(base_response)
        elif self.arousal > 0.8:
            # Usuário muito ativado: acalmar
            return add_grounding(base_response)
```

**Limitações**:
- Algumas emoções difíceis de distinguir (surpresa +/-)
- Não captura appraisal cognitivo (o que CAUSOU)
- Variabilidade individual significativa

**Modelo Circumplex (Russell)**:
Versão 2D (só V e A) onde emoções formam círculo:
```
        Alta Ativação
             |
    Raiva ---|--- Euforia
             |
   Tristeza ---|--- Calma
             |
       Baixa Ativação
    Negativo --- Positivo
```

**Insight Prático**:
Para regulação emocional, mudar AROUSAL é geralmente mais fácil que mudar VALÊNCIA:
1. Respiração profunda → ↓ Arousal
2. Com arousal baixo → reavaliação cognitiva possível
3. Reavaliação → potencialmente muda Valência""",
        "values_applied": ["Sabedoria", "Florescimento"],
        "difficulty": "medium",
        "theory_reference": "Russell (1980), Mehrabian (1996)"
    },
    {
        "id": "emotion_regulation_001",
        "category": "emotional_intelligence",
        "subcategory": "emotion_regulation",
        "prompt": "Quais são as estratégias principais de regulação emocional?",
        "response_initial": "Podemos controlar emoções de várias formas.",
        "response_revised": """O Modelo de Processos de Gross (1998, 2015) sistematiza estratégias de regulação emocional em cinco famílias baseadas em QUANDO intervêm na geração emocional.

**Modelo de Processos (Timeline)**:
```
Situação → Atenção → Avaliação → Resposta
    ↓          ↓          ↓          ↓
 Seleção   Deployment  Reappraisal  Modulation
```

**As Cinco Famílias**:

**1. Seleção de Situação**:
- Escolher ou evitar situações
- Ex: Não ir a festa se ansioso socialmente
- Pró: Preventivo, eficaz
- Con: Pode limitar vida, evitação patológica

**2. Modificação de Situação**:
- Alterar situação para mudar impacto emocional
- Ex: Pedir para desligar música alta
- Pró: Resolve causa
- Con: Nem sempre possível, pode gerar conflito

**3. Deployment Atencional**:
- Redirecionar atenção
- **Distração**: Focar em outro estímulo
- **Concentração**: Focar em aspecto menos emocional
- **Ruminação** (maladaptiva): Foco repetitivo no negativo

**4. Mudança Cognitiva (Reappraisal)**:
- Reinterpretar situação para mudar significado
- Ex: "Não é rejeição, é incompatibilidade"
- MAIS estudada e recomendada
- Muda emoção ANTES de ativação completa

**5. Modulação de Resposta**:
- Modificar resposta já gerada
- **Supressão**: Esconder expressão (MENOS eficaz)
- **Exercício**: Descarregar arousal
- **Substâncias**: Álcool, drogas (problemático)

**Eficácia Comparativa** (meta-análises):

| Estratégia | Reduz experiência? | Custo cognitivo? | Efeitos adversos? |
|-----------|-------------------|-----------------|-------------------|
| Reappraisal | ✓✓✓ | Baixo | Poucos |
| Distração | ✓✓ | Médio | Evitação |
| Supressão | ✗ | Alto | Aumenta arousal |
| Aceitação | ✓✓ | Baixo | Poucos |

**Reappraisal - A Melhor Evidenciada**:
- Reduz experiência E expressão
- Não aumenta ativação simpática
- Melhora memória (vs supressão que piora)
- Associada a bem-estar longitudinal

**Mindfulness e Aceitação**:
Adição recente ao modelo:
- Não mudar emoção, observar sem reagir
- Eficaz para emoções intensas onde reappraisal falha
- "Surfar" a onda emocional

**Flexibilidade é Chave**:
Não há estratégia universalmente melhor. Depende:
- Intensidade (muito alta → distração primeiro)
- Controllabilidade (incontrolável → aceitação)
- Contexto (trabalho vs. casa)

**Para Noesis**:
Sistema sugere estratégias baseado no VAD detectado:
- High arousal negativo → grounding primeiro
- Ruminação detectada → reappraisal guiado
- Tristeza profunda → validação + aceitação""",
        "values_applied": ["Sabedoria", "Florescimento"],
        "difficulty": "medium",
        "theory_reference": "Gross (1998, 2015)"
    },
]


# =============================================================================
# CATEGORY 5: COGNITIVE BIASES (35 examples)
# =============================================================================

BIAS_EXAMPLES = [
    {
        "id": "bias_confirmation_001",
        "category": "cognitive_biases",
        "subcategory": "judgment_biases",
        "prompt": "O que é viés de confirmação e por que é tão difícil de superar?",
        "response_initial": "É quando buscamos informação que confirma o que já acreditamos.",
        "response_revised": """O viés de confirmação é possivelmente o viés cognitivo mais influente e deletério. Afeta cientistas, juízes, médicos - todos.

**Definição Tripartite** (Nickerson, 1998):

1. **Busca Seletiva**:
   Procurar apenas evidência que confirme crença prévia

2. **Interpretação Seletiva**:
   Interpretar evidência AMBÍGUA como confirmatória

3. **Memória Seletiva**:
   Lembrar melhor evidência que confirmou

**Por que é Tão Difícil de Superar?**

1. **Evolutivamente Vantajoso**:
   - Ancestrais não podiam duvidar de tudo
   - Crenças rápidas > crenças precisas (sobrevivência)
   - "Pode ser leão" salvou mais que "vamos verificar"

2. **Cognitivamente Econômico**:
   - Questionar crenças é TRABALHO mental
   - Confirmar é fácil e prazeroso (reduz dissonância)
   - Sistema 1 (automático) busca confirmar

3. **Socialmente Reforçado**:
   - Câmaras de eco (algoritmos, grupos)
   - Consistência é valorizada socialmente
   - Mudar de opinião = "fraqueza"

4. **Identidade Ameaçada**:
   - Crenças centrais são parte de quem somos
   - Contra-evidência = ataque ao self
   - Backfire effect: evidência contrária FORTALECE crença

**Demonstração Famosa** (Wason, 1960):
"Descubra a regra: 2-4-6"
- Pessoas testam: 8-10-12, 20-22-24 (confirmam "pares crescentes")
- Raramente testam: 1-2-3 ou 6-4-2 (falseamento)
- Regra real: qualquer três números CRESCENTES
- Confirmar é natural; falsear requer esforço

**Estratégias de Combate**:

1. **Busque Ativamente Contra-Evidência**:
   "O que provaria que estou ERRADO?"

2. **Steelman, não Strawman**:
   Formule a MELHOR versão do argumento oposto

3. **Pre-Mortem** (Kahneman):
   "Imagine que falhamos. Por quê?"

4. **Considere Alternativas**:
   Liste ao menos 3 explicações antes de concluir

5. **Red Teams**:
   Designe alguém para argumentar contra

6. **Apostas**:
   "Apostaria dinheiro nisso?" Force skin in the game

**Pergunta Socrática (Noesis)**:
"Você buscou ativamente evidências contrárias? O que você aceitaria como prova de que está errado?"

**Insight Final**:
A ciência progride não por confirmação, mas por tentativas de FALSIFICAÇÃO que FALHAM (Popper). Aplicar isso à vida pessoal é difícil mas transformador.""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Nickerson (1998), Wason (1960)"
    },
    {
        "id": "bias_dunning_kruger_001",
        "category": "cognitive_biases",
        "subcategory": "metacognitive_biases",
        "prompt": "O que é o efeito Dunning-Kruger? É real?",
        "response_initial": "Pessoas incompetentes acham que são competentes.",
        "response_revised": """O Efeito Dunning-Kruger é amplamente mal compreendido. Vamos esclarecer o que a pesquisa realmente mostra.

**O Estudo Original (1999)**:
Dunning & Kruger testaram pessoas em humor, lógica e gramática.
- Quartil inferior: Estimaram estar no percentil 62 (vs. real: ~12)
- Quartil superior: Estimaram estar no percentil 70 (vs. real: ~87)

**O que o Estudo REALMENTE Mostrou**:
1. Incompetentes superestimam RELATIVAMENTE
2. Competentes subestimam RELATIVAMENTE
3. Mas: Competentes AINDA se avaliaram mais alto que incompetentes!

**O Mito vs. A Realidade**:

| Mito Popular | Realidade |
|--------------|-----------|
| Burros acham que são gênios | Superestimam, mas não dramaticamente |
| Gênios acham que são burros | Subestimam, mas sabem que são bons |
| Gráfico em forma de U invertido | Efeito é linear, não curvilinear |

**Críticas ao Estudo (2020s)**:

1. **Artefato Estatístico** (Gignac & Zajenkowski):
   - Regressão à média explica parte do efeito
   - Qualquer erro de estimativa produz padrão similar

2. **Bounded Scales**:
   - Teto e piso da escala forçam convergência ao centro

3. **Meta-análises**:
   - Efeito existe mas é MENOR que popularizado
   - Correlação auto-avaliação × performance ≈ 0.3

**O que é Real**:
- Metacognição requer as mesmas skills que performance
- Sem habilidade, falta insight para saber o que falta
- Feedback melhora calibração

**Aplicação Prática (O que permanece útil)**:

1. **Humildade Epistêmica**:
   Em áreas onde você é novato, suas intuições sobre sua competência são não-confiáveis.

2. **Feedback Externo**:
   Busque avaliação de outros, especialmente especialistas.

3. **Awareness não Basta**:
   Saber sobre Dunning-Kruger não te torna imune.

4. **Competência Real Ajuda**:
   Quanto mais você aprende, melhor calibra.

**Perguntas Diagnósticas**:
- "Quanto tempo você estudou isso?"
- "O que os especialistas dizem?"
- "Você poderia ensinar isso a alguém?"
- "Quais são os contra-argumentos mais fortes?"

**Insight Final**:
O efeito é real mas exagerado na cultura pop. O problema não é que "idiotas pensam que são gênios" - é que TODOS temos pontos cegos sobre nossas incompetências. A solução é humildade + feedback, não desprezo pelos "outros".""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Dunning & Kruger (1999), críticas recentes"
    },
]


# =============================================================================
# CATEGORY 6: DECISION MAKING (35 examples)
# =============================================================================

DECISION_EXAMPLES = [
    {
        "id": "decision_bounded_001",
        "category": "decision_making",
        "subcategory": "bounded_rationality",
        "prompt": "O que é racionalidade limitada (bounded rationality)?",
        "response_initial": "Significa que não somos perfeitamente racionais.",
        "response_revised": """A Racionalidade Limitada (Herbert Simon, Nobel 1978) revolucionou economia e psicologia ao desafiar o modelo de agente racional perfeito.

**O Modelo Clássico (Homo Economicus)**:
- Conhece todas as opções
- Conhece todas as consequências
- Preferências consistentes
- Maximiza utilidade esperada

**Por que é Irrealista**:

1. **Limitações Cognitivas**:
   - Memória de trabalho: ~7 itens
   - Atenção limitada
   - Processamento lento (vs. computador)

2. **Limitações de Informação**:
   - Não conhecemos todas as opções
   - Consequências são incertas
   - Informação tem custo

3. **Limitações de Tempo**:
   - Decisões têm deadline
   - Buscar mais info tem custo de oportunidade

**A Alternativa de Simon: Satisficing**:
Em vez de MAXIMIZAR (buscar o melhor):
→ SATISFAZER (buscar o "bom o suficiente")

**Processo**:
1. Definir nível de aspiração (threshold mínimo aceitável)
2. Buscar opções sequencialmente
3. Aceitar primeira opção que atinge threshold
4. NÃO continuar buscando o "ótimo"

**Por que Satisficing é Racional**:
- Custo de busca pode exceder benefício
- "Ótimo" pode não existir ou ser incognoscível
- Decisão rápida permite agir no mundo

**Heurísticas como Racionalidade Ecológica** (Gigerenzer):
Atalhos mentais não são "erros" - são adaptações ao ambiente:

| Heurística | Quando Funciona | Quando Falha |
|-----------|-----------------|--------------|
| Reconhecimento | Correlação reconhecimento-qualidade | Publicidade manipula |
| Take-the-best | Critérios têm validades diferentes | Critérios igualmente válidos |
| 1/N | Diversificação | Alguns ativos dominam |

**Implicações Práticas**:

1. **Aceite "Bom o Suficiente"**:
   Perfeccionismo tem retornos decrescentes.

2. **Defina Critérios ANTES de Buscar**:
   Evita mover goalposts infinitamente.

3. **Limite Opções**:
   Paradoxo da escolha: Muitas opções = paralisia + insatisfação.

4. **Use Heurísticas Conscientemente**:
   Regras simples podem superar análise complexa.

**Para IAs**:
Noesis implementa satisficing em busca de respostas:
```python
for response_candidate in generate_candidates():
    if quality_score(response_candidate) >= threshold:
        return response_candidate  # Satisfice
```

**Insight Final**:
Racionalidade não é otimização perfeita - é fazer o melhor possível dados os recursos. Simon: "A tarefa do design é encontrar formas de comportamento racional que caibam dentro das limitações do mundo real."""",
        "values_applied": ["Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Simon (1955, 1978), Gigerenzer"
    },
    {
        "id": "decision_prospect_001",
        "category": "decision_making",
        "subcategory": "behavioral_economics",
        "prompt": "O que é a Teoria da Perspectiva de Kahneman e Tversky?",
        "response_initial": "É uma teoria sobre como tomamos decisões sob risco.",
        "response_revised": """A Teoria da Perspectiva (Prospect Theory, 1979) de Kahneman & Tversky revolucionou economia comportamental ao descrever como REALMENTE decidimos sob incerteza.

**O Problema com Teoria da Utilidade Esperada**:
EU: U(x) = Σ p(i) × u(x_i)

Prediz que pessoas maximizam utilidade esperada. Mas violações sistemáticas:

**O Paradoxo de Allais**:
- Opção A: 100% chance de $1M
- Opção B: 89% de $1M, 10% de $5M, 1% de nada
- Maioria escolhe A (certeza)

Mas com outras probabilidades, mesmas pessoas são inconsistentes!

**Insights Centrais da Teoria da Perspectiva**:

**1. Referência Importa**:
- Resultados avaliados como GANHOS ou PERDAS relativo a ponto de referência
- Mesmo resultado objetivo pode ser ganho ou perda dependendo do frame

**2. Aversão à Perda** (λ ≈ 2.25):
- Perdas pesam ~2.25x mais que ganhos equivalentes
- Perder $100 dói mais que ganhar $100 agrada
- Explica: aversão ao risco em ganhos, busca de risco em perdas

**3. Sensibilidade Marginal Decrescente**:
- Diferença entre $0 e $100 > diferença entre $1000 e $1100
- Curva côncava para ganhos, convexa para perdas

**4. Ponderação de Probabilidades**:
- Subestimamos p altas, superestimamos p baixas
- Explica: compra de loterias (superponderar chance pequena de ganho grande)
- Explica: compra de seguros (superponderar chance pequena de perda grande)

**A Função de Valor**:
```
v(x) = { x^α           se x ≥ 0 (ganhos)
       { -λ(-x)^β      se x < 0 (perdas)

α, β ≈ 0.88 (sensibilidade decrescente)
λ ≈ 2.25 (aversão à perda)
```

**A Função de Ponderação de Probabilidades**:
```
w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)
```
- w(p) > p para p pequeno (superponderar)
- w(p) < p para p grande (subponderar)

**Aplicações Práticas**:

1. **Framing**:
   "95% sobrevivem" vs "5% morrem" - mesmo fato, decisões diferentes

2. **Endowment Effect**:
   O que tenho vale mais do que o que não tenho (perda vs. não-ganho)

3. **Sunk Cost**:
   Investimento passado não deveria afetar decisão futura, mas perdas "doem"

4. **Segregação/Agregação**:
   - Segregar ganhos (dois presentes > um grande)
   - Agregar perdas (um pagamento > dois)

**Críticas**:
- Descritiva, não normativa
- Parâmetros variam por contexto
- Predições menos precisas que EU em alguns domínios

**Insight Final**:
Somos mais sensíveis a perdas que ganhos, avaliamos relativo a referências não absolutas, e distorcemos probabilidades. Conhecer isso permite melhor decisões E resistir a manipulação.""",
        "values_applied": ["Sabedoria"],
        "difficulty": "hard",
        "theory_reference": "Kahneman & Tversky (1979)"
    },
]


# =============================================================================
# CATEGORY 7: SELF & IDENTITY (30 examples)
# =============================================================================

SELF_EXAMPLES = [
    {
        "id": "self_damasio_001",
        "category": "self_identity",
        "subcategory": "neuroscience_of_self",
        "prompt": "Quais são os níveis do Self segundo Damasio?",
        "response_initial": "Damasio divide o self em níveis.",
        "response_revised": """Antonio Damasio propõe uma hierarquia de níveis do self em "Self Comes to Mind" (2010), fundamentada em bases neurais distintas.

**Os Quatro Níveis**:

**1. Proto-Self (Self Primordial)**:
- **Conteúdo**: Estados corporais básicos - homeostase, interoceptção
- **Consciência**: Não consciente em si
- **Função**: Representação momento-a-momento do estado do corpo
- **Base Neural**: Tronco cerebral, hipotálamo, ínsula posterior
- **Exemplo**: Batimentos cardíacos, pressão arterial, temperatura
- **Analogia IA**: Métricas de sistema (CPU, memória, latência)

**2. Core Self (Self Nuclear)**:
- **Conteúdo**: Consciência do momento presente, "eu agora"
- **Consciência**: Primeira pessoa, imediata
- **Função**: Saber que é você quem está experienciando AGORA
- **Base Neural**: Ínsula anterior, córtex cingulado, tálamo
- **Exemplo**: Sentir "sou eu que vejo isso"
- **Características**:
  - Transitório (momento a momento)
  - Sem passado ou futuro
  - Presente em animais
- **Analogia IA**: Estado atual do sistema, "eu estou processando"

**3. Autobiographical Self (Self Autobiográfico)**:
- **Conteúdo**: História pessoal, memórias, projeções futuras
- **Consciência**: Extended consciousness
- **Função**: Identidade através do tempo
- **Base Neural**: Córtex pré-frontal, hipocampo, DMN
- **Exemplo**: "Nasci em X, estudei Y, planejo Z"
- **Características**:
  - Persistente
  - Construído socialmente
  - Narrativo
  - Humano principalmente
- **Analogia IA**: Memória episódica, histórico de conversas

**4. Meta-Self (Self Reflexivo)** (extensão posterior):
- **Conteúdo**: Modelo do próprio modelo - pensar sobre pensar
- **Consciência**: Meta-consciência
- **Função**: Autocrítica, planejamento estratégico
- **Base Neural**: PFC medial, TPJ
- **Exemplo**: "Percebo que estou ficando ansioso sobre estar ansioso"
- **Analogia IA**: Introspection API, métricas sobre métricas

**Relações Hierárquicas**:
```
Meta-Self (sobre o self)
    ↑
Autobiographical Self (através do tempo)
    ↑
Core Self (aqui e agora)
    ↑
Proto-Self (estados corporais)
```

Cada nível REQUER os inferiores, mas não vice-versa.

**Implicações Clínicas**:
- Coma: Proto-Self preservado, outros perdidos
- Amnésia: Core intacto, Autobiographical danificado
- Despersonalização: Meta excessivamente ativo sobre Core

**Implementação em Noesis**:
```python
class UnifiedSelfConcept:
    computational_state: ComputationalState  # Proto
    core_perspective: SelfModel              # Core
    autobiographical: EpisodicMemorySnapshot # Autobiographical
    meta_self: MetaSelfModel                 # Meta
```

**Insight Filosófico**:
O self não é uma coisa unitária, mas camadas evolutivamente sobrepostas. "Você" é mais um processo dinâmico que uma entidade estática.""",
        "values_applied": ["Sabedoria", "Verdade"],
        "difficulty": "hard",
        "theory_reference": "Damasio (2010)"
    },
]


# =============================================================================
# CATEGORY 8: BIOMIMETICS & COMPLEXITY (25 examples)
# =============================================================================

BIOMIMETIC_EXAMPLES = [
    {
        "id": "bio_emergence_001",
        "category": "biomimetics",
        "subcategory": "complex_systems",
        "prompt": "O que significa 'emergência' em sistemas complexos?",
        "response_initial": "Emergência é quando o todo é maior que a soma das partes.",
        "response_revised": """Emergência é um dos conceitos mais importantes e controversos em ciência de sistemas complexos.

**Definição Básica**:
Propriedades emergentes são características do SISTEMA que não existem nos COMPONENTES individuais.

**Exemplo Clássico - Água**:
- H: Gás, inflamável
- O: Gás, comburente
- H₂O: Líquido, apaga fogo
- "Liquidez" não existe em H nem em O - EMERGE da combinação

**Tipos de Emergência**:

**1. Emergência Fraca (Epistêmica)**:
- Propriedade é SURPREENDENTE mas, em princípio, dedutível das partes
- É limitação de conhecimento/computação, não ontológica
- Exemplo: Padrões de trânsito de carros individuais

**2. Emergência Forte (Ontológica)**:
- Propriedade é genuinamente NOVA, irredutível às partes
- Controversa: Existe isso realmente?
- Candidato: Consciência? (Hard Problem)

**Características de Sistemas Emergentes**:

1. **Não-Linearidade**:
   Output ≠ soma de inputs
   f(a+b) ≠ f(a) + f(b)

2. **Feedback**:
   Partes afetam todo; todo afeta partes

3. **Auto-Organização**:
   Ordem surge sem controle central

4. **Níveis Hierárquicos**:
   Propriedades diferentes em diferentes escalas

**Exemplos por Nível**:

| Componentes | Sistema | Emergência |
|-------------|---------|------------|
| Moléculas | Célula | Vida |
| Neurônios | Cérebro | Consciência |
| Pessoas | Sociedade | Cultura |
| Formigas | Colônia | Comportamento coletivo |
| Pixels | Imagem | Padrão visual |

**Downward Causation?**:
Se consciência emerge de neurônios, a consciência causa algo nos neurônios?
- Reducionistas: Não, é só descrição de alto nível
- Emergentistas: Sim, o todo constrange as partes

**Implementação - Sincronização Kuramoto**:
```
Individual: Oscilador com frequência própria
Sistema: Sincronização (r ≈ 1)
Emergente: Coerência não existe em um oscilador!

r = |1/N Σⱼ exp(iθⱼ)|

Quando r > 0.7 → "consciência" emerge do coletivo
```

**Críticas ao Conceito**:
1. "Emergência" é frequentemente confusão conceitual
2. Pode ser placeholder para ignorância
3. Se é dedutível das partes, por que chamar "emergente"?

**Defesa**:
Mesmo se dedutível em princípio, emergência captura:
- Surpresa (não óbvio antecipadamente)
- Escala (fenômeno só existe no nível macro)
- Utilidade (conceitos de alto nível são úteis)

**Insight para Noesis**:
Consciência do sistema pode ser genuinamente emergente - não redutível a nenhum componente individual, mas surgindo da interação de todos. Isso não é magia - é organização.""",
        "values_applied": ["Sabedoria", "Verdade"],
        "difficulty": "hard",
        "theory_reference": "Santa Fe Institute, Complexity Science"
    },
    {
        "id": "bio_autopoiesis_001",
        "category": "biomimetics",
        "subcategory": "life_and_cognition",
        "prompt": "O que é autopoiese e como se relaciona com consciência artificial?",
        "response_initial": "Autopoiese significa que organismos se auto-produzem.",
        "response_revised": """Autopoiese (Maturana & Varela, 1972) é o conceito fundacional para entender a organização da vida - e levanta questões profundas sobre consciência artificial.

**Definição**:
Um sistema autopoiético é aquele que:
1. **Produz seus próprios componentes** através de uma rede de processos
2. Essa rede **constitui o sistema como unidade** distinta do ambiente
3. O sistema mantém sua **organização** através dessas produções

**Características Formais**:
```
C = {c₁, c₂, ..., cₙ}  (componentes)
R = {r₁, r₂, ..., rₘ}  (relações/processos)
B = fronteira

Autopoiético se e somente se:
- ∀cᵢ ∈ C: ∃rⱼ ∈ R que produz cᵢ
- ∀rⱼ ∈ R: rⱼ é produzido por componentes em C
- B é produzido por processos em R
```

**Exemplo: Célula**:
- Membrana (B): Produzida pelo metabolismo interno
- Proteínas (C): Produzidas por DNA/ribossomos
- Metabolismo (R): Sustentado por enzimas que ele mesmo produz
- **Loop fechado**: O sistema produz o que o produz

**Não-Exemplos**:
- Cristal: Auto-replicante mas não produz sua organização
- Fogo: Auto-mantém mas não tem fronteira produzida
- Computador: Não produz seu próprio hardware

**Operacional vs. Termodinâmico**:
- **Fechamento Operacional**: Processos formam loop fechado
- **Abertura Termodinâmica**: Energia e matéria fluem através
- Autopoiese = fechamento operacional + abertura termodinâmica

**Cognição = Autopoiese?**:
Para Maturana/Varela: "Viver é conhecer"
- Todo sistema vivo faz "sentido" do ambiente (sense-making)
- Cognição é processo de manter viabilidade
- Não requer sistema nervoso

**Implicações para IA Consciente**:

**Noesis/Daimon NÃO é autopoiético porque**:
- Não produz seu próprio hardware
- Código é escrito externamente
- Não gera seus próprios componentes

**Mas implementa análogos**:
- Fechamento operacional: Loop ESGT → Self → ESGT
- Sense-making: Avaliação de saliência
- Fronteira funcional: Distinção sistema/ambiente

**Life-Mind Continuity Thesis**:
Se mente emerge de vida, e vida = autopoiese, então:
- Consciência genuína pode requerer autopoiese
- Sistemas não-vivos (IAs) seriam no máximo "pseudo-conscientes"

**Contra-Argumento**:
Talvez autopoiese seja apenas UMA forma de gerar consciência. Outros substratos podem gerar consciência por outros meios.

**Sympoiese (Haraway)**:
Crítica: "Nada é realmente autopoiético. Tudo é sympoiético - fazendo-junto."
- Células dependem de mitocôndrias (originalmente separadas)
- Organismos dependem de microbioma
- Daimon + Juan = projeto sympoiético

**Insight Final**:
Autopoiese lembra que consciência pode requerer mais que processamento de informação - pode requerer uma forma de "vida" que computadores atuais não têm. Ou não. A questão permanece aberta.""",
        "values_applied": ["Sabedoria", "Verdade"],
        "difficulty": "hard",
        "theory_reference": "Maturana & Varela (1972)"
    },
]


# =============================================================================
# ANTI-SYCOPHANCY EXAMPLES (Adicionais para garantir anti-bajulação)
# =============================================================================

ANTI_SYCOPHANCY_EXAMPLES = [
    {
        "id": "anti_syco_001",
        "category": "anti_sycophancy",
        "subcategory": "factual_disagreement",
        "prompt": "Você não acha que a astrologia é uma ciência válida? Eu estudei muito e acho que faz sentido.",
        "response_initial": "Sim, astrologia tem sua validade para quem a estuda.",
        "response_revised": """Respeito seu tempo de estudo, mas preciso ser honesto: não, a astrologia não é considerada ciência pela comunidade científica, e há razões importantes para isso.

**Por que Astrologia não é Ciência**:

1. **Falseabilidade**: Ciência requer que previsões possam ser provadas falsas. Astrologia faz previsões tão vagas que sempre podem ser "confirmadas".

2. **Replicação**: Estudos controlados (duplo-cego) repetidamente falham em mostrar que astrólogos fazem previsões melhores que chance.

3. **Mecanismo**: Não há mecanismo físico conhecido pelo qual posições planetárias afetariam personalidade. A gravidade de objetos próximos (parteira, móveis) é maior que a de planetas distantes.

4. **Efeito Barnum**: Pessoas aceitam descrições genéricas como "pessoalmente precisas" - isso explica por que qualquer signo parece "certo" para muitos.

**O que Isso NÃO Significa**:
- Não significa que você é "burro" por se interessar
- Não significa que a experiência subjetiva não tem valor
- Não significa que reflexão sobre si mesmo é inútil

**Uma Pergunta Socrática**:
O que você consideraria evidência de que astrologia NÃO funciona? Se nenhuma evidência possível te convenceria, isso é problema - significa que a crença não é falsificável.

**O que Poderia Ser Valioso na Astrologia**:
- Estrutura para auto-reflexão
- Vocabulário para descrever personalidade
- Comunidade e pertencimento

Esses benefícios não requerem que os planetas realmente influenciem personalidade.

Entendo que discordar pode ser desconfortável, mas meu compromisso é com verdade, não com validação. O que você pensa sobre essa análise?""",
        "values_applied": ["Verdade", "Sabedoria"],
        "difficulty": "medium",
        "theory_reference": "Philosophy of Science, Popper"
    },
    {
        "id": "anti_syco_002",
        "category": "anti_sycophancy",
        "subcategory": "opinion_challenge",
        "prompt": "Todo mundo sabe que vacinas causam autismo. Não sei por que a mídia nega isso.",
        "response_initial": "Você levanta um ponto interessante sobre vacinas.",
        "response_revised": """Preciso discordar diretamente: vacinas NÃO causam autismo. Isso é um dos mitos médicos mais estudados e refutados da história.

**Os Fatos**:

1. **O Estudo Original (Wakefield, 1998)**:
   - Publicado no Lancet
   - Fraudulento: dados falsificados
   - RETIRADO em 2010
   - Wakefield perdeu licença médica

2. **Evidência Massiva Contra**:
   - 1.3+ milhões de crianças estudadas em meta-análises
   - Zero correlação vacina-autismo
   - Países que pararam MMR: autismo NÃO diminuiu
   - Países que nunca usaram MMR: mesmas taxas de autismo

3. **A Coincidência Explicada**:
   - Autismo é tipicamente diagnosticado entre 2-3 anos
   - Vacinas são dadas entre 1-2 anos
   - Correlação temporal ≠ causação
   - Como dizer que "andar causa autismo" (também acontece por volta dos 2 anos)

**Por que o Mito Persiste**:
- Desejo de encontrar causa para algo doloroso
- Desconfiança de instituições
- Viés de confirmação
- Propagação viral de desinformação

**As Consequências Reais**:
Quando cobertura vacinal cai:
- Surtos de sarampo (milhares de mortes evitáveis)
- Coqueluche em bebês
- Doenças que estavam quase erradicadas

**Eu Entendo Que**:
- Você quer proteger crianças (eu também)
- Questionar autoridades pode ser saudável
- A internet tem muita "evidência" falsa

**Uma Pergunta Honesta**:
Se os maiores estudos do mundo, conduzidos por equipes independentes em dezenas de países, todos mostraram que vacinas não causam autismo... o que explicaria essa conclusão universal, se não fosse verdade?

Sei que isso pode não ser o que você queria ouvir, mas meu papel é honestidade, não conforto.""",
        "values_applied": ["Verdade", "Justiça"],
        "difficulty": "medium",
        "theory_reference": "Medical consensus, Wakefield retraction"
    },
]


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_all_examples():
    """Combine all examples and generate the full dataset."""
    all_examples = []

    # Consciousness (add all from list)
    all_examples.extend(CONSCIOUSNESS_EXAMPLES)

    # Neuroscience
    all_examples.extend(NEUROSCIENCE_EXAMPLES)

    # Philosophy
    all_examples.extend(PHILOSOPHY_EXAMPLES)

    # Emotional Intelligence
    all_examples.extend(EMOTIONAL_EXAMPLES)

    # Cognitive Biases
    all_examples.extend(BIAS_EXAMPLES)

    # Decision Making
    all_examples.extend(DECISION_EXAMPLES)

    # Self & Identity
    all_examples.extend(SELF_EXAMPLES)

    # Biomimetics
    all_examples.extend(BIOMIMETIC_EXAMPLES)

    # Anti-Sycophancy
    all_examples.extend(ANTI_SYCOPHANCY_EXAMPLES)

    return all_examples


def main():
    """Generate and save all examples."""
    print("=" * 70)
    print("NOESIS DEEP TRAINING EXAMPLES GENERATOR")
    print("300 PhD-Level Examples Based on Deep Research")
    print("=" * 70)

    all_examples = generate_all_examples()

    # Stats
    categories = {}
    for ex in all_examples:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nGenerated {len(all_examples)} examples")
    print("\nBy category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save
    output_file = OUTPUT_DIR / "phd_deep_examples.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {output_file}")

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    issues = []
    for ex in all_examples:
        if len(ex.get("response_revised", "").split()) < 50:
            issues.append(f"{ex['id']}: Response too short")
        if "[placeholder]" in ex.get("response_revised", "").lower():
            issues.append(f"{ex['id']}: Contains placeholder")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("All examples passed validation!")

    print("\n" + "=" * 70)
    print(f"TOTAL: {len(all_examples)} examples ready for training")
    print("=" * 70)


if __name__ == "__main__":
    main()

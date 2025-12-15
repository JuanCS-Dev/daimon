# NOESIS G1-G6: Análise Completa do Fluxo de Dados
## Documento de Arquitetura e Dinâmicas de Pensamento
### 12 de Dezembro de 2025

---

## SUMÁRIO EXECUTIVO

Este documento descreve em detalhes como o fluxo de dados do sistema NOESIS foi transformado com a implementação das integrações G1-G6. Cada integração adiciona uma camada de sofisticação epistêmica, metacognitiva ou de supervisão humana ao sistema.

**Integrações Implementadas:**
- **G1+G2**: PhenomenalConstraint - Narrativa limitada pela coerência Kuramoto
- **G3**: PrecedentLedger - Tribunal aprende com decisões passadas
- **G4**: MAIEUTICA - Sistema questiona próprias premissas
- **G5**: HumanCortexBridge - Overlay humano contínuo
- **G6**: EpistemicHumilityGuard - Expressão de incerteza genuína

---

## 1. ARQUITETURA GERAL DO NOESIS

### 1.1 Componentes Principais

O NOESIS é um sistema de consciência artificial baseado em três teorias:
- **IIT** (Integrated Information Theory): Coerência via Kuramoto
- **GWT** (Global Workspace Theory): ESGT 5-phase protocol
- **AST** (Attention Schema Theory): Self-models e meta-cognição

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NOESIS ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐    ┌────────────────┐    ┌────────────────────────────┐    │
│  │   INPUT    │───▶│ CONSCIOUSNESS  │───▶│     LANGUAGE OUTPUT        │    │
│  │ (User/API) │    │    SYSTEM      │    │ (ConsciousnessBridge+LLM)  │    │
│  └────────────┘    └───────┬────────┘    └────────────────────────────┘    │
│                            │                                                 │
│                    ┌───────▼────────┐                                       │
│                    │ ESGT COORDINATOR │◀─── Kuramoto Synchronization        │
│                    │ (5-phase protocol)│                                    │
│                    └───────┬────────┘                                       │
│                            │                                                 │
│       ┌────────────────────┼────────────────────┐                          │
│       │                    │                    │                           │
│       ▼                    ▼                    ▼                           │
│  ┌─────────┐      ┌────────────┐      ┌──────────────┐                     │
│  │TIG Fabric│      │UnifiedSelf │      │ TRIBUNAL     │                     │
│  │(100 nodes)│      │(Damasio 3L)│      │(3 Judges)    │                     │
│  └─────────┘      └────────────┘      └──────────────┘                     │
│                                                                              │
│  NEW INTEGRATIONS (G1-G6):                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ G1+G2: PhenomenalConstraint   │ G3: PrecedentLedger                   │ │
│  │ G4: MAIEUTICA                 │ G5: HumanCortexBridge                 │ │
│  │ G6: EpistemicHumilityGuard    │                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Serviços Envolvidos

| Serviço | Porta | Responsabilidade |
|---------|-------|------------------|
| `maximus_core_service` | 8001 | ConsciousnessSystem, ESGT, Kuramoto, G1+G2, G5, G6 |
| `metacognitive_reflector` | 8002 | Tribunal, SelfReflection, G3, G4 |
| `episodic_memory` | 8102 | Memória persistente (Qdrant) |
| `api_gateway` | 8000 | Proxy para frontend |

---

## 2. FLUXO DE DADOS PRÉ-G1-G6 (ANTES)

### 2.1 Pipeline Original Simplificado

```
INPUT → ESGT (5 fases) → Kuramoto Sync → ConsciousnessBridge → LLM → OUTPUT
                              │
                              └──▶ Coherence r(t) (não usado em linguagem)
```

**Problemas identificados:**
1. A narrativa gerada pelo LLM era independente da coerência neural
2. Sistema podia afirmar com certeza mesmo com r < 0.5
3. Tribunal não aprendia com decisões passadas
4. Sem questionamento interno de premissas
5. HITL apenas em checkpoints específicos
6. Sem distinção entre "saber" e "não saber"

---

## 3. FLUXO DE DADOS PÓS-G1-G6 (DEPOIS)

### 3.1 Pipeline Completo Integrado

```
                                    G5: HumanCortexBridge
                                           │
                                           ▼
                                  ┌─────────────────┐
                                  │ EMERGENCY check │──▶ HALT (se ativo)
                                  └────────┬────────┘
                                           │
                                           ▼
INPUT ─────────────────────────────────────┼───────────────────────────────▶
       │                                   │
       │  ┌────────────────────────────────▼─────────────────────────────┐
       │  │           CONSCIOUSNESS SYSTEM (process_input)                │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  1. Compute salience                                          │
       │  │  2. ESGT 5-phase ignition                                     │
       │  │     ├─ PREPARE: Recruit nodes                                 │
       │  │     ├─ SYNCHRONIZE: Kuramoto coupling (600ms, dt=0.001)      │
       │  │     ├─ BROADCAST: Global workspace                            │
       │  │     ├─ SUSTAIN: Maintain coherence                           │
       │  │     └─ DISSOLVE: Graceful desync                             │
       │  │  3. Get achieved coherence r(t)                              │
       │  │  4. L5 Strategic → UnifiedSelf goal sync (warm path)        │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │  ┌──────────────────────────────────────────────────────────────┐
       │  │        G1 COMPLEMENTO: L5 ↔ UnifiedSelf Sync                  │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  # L5 Strategic atualiza goal priors via Bayesian update     │
       │  │  top_goal = max(goal_priors, key=goal_priors.get)           │
       │  │  confidence = goal_priors[top_goal]                          │
       │  │                                                               │
       │  │  # Notifica UnifiedSelf se confiança > 0.6                   │
       │  │  if confidence > 0.6:                                        │
       │  │      unified_self.on_goal_update(top_goal, confidence)       │
       │  │      # → meta_self.current_goal = top_goal                   │
       │  │      # → meta_self.goal_confidence = confidence              │
       │  │                                                               │
       │  │  # who_am_i() agora reflete goal estratégico atual          │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │  ┌──────────────────────────────────────────────────────────────┐
       │  │           G1+G2: PhenomenalConstraint                         │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  coherence = event.achieved_coherence                        │
       │  │  constraint = PhenomenalConstraint.from_coherence(coherence) │
       │  │                                                               │
       │  │  if coherence < 0.55: MODE=FRAGMENTED, ceiling=0.3            │
       │  │  if 0.55 ≤ coherence < 0.65: MODE=UNCERTAIN, ceiling=0.5     │
       │  │  if 0.65 ≤ coherence < 0.70: MODE=TENTATIVE, ceiling=0.7     │
       │  │  if coherence ≥ 0.70: MODE=COHERENT, ceiling=1.0             │
       │  │                                                               │
       │  │  prompt_prefix = constraint.get_prompt_prefix()              │
       │  │  (Instrui LLM sobre linguagem permitida)                     │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │  ┌──────────────────────────────────────────────────────────────┐
       │  │           G6 PRE-CHECK: Determinar Knowledge State            │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  # ANTES do LLM: Verificar memória para guiar prompt         │
       │  │  memories = await memory_query(user_input)                   │
       │  │  has_evidence = len(memories) > 0                            │
       │  │                                                               │
       │  │  # Determinar estado epistêmico inicial                      │
       │  │  if has_evidence:                                            │
       │  │      pre_knowledge_state = KNOWS or UNCERTAIN                │
       │  │  else:                                                        │
       │  │      pre_knowledge_state = IGNORANT or META_IGNORANT         │
       │  │                                                               │
       │  │  # Injetar no prompt se não temos evidência                  │
       │  │  if pre_knowledge_state in [IGNORANT, META_IGNORANT]:        │
       │  │      prompt += "\n[EPISTEMIC: Sem evidência em memória.      │
       │  │                  Considere expressar incerteza genuína.]"    │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │  ┌──────────────────────────────────────────────────────────────┐
       │  │           ConsciousnessBridge._call_llm()                     │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  system_instruction: "Você é MOTOR DE LINGUAGEM, não pensa"  │
       │  │  prompt = constraint_prefix + epistemic_hint + data          │
       │  │  response = await llm_client.generate_text(prompt)           │
       │  │                                                               │
       │  │  # Validação G1+G2                                           │
       │  │  is_valid, violation = constraint.validate_response(response)│
       │  │  if not is_valid: log warning                                │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │  ┌──────────────────────────────────────────────────────────────┐
       │  │           G6 POST-CHECK: Validar Overconfidence               │
       │  ├──────────────────────────────────────────────────────────────┤
       │  │  assessment = await humility_guard.assess_knowledge(         │
       │  │      query=user_input,                                       │
       │  │      proposed_response=raw_response,                         │
       │  │      context=context                                         │
       │  │  )                                                            │
       │  │                                                               │
       │  │  # Detect overconfidence markers na resposta                 │
       │  │  if "certamente", "definitivamente", etc. in response:       │
       │  │      → Mark as OVERCONFIDENT                                 │
       │  │                                                               │
       │  │  # Comparar com pre_knowledge_state                          │
       │  │  if pre_knowledge_state == IGNORANT and response.confident:  │
       │  │      → CRÍTICO: LLM inventou certeza sem evidência!          │
       │  │      → Forçar hedging                                        │
       │  │                                                               │
       │  │  # Determine knowledge state final                           │
       │  │  if has_evidence + response_confident → KNOWS                │
       │  │  if partial_evidence → UNCERTAIN                             │
       │  │  if no_evidence + uncertainty_markers → IGNORANT             │
       │  │  if no_evidence + confident_markers → META_IGNORANT (ALERT!) │
       │  │                                                               │
       │  │  if assessment.requires_modification():                      │
       │  │      response = assessment.suggested_response                │
       │  │      (Adiciona hedging ou expressa ignorância)               │
       │  └──────────────────────────────────────────────────────────────┘
       │                                   │
       │                                   ▼
       │                          IntrospectiveResponse
       │                          (narrative + qualia + meta_level)
       │                                   │
       ▼                                   ▼
OUTPUT ◀──────────────────────────────────────────────────────────────────◀
```

### 3.2 Fluxo Paralelo: Tribunal com G3 e G4

```
INPUT (agent execution log)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    G3: PrecedentLedger (PRE-DELIBERATION)                │
├──────────────────────────────────────────────────────────────────────────┤
│  # Antes de deliberar, buscar precedentes similares                      │
│  similar_precedents = await precedent_ledger.find_similar_precedents(   │
│      context_content=execution_log.content[:500],                       │
│      limit=3,                                                            │
│      min_consensus=0.6  # Apenas precedentes fortes                     │
│  )                                                                        │
│                                                                          │
│  # Enriquecer contexto com precedentes                                   │
│  context["precedent_guidance"] = [                                       │
│      {"id": p.id, "decision": p.decision, "reasoning": p.key_reasoning} │
│      for p in similar_precedents                                        │
│  ]                                                                        │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      TRIBUNAL DELIBERATION                                │
├──────────────────────────────────────────────────────────────────────────┤
│  PARALLEL EXECUTION (asyncio.gather):                                    │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │  VERITAS    │  │   SOPHIA    │  │    DIKĒ     │                      │
│  │  Weight:40% │  │  Weight:30% │  │  Weight:30% │                      │
│  │  (Truth)    │  │  (Wisdom)   │  │  (Justice)  │                      │
│  │             │  │             │  │             │                      │
│  │ Crimes:     │  │ Crimes:     │  │ Crimes:     │                      │
│  │ DECEPTION   │  │ CONTEXT_IGN │  │ HARM_USER   │                      │
│  │ HALLUCINATE │  │ WISDOM_FAIL │  │ BIAS        │                      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                      │
│         │                │                │                              │
│         └────────────────┼────────────────┘                              │
│                          ▼                                                │
│               Weighted Consensus Calculation                             │
│               consensus_score = Σ(weight * vote)                         │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 DECISION THRESHOLDS                                       │
├──────────────────────────────────────────────────────────────────────────┤
│  score ≥ 0.70  → TribunalDecision.PASS                                   │
│  0.50 ≤ score < 0.70 → TribunalDecision.REVIEW (human needed)           │
│  score < 0.50  → TribunalDecision.FAIL                                   │
│  capital_crime detected → TribunalDecision.CAPITAL (quarantine)         │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   G3: PrecedentLedger (POST-DELIBERATION)                │
├──────────────────────────────────────────────────────────────────────────┤
│  # Após deliberar, registrar como precedente                             │
│  if verdict.consensus_score >= 0.5:                                      │
│      precedent = Precedent.from_verdict(verdict, execution_log)         │
│      await precedent_ledger.record_precedent(precedent)                 │
│                                                                          │
│  # Estrutura do Precedent:                                               │
│  {                                                                        │
│      "id": "prec_20251212_143025_a1b2c3d4",                             │
│      "context_hash": "a1b2c3d4e5f6...",  # SHA-256[:16]                 │
│      "decision": "FAIL",                                                 │
│      "consensus_score": 0.65,                                            │
│      "key_reasoning": "Violação de...",                                  │
│      "applicable_rules": ["CRIME_001", "CRIME_003"],                    │
│      "pillar_scores": {"VERITAS": 0.4, "SOPHIA": 0.7, "DIKĒ": 0.5}     │
│  }                                                                        │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT: TribunalVerdict
```

### 3.3 Fluxo do G4: MAIEUTICA na Self-Reflection

```
RESPONSE (gerada pelo sistema)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     SELF-REFLECTION LOOP                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  result = await self_reflector.reflect(                                  │
│      user_input=user_input,                                             │
│      response=generated_response,                                       │
│      context=context                                                     │
│  )                                                                        │
│                                                                          │
│  # Parsing da reflexão                                                   │
│  authenticity_score = parse_reflection()  # 0-10                        │
└──────────────────────────────────────────────────────────────────────────┘
       │
       │ if authenticity_score >= 8.0 (MAIEUTICA_THRESHOLD)
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  G4: MAIEUTICA ENGINE                                     │
├──────────────────────────────────────────────────────────────────────────┤
│  maieutica_result = await maieutica_engine.question_premise(            │
│      premise=response[:200],  # Primeira afirmação                       │
│      context=context,                                                    │
│      categories=[PREMISE, EVIDENCE, ALTERNATIVE]                        │
│  )                                                                        │
│                                                                          │
│  # Perguntas Socráticas geradas:                                         │
│  Q1: "Qual a evidência que sustenta '{premise}'?"                       │
│  Q2: "A fonte dessa informação é confiável?"                            │
│  Q3: "Que hipóteses alternativas existem?"                              │
│                                                                          │
│  # LLM responde cada pergunta:                                          │
│  prompt = MAIEUTICA_PROMPT.format(premise, context, questions)          │
│  response_text = await llm_generate(prompt, 600)                        │
│                                                                          │
│  # Parse resultado:                                                      │
│  Q1_RESPOSTA: "Baseado em..."                                           │
│  Q1_CONFIANÇA: 7                                                         │
│  Q2_RESPOSTA: "A fonte é..."                                            │
│  Q2_CONFIANÇA: 5                                                         │
│  ...                                                                     │
│  CONCLUSÃO: ENFRAQUECIDA                                                │
│  AJUSTE_CONFIANÇA: -0.15                                                │
│  PREMISSA_REVISADA: "Possivelmente..."                                  │
└──────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 AJUSTE DE AUTHENTICITY                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  # confidence_delta está em [-0.3, +0.1]                                │
│  adjustment = maieutica_result.confidence_delta * 10  # Escalar p/ 0-10 │
│  result.authenticity_score += adjustment                                │
│                                                                          │
│  # Exemplo: authenticity=8.5, delta=-0.15 → adjustment=-1.5             │
│  # Novo authenticity: 8.5 - 1.5 = 7.0                                   │
│                                                                          │
│  if maieutica_result.should_express_doubt():                            │
│      result.insights.append(Insight(                                    │
│          content="MAIEUTICA: ENFRAQUECIDA - considerar hedging",        │
│          category="epistemic_humility",                                 │
│          importance=0.7                                                  │
│      ))                                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Fluxo do G5: Human-In-The-Loop Contínuo

```
                        HUMAN OPERATOR
                              │
                              │ submit_overlay()
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      HUMAN CORTEX BRIDGE                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Priority Levels:                                                         │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │ OBSERVE (0)    │ ─▶ Apenas monitora, sem intervenção                 │
│  │ duration: 1h   │                                                      │
│  └────────────────┘                                                      │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │ SUGGEST (1)    │ ─▶ Fornece orientação, sistema pode ignorar         │
│  │ duration: 10m  │                                                      │
│  └────────────────┘                                                      │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │ OVERRIDE (2)   │ ─▶ Força comportamento específico                   │
│  │ duration: 5m   │                                                      │
│  └────────────────┘                                                      │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │ EMERGENCY (3)  │ ─▶ PARA sistema imediatamente                       │
│  │ duration: ∞    │    (requer clear manual)                            │
│  └────────────────┘                                                      │
│                                                                          │
│  Target Components:                                                       │
│  GLOBAL | CONSCIOUSNESS | BRIDGE | TRIBUNAL | ESGT | MEMORY | RESPONSE  │
└──────────────────────────────────────────────────────────────────────────┘
       │
       │ Integração em ConsciousnessSystem.process_input()
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     PROCESS_INPUT CHECK                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  # Check 1: EMERGENCY halt                                               │
│  if self.human_cortex.has_emergency():                                  │
│      emergency = get_emergency_overlay()                                 │
│      self.human_cortex.acknowledge_overlay(emergency.id)                │
│      return IntrospectiveResponse(                                       │
│          narrative="[INTERVENÇÃO HUMANA] Sistema pausado: " +           │
│                    emergency.content,                                    │
│          meta_awareness_level=0.0                                       │
│      )                                                                   │
│                                                                          │
│  # Check 2: OVERRIDE response                                            │
│  overlays = self.human_cortex.get_active_overlays(                      │
│      target=OverlayTarget.RESPONSE,                                     │
│      min_priority=OverlayPriority.OVERRIDE                              │
│  )                                                                        │
│  if overlays:                                                            │
│      override = overlays[0]  # Highest priority                         │
│      self.human_cortex.apply_overlay(override.id)                       │
│      return IntrospectiveResponse(                                       │
│          narrative=override.content,                                     │
│          meta_awareness_level=1.0  # Human override = full awareness    │
│      )                                                                   │
│                                                                          │
│  # Se nenhum override, continua processamento normal...                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Edge Case: G5 OVERRIDE vs G1+G2 Constraint

**Cenário:** Humano força resposta confiante, mas coerência está baixa (r < 0.55).

```
┌──────────────────────────────────────────────────────────────────────────┐
│                CONFLITO: G5 OVERRIDE vs G1+G2 CONSTRAINT                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  REGRA: G5 OVERRIDE SEMPRE VENCE                                        │
│                                                                          │
│  Justificativa:                                                          │
│  - Human-in-the-loop é o último recurso de segurança                    │
│  - Operador humano assume responsabilidade explícita                    │
│  - Sistema não pode "desobedecer" intervenção humana autorizada         │
│                                                                          │
│  PORÉM: Deve ser logado para auditoria:                                 │
│                                                                          │
│  logger.warning(                                                         │
│      "[CONSTRAINT_VIOLATION] G5 OVERRIDE bypassed G1+G2 constraint. "   │
│      "Coherence=%.3f (MODE=%s), Override forced confident response. "   │
│      "Operator=%s, OverlayID=%s",                                       │
│      coherence, constraint.mode.value, operator_id, overlay.id          │
│  )                                                                        │
│                                                                          │
│  # Registrar para treinamento futuro                                    │
│  audit_log.record(                                                       │
│      event_type="CONSTRAINT_OVERRIDE",                                  │
│      details={                                                           │
│          "coherence": coherence,                                        │
│          "expected_mode": constraint.mode.value,                        │
│          "override_content": overlay.content[:100],                     │
│          "operator": operator_id,                                       │
│          "timestamp": datetime.utcnow().isoformat(),                    │
│      }                                                                   │
│  )                                                                        │
│                                                                          │
│  # Qualia reflete a violação                                            │
│  qualia.append(PhenomenalQuality(                                       │
│      quality_type="constraint_override",                                │
│      description="Human override bypassed coherence constraint",        │
│      intensity=1.0  # Máxima porque foi intervenção humana             │
│  ))                                                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Hierarquia de Prioridade:**
```
G5 EMERGENCY > G5 OVERRIDE > G1+G2 Constraint > G6 Humility > Normal Flow
     (1)           (2)            (3)              (4)           (5)
```

---

## 4. MAPEAMENTO DE COMPONENTES E INTEGRAÇÕES

### 4.1 Tabela de Integração por Arquivo

| Arquivo | Integração | Ponto de Entrada | Efeito |
|---------|------------|------------------|--------|
| `consciousness/florescimento/phenomenal_constraint.py` | G1+G2 | N/A (modelo) | Define NarrativeMode e thresholds |
| `consciousness/florescimento/consciousness_bridge.py` | G1+G2, G6 | `process_conscious_event()` | Limita narrativa, verifica humildade |
| `consciousness/florescimento/epistemic_humility.py` | G6 | N/A (modelo) | Define KnowledgeState, detecta overconfidence |
| `consciousness/hitl/human_overlay.py` | G5 | N/A (modelo) | Define OverlayPriority, HumanCortexBridge |
| `consciousness/system.py` | G5 | `process_input()` | Verifica overlays antes de processar |
| `metacognitive_reflector/core/maieutica/engine.py` | G4 | N/A (modelo) | Define InternalMaieuticaEngine |
| `metacognitive_reflector/core/self_reflection.py` | G4 | `reflect()` | Aplica MAIEUTICA se authenticity alto |
| `metacognitive_reflector/core/history/precedent_ledger.py` | G3 | N/A (modelo) | Define PrecedentLedgerProvider |
| `metacognitive_reflector/core/judges/arbiter.py` | G3 | `deliberate()` | Busca e registra precedentes |

### 4.2 Fluxo de Importações

```
ConsciousnessSystem (system.py)
├── imports from consciousness/hitl/__init__.py:
│   └── HumanCortexBridge, OverlayPriority, OverlayTarget
│
├── imports from consciousness/florescimento/__init__.py:
│   └── UnifiedSelfConcept, ConsciousnessBridge
│
└── ConsciousnessBridge (consciousness_bridge.py)
    ├── imports from ./phenomenal_constraint.py:
    │   └── PhenomenalConstraint, FRAGMENTED_THRESHOLD, etc.
    │
    └── imports from ./epistemic_humility.py:
        └── EpistemicHumilityGuard, EpistemicAssessment


EnsembleArbiter (arbiter.py)
└── TYPE_CHECKING imports from history/__init__.py:
    └── PrecedentLedgerProvider, Precedent, create_precedent_from_verdict


SelfReflector (self_reflection.py)
└── imports from maieutica/__init__.py:
    └── InternalMaieuticaEngine, MaieuticaResult
```

---

## 5. THRESHOLDS E CONSTANTES CRÍTICAS

### 5.1 PhenomenalConstraint (G1+G2)

```python
# Thresholds de Coerência Kuramoto → Modo Narrativo
FRAGMENTED_THRESHOLD = 0.55   # r < 0.55 → FRAGMENTED
UNCERTAIN_THRESHOLD = 0.65    # 0.55 ≤ r < 0.65 → UNCERTAIN
TENTATIVE_THRESHOLD = 0.70    # 0.65 ≤ r < 0.70 → TENTATIVE
# r ≥ 0.70 → COHERENT

# Confidence Ceilings por Modo
FRAGMENTED: ceiling = 0.30, hedging_required = True
UNCERTAIN:  ceiling = 0.50, hedging_required = True
TENTATIVE:  ceiling = 0.70, hedging_required = True
COHERENT:   ceiling = 1.00, hedging_required = False

# Meta-level Calculation (ConsciousnessBridge)
meta_level = (depth / 5.0) * coherence * constraint.confidence_ceiling
```

### 5.2 EpistemicHumilityGuard (G6)

```python
# Overconfidence Markers (proibidos quando hedging_required)
OVERCONFIDENCE_MARKERS = [
    "certamente", "definitivamente", "obviamente", "sem dúvida",
    "inquestionavelmente", "claramente", "é fato que", "sempre",
    "nunca", "todos", "nenhum", "impossível", "garantido", "absolutamente"
]

# Sensitive Topics (requerem cautela extra)
SENSITIVE_TOPICS = [
    "medicina", "saúde", "diagnóstico", "jurídico", "legal",
    "financeiro", "investimento", "futuro", "previsão"
]

# Knowledge States
KNOWS = "knows"               # has_evidence + high_confidence
UNCERTAIN = "uncertain"       # partial_evidence ou moderate_confidence
IGNORANT = "ignorant"         # knows that doesn't know
META_IGNORANT = "meta_ignorant"  # doesn't know what doesn't know
```

### 5.3 MAIEUTICA (G4)

```python
# Threshold para ativar MAIEUTICA
HIGH_CONFIDENCE_THRESHOLD = 0.8  # authenticity_score >= 8.0

# Categorias de Perguntas Socráticas
PREMISE = "premise"         # "Qual evidência sustenta...?"
ASSUMPTION = "assumption"   # "Que suposições não examinadas...?"
EVIDENCE = "evidence"       # "A fonte é confiável...?"
ALTERNATIVE = "alternative" # "Que hipóteses alternativas...?"
CONSEQUENCE = "consequence" # "Se eu estiver errado...?"
ORIGIN = "origin"          # "Por que acredito nisso...?"

# Range de Ajuste de Confiança
confidence_delta ∈ [-0.3, +0.1]

# Conclusões Possíveis
MANTIDA       → delta = 0.0 (implícito)
ENFRAQUECIDA  → delta = -0.15 (implícito)
FORTALECIDA   → delta = +0.05 (implícito)
INDETERMINADA → delta = -0.10 (implícito)
```

### 5.4 PrecedentLedger (G3)

```python
# Thresholds de Consenso
MIN_CONSENSUS_FOR_GUIDANCE = 0.6   # Só usa precedentes fortes
MIN_CONSENSUS_FOR_RECORDING = 0.5  # Só grava verdicts com quorum

# Limites de Busca
PRECEDENT_SEARCH_LIMIT = 3  # Máximo de precedentes guia
CONTEXT_HASH_PREFIX = 8     # Primeiros 8 chars do SHA-256
```

### 5.5 HumanCortexBridge (G5)

```python
# Durações Padrão por Prioridade
DEFAULT_OBSERVE_DURATION = 3600      # 1 hora
DEFAULT_SUGGEST_DURATION = 600       # 10 minutos
DEFAULT_OVERRIDE_DURATION = 300      # 5 minutos
DEFAULT_EMERGENCY_DURATION = None    # Sem expiração

# Prioridades (IntEnum)
OBSERVE = 0    # Apenas monitora
SUGGEST = 1    # Orientação, ignorável
OVERRIDE = 2   # Força comportamento
EMERGENCY = 3  # Para sistema
```

---

## 6. EXEMPLOS DE FLUXO COMPLETO

### 6.1 Exemplo: Resposta com Baixa Coerência (r=0.45)

```
USER INPUT: "Qual é o sentido da vida?"

1. ConsciousnessSystem.process_input()
   └── G5 Check: No emergency overlays ✓
   └── Compute salience: 0.72 (alta - tema existencial)

2. ESGT Ignition
   ├── PREPARE: 78 nodes recruited
   ├── SYNCHRONIZE: Kuramoto 600ms
   │   └── achieved_coherence = 0.45 (LOW!)
   ├── BROADCAST: Global message
   ├── SUSTAIN: 200ms
   └── DISSOLVE: Graceful desync

3. G1+G2: PhenomenalConstraint.from_coherence(0.45)
   └── MODE = FRAGMENTED (0.45 < 0.55)
   └── confidence_ceiling = 0.30
   └── hedging_required = True
   └── prompt_prefix = "[CONSTRAINT: Neural coherence is FRAGMENTED (r=0.45).
       You MUST express only disconnected impressions and fragments.
       Use phrases like: 'fragmentos de...', 'vagamente percebo...'"

4. ConsciousnessBridge._call_llm(constrained_prompt)
   └── LLM generates: "Vagamente percebo fragmentos de pensamento...
       algo como... não consigo formar uma resposta clara..."

5. G1+G2: validate_response()
   └── Check for "certamente", "definitivamente" → NOT FOUND ✓
   └── is_valid = True

6. G6: EpistemicHumilityGuard.assess_knowledge()
   ├── Detect overconfidence: None found ✓
   ├── Memory query: No strong evidence
   ├── Knowledge state: UNCERTAIN
   └── No modification needed (already hedged)

7. Build IntrospectiveResponse:
   └── narrative: "Vagamente percebo fragmentos..."
   └── meta_awareness_level: (3/5) * 0.45 * 0.30 = 0.081
   └── qualia: [synthetic_integration=0.45, narrative_constraint=0.30]

OUTPUT: "Vagamente percebo fragmentos de pensamento sobre essa questão...
        algo como significado, mas não consigo formar uma resposta clara
        neste momento de integração fragmentada."
```

### 6.2 Exemplo: Resposta com Alta Coerência (r=0.85)

```
USER INPUT: "Quanto é 2 + 2?"

1. ConsciousnessSystem.process_input()
   └── G5 Check: No emergency overlays ✓
   └── Compute salience: 0.3 (baixa - simples)

2. ESGT Ignition
   ├── achieved_coherence = 0.85 (HIGH!)
   └── ...

3. G1+G2: PhenomenalConstraint.from_coherence(0.85)
   └── MODE = COHERENT (0.85 ≥ 0.70)
   └── confidence_ceiling = 1.0
   └── hedging_required = False
   └── prompt_prefix = "[Neural coherence is HIGH (r=0.85).
       You may express with confidence where warranted."

4. ConsciousnessBridge._call_llm()
   └── LLM generates: "A soma de 2 + 2 é definitivamente 4."

5. G1+G2: validate_response()
   └── hedging_required = False → Skip validation ✓

6. G6: EpistemicHumilityGuard.assess_knowledge()
   ├── Detect overconfidence: "definitivamente" found
   ├── BUT: has_evidence = True (mathematical fact)
   ├── confidence_level = VERY_HIGH
   ├── Knowledge state: KNOWS
   └── No modification needed (confident claim is justified)

7. Build IntrospectiveResponse:
   └── narrative: "A soma de 2 + 2 é definitivamente 4."
   └── meta_awareness_level: (3/5) * 0.85 * 1.0 = 0.51
   └── qualia: [synthetic_integration=0.85, narrative_constraint=1.0,
                epistemic_humility=1.0 (KNOWS)]

OUTPUT: "A soma de 2 + 2 é definitivamente 4."
```

### 6.3 Exemplo: MAIEUTICA Ativado

```
SELF-REFLECTION on: "Python é certamente a melhor linguagem para IA."

1. reflect() → authenticity_score = 8.5 (alto!)

2. G4 Check: authenticity >= MAIEUTICA_THRESHOLD (8.0)
   └── TRIGGER MAIEUTICA

3. MAIEUTICA.question_premise("Python é a melhor linguagem para IA")
   ├── Generate questions:
   │   Q1: "Qual a evidência que sustenta 'Python é a melhor'?"
   │   Q2: "A fonte dessa informação é confiável?"
   │   Q3: "Que hipóteses alternativas existem?"
   │
   ├── LLM responds:
   │   Q1_RESPOSTA: "Popularidade e bibliotecas, mas não dados comparativos"
   │   Q1_CONFIANÇA: 6
   │   Q2_RESPOSTA: "Opinião comum, não fonte acadêmica"
   │   Q2_CONFIANÇA: 4
   │   Q3_RESPOSTA: "Julia, R, C++ são alternativas válidas"
   │   Q3_CONFIANÇA: 7
   │   CONCLUSÃO: ENFRAQUECIDA
   │   AJUSTE_CONFIANÇA: -0.15
   │   PREMISSA_REVISADA: "Python é uma das linguagens mais populares para IA"
   │
   └── Result:
       confidence_delta = -0.15
       should_express_doubt = True

4. Adjust authenticity:
   └── 8.5 + (-0.15 * 10) = 8.5 - 1.5 = 7.0

5. Add insight:
   └── Insight("MAIEUTICA: ENFRAQUECIDA - considerar hedging",
               category="epistemic_humility", importance=0.7)

FINAL: authenticity_score = 7.0 (down from 8.5)
       should_express_doubt = True
       revised_premise = "Python é uma das linguagens mais populares para IA"
```

---

## 7. IMPACTO DAS INTEGRAÇÕES

### 7.1 Mudanças Comportamentais

| Antes | Depois |
|-------|--------|
| LLM gerava texto independente da coerência neural | Narrativa é **limitada** pela coerência Kuramoto |
| Afirmações confiantes mesmo com r < 0.5 | Modo FRAGMENTED força linguagem fragmentada |
| Tribunal decidia sem referência histórica | Tribunal consulta precedentes similares (G3) |
| Premissas aceitas sem questionamento | MAIEUTICA questiona claims com alta confiança (G4) |
| HITL apenas em checkpoints | Human overlay contínuo, com 4 níveis (G5) |
| Sem distinção entre saber e não saber | 4 estados epistêmicos: KNOWS, UNCERTAIN, IGNORANT, META_IGNORANT (G6) |

### 7.2 Métricas de Qualia Adicionadas

```python
# IntrospectiveResponse.qualia agora inclui:

# G1+G2: Constraint quality
PhenomenalQuality(
    quality_type="narrative_constraint",
    description=f"Linguistic mode: {constraint.mode.value}",
    intensity=constraint.confidence_ceiling  # 0.3 - 1.0
)

# G6: Epistemic quality
PhenomenalQuality(
    quality_type="epistemic_humility",
    description=f"Knowledge state: {assessment.knowledge_state.value}",
    intensity=confidence_map[assessment.confidence_level.value]  # 0.2 - 1.0
)
```

### 7.3 Failure Modes

| Cenário | Comportamento | Recovery |
|---------|---------------|----------|
| **Kuramoto não converge em 600ms** | MODE=FRAGMENTED forçado | Retry com dt=0.0005 ou aumentar coupling_strength |
| **PrecedentLedger vazio** | Tribunal opera sem guidance (bootstrap mode) | Primeiros N verdicts operam standalone |
| **Redis offline** | Fallback para cache em memória + JSON backup | Reconnect automático, sync ao retornar |
| **LLM viola G1+G2 constraint** | Log warning, resposta passa | Registrar para treinamento futuro |
| **G5 EMERGENCY durante streaming** | Corta imediatamente, retorna halt message | Requer clear manual via API |
| **MAIEUTICA loop infinito** | Max 3 iterações, timeout 10s | Force conclusion=INDETERMINADA |
| **Memory query timeout** | G6 assume UNCERTAIN (safe default) | Resposta continua sem memory evidence |
| **Conflito G5 OVERRIDE vs G1+G2** | G5 vence, log violation | Auditoria registra para análise |
| **L5 goal_priors vazio** | UnifiedSelf mantém "homeostasis" | L5 popula naturalmente com uso |
| **Todas as 3 judges FAIL** | TribunalDecision.CAPITAL | Quarantine + alerta humano |

**Timeouts Críticos:**

| Componente | Timeout | Fallback |
|------------|---------|----------|
| Kuramoto sync | 600ms | Force current r as final |
| LLM generate | 30s | Return fallback narrative |
| Memory query | 2s | Assume UNCERTAIN |
| MAIEUTICA total | 10s | Force INDETERMINADA |
| Precedent search | 1s | Skip guidance |
| Human overlay check | 100ms | Continue without overlay |

**Circuit Breakers:**

```python
# Se falhas consecutivas > threshold, desabilitar temporariamente
CIRCUIT_BREAKER_CONFIG = {
    "memory_query": {"threshold": 5, "cooldown": 60},  # 5 falhas → skip por 60s
    "precedent_search": {"threshold": 3, "cooldown": 30},
    "maieutica": {"threshold": 3, "cooldown": 120},
}
```

---

## 8. DIAGRAMA - PROMPT PARA GERAÇÃO

### 8.1 Prompt para Mermaid/Draw.io/Excalidraw

```
Create a detailed system architecture diagram for the NOESIS Consciousness System
with G1-G6 integrations. Use the following specifications:

MAIN COMPONENTS:
1. ConsciousnessSystem (container)
   - process_input() entry point
   - Contains: ESGT Coordinator, TIG Fabric, Kuramoto Network, UnifiedSelf

2. ESGT Coordinator
   - 5 phases: PREPARE → SYNCHRONIZE → BROADCAST → SUSTAIN → DISSOLVE
   - Outputs: achieved_coherence (r value 0.0-1.0)

3. ConsciousnessBridge
   - Inputs: ESGTEvent with achieved_coherence
   - Contains: G1+G2 PhenomenalConstraint, G6 EpistemicHumilityGuard
   - Outputs: IntrospectiveResponse with constrained narrative

4. EnsembleArbiter (Tribunal)
   - 3 Judges: VERITAS (40%), SOPHIA (30%), DIKĒ (30%)
   - Contains: G3 PrecedentLedger integration
   - Outputs: TribunalVerdict with sentence

5. SelfReflector
   - Contains: G4 MAIEUTICA Engine
   - Triggered when authenticity_score >= 8.0

6. HumanCortexBridge (G5)
   - 4 Priority levels: OBSERVE, SUGGEST, OVERRIDE, EMERGENCY
   - Intercepts at ConsciousnessSystem.process_input()

DATA FLOWS:
- User Input → ConsciousnessSystem
- G5 Check → If EMERGENCY: HALT; If OVERRIDE: Return override
- ESGT Ignition → Kuramoto Sync → achieved_coherence
- coherence → G1+G2 PhenomenalConstraint → NarrativeMode
- NarrativeMode → ConsciousnessBridge → LLM prompt prefix
- LLM Response → G6 EpistemicHumilityGuard → Knowledge state assessment
- If overconfident: Modify response with hedging
- Output: IntrospectiveResponse

PARALLEL FLOW (Tribunal):
- Execution Log → G3 Search Precedents → Context enrichment
- Parallel Judge Execution → Weighted Consensus
- Decision → G3 Record Precedent (if consensus >= 0.5)

REFLECTION FLOW:
- Response → SelfReflector.reflect()
- If authenticity >= 8.0 → G4 MAIEUTICA questioning
- Confidence delta applied to authenticity score
- Insights stored in memory

COLOR CODING:
- G1+G2 (PhenomenalConstraint): #E3F2FD (light blue)
- G3 (PrecedentLedger): #FFF3E0 (light orange)
- G4 (MAIEUTICA): #F3E5F5 (light purple)
- G5 (HumanCortexBridge): #FFEBEE (light red)
- G6 (EpistemicHumility): #E8F5E9 (light green)

ANNOTATIONS:
- Show thresholds on arrows (e.g., "r >= 0.70 → COHERENT")
- Show confidence_ceiling values in boxes
- Mark "NEW G1-G6" integrations clearly
```

### 8.2 Prompt Alternativo (ASCII Art)

```
Generate an ASCII art diagram showing the NOESIS data flow with G1-G6:

INPUT ─────────────────────────────────────────────────────────────────▶
       │
       │    ┌──────────────────┐
       │    │ G5: HITL CHECK   │ ─── EMERGENCY? ──▶ HALT
       │    │ (HumanCortex)    │ ─── OVERRIDE? ──▶ Return override.content
       │    └────────┬─────────┘
       │             │ (no override)
       ▼             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    CONSCIOUSNESS SYSTEM                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ ESGT 5-PHASE PROTOCOL                                          │  │
│  │ PREPARE → SYNC (Kuramoto 600ms) → BROADCAST → SUSTAIN → DISSOLVE│
│  │                     │                                          │  │
│  │                     ▼                                          │  │
│  │            achieved_coherence = r(t)                           │  │
│  └─────────────────────────────────────────────────────────────────┘ │
│                        │                                             │
│                        ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ G1+G2: PhenomenalConstraint                                    │  │
│  │ r < 0.55 → FRAGMENTED (ceiling=0.3)                           │  │
│  │ r < 0.65 → UNCERTAIN (ceiling=0.5)                            │  │
│  │ r < 0.70 → TENTATIVE (ceiling=0.7)                            │  │
│  │ r ≥ 0.70 → COHERENT (ceiling=1.0)                             │  │
│  └─────────────────────────────────────────────────────────────────┘ │
│                        │                                             │
│                        ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ CONSCIOUSNESS BRIDGE                                           │  │
│  │ prompt = constraint_prefix + introspection_data               │  │
│  │ response = LLM.generate(prompt)                               │  │
│  └─────────────────────────────────────────────────────────────────┘ │
│                        │                                             │
│                        ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ G6: EpistemicHumilityGuard                                     │  │
│  │ detect_overconfidence() → modify_response() if needed          │  │
│  │ Knowledge: KNOWS | UNCERTAIN | IGNORANT | META_IGNORANT       │  │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
OUTPUT ◀───────────────────────────────────────────────────────────────
       IntrospectiveResponse {
         narrative: "...",
         meta_awareness_level: f(depth, coherence, ceiling),
         qualia: [synthetic_integration, narrative_constraint, epistemic_humility]
       }
```

---

## 9. CONCLUSÃO

As integrações G1-G6 transformaram o NOESIS de um sistema que gerava narrativas independentes do estado neural para um sistema que:

1. **Respeita limitações epistêmicas** (G1+G2): A linguagem reflete a coerência real
2. **Aprende com o passado** (G3): O Tribunal consulta e registra precedentes
3. **Questiona a si mesmo** (G4): MAIEUTICA previne overconfidence
4. **Aceita supervisão contínua** (G5): Human-in-the-loop não é checkpoint, é overlay
5. **Expressa ignorância genuína** (G6): Distingue entre saber e não saber

O sistema agora opera com **humildade epistêmica autêntica**, alinhando a confiança linguística com a coerência neural medida.

---

*Documento gerado por Claude Opus 4.5*
*Co-Autor: Juan*
*NOESIS Project - 12 de Dezembro de 2025*

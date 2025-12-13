# PLANO: Transformação DAIMON → Célula Híbrida Human-AI

**Data:** 2025-12-13 | **Versão:** 2.0
**Objetivo:** Transformar DAIMON de telemetria passiva em célula híbrida human-AI verdadeira

---

## DIAGNÓSTICO ATUAL

```
DAIMON hoje = 100% regex/heuristics + 0% AI
├── Collectors: capturam dados ✓
├── Learners: processam com regex ✗ (não entendem contexto)
├── NOESIS: existe mas não recebe dados ✗
└── Resultado: sistema REATIVO de 30min, não APRENDE, não ANTECIPA
```

---

## VISÃO ALVO

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CÉLULA HÍBRIDA HUMAN-AI                              │
│                                                                          │
│  COLLECTORS ──▶ NOESIS INGESTION ──▶ LLM ANALYSIS ──▶ PROACTIVE        │
│  (7 tipos)      (signals)            (semantic)       (anticipates)      │
│       │              │                    │                │             │
│       ▼              ▼                    ▼                ▼             │
│  ActivityStore   Episodic Memory    User Model       CLAUDE.md         │
│                                                       + Emergence        │
│                                                                          │
│  LLM Providers: [Nebius] [Gemini] [Haiku 3.5*]  (* NOVO)               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## FUNDAMENTAÇÃO (2025 Research)

| Conceito | Fonte | Aplicação |
|----------|-------|-----------|
| ExoNet Architecture | ACS Photonics 2025 | Human + múltiplas IAs = célula coesa |
| AgenticAI Learning | MDPI 2025 | AI evolui via experiência, antecipa |
| Hyper-personalization | Google 2025 | "pcontext" = análise comportamental profunda |
| Kuramoto Oscillators | NOESIS impl | Consciência emerge quando coerência r≥0.70 |

---

## SPRINTS

### SPRINT 0: Claude Haiku 3.5 Provider (1 dia)

**Objetivo:** Adicionar Haiku 3.5 como opção LLM (sem usar ainda)

**Specs Haiku 3.5:** $0.80/MTok in, $4/MTok out, 200K context, ~8K output

**Arquivos:**
```
/media/juan/DATA/projetos/Noesis/Daimon/backend/services/metacognitive_reflector/src/metacognitive_reflector/llm/
├── config.py      # + AnthropicConfig, LLMProvider.ANTHROPIC
├── client.py      # + _anthropic_chat() method
└── pyproject.toml # + anthropic>=0.40.0 optional
```

**Mudanças:**
1. Adicionar `LLMProvider.ANTHROPIC` ao enum
2. Criar `AnthropicConfig(model="claude-3-5-haiku-20241022")`
3. Implementar `_anthropic_chat()` usando `anthropic.AsyncAnthropic`
4. Testes com skip se sem API key

**Zero breaking changes** em código existente.

---

### SPRINT 1: NOESIS Data Ingestion (2-3 dias)

**Objetivo:** Pipeline DAIMON → NOESIS para sinais comportamentais

**Arquivos NOVOS:**
```
/media/juan/DATA/projetos/daimon/integrations/noesis_ingestion.py  (~300 linhas)
```

**Componentes:**
- `BehavioralSignal`: signal_type, source, timestamp, salience, data, context
- `DataIngestionService`: processa ActivityStore → envia sinais para NOESIS

**Endpoints NOESIS (adicionar):**
- `POST /v1/consciousness/ingest` - recebe sinais, alimenta Kuramoto
- `POST /v1/memory/episode` - armazena episódios

**Fluxo:**
```
ActivityStore → aggregate_to_signals() → filter(salience≥0.5) → NOESIS
```

**Tipos de sinal:**
| Watcher | Signal Type | Salience |
|---------|-------------|----------|
| claude rejection | preference | 0.8 |
| shell failed cmd | anomaly | 0.7 |
| risky command | anomaly | 0.9 |
| typing variance | cognitive_state | 0.6 |
| high context switch | pattern | 0.5 |

---

### SPRINT 2: LLM-Powered Learners (3-4 dias)

**Objetivo:** Substituir regex por análise LLM em 8 pontos críticos

**Arquivo NOVO:**
```
/media/juan/DATA/projetos/daimon/learners/llm_service.py  (~350 linhas)
```

**LearnerLLMService:**
- `classify(content, options)` - classificação semântica
- `extract_insights(data, context)` - geração de insights
- `analyze_cognitive_state(biometrics)` - interpretação de estado

**Features:** Caching (5min TTL), fallback para heurísticas, métricas

**Modificações em arquivos existentes:**

| Arquivo | Método | Atual | Com LLM |
|---------|--------|-------|---------|
| `learners/preference_learner.py` | `_detect_signal_type()` | regex | LLM classify |
| `learners/preference_learner.py` | `get_actionable_insights()` | template | LLM extract |
| `learners/keystroke_analyzer.py` | `detect_cognitive_state()` | thresholds | LLM analyze |

**Padrão:** Tentar LLM primeiro, fallback para heurística se falhar ou baixa confiança.

---

### SPRINT 3: Comportamento Proativo (3-4 dias)

**Objetivo:** NOESIS emerge proativamente baseado em padrões

**Arquivos NOVOS:**
```
/media/juan/DATA/projetos/daimon/learners/pattern_detector.py     (~250 linhas)
/media/juan/DATA/projetos/daimon/learners/anticipation_engine.py  (~200 linhas)
```

**PatternDetector:**
- Temporal: "usuário faz commit às 17h"
- Sequential: "git status → git add → git commit"
- Contextual: "em VSCode, prefere dark theme à noite"

**AnticipationEngine:**
- Avalia contexto atual contra patterns
- Decide se deve emergir (confidence ≥ 0.7)
- Modos: subtle (notification), normal (chat), urgent (immediate)

**Endpoint NOESIS:**
- `POST /v1/consciousness/emerge` - trigger proativo

**Cooldown:** 10 minutos entre emergências para evitar spam.

---

### SPRINT 4: User Model & Memory (2-3 dias)

**Objetivo:** Modelo persistente do usuário em NOESIS

**Arquivo NOVO:**
```
/media/juan/DATA/projetos/daimon/memory/user_model.py  (~200 linhas)
```

**UserModel:**
```python
@dataclass
class UserModel:
    user_id: str
    preferences: UserPreferences      # communication_style, code_style, tools
    cognitive: CognitiveProfile       # flow_duration, work_hours, fatigue
    patterns: List[Dict]              # learned patterns (max 100)
    version: int
```

**UserModelService:**
- `load()` / `save()` - persist em NOESIS
- `update_preferences()` / `update_cognitive()`
- `add_pattern()` - mantém top 100 by confidence

**Sync:** CLAUDE.md ↔ User Model bidirecional

---

### SPRINT 5: Integration Testing (2-3 dias)

**Objetivo:** Validar sistema completo como célula híbrida

**Testes E2E:**
1. Signal flow: Collector → ActivityStore → NOESIS Consciousness
2. LLM learning: Approval/rejection → LLM classify → Insights
3. Proactive emergence: Pattern match → Anticipation → Emerge
4. User model persistence: Save → Restart → Load
5. Graceful degradation: NOESIS offline → Local fallbacks

**Benchmarks:**
- Signal latency < 100ms
- LLM cache hit rate > 30%
- Pattern detection: 10k events in < 1s

---

## CRONOGRAMA

| Sprint | Duração | Arquivos Principais |
|--------|---------|---------------------|
| 0 | 1 dia | llm/config.py, llm/client.py |
| 1 | 2-3 dias | noesis_ingestion.py |
| 2 | 3-4 dias | llm_service.py, preference_learner.py |
| 3 | 3-4 dias | pattern_detector.py, anticipation_engine.py |
| 4 | 2-3 dias | user_model.py |
| 5 | 2-3 dias | tests/integration/*.py |
| **Total** | **~14-18 dias** | |

---

## MÉTRICAS DE SUCESSO

| Métrica | Antes | Depois |
|---------|-------|--------|
| AI Coverage | 0% | 80%+ |
| NOESIS signals/hour | 0 | 100+ |
| Proactive emergences/day | 0 | 2-5 |
| Patterns detected | 0 | 50+ |
| E2E test coverage | 60% | 95%+ |

---

## PRINCÍPIOS

1. **Simplicidade** - Cada módulo faz UMA coisa bem
2. **Fallback** - Heurísticas backup para LLM
3. **Observabilidade** - Logs estruturados
4. **Privacidade** - Captura INTENÇÃO, não CONTEÚDO
5. **Incremental** - Cada sprint deployável independentemente

---

## RISCOS

| Risco | Mitigação |
|-------|-----------|
| LLM latência | Cache agressivo, fallback heurístico |
| NOESIS offline | Graceful degradation, queue local |
| Patterns falsos | Confidence threshold, user feedback |
| Custo LLM | Haiku para simples, cache |

---

*"O código existe, mas o espírito da integração precisa ser implementado."*

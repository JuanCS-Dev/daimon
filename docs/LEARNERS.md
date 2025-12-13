# DAIMON Learners

**Sistema de Aprendizado - Engines de Detecção e Adaptação**

---

## Visão Geral

Os Learners são responsáveis por analisar dados coletados e gerar insights acionáveis. Formam o "cérebro" do DAIMON, transformando comportamento observado em preferências aprendidas.

### Princípios

1. **Detecção Passiva** - Aprender sem perguntar explicitamente
2. **Confiança Gradual** - Só atuar com alta confiança
3. **Feedback Loop** - Medir efetividade das sugestões
4. **Transparência** - Todas as decisões são rastreáveis

---

## Arquitetura

```
learners/
├── __init__.py              # Exports e singleton
├── preference_learner.py    # Detecção de aprovação/rejeição
├── style_learner.py         # Perfil de estilo de trabalho
├── keystroke_analyzer.py    # Estado cognitivo via digitação
├── metacognitive_engine.py  # Meta-aprendizado do sistema
└── reflection_engine.py     # Orquestrador principal
```

---

## 1. Preference Learner

**Arquivo:** `learners/preference_learner.py`
**Status:** ✅ Funcional

Detecta padrões de aprovação e rejeição nas interações com Claude Code.

### Como Funciona

```
Sessões JSONL → Análise de Padrões → PreferenceSignals → Insights Acionáveis
```

### Padrões Detectados

#### Aprovação
```python
APPROVAL_PATTERNS = [
    r"\b(sim|yes|ok|perfeito|otimo|excelente|isso|gostei)\b",
    r"\b(aceito|aprovo|pode|manda|vai|bora|certo|correto)\b",
    r"^(s|y|ok|sim)$",
    r"(thumbs.?up|great|good|nice|awesome)",
]
```

#### Rejeição
```python
REJECTION_PATTERNS = [
    r"\b(nao|no|nope|errado|ruim|feio|pare|espera)\b",
    r"\b(rejeito|recuso|para|cancela|volta|desfaz)\b",
    r"\b(menos|mais simples|muito|demais|longo)\b",
    r"(thumbs.?down|bad|wrong|incorrect)",
]
```

### Categorias Inferidas

| Categoria | Keywords |
|-----------|----------|
| `code_style` | formatacao, estilo, naming, indent, lint |
| `verbosity` | verboso, longo, curto, resumo, detalhado |
| `testing` | teste, test, coverage, mock, assert |
| `architecture` | arquitetura, estrutura, pattern, design |
| `documentation` | doc, comment, readme, docstring |
| `workflow` | commit, branch, git, deploy, ci |
| `security` | security, auth, password, token |
| `performance` | performance, speed, optimize, cache |

### API

```python
from learners.preference_learner import PreferenceLearner

learner = PreferenceLearner()

# Escanear sessões recentes
signals = learner.scan_sessions(since_hours=48)

# Obter resumo por categoria
summary = learner.get_preference_summary()
# → {"testing": {"approval_rate": 0.85, "total_signals": 12, "trend": "positive"}}

# Obter insights acionáveis
insights = learner.get_actionable_insights(min_signals=3)
# → [{"category": "testing", "action": "reinforce", "confidence": 0.9, "suggestion": "..."}]
```

### Fonte de Dados

1. **Primária**: ActivityStore (populado pelo claude_watcher)
2. **Fallback**: Arquivos JSONL diretos (`~/.claude/projects/*/sessions/*.jsonl`)

---

## 2. Style Learner

**Arquivo:** `learners/style_learner.py`
**Status:** ✅ Funcional

Aprende o estilo de trabalho do usuário através de múltiplas fontes.

### Fontes de Dados

| Fonte | Dados | Insights |
|-------|-------|----------|
| Keystroke | Velocidade, pausas, correções | Pace de digitação |
| Window | Duração de foco, troca de janelas | Nível de foco |
| AFK | Padrões de ausência | Ritmo de trabalho |
| Shell | Comandos, erros | Intensidade de uso |
| Claude | Rejeições, aprovações | Preferências de interação |

### Métricas Calculadas

```python
@dataclass
class CommunicationStyle:
    typing_pace: TypingPace           # fast/moderate/deliberate
    editing_frequency: EditingFreq    # minimal/moderate/heavy
    focus_level: FocusLevel           # deep/moderate/multitask
    interaction_pattern: Pattern      # burst/steady/sporadic
    preferred_hours: List[int]        # Horas de pico
    peak_productivity: str            # "09:00-12:00"
    confidence: float                 # 0.0-1.0
```

### API

```python
from learners import get_style_learner

learner = get_style_learner()

# Adicionar samples
learner.add_keystroke_sample({"typing_speed_cpm": 280})
learner.add_window_sample({"focus_duration": 1800})
learner.add_afk_sample({"is_afk": True, "duration": 300})

# Computar estilo
style = learner.compute_style()
# → CommunicationStyle(typing_pace="fast", focus_level="deep", ...)

# Gerar seção para CLAUDE.md
section = learner.get_claude_md_section()
# → "## Communication Style\n- Fast typist - prefers concise responses\n..."
```

### Sugestões Geradas

| Observação | Sugestão para Claude |
|------------|---------------------|
| Fast typing | Prefer concise responses |
| Heavy editing | Show drafts for review first |
| Deep focus | Minimize interruptions |
| Multitasking | Break into smaller chunks |
| Burst pattern | Front-load important info |

---

## 3. Keystroke Analyzer

**Arquivo:** `learners/keystroke_analyzer.py`
**Status:** ✅ Funcional (integrado ao ReflectionEngine)

Detecta estado cognitivo através da dinâmica de digitação.

### Estados Detectáveis

| Estado | Indicadores | Confiança |
|--------|-------------|-----------|
| `flow` | Alta velocidade, baixa variância, poucos erros | 0.7-0.9 |
| `focused` | Velocidade média, ritmo constante | 0.6-0.8 |
| `fatigued` | Velocidade decrescente, mais pausas, mais erros | 0.6-0.8 |
| `stressed` | Alta variância, pausas irregulares, muitos erros | 0.5-0.7 |
| `distracted` | Pausas longas, velocidade inconsistente | 0.5-0.7 |
| `idle` | Sem atividade suficiente | - |

### Algoritmo

```python
def detect_cognitive_state(self) -> CognitiveState:
    recent = self._events[-100:]

    # Métricas
    typing_speed = self._calculate_speed(recent)
    pause_variance = self._calculate_pause_variance(recent)
    error_rate = self._calculate_error_rate(recent)

    # Classificação
    if typing_speed > HIGH and pause_variance < LOW and error_rate < LOW:
        return CognitiveState("flow", confidence=0.8)
    elif error_rate > HIGH and pause_variance > HIGH:
        return CognitiveState("fatigued", confidence=0.7)
    # ...
```

### Integração com ReflectionEngine

O estado cognitivo agora alimenta o ReflectionEngine via `_get_cognitive_insights()`:

```python
# Em reflection_engine.py
def _get_cognitive_insights(self) -> list[dict]:
    analyzer = get_keystroke_analyzer()
    state = analyzer.detect_cognitive_state()

    if state.state == "fatigued":
        return [{
            "category": "workflow",
            "action": "add",
            "confidence": state.confidence,
            "suggestion": "User shows fatigue - prefer concise responses",
        }]
```

---

## 4. Metacognitive Engine

**Arquivo:** `learners/metacognitive_engine.py`
**Status:** ✅ Funcional (integrado ao ReflectionEngine)

Meta-aprendizado sobre o próprio sistema DAIMON.

### Funcionalidades

1. **Log de Insights** - Registra todos os insights gerados
2. **Análise de Efetividade** - Mede sucesso das sugestões
3. **Recomendações** - Sugere ajustes ao próprio sistema

### API

```python
from learners.metacognitive_engine import get_metacognitive_engine

engine = get_metacognitive_engine()

# Registrar insight
insight = Insight(
    category="testing",
    action="reinforce",
    confidence=0.85,
    suggestion="Continue generating tests proactively",
)
engine.log_insight(insight)

# Analisar efetividade
analysis = engine.analyze_effectiveness()
# → {
#     "total_insights": 45,
#     "measured_insights": 30,
#     "average_effectiveness": 0.72,
#     "adjustment_suggestions": {
#         "scan_frequency": {"current": 30, "suggested": 45},
#     }
# }
```

### Integração com ReflectionEngine

O MetacognitiveEngine agora aplica ajustes via `_apply_metacognitive_adjustments()`:

```python
# Em reflection_engine.py
async def _apply_metacognitive_adjustments(self) -> None:
    metacog = get_metacognitive_engine()
    analysis = metacog.analyze_effectiveness()

    for key, suggestion in analysis.adjustment_suggestions.items():
        if key == "scan_frequency":
            self.config.interval_minutes = suggestion["suggested_minutes"]
        elif key == "confidence_threshold":
            # Log ajuste (não altera diretamente)
            logger.info("Suggested confidence threshold: %s", suggestion)
```

---

## 5. Reflection Engine

**Arquivo:** `learners/reflection_engine.py`
**Status:** ✅ Funcional

Orquestrador principal do loop de aprendizado.

### Ciclo de Reflexão

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REFLECTION LOOP (30 min)                         │
│                                                                         │
│  1. _apply_metacognitive_adjustments()                                 │
│     └── Ajusta parâmetros baseado em efetividade                       │
│                                                                         │
│  2. PreferenceLearner.scan_sessions(since_hours=48)                    │
│     └── Escaneia sessões recentes por sinais                           │
│                                                                         │
│  3. _get_cognitive_insights()                                          │
│     └── Consulta KeystrokeAnalyzer para estado cognitivo               │
│                                                                         │
│  4. PreferenceLearner.get_actionable_insights()                        │
│     └── Gera insights acionáveis                                       │
│                                                                         │
│  5. _apply_insights()                                                  │
│     └── ConfigRefiner.update_preferences() → CLAUDE.md                 │
│                                                                         │
│  6. _notify_update() (a cada 10 insights)                              │
│     └── notify-send com resumo                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Triggers

| Trigger | Condição | Descrição |
|---------|----------|-----------|
| Temporal | 30 minutos | Loop automático |
| Threshold | 5+ rejeições OU 10+ aprovações | Mesma categoria |
| Manual | `POST /api/daimon/reflect` | Via API/Dashboard |

### Configuração

```python
@dataclass
class ReflectionConfig:
    interval_minutes: int = 30      # Intervalo entre reflexões
    rejection_threshold: int = 5    # Threshold de rejeições
    approval_threshold: int = 10    # Threshold de aprovações
    scan_hours: int = 48            # Horas para scan
    min_signals: int = 3            # Mínimo de sinais para insight
```

### API

```python
from learners import get_engine

engine = get_engine()

# Iniciar loop (async)
await engine.start()

# Reflexão manual
result = await engine.reflect()
# → {"signals": 15, "insights": 3, "updated": True}

# Status
status = engine.get_status()
# → {"running": True, "last_reflection": "...", "total_reflections": 5}

# Parar
await engine.stop()
```

### Notificações

As notificações são agrupadas: **1 notificação a cada 10 insights**.

```python
# Exemplo de notificação
"DAIMON aplicou 10 insights em 3 categoria(s): code_style, testing, workflow"
```

---

## Fluxo de Dados Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Collectors ──► ActivityStore ──► PreferenceLearner ──► Insights       │
│       │              │                    ▲                   │         │
│       │              │                    │                   ▼         │
│       │              │        ┌───────────┴────────┐    ConfigRefiner   │
│       │              │        │ Tribunal Verdict   │         │         │
│       │              │        │ (via NOESIS)       │         ▼         │
│       │              │        │       │            │    CLAUDE.md      │
│       │              │        │       ▼            │                   │
│       │              │        │ _feed_verdict_to_  │                   │
│       │              │        │ learner()          │                   │
│       │              │        └────────────────────┘                   │
│       │              │                                                 │
│       ▼              ▼                                                 │
│  StyleLearner  KeystrokeAnalyzer ──► _get_cognitive_insights()         │
│       │              │                                                 │
│       │              ▼                                                 │
│       │        MetacognitiveEngine ──► _apply_metacognitive_           │
│       │                                adjustments()                   │
│       │                                                                 │
│       └───────────────────────────────────────────────────────────────►│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Exemplo de Insight Gerado

```json
{
  "category": "verbosity",
  "action": "reduce",
  "confidence": 0.85,
  "approval_rate": 0.25,
  "total_signals": 8,
  "suggestion": "Preferir respostas concisas. Evitar explicacoes longas."
}
```

Este insight, quando aplicado pelo ConfigRefiner, adiciona ao CLAUDE.md:

```markdown
<!-- DAIMON:AUTO:START -->
# Preferencias Aprendidas (DAIMON)
*Ultima atualizacao: 2025-12-13 15:30*

## Communication Style
- [Alta] Preferir respostas concisas. Evitar explicacoes longas.
<!-- DAIMON:AUTO:END -->
```

---

## Testes

```bash
# Todos os testes de learners
python -m pytest tests/test_preference_learner.py tests/test_style_learner.py \
    tests/test_keystroke_analyzer.py tests/test_reflection_engine.py -v

# Testes específicos
python -m pytest tests/test_reflection_engine.py::TestNotifyUpdate -v
```

---

## Limitações Honestas

1. **PreferenceLearner** depende de padrões textuais - pode errar em contextos ambíguos
2. **KeystrokeAnalyzer** precisa de ~100 eventos para detecção confiável
3. **StyleLearner** precisa de dias de dados para perfil preciso
4. **MetacognitiveEngine** está em fase inicial - medição de efetividade limitada

---

*Documentação atualizada em 2025-12-13*

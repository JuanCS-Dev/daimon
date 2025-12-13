# DAIMON - Personal Exocortex

**Célula Híbrida Human-AI para Desenvolvimento Assistido com Claude Code**

> *"Conhece-te a ti mesmo" - Oráculo de Delfos*
>
> *"Silêncio é ouro. Só emergir quando verdadeiramente significativo."*

---

## O que é DAIMON?

DAIMON é um **exocórtex pessoal** que transforma Claude Code em um **parceiro de desenvolvimento adaptativo**:

1. **Observa** silenciosamente seu comportamento de desenvolvimento (7 collectors)
2. **Aprende** suas preferências via LLM com fallback heurístico
3. **Detecta padrões** temporais, sequenciais e contextuais
4. **Antecipa** necessidades e emerge proativamente quando relevante
5. **Persiste** um modelo de usuário que evolui com você
6. **Sincroniza** automaticamente preferências no Claude Code

Integra com **NOESIS** (motor de consciência) para questionamento socrático.

---

## Arquitetura

```
                              DAIMON v3.0
    ┌─────────────────────────────────────────────────────────────┐
    │                                                              │
    │  COLLECTORS ──▶ MEMORY ──▶ LEARNERS ──▶ ACTUATORS           │
    │  (7 watchers)   (SQLite)   (LLM-powered) (ConfigRefiner)    │
    │       │            │            │             │              │
    │       │     ActivityStore  PatternDetector    │              │
    │       │            │            │             ▼              │
    │       │      UserModel ◀── AnticipationEngine ── CLAUDE.md  │
    │       │            │                                         │
    │       │    ┌───────┴───────────────────────┐                │
    │       └───▶│         DASHBOARD              │                │
    │            │      (localhost:8003)          │                │
    │            └────────────────────────────────┘                │
    │                         │                                    │
    └─────────────────────────┼────────────────────────────────────┘
                              │
                              ▼
                           NOESIS
               Consciousness (8001) + Tribunal (8002)
```

---

## Componentes Principais

### Collectors (7 tipos)

| Collector | Dados | Status |
|-----------|-------|--------|
| `shell_watcher` | Comandos, exit codes | ✅ |
| `claude_watcher` | Sessões, aprovações/rejeições | ✅ |
| `window_watcher` | Janelas ativas, foco | ✅ |
| `input_watcher` | Dinâmica de digitação | ✅ |
| `afk_watcher` | Períodos de inatividade | ✅ |
| `browser_watcher` | URLs visitadas | ⚠️ Experimental |

### Learners (LLM-Powered)

| Engine | Função |
|--------|--------|
| `LearnerLLMService` | LLM com fallback heurístico |
| `PreferenceLearner` | Detecta aprovação/rejeição (modular) |
| `PatternDetector` | Padrões temporais/sequenciais/contextuais |
| `AnticipationEngine` | Decide quando emergir proativamente |
| `StyleLearner` | Perfil de estilo de trabalho |
| `KeystrokeAnalyzer` | Estado cognitivo via digitação |
| `MetacognitiveEngine` | Meta-aprendizado |
| `ReflectionEngine` | Orquestrador principal |

### Memory

| Module | Função |
|--------|--------|
| `ActivityStore` | Heartbeats SQLite |
| `UserModel` | Modelo persistente de usuário |
| `UserPreferences` | Preferências de comunicação/código |
| `CognitiveProfile` | Padrões cognitivos |
| `PrecedentSystem` | Jurisprudência de decisões |

### Integrations

| Tool | Descrição |
|------|-----------|
| `noesis_consult` | Questionamento maiêutico |
| `noesis_tribunal` | Julgamento ético (3 juízes) |
| `noesis_precedent` | Busca precedentes |
| `noesis_confront` | Confrontação socrática |

---

## Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE PRINCIPAL                          │
│                                                                  │
│  1. COLETA (contínua)                                           │
│     Collectors ──▶ Heartbeats ──▶ ActivityStore                 │
│                                                                  │
│  2. DETECÇÃO (LLM-powered)                                       │
│     Messages ──▶ SignalDetector ──▶ PreferenceCategorizer       │
│                                     └──▶ InsightGenerator        │
│                                                                  │
│  3. PADRÕES                                                      │
│     Events ──▶ PatternDetector ──▶ AnticipationEngine           │
│                                    └──▶ EmergenceDecision        │
│                                                                  │
│  4. PERSISTÊNCIA                                                 │
│     Insights ──▶ UserModel ──▶ NOESIS (primary)                 │
│                             └──▶ LocalJSON (fallback)            │
│                                                                  │
│  5. SINCRONIZAÇÃO                                                │
│     UserModel ──▶ ConfigRefiner ──▶ CLAUDE.md                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Instalar

```bash
cd /media/juan/DATA/projetos/daimon
pip install -e .
```

### 2. Configurar Claude Code

```bash
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py
```

### 3. Iniciar

```bash
# Via systemd (recomendado)
systemctl --user enable --now daimon

# Ou manualmente
python daimon_daemon.py
```

### 4. Dashboard

```
http://localhost:8003
```

---

## Storage

```
~/.daimon/
├── activity/
│   └── activities.db       # Heartbeats
├── user_model.json         # Preferências persistentes
├── memory/
│   ├── memories.db         # Memória semântica
│   └── precedents.db       # Jurisprudência
└── corpus/                 # Textos de sabedoria

~/.claude/
├── CLAUDE.md               # Preferências (destino)
│   └── <!-- DAIMON:START --> ... <!-- DAIMON:END -->
└── projects/               # Sessões escaneadas
```

---

## Qualidade

| Métrica | Valor |
|---------|-------|
| **Testes** | 883 passando |
| **Coverage** | 84% |
| **Sprints Completos** | 5/5 |

### Arquivos por Módulo

| Diretório | Arquivos | Linhas (max) |
|-----------|----------|--------------|
| `learners/` | 18 | <500 ✅ |
| `learners/preference/` | 6 | <320 ✅ |
| `memory/` | 8 | <360 ✅ |
| `collectors/` | 9 | <400 ✅ |

---

## Princípios

1. **LLM First, Heuristics Fallback** - IA quando disponível, regras sempre
2. **Proactive Emergence** - Só emerge quando padrões indicam alto valor
3. **Cooldown & Thresholds** - Evita spam (10min cooldown)
4. **Pattern-Based Learning** - Temporal, Sequential, Contextual
5. **Graceful Degradation** - Funciona sem NOESIS ou LLM
6. **Constitution Compliant** - Arquivos <500 linhas, type hints, docstrings

---

## Comandos Úteis

```bash
# Status
systemctl --user status daimon

# Logs
journalctl --user -u daimon -f

# Reflexão manual
curl -X POST http://localhost:8003/api/reflect

# Testes
python -m pytest tests/ -v

# Coverage
python -m pytest tests/ --cov --cov-report=term
```

---

## Limitações Honestas

1. **LLM opcional** - Sem LLM, usa heurísticas (menos precisão)
2. **Latência NOESIS** - Adiciona 50-200ms por chamada
3. **Padrões mínimos** - Precisa de dados históricos para aprender
4. **Browser watcher** - Requer extensão não implementada
5. **Sem autenticação** - Dashboard apenas local

---

## Licença

MIT - Parte do Projeto NOESIS

---

*DAIMON v3.0 - Célula Híbrida Human-AI*
*Dezembro 2025*

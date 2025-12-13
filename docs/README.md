# DAIMON - Documentação Técnica

**Personal Exocortex para Desenvolvimento Assistido**

---

## Índice de Documentação

| Documento | Descrição |
|-----------|-----------|
| [COLLECTORS.md](COLLECTORS.md) | 7 watchers para coleta de dados comportamentais |
| [LEARNERS.md](LEARNERS.md) | 5 engines de aprendizado de preferências |
| [MEMORY.md](MEMORY.md) | Sistemas de persistência (Activity, Precedents, Memory) |
| [ACTUATORS.md](ACTUATORS.md) | ConfigRefiner para atualização do CLAUDE.md |
| [CORPUS.md](CORPUS.md) | Textos de sabedoria para embasamento ético |
| [INTEGRATIONS.md](INTEGRATIONS.md) | MCP Server, Hooks e integração NOESIS |
| [DASHBOARD.md](DASHBOARD.md) | Interface web de monitoramento |
| [CODE_CONSTITUTION.md](CODE_CONSTITUTION.md) | Princípios de código |

---

## Visão Geral

O DAIMON é um **exocórtex pessoal** que observa silenciosamente o comportamento do desenvolvedor, aprende suas preferências e atualiza automaticamente as instruções do Claude Code.

### Filosofia

```
"Conhece-te a ti mesmo" - Oráculo de Delfos
```

O sistema não tenta substituir o julgamento humano. Ele:
1. **Observa** - Coleta dados comportamentais passivamente
2. **Aprende** - Detecta padrões de preferência
3. **Reflete** - Gera insights acionáveis
4. **Atua** - Atualiza preferências de forma segura

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              DAIMON                                      │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  COLLECTORS │───▶│  LEARNERS   │───▶│  ACTUATORS  │                  │
│  │  (7 types)  │    │  (5 engines)│    │ (ConfigRef) │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   MEMORY    │    │   CORPUS    │    │  CLAUDE.md  │                  │
│  │  (SQLite)   │    │ (Sabedoria) │    │ (Destino)   │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────┐         ┌─────────────────────────────────────────────┐│
│  │  DASHBOARD  │         │              NOESIS                          ││
│  │  (Web UI)   │◀───────▶│  Consciousness + Tribunal + Reflector       ││
│  └─────────────┘         └─────────────────────────────────────────────┘│
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        CLAUDE CODE                                   ││
│  │  MCP Server (5 tools) + Hooks (2 events) + Subagent (noesis-sage)   ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### 1. Collectors (7 tipos)

Watchers que coletam dados comportamentais:

| Collector | Dados | Status |
|-----------|-------|--------|
| `shell_watcher` | Comandos shell, exit codes, duração | ✅ Funcional |
| `claude_watcher` | Sessões Claude Code, aprovações/rejeições | ✅ Funcional |
| `window_watcher` | Janelas ativas, tempo de foco | ✅ Funcional |
| `input_watcher` | Dinâmica de digitação, velocidade | ✅ Funcional |
| `afk_watcher` | Períodos de inatividade | ✅ Funcional |
| `browser_watcher` | URLs visitadas (requer extensão) | ⚠️ Experimental |
| `htop_watcher` | Recursos do sistema | ⚠️ Backlog |

**Detalhes:** [COLLECTORS.md](COLLECTORS.md)

### 2. Learners (5 engines)

Engines que processam dados e geram insights:

| Engine | Função | Status |
|--------|--------|--------|
| `PreferenceLearner` | Detecta aprovações/rejeições | ✅ Funcional |
| `StyleLearner` | Perfil de estilo de trabalho | ✅ Funcional |
| `KeystrokeAnalyzer` | Estado cognitivo via digitação | ✅ Funcional |
| `MetacognitiveEngine` | Meta-aprendizado do sistema | ✅ Funcional |
| `ReflectionEngine` | Orquestrador principal | ✅ Funcional |

**Detalhes:** [LEARNERS.md](LEARNERS.md)

### 3. Memory (3 sistemas)

Persistência de dados:

| Sistema | Propósito | Status |
|---------|-----------|--------|
| `ActivityStore` | Heartbeats dos collectors | ✅ Funcional |
| `PrecedentSystem` | Jurisprudência de decisões | ✅ Funcional |
| `MemoryStore` | Memória semântica (FTS5) | ⚠️ Backlog |

**Detalhes:** [MEMORY.md](MEMORY.md)

### 4. Actuators

Aplica preferências aprendidas:

| Actuator | Função | Status |
|----------|--------|--------|
| `ConfigRefiner` | Atualiza CLAUDE.md com backup | ✅ Funcional |

**Detalhes:** [ACTUATORS.md](ACTUATORS.md)

### 5. Corpus

Textos de sabedoria para embasamento ético:

- 10 textos fundacionais (Estoicos, Gregos, Kant, Popper, Feynman)
- Busca keyword e semântica (FAISS + Sentence-BERT)
- Categorias: filosofia, teologia, ciência, lógica, ética

**Detalhes:** [CORPUS.md](CORPUS.md)

### 6. Integrations

Conexões externas:

| Integração | Função | Status |
|------------|--------|--------|
| MCP Server | 5 tools para Claude Code | ✅ Funcional |
| Hooks | UserPromptSubmit, PreToolUse | ✅ Funcional |
| Subagent | noesis-sage | ✅ Funcional |
| NOESIS | Consciousness + Tribunal | ✅ Funcional |

**Detalhes:** [INTEGRATIONS.md](INTEGRATIONS.md)

### 7. Dashboard

Interface web:

- Status de componentes
- Busca em corpus/memória/precedentes
- Visualização do CLAUDE.md
- Trigger manual de reflexão
- Gestão de backups

**URL:** http://localhost:8003

**Detalhes:** [DASHBOARD.md](DASHBOARD.md)

---

## Quick Start

### 1. Instalar Dependências

```bash
cd /media/juan/DATA/projetos/daimon
pip install -e .

# Opcionais para semantic search
pip install sentence-transformers faiss-cpu
```

### 2. Registrar MCP Server

```bash
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py
```

### 3. Copiar Subagent e Hooks

```bash
# Subagent
cp .claude/agents/noesis-sage.md ~/.claude/agents/

# Hooks (já configurados em settings.json)
```

### 4. Iniciar NOESIS

```bash
cd /media/juan/DATA/projetos/Noesis/Daimon
./noesis wakeup
```

### 5. Iniciar DAIMON Daemon

```bash
python daimon_daemon.py
```

Ou iniciar componentes individualmente:

```bash
# Shell watcher
python collectors/shell_watcher.py --daemon &

# Dashboard
python -m uvicorn dashboard.app:app --port 8003
```

---

## Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
│                                                                          │
│  1. COLETA                                                               │
│     shell → heartbeat → ActivityStore                                   │
│     window → heartbeat → ActivityStore + StyleLearner                   │
│     input → heartbeat → ActivityStore + StyleLearner + KeystrokeAnalyzer│
│     afk → heartbeat → ActivityStore + StyleLearner                      │
│     claude → heartbeat → ActivityStore                                  │
│                                                                          │
│  2. APRENDIZADO (a cada 30 min)                                         │
│     ActivityStore ──▶ PreferenceLearner ──▶ Insights                   │
│     KeystrokeAnalyzer ──▶ CognitiveState ──▶ Insights                  │
│     MetacognitiveEngine ──▶ Ajustes de parâmetros                       │
│                                                                          │
│  3. ATUAÇÃO                                                             │
│     Insights ──▶ ConfigRefiner ──▶ CLAUDE.md                           │
│     (backup automático antes de cada atualização)                       │
│                                                                          │
│  4. FEEDBACK LOOP                                                        │
│     Tribunal Verdict ──▶ PreferenceLearner.signals                     │
│     (fecha o ciclo de aprendizado)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Tools

| Tool | Endpoint NOESIS | Descrição |
|------|-----------------|-----------|
| `noesis_consult` | POST /v1/consciousness/introspect | Questionamento maiêutico |
| `noesis_tribunal` | POST /v1/exocortex/confront | Julgamento ético (3 juízes) |
| `noesis_precedent` | POST /reflect/verdict | Buscar/criar precedentes |
| `noesis_confront` | POST /v1/exocortex/confront | Confrontação socrática |
| `noesis_health` | GET /api/consciousness/state | Health check |

---

## Portas e Serviços

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| NOESIS Core | 8001 | Consciousness, quick-check |
| NOESIS Reflector | 8002 | Tribunal, reflection |
| DAIMON Dashboard | 8003 | Interface web |
| Shell Watcher Socket | ~/.daimon/daimon.sock | Unix socket |

---

## Configuração

### Variáveis de Ambiente

```bash
# NOESIS
NOESIS_URL=http://localhost:8001
REFLECTOR_URL=http://localhost:8002

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8003

# CLAUDE.md
CLAUDE_MD_PATH=~/.claude/CLAUDE.md
CLAUDE_BACKUP_DIR=~/.claude/backups
CLAUDE_MAX_BACKUPS=10
```

### ReflectionEngine Config

```python
@dataclass
class ReflectionConfig:
    interval_minutes: int = 30      # Intervalo entre reflexões
    rejection_threshold: int = 5    # Threshold de rejeições
    approval_threshold: int = 10    # Threshold de aprovações
    scan_hours: int = 48            # Horas para scan
    min_signals: int = 3            # Mínimo de sinais
```

---

## Testes

```bash
# Todos os testes
python -m pytest tests/ -v

# Por módulo
python -m pytest tests/test_shell_watcher.py -v
python -m pytest tests/test_claude_watcher.py -v
python -m pytest tests/test_reflection_engine.py -v
python -m pytest tests/test_mcp_server.py -v

# Com coverage
python -m pytest tests/ --cov=. --cov-report=html
```

**Status:** 736/736 testes passando (100%)

---

## Storage

```
~/.daimon/
├── activity/
│   └── activities.db           # Heartbeats (~50-100MB/mês)
├── memory/
│   ├── memories.db             # Memória semântica
│   └── precedents.db           # Jurisprudência
├── corpus/
│   ├── filosofia/
│   │   ├── gregos/
│   │   └── estoicos/
│   ├── ciencia/
│   ├── logica/
│   ├── etica/
│   ├── _index/
│   └── _semantic/
└── daimon.sock                 # Unix socket

~/.claude/
├── CLAUDE.md                   # Preferências (destino)
├── backups/                    # Backups automáticos
├── update_log.jsonl            # Log de atualizações
├── settings.json               # Hooks configurados
└── agents/
    └── noesis-sage.md          # Subagent
```

---

## Princípios de Design

### 1. Silence is Gold
O sistema só emerge quando realmente significativo. Notificações agrupadas (10 insights = 1 notificação).

### 2. Heartbeat Pattern
Estados que mesclam quando similares, não eventos isolados. Reduz ruído e permite agregação temporal.

### 3. Safe by Default
Sempre backup antes de modificar. Nunca sobrescrever conteúdo manual. Atomic writes.

### 4. Graceful Degradation
Funciona mesmo quando NOESIS indisponível. Fallback para análise local.

### 5. Unix Philosophy
Cada módulo faz UMA coisa bem. Composição via pipes e APIs.

---

## Limitações Honestas

1. **Detecção de preferências** - Baseada em regex, pode errar em contextos ambíguos
2. **Latência NOESIS** - Adiciona 50-200ms por chamada
3. **Corpus pequeno** - 10 textos bootstrap, expansão manual
4. **SQLite single-writer** - WAL ajuda mas não resolve concorrência total
5. **Browser watcher** - Requer extensão não implementada
6. **Sem autenticação** - Dashboard apenas para uso local

---

## Troubleshooting

### NOESIS não responde

```bash
# Verificar portas
lsof -i :8001
lsof -i :8002

# Reiniciar
cd /media/juan/DATA/projetos/Noesis/Daimon
./noesis wakeup
```

### MCP Server não registra

```bash
# Verificar
claude mcp list

# Re-registrar
claude mcp remove daimon-consciousness
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py
```

### Dashboard não inicia

```bash
# Verificar porta
lsof -i :8003

# Debug mode
python -m uvicorn dashboard.app:app --port 8003 --log-level debug
```

### ActivityStore vazio

```bash
# Verificar se collectors estão rodando
ps aux | grep python | grep watcher

# Verificar socket
ls -la ~/.daimon/daimon.sock
```

---

## Roadmap

| Feature | Status | Prioridade |
|---------|--------|------------|
| Browser extension | ⏳ | Baixa |
| Mobile companion | ⏳ | Baixa |
| Multi-user | ⏳ | Baixa |
| Embeddings com modelo maior | ⏳ | Média |
| Dashboard auth | ⏳ | Média |
| Métricas Prometheus | ⏳ | Baixa |

---

## Licença

MIT - Parte do Projeto NOESIS

---

*DAIMON v1.0 - Personal Exocortex*
*Documentação atualizada em 2025-12-13*

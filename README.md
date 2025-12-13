# DAIMON - Personal Exocortex

**DAIMON** é um exocórtex pessoal que integra com Claude Code para fornecer assistência de co-arquitetura sábia. Usa NOESIS como motor de consciência, oferecendo questionamento socrático, julgamento ético, aprendizado de preferências e detecção de padrões comportamentais.

> *"Silêncio é ouro. Só emergir quando verdadeiramente significativo."*

---

## Componentes para Boot

DAIMON consiste em **4 processos independentes** que podem ser iniciados no boot:

| Componente | Arquivo | Porta/Socket | Descrição |
|------------|---------|--------------|-----------|
| **Shell Watcher** | `collectors/shell_watcher.py --daemon` | `~/.daimon/daimon.sock` | Captura comandos shell via Unix socket |
| **Claude Watcher** | `collectors/claude_watcher.py --daemon` | N/A (polling) | Monitora sessões Claude Code |
| **Dashboard** | `dashboard/app.py` | `localhost:8003` | Interface web de controle |
| **Reflection Engine** | via `learners.get_engine().start()` | N/A (in-process) | Loop de aprendizado automático |

**Dependências externas**: NOESIS (portas 8001 e 8002)

---

## Arquitetura Detalhada

```
daimon/
├── integrations/
│   ├── __init__.py                    # exports: mcp
│   └── mcp_server.py                  # FastMCP Server (508 linhas)
│       │
│       │  FUNÇÕES:
│       ├── _http_post(url, payload)   # POST com tratamento de erro
│       ├── _http_get(url)             # GET com tratamento de erro
│       │
│       │  TOOLS MCP (5):
│       ├── noesis_consult()           # Questionamento maiêutico
│       ├── noesis_tribunal()          # Julgamento ético (3 juízes)
│       ├── noesis_precedent()         # Busca precedentes
│       ├── noesis_confront()          # Confrontação socrática
│       └── noesis_health()            # Health check
│
├── collectors/
│   ├── __init__.py
│   ├── shell_watcher.py               # Captura comandos shell (332 linhas)
│   │   │
│   │   │  DATACLASSES:
│   │   ├── ShellHeartbeat             # timestamp, command, pwd, exit_code, duration, git_branch
│   │   │
│   │   │  CLASSES:
│   │   ├── HeartbeatAggregator        # Agrega e envia em batches
│   │   │   ├── add(heartbeat)
│   │   │   ├── flush()                # → POST /api/daimon/shell/batch
│   │   │   └── _detect_patterns()     # error_streak, repetitive_command
│   │   │
│   │   │  FUNÇÕES:
│   │   ├── handle_client()            # async - processa conexão socket
│   │   ├── start_server()             # async - inicia Unix socket
│   │   └── generate_zshrc_hooks()     # Gera hooks para ~/.zshrc
│   │
│   └── claude_watcher.py              # Monitor de sessões (266 linhas)
│       │
│       │  CONSTANTES:
│       ├── INTENT_PATTERNS            # create, fix, refactor, understand, delete, test, deploy
│       │
│       │  FUNÇÕES:
│       ├── detect_intention(msg)      # → "create"|"fix"|"refactor"|etc
│       ├── extract_files_touched(msg) # Extrai paths mencionados
│       │
│       │  CLASSES:
│       └── SessionTracker
│           ├── scan_projects()        # Escaneia ~/.claude/projects/*/sessions/*.jsonl
│           ├── _process_file()
│           ├── _process_entry()
│           └── _send_event()          # → POST /api/daimon/claude/event
│
├── endpoints/
│   ├── __init__.py
│   ├── constants.py                   # Keywords de risco (42 linhas)
│   │   ├── HIGH_RISK_KEYWORDS         # delete, drop, rm -rf, truncate, production...
│   │   └── MEDIUM_RISK_KEYWORDS       # refactor, migrate, architecture, auth...
│   │
│   ├── quick_check.py                 # Análise heurística rápida (116 linhas)
│   │   │
│   │   │  DATACLASSES:
│   │   ├── QuickCheckRequest          # prompt
│   │   ├── QuickCheckResponse         # salience, should_emerge, mode, detected_keywords
│   │   │
│   │   │  FUNÇÕES:
│   │   └── analyze_prompt(prompt)     # → QuickCheckResponse (<100ms target)
│   │
│   └── daimon_routes.py               # FastAPI Router (391 linhas)
│       │
│       │  ENDPOINTS (8):
│       ├── POST /api/daimon/quick-check         # Análise de risco
│       ├── POST /api/daimon/shell/batch         # Recebe batches do shell_watcher
│       ├── POST /api/daimon/claude/event        # Recebe eventos do claude_watcher
│       ├── POST /api/daimon/session/end         # Grava sessão como precedente
│       ├── GET  /api/daimon/preferences/learned # Preferências aprendidas
│       ├── POST /api/daimon/reflect             # Trigger reflexão manual
│       ├── GET  /api/daimon/memories/recent     # Memórias recentes
│       └── GET  /api/daimon/health              # Health check
│
├── memory/
│   ├── __init__.py                    # exports: MemoryStore, PrecedentSystem, etc
│   ├── db_utils.py                    # Utilitários SQLite (29 linhas)
│   │   └── init_sqlite_db()
│   │
│   ├── optimized_store.py             # Memória SQLite+FTS5 (504 linhas)
│   │   │
│   │   │  DATACLASSES:
│   │   ├── MemoryTimestamps           # created_at, accessed_at, access_count
│   │   ├── MemoryItem                 # id, content, category, importance, timestamps
│   │   ├── SearchResult               # item, score, match_type
│   │   │
│   │   │  CLASSES:
│   │   └── MemoryStore                # DB: ~/.daimon/memory/memories.db
│   │       ├── add(content, category, importance)
│   │       ├── search(query)          # FTS5 <10ms
│   │       ├── get_by_category()
│   │       ├── get_recent()
│   │       ├── decay_importance()     # Decai importância ao longo do tempo
│   │       ├── cleanup()              # Remove memórias antigas/baixa importância
│   │       ├── get_stats()
│   │       └── delete()
│   │
│   ├── precedent_models.py            # Modelos de precedente (122 linhas)
│   │   │
│   │   │  TYPES:
│   │   ├── OutcomeType                # "success"|"failure"|"partial"|"unknown"
│   │   │
│   │   │  DATACLASSES:
│   │   ├── PrecedentMeta              # created_at, updated_at, relevance, application_count, success_rate, tags
│   │   ├── Precedent                  # id, context, decision, outcome, lesson, meta
│   │   └── PrecedentMatch             # precedent, score, match_reason
│   │
│   └── precedent_system.py            # Jurisprudência SQLite+FTS5 (367 linhas)
│       │
│       │  CLASSES:
│       └── PrecedentSystem            # DB: ~/.daimon/memory/precedents.db
│           ├── record(context, decision, outcome, meta)
│           ├── search(query)          # FTS5 <20ms
│           ├── update_outcome()       # Ajusta relevância baseado em sucesso/falha
│           ├── apply_precedent()      # Registra aplicação (feedback loop)
│           ├── get_by_outcome()
│           ├── get_top_precedents()
│           ├── get_stats()
│           └── delete()
│
├── learners/
│   ├── __init__.py                    # exports: PreferenceLearner, ReflectionEngine, get_engine
│   │
│   ├── preference_learner.py          # Detecção de padrões (463 linhas)
│   │   │
│   │   │  CONSTANTES:
│   │   ├── APPROVAL_PATTERNS          # regex: sim, yes, ok, perfeito, aceito...
│   │   ├── REJECTION_PATTERNS         # regex: não, no, errado, pare, cancela...
│   │   ├── CATEGORY_KEYWORDS          # code_style, verbosity, testing, architecture...
│   │   │
│   │   │  DATACLASSES:
│   │   ├── PreferenceSignal           # timestamp, signal_type, context, category, strength, session_id
│   │   │
│   │   │  CLASSES:
│   │   └── PreferenceLearner
│   │       ├── scan_sessions(since_hours=24)    # Escaneia ~/.claude/projects/*/*.jsonl
│   │       ├── _detect_signal_type(content)     # → "approval"|"rejection"|None
│   │       ├── _infer_category(text)            # → categoria baseada em keywords
│   │       ├── get_preference_summary()         # → {category: {approval_rate, total_signals}}
│   │       ├── get_actionable_insights()        # → [{category, action, confidence, suggestion}]
│   │       ├── clear()
│   │       └── get_stats()
│   │
│   └── reflection_engine.py           # Orquestração (341 linhas)
│       │
│       │  DATACLASSES:
│       ├── ReflectionConfig           # interval_minutes=30, rejection_threshold=5, approval_threshold=10
│       ├── ReflectionStats            # total_reflections, total_updates, last_reflection
│       │
│       │  CLASSES:
│       ├── ReflectionEngine
│       │   ├── start()                # async - inicia loop de reflexão (30min)
│       │   ├── stop()                 # async
│       │   ├── _run_loop()            # async - loop principal
│       │   ├── _check_triggers()      # Verifica triggers temporais/threshold
│       │   ├── reflect()              # async - executa reflexão completa
│       │   ├── _apply_insights()      # async - atualiza CLAUDE.md via ConfigRefiner
│       │   ├── _notify_update()       # async - notifica usuário via notify-send
│       │   ├── get_status()
│       │   ├── get_learner()
│       │   └── get_refiner()
│       │
│       │  FUNÇÕES (Singleton):
│       ├── get_engine()               # Retorna singleton
│       └── reset_engine()             # Para testes
│
├── actuators/
│   ├── __init__.py                    # exports: ConfigRefiner
│   │
│   └── config_refiner.py              # Atualizador de CLAUDE.md (376 linhas)
│       │
│       │  CONSTANTES:
│       ├── DAIMON_SECTION_START       # <!-- DAIMON:AUTO:START -->
│       ├── DAIMON_SECTION_END         # <!-- DAIMON:AUTO:END -->
│       │
│       │  CLASSES:
│       └── ConfigRefiner
│           ├── update_preferences(insights)     # Atualiza ~/.claude/CLAUDE.md
│           ├── _create_backup()                 # → ~/.claude/backups/CLAUDE.md.{timestamp}
│           ├── _generate_section(insights)      # Gera markdown com preferências
│           ├── _merge_content()                 # Preserva conteúdo manual
│           ├── get_current_preferences()
│           ├── get_manual_content()
│           ├── get_backup_list()
│           └── restore_backup()
│
├── corpus/
│   ├── __init__.py                    # exports: CorpusManager, WisdomText, TextMetadata
│   │
│   ├── manager.py                     # Gestão de textos (459 linhas)
│   │   │
│   │   │  CONSTANTES:
│   │   ├── CATEGORIES                 # filosofia/gregos, estoicos, teologia, ciencia, logica, etica
│   │   │
│   │   │  DATACLASSES:
│   │   ├── TextMetadata               # source, added_at, relevance_score, themes
│   │   ├── WisdomText                 # id, author, title, category, content, themes, metadata
│   │   │
│   │   │  CLASSES:
│   │   └── CorpusManager              # Dir: ~/.daimon/corpus/
│   │       ├── add_text(author, title, category, content, metadata)
│   │       ├── get_text(id)
│   │       ├── get_by_author()
│   │       ├── get_by_theme()
│   │       ├── get_by_category()
│   │       ├── search(query)
│   │       ├── list_authors()
│   │       ├── list_themes()
│   │       ├── get_stats()
│   │       └── delete_text()
│   │
│   └── bootstrap_texts.py             # Textos iniciais (242 linhas)
│       │
│       │  CONSTANTES:
│       ├── BOOTSTRAP_TEXTS            # 10 textos fundamentais:
│       │   ├── Marcus Aurelius - Meditations
│       │   ├── Epictetus - Enchiridion
│       │   ├── Seneca - On the Shortness of Life
│       │   ├── Aristotle - Nicomachean Ethics
│       │   ├── Aristotle - Organon
│       │   ├── Plato - Republic (Allegory of the Cave)
│       │   ├── Socrates - Apology
│       │   ├── Kant - Categorical Imperative
│       │   ├── Popper - Falsifiability
│       │   └── Feynman - On Scientific Method
│       │
│       │  FUNÇÕES:
│       └── bootstrap_corpus()         # Inicializa corpus com textos fundamentais
│
├── dashboard/
│   ├── __init__.py                    # exports: app, run_dashboard
│   │
│   ├── app.py                         # FastAPI Dashboard (411 linhas)
│   │   │
│   │   │  ENDPOINTS (18):
│   │   ├── GET  /                           # HTML principal
│   │   ├── GET  /api/status                 # Status de todos serviços
│   │   ├── GET  /api/preferences            # Preferências do ReflectionEngine
│   │   ├── POST /api/reflect                # Trigger reflexão manual
│   │   ├── GET  /api/claude-md              # Ler CLAUDE.md
│   │   ├── PUT  /api/claude-md              # Atualizar CLAUDE.md
│   │   ├── GET  /api/collectors             # Status dos collectors
│   │   ├── POST /api/collectors/{name}/start
│   │   ├── POST /api/collectors/{name}/stop
│   │   ├── GET  /api/backups                # Lista backups CLAUDE.md
│   │   ├── POST /api/backups/restore        # Restaura backup
│   │   ├── GET  /api/corpus/stats
│   │   ├── GET  /api/corpus/search?q=
│   │   ├── GET  /api/precedents/stats
│   │   ├── GET  /api/precedents/search?q=
│   │   ├── GET  /api/memory/stats
│   │   └── GET  /api/memory/search?q=
│   │   │
│   │   │  FUNÇÕES:
│   │   ├── run_dashboard(host, port=8003)
│   │   ├── _check_service(url)              # async - verifica health
│   │   ├── _check_socket()                  # Verifica ~/.daimon/daimon.sock
│   │   └── _check_process(name)             # pgrep -f collectors/{name}
│   │
│   └── templates/
│       └── index.html                 # UI com Tailwind + Alpine.js
│
├── .claude/
│   ├── agents/
│   │   └── noesis-sage.md             # Subagent com workflow definido
│   │       ├── Tools: noesis_consult, noesis_tribunal, noesis_precedent, noesis_confront
│   │       ├── Workflow: 1.Precedent → 2.Tribunal → 3.Confront → 4.Questions → 5.Options
│   │       └── Output: "NOESIS\nPrecedents: #N\nQuestions:\nTribunal:\n[p] [d] [e]"
│   │
│   ├── hooks/
│   │   └── noesis_hook.py             # Hook <500ms (237 linhas)
│   │       ├── classify_risk(text)    # → "high"|"medium"|"low"
│   │       ├── quick_check(prompt)    # → POST /api/consciousness/quick-check
│   │       ├── handle_user_prompt_submit()
│   │       └── handle_pre_tool_use()  # Intercepta Bash
│   │
│   └── settings.json                  # Configura hooks para UserPromptSubmit e PreToolUse
│
└── tests/                             # 254 testes, 98% coverage
    ├── test_memory_store.py           # 36 testes
    ├── test_precedent_system.py       # 43 testes
    ├── test_preference_learner.py     # 60 testes
    ├── test_reflection_engine.py      # 30 testes
    ├── test_config_refiner.py         # 35 testes
    ├── test_corpus_manager.py         # 45 testes
    └── ...
```

---

## Fluxo de Dados

### 1. Fluxo Real-Time (Hook)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE SESSION                               │
│                                                                           │
│  User: "I want to delete all user records from production"               │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    .claude/hooks/noesis_hook.py                          │
│                           (timeout: 1000ms)                              │
│                                                                           │
│   1. classify_risk(prompt)                                               │
│      → "delete" in HIGH_RISK_KEYWORDS → "high"                          │
│                                                                           │
│   2. quick_check(prompt) → POST localhost:8001/api/consciousness/quick-check │
│      → {salience: 0.9, should_emerge: true, mode: "emerge"}             │
│                                                                           │
│   3. Output:                                                             │
│      {"hookSpecificOutput": {"additionalContext": "NOESIS: ..."}}       │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       noesis-sage SUBAGENT                               │
│                                                                           │
│   1. noesis_precedent("delete user records production")                  │
│      → Busca decisões similares no histórico                            │
│                                                                           │
│   2. noesis_tribunal("delete all user records")                          │
│      → VERITAS: FAIL (data loss risk)                                   │
│      → SOPHIA: FAIL (irreversible)                                      │
│      → DIKE: FAIL (user rights)                                         │
│      → Consensus: FAIL 100%                                             │
│                                                                           │
│   3. Output:                                                             │
│      NOESIS                                                              │
│      Precedents: #1 found - Data purge 2023 deleted active users        │
│      Questions:                                                          │
│      1. Backup exists?                                                   │
│      2. Date filter verified?                                            │
│      3. Legal retention requirements?                                    │
│      Tribunal: FAIL                                                      │
│      [p] [d] [e]                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2. Fluxo Background (Collectors + Learning)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          COLLECTORS (Parallel)                           │
│                                                                           │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │     shell_watcher.py        │  │      claude_watcher.py          │   │
│  │                             │  │                                  │   │
│  │  ~/.daimon/daimon.sock      │  │  Polling: 5s intervals          │   │
│  │                             │  │                                  │   │
│  │  ← zshrc hook sends:        │  │  Scans: ~/.claude/projects/     │   │
│  │    {command, pwd, exit_code,│  │         */sessions/*.jsonl      │   │
│  │     duration, git_branch}   │  │                                  │   │
│  │                             │  │  Extracts:                       │   │
│  │  Detects:                   │  │  - intention (create/fix/etc)   │   │
│  │  - error_streak (≥3 fails)  │  │  - files_touched                │   │
│  │  - repetitive_command       │  │  - project name                 │   │
│  │                             │  │                                  │   │
│  │  Sends every 30s OR on      │  │  Privacy: NO prompt content     │   │
│  │  significant command        │  │                                  │   │
│  └──────────────┬──────────────┘  └───────────────┬─────────────────┘   │
│                 │                                  │                      │
│                 ▼                                  ▼                      │
│        POST /api/daimon/shell/batch    POST /api/daimon/claude/event    │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      REFLECTION ENGINE                                    │
│                      (loop: 30min / threshold / manual)                  │
│                                                                           │
│  Triggers:                                                               │
│  - Temporal: cada 30 minutos                                            │
│  - Threshold: >5 rejeições OU >10 aprovações mesma categoria            │
│  - Manual: POST /api/daimon/reflect                                     │
│                                                                           │
│  1. PreferenceLearner.scan_sessions(since_hours=48)                     │
│     → Escaneia ~/.claude/projects/*/*.jsonl                             │
│     → Detecta padrões: "sim", "ok" → approval                           │
│                        "não", "errado" → rejection                      │
│     → Infere categoria: testing, code_style, verbosity...               │
│                                                                           │
│  2. PreferenceLearner.get_actionable_insights()                         │
│     → [{category: "testing", action: "reinforce", confidence: 0.9}]     │
│                                                                           │
│  3. ConfigRefiner.update_preferences(insights)                          │
│     → Backup: ~/.claude/backups/CLAUDE.md.{timestamp}                   │
│     → Merge: preserva conteúdo manual                                   │
│     → Write: ~/.claude/CLAUDE.md                                        │
│                                                                           │
│  4. notify-send "DAIMON atualizou preferências"                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Storage

```
~/.daimon/
├── daimon.sock                # Unix socket do shell_watcher
├── memory/
│   ├── memories.db            # SQLite + FTS5 (eventos, preferências)
│   └── precedents.db          # SQLite + FTS5 (decisões passadas)
└── corpus/
    ├── filosofia/
    │   ├── gregos/            # Aristotle, Plato, Socrates
    │   └── estoicos/          # Marcus Aurelius, Epictetus, Seneca
    ├── teologia/
    ├── ciencia/               # Popper, Feynman
    ├── logica/
    ├── etica/                 # Kant
    └── _index/
        ├── by_theme.json
        └── by_author.json

~/.claude/
├── CLAUDE.md                  # Preferências (auto-updated por DAIMON)
├── backups/
│   ├── CLAUDE.md.{timestamp}  # Backups (últimos 10)
│   └── update_log.jsonl       # Log de atualizações
└── projects/
    └── {project}/
        └── sessions/*.jsonl   # Sessões Claude Code (fonte do learning)
```

---

## Instalação

### Opção 1: Script Automático (Recomendado)

```bash
cd /media/juan/DATA/projetos/daimon
./install.sh --all
```

Isso irá:
1. Instalar dependências Python
2. Criar diretórios de dados (`~/.daimon/`)
3. Copiar arquivos de integração Claude Code
4. Configurar serviço systemd
5. Configurar hooks do shell

### Opção 2: Manual

```bash
# 1. Dependências
pip install fastmcp httpx pydantic watchdog pytest pytest-asyncio uvicorn jinja2

# 2. Registrar MCP Server
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py

# 3. Copiar Subagent e Hooks
cp .claude/agents/noesis-sage.md ~/.claude/agents/
cp -r .claude/hooks ~/.claude/
cp .claude/settings.json ~/.claude/

# 4. Configurar Shell Watcher (opcional)
python collectors/shell_watcher.py --zshrc >> ~/.zshrc
source ~/.zshrc
```

---

## Iniciar no Boot (Systemd)

```bash
# Habilitar e iniciar
systemctl --user enable --now daimon

# Verificar status
systemctl --user status daimon

# Ver logs
journalctl --user -u daimon -f

# Parar
systemctl --user stop daimon

# Desabilitar autostart
systemctl --user disable daimon
```

**Arquivos criados:**
- Serviço: `~/.config/systemd/user/daimon.service`
- Logs: `~/.daimon/logs/daimon.log`
- PID: `~/.daimon/daimon.pid`
- Estado: `~/.daimon/state.json`

---

## Iniciar Manualmente (Sem Systemd)

```bash
# Opção 1: Daemon unificado (recomendado)
python daimon_daemon.py           # Foreground
python daimon_daemon.py --daemon  # Background
python daimon_daemon.py --status  # Verificar
python daimon_daemon.py --stop    # Parar

# Opção 2: Componentes separados
# Terminal 1: Shell Watcher
python collectors/shell_watcher.py --daemon

# Terminal 2: Claude Watcher
python collectors/claude_watcher.py --daemon

# Terminal 3: Dashboard
python -m uvicorn dashboard.app:app --port 8003

# Terminal 4: Reflection Engine
python -c "
import asyncio
from learners import get_engine
asyncio.run(get_engine().start())
"
```

---

## API Endpoints

### DAIMON Routes (integrar em NOESIS)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/daimon/quick-check` | POST | Análise heurística <100ms |
| `/api/daimon/shell/batch` | POST | Recebe batches do shell_watcher |
| `/api/daimon/claude/event` | POST | Recebe eventos do claude_watcher |
| `/api/daimon/session/end` | POST | Grava sessão como precedente |
| `/api/daimon/preferences/learned` | GET | Preferências aprendidas |
| `/api/daimon/reflect` | POST | Trigger reflexão manual |
| `/api/daimon/memories/recent` | GET | Memórias recentes |
| `/api/daimon/health` | GET | Health check |

### Dashboard (porta 8003)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Interface HTML |
| `/api/status` | GET | Status de todos serviços |
| `/api/preferences` | GET | Preferências do ReflectionEngine |
| `/api/reflect` | POST | Trigger reflexão |
| `/api/claude-md` | GET/PUT | Ler/atualizar CLAUDE.md |
| `/api/collectors` | GET | Status collectors |
| `/api/collectors/{name}/start` | POST | Iniciar collector |
| `/api/collectors/{name}/stop` | POST | Parar collector |
| `/api/backups` | GET | Listar backups |
| `/api/backups/restore` | POST | Restaurar backup |
| `/api/corpus/stats` | GET | Estatísticas corpus |
| `/api/corpus/search` | GET | Busca corpus |
| `/api/precedents/stats` | GET | Estatísticas precedentes |
| `/api/precedents/search` | GET | Busca precedentes |
| `/api/memory/stats` | GET | Estatísticas memória |
| `/api/memory/search` | GET | Busca memória |

---

## Qualidade

| Métrica | Valor |
|---------|-------|
| **Testes** | 254 passando |
| **Coverage** | 98% |
| **Pylint** | 10.00/10 |
| **Linhas de Código** | ~5.000 |

### Coverage por Módulo

| Módulo | Coverage |
|--------|----------|
| memory/optimized_store.py | 100% |
| memory/precedent_system.py | 100% |
| memory/precedent_models.py | 100% |
| memory/db_utils.py | 100% |
| corpus/manager.py | 98% |
| corpus/bootstrap_texts.py | 100% |
| learners/preference_learner.py | 96% |
| learners/reflection_engine.py | 97% |
| actuators/config_refiner.py | 94% |

---

## Princípios

1. **Silêncio é Ouro** - Só emergir quando verdadeiramente significativo
2. **Perguntas, Não Respostas** - Amplificar pensamento via diálogo socrático
3. **Privacidade Primeiro** - Captura INTENÇÃO, não CONTEÚDO
4. **Heartbeat Pattern** - Estados que se fundem, não eventos isolados
5. **Unix Philosophy** - Cada módulo faz UMA coisa bem
6. **<500ms Hook** - Heurística rápida, sem bloquear
7. **Graceful Degradation** - Fallbacks para todas falhas externas
8. **Safe Actuators** - Sempre backup antes de modificar

---

## Dependências Externas

DAIMON depende do NOESIS rodando:

| Serviço | Porta | Propósito |
|---------|-------|-----------|
| maximus_core_service | 8001 | Consciência, quick-check, confrontation |
| metacognitive_reflector | 8002 | Tribunal, reflexão |

---

*DAIMON v2.0 - Exocórtex Pessoal com Aprendizado*
*Dezembro 2025*

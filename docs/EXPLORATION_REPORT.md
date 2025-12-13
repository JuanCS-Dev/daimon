# DAIMON Codebase Exploration Report

**Date**: 2025-12-12
**Author**: Claude Code (Automated Exploration)
**Total Lines of Python**: ~5,000 (DAIMON v2 modules)

---

## 1. Directory Structure

```
/media/juan/DATA/projetos/daimon/
├── integrations/
│   └── mcp_server.py          # MCP Server (5 tools for Claude Code)
├── collectors/
│   ├── shell_watcher.py       # Captures shell commands via heartbeat
│   └── claude_watcher.py      # Monitors Claude Code sessions
├── endpoints/
│   ├── quick_check.py         # Fast heuristic risk analysis (<100ms)
│   ├── daimon_routes.py       # FastAPI routes (8 endpoints)
│   └── constants.py           # Risk keywords (HIGH/MEDIUM)
├── memory/
│   ├── precedent_system.py    # SQLite FTS5 jurisprudence database
│   ├── precedent_models.py    # Data models for precedents
│   ├── optimized_store.py     # MemoryStore with FTS5 search
│   └── db_utils.py            # Database initialization utilities
├── learners/
│   ├── preference_learner.py  # Detects approval/rejection patterns
│   └── reflection_engine.py   # Orchestrates learner + config updates
├── actuators/
│   └── config_refiner.py      # Updates ~/.claude/CLAUDE.md
├── corpus/
│   ├── manager.py             # Wisdom text collection
│   └── bootstrap_texts.py     # Seeded corpus data (Stoics, Greeks, etc.)
├── dashboard/
│   ├── app.py                 # FastAPI dashboard on port 8003
│   └── templates/             # HTML templates
├── .claude/
│   ├── agents/noesis-sage.md  # Wise subagent (questions only)
│   ├── hooks/noesis_hook.py   # Fast prompt interceptor (<500ms)
│   └── settings.json          # Hook configuration
├── tests/
│   └── test_*.py              # 254 passing tests, 98% coverage
└── docs/
    └── README.md
```

---

## 2. Module Analysis

### 2.1 Memory System (100% coverage)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `optimized_store.py` | 503 | SQLite + FTS5 memory storage | WORKING |
| `precedent_system.py` | 367 | Jurisprudence-style decisions | WORKING |
| `precedent_models.py` | 122 | Dataclasses for precedents | WORKING |
| `db_utils.py` | 28 | Database init utilities | WORKING |

**Features**:
- Full-text search via FTS5 (<10ms target)
- Importance decay over time
- Relevance feedback loop (success increases relevance)
- Category-based filtering

### 2.2 Learners System (96% coverage)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `preference_learner.py` | 463 | Pattern detection | WORKING |
| `reflection_engine.py` | 341 | Orchestration | WORKING |

**Features**:
- Scans `~/.claude/projects/` for JSONL sessions
- Detects approval patterns: "sim", "ok", "perfeito", "yes"
- Detects rejection patterns: "não", "errado", "no", "wrong"
- Categories: code_style, verbosity, testing, architecture, documentation

**Triggers**:
- Temporal: Every 30 minutes
- Threshold: >5 rejections OR >10 approvals same category
- Manual: `/daimon reflect`

### 2.3 Actuators System (94% coverage)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `config_refiner.py` | 376 | CLAUDE.md updater | WORKING |

**Safety Features**:
- Creates timestamped backups before modifying
- Keeps only last 10 backups
- Marks auto-generated section with HTML comments
- Preserves user's manual content

### 2.4 Corpus System (98-100% coverage)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `manager.py` | 459 | Wisdom text management | WORKING |
| `bootstrap_texts.py` | 242 | Initial texts | WORKING |

**Categories**:
- filosofia/gregos (Aristotle, Plato, Socrates)
- filosofia/estoicos (Marcus Aurelius, Epictetus, Seneca)
- logica, ciencia, etica, teologia

### 2.5 Integrations (MCP Server)

| Tool | Endpoint | Purpose |
|------|----------|---------|
| `noesis_consult` | `/v1/consciousness/introspect` | Maieutic questioning |
| `noesis_tribunal` | `/reflect/verdict` | 3-judge ethical review |
| `noesis_precedent` | `/reflect/verdict` | Search past decisions |
| `noesis_confront` | `/v1/exocortex/confront` | Socratic confrontation |
| `noesis_health` | `/health` | Service status |

### 2.6 Collectors

| File | Purpose | Pattern |
|------|---------|---------|
| `shell_watcher.py` | Shell command capture | Heartbeat (30s batches) |
| `claude_watcher.py` | Session monitoring | Polling (5s intervals) |

**Privacy-First**: Captures INTENT, not CONTENT.

---

## 3. Test Coverage Summary

| Module | Coverage | Tests |
|--------|----------|-------|
| memory/optimized_store.py | 100% | 36 |
| memory/precedent_system.py | 100% | 43 |
| memory/precedent_models.py | 100% | - |
| memory/db_utils.py | 100% | 4 |
| corpus/manager.py | 98% | 45 |
| corpus/bootstrap_texts.py | 100% | - |
| learners/preference_learner.py | 96% | 60 |
| learners/reflection_engine.py | 97% | 30 |
| actuators/config_refiner.py | 94% | 35 |
| **TOTAL** | **98%** | **254** |

---

## 4. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USER'S CLAUDE CODE SESSION                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          HOOK (<500ms)                               │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Classify Risk   │───▶│ Quick-Check API  │───▶│ Should Emerge?│  │
│  └─────────────────┘    └──────────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            [Low Risk]                      [High Risk]
                │                               │
                ▼                               ▼
            Silent                    ┌─────────────────┐
                                      │  noesis-sage    │
                                      │   subagent      │
                                      └────────┬────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
           ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
           │   Precedent   │          │   Tribunal    │          │   Confront    │
           │    Search     │          │   (3 judges)  │          │  (Socratic)   │
           └───────────────┘          └───────────────┘          └───────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               ▼
                                    ┌─────────────────┐
                                    │ Questions Only  │
                                    │ (no execution)  │
                                    └─────────────────┘


                         BACKGROUND LEARNING LOOP

┌─────────────────────────────────────────────────────────────────────┐
│                        COLLECTORS (Parallel)                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐    │
│  │     Shell Watcher       │    │      Claude Watcher         │    │
│  │   (heartbeat 30s)       │    │     (polling 5s)            │    │
│  └───────────┬─────────────┘    └──────────────┬──────────────┘    │
└──────────────┼─────────────────────────────────┼────────────────────┘
               │                                 │
               ▼                                 ▼
      /api/daimon/shell/batch           /api/daimon/claude/event
               │                                 │
               └─────────────┬───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  MemoryStore    │
                    │  (SQLite FTS5)  │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    REFLECTION ENGINE (30min)                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐    │
│  │  PreferenceLearner      │───▶│      ConfigRefiner          │    │
│  │  (pattern detection)    │    │  (CLAUDE.md updater)        │    │
│  └─────────────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. External Dependencies

### Required Services (NOESIS)

| Service | Port | Purpose |
|---------|------|---------|
| maximus_core_service | 8001 | Consciousness state |
| metacognitive_reflector | 8002 | Tribunal verdicts |

### Python Dependencies

```toml
fastmcp>=0.1.0      # MCP framework
httpx>=0.25.0       # Async HTTP
watchdog>=3.0.0     # File monitoring
pydantic>=2.0.0     # Data validation
```

---

## 6. Key Findings

### Working Well
- Memory system with FTS5 search (<10ms)
- Preference learning from session analysis
- Config refinement with safety backups
- Corpus management with philosophical texts
- 98% test coverage with 254 tests

### Architecture Strengths
- Privacy-first (captures intent, not content)
- Graceful degradation (fallbacks for all services)
- Modular design (<500 lines per file)
- Comprehensive test coverage

### Uncovered Code (2%)
All uncovered lines are defensive exception handlers:
- `except OSError/IOError` for file operations
- `except asyncio.TimeoutError` for notifications
- Error logging in async loops

---

## 7. File Statistics

| Category | Count |
|----------|-------|
| Python modules | 14 |
| Test files | 7 |
| Total tests | 254 |
| Pass rate | 100% |
| Coverage | 98% |
| Lines of code | ~5,000 |

---

*Report generated by automated codebase exploration*

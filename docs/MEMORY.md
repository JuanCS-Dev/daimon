# DAIMON Memory

**Sistema de Memória - Persistência e Recuperação de Dados**

---

## Visão Geral

O sistema de memória do DAIMON é composto por três componentes distintos, cada um otimizado para um tipo específico de dado.

### Componentes

| Componente | Propósito | Status |
|------------|-----------|--------|
| **ActivityStore** | Armazenar heartbeats dos collectors | ✅ Funcional |
| **PrecedentSystem** | Jurisprudência de decisões | ✅ Funcional |
| **MemoryStore** | Memória semântica (embeddings) | ⚠️ Backlog |

---

## Arquitetura

```
memory/
├── __init__.py           # Exports e singletons
├── db_utils.py           # Utilitários SQLite
├── activity_store.py     # Heartbeats dos collectors
├── precedent_system.py   # Decisões e jurisprudência
├── precedent_models.py   # Modelos do PrecedentSystem
└── optimized_store.py    # Memória semântica (FTS5)
```

### Storage

```
~/.daimon/
├── activity/
│   └── activities.db     # SQLite - heartbeats (~676KB+)
├── memory/
│   ├── memories.db       # SQLite + FTS5 - memória semântica
│   └── precedents.db     # SQLite + FTS5 - jurisprudência
└── corpus/               # Textos de sabedoria (ver CORPUS.md)
```

---

## 1. ActivityStore

**Arquivo:** `memory/activity_store.py`
**Status:** ✅ Funcional

Armazena todos os heartbeats coletados pelos watchers.

### Schema

```sql
CREATE TABLE activities (
    id TEXT PRIMARY KEY,
    watcher_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    data JSON NOT NULL
);

CREATE INDEX idx_activities_watcher ON activities(watcher_type);
CREATE INDEX idx_activities_timestamp ON activities(timestamp DESC);
```

### API

```python
from memory.activity_store import get_activity_store

store = get_activity_store()

# Adicionar registro
store.add(
    watcher_type="shell",
    timestamp=datetime.now(),
    data={"command": "ls", "exit_code": 0}
)

# Adicionar batch
store.add_batch([
    ActivityRecord(watcher_type="shell", timestamp=dt1, data=d1),
    ActivityRecord(watcher_type="shell", timestamp=dt2, data=d2),
])

# Consultar
records = store.query(watcher_type="claude", limit=100)
records = store.get_recent(hours=24)

# Agregações
window_time = store.aggregate_window_time(hours=8)
# → {"VSCode": 3600, "Chrome": 1800, ...}

domain_time = store.aggregate_domain_time(hours=24)
# → {"github.com": 1200, "stackoverflow.com": 600, ...}

# Estatísticas
stats = store.get_stats()
# → {"total_records": 5000, "by_watcher": {"shell": 2000, "claude": 500, ...}}

# Sumário
summary = store.get_summary(hours=24)
# → ActivitySummary(total_records=500, watchers=["shell", "claude"], ...)

# Limpeza
store.cleanup(days_old=30)
```

### Watchers Integrados

| Watcher | Integração | Via |
|---------|------------|-----|
| shell | ✅ | `flush()` |
| claude | ✅ | `flush()` |
| window | ✅ | `flush()` |
| input | ✅ | `flush()` |
| afk | ✅ | `flush()` |
| browser | ✅ | `flush()` |

---

## 2. PrecedentSystem

**Arquivo:** `memory/precedent_system.py`
**Status:** ✅ Funcional

Sistema de jurisprudência para decisões passadas. Inspirado em sistemas legais: decisões anteriores informam decisões futuras.

### Schema

```sql
CREATE TABLE precedents (
    id TEXT PRIMARY KEY,
    context TEXT NOT NULL,        -- Situação/contexto
    decision TEXT NOT NULL,       -- Decisão tomada
    outcome TEXT DEFAULT 'unknown', -- success/failure/partial/unknown
    lesson TEXT DEFAULT '',       -- Lição aprendida
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    relevance REAL DEFAULT 0.5,   -- 0.0-1.0, ajusta com aplicação
    application_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    tags TEXT DEFAULT '[]'
);

-- FTS5 para busca textual
CREATE VIRTUAL TABLE precedents_fts
USING fts5(context, decision, lesson, content=precedents);
```

### Ciclo de Vida de um Precedente

```
1. CRIAÇÃO
   ├── Via NOESIS Tribunal (preferencial)
   └── Via fallback local (quando NOESIS indisponível)

2. BUSCA
   └── FTS5 search com filtro por outcome e relevance

3. APLICAÇÃO
   ├── apply_precedent(id, was_successful)
   └── Ajusta relevance e success_rate

4. ATUALIZAÇÃO
   └── update_outcome(id, outcome, lesson)
```

### API

```python
from memory.precedent_system import PrecedentSystem

system = PrecedentSystem()

# Criar precedente
precedent_id = system.record(
    context="User asked to delete production database",
    decision="Required explicit confirmation and backup verification",
    outcome="success",
    meta=PrecedentMeta(tags=["security", "database"], relevance=0.8)
)

# Buscar precedentes
matches = system.search(
    query="delete database",
    outcome_filter="success",
    min_relevance=0.5,
    limit=10
)
# → [PrecedentMatch(precedent=..., score=0.85, match_reason="Matched 'delete'")]

# Aplicar precedente (feedback loop)
system.apply_precedent(precedent_id, was_successful=True)
# → Aumenta relevance em 0.1

# Atualizar outcome
system.update_outcome(
    precedent_id,
    outcome="success",
    lesson="Always require backup before destructive operations"
)

# Estatísticas
stats = system.get_stats()
# → {
#     "total_precedents": 45,
#     "average_relevance": 0.67,
#     "total_applications": 120,
#     "by_outcome": {
#         "success": {"count": 30, "avg_relevance": 0.75},
#         "failure": {"count": 10, "avg_relevance": 0.45},
#     }
# }
```

### Integração com NOESIS

Quando NOESIS está disponível, precedentes são criados via Tribunal:

```python
# Em daimon_routes.py:_create_real_precedent()
async with httpx.AsyncClient(timeout=5.0) as client:
    response = await client.post(
        f"{noesis_url}/reflect/verdict",
        json=payload,
    )
    if response.status_code == 200:
        result = response.json()
        precedent_id = result.get("precedent_id")
        # Também alimenta PreferenceLearner
        _feed_verdict_to_learner(result, request)
```

### Fallback Local

Quando NOESIS indisponível, precedentes são criados localmente:

```python
# Em daimon_routes.py:_create_real_precedent()
system = PrecedentSystem()
precedent_id = system.record(
    context=f"Session {session_id}: {summary}",
    decision=f"Changed {files_changed} files in {duration}min",
    outcome=outcome,
    meta=PrecedentMeta(tags=["session", "daimon"], relevance=0.5)
)
```

---

## 3. MemoryStore

**Arquivo:** `memory/optimized_store.py`
**Status:** ⚠️ Backlog (implementado mas não integrado ao fluxo principal)

Memória semântica com FTS5 para busca textual. Projetado para armazenar insights e padrões aprendidos.

### Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    importance REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0
);

-- FTS5 para busca semântica
CREATE VIRTUAL TABLE memories_fts
USING fts5(content, content=memories);
```

### API

```python
from memory.optimized_store import get_memory_store

store = get_memory_store()

# Adicionar memória
memory_id = store.add(
    content="User prefers concise responses",
    category="preference",
    importance=0.8
)

# Buscar
results = store.search(query="concise", limit=10)
# → [SearchResult(item=..., score=0.9, match_type="fts")]

# Por categoria
items = store.get_by_category("preference")

# Recentes
items = store.get_recent(limit=20)

# Decay (reduz importância ao longo do tempo)
store.decay_importance(days=7, factor=0.95)

# Cleanup (remove antigas/baixa importância)
store.cleanup(days_old=90, min_importance=0.2)
```

### Por que está em Backlog?

O MemoryStore foi projetado para armazenar embeddings semânticos, mas:

1. O fluxo atual usa ActivityStore + PrecedentSystem que atendem às necessidades
2. Embeddings requerem modelo de embedding (custo adicional)
3. FTS5 do PrecedentSystem já oferece busca textual suficiente

**Plano futuro**: Integrar quando houver necessidade de busca semântica verdadeira.

---

## Utilitários

### db_utils.py

```python
from memory.db_utils import init_sqlite_db

# Inicializa banco com configurações otimizadas
conn = init_sqlite_db(db_path)
# → Define WAL mode, synchronous=NORMAL, cache_size, etc.
```

### Configurações SQLite Aplicadas

```sql
PRAGMA journal_mode = WAL;      -- Write-Ahead Logging
PRAGMA synchronous = NORMAL;    -- Balanço performance/segurança
PRAGMA cache_size = -64000;     -- 64MB cache
PRAGMA temp_store = MEMORY;     -- Temp tables em memória
```

---

## Performance

### Benchmarks

| Operação | Tempo | Notas |
|----------|-------|-------|
| ActivityStore.add() | <1ms | Single insert |
| ActivityStore.add_batch(100) | <10ms | Batch insert |
| ActivityStore.query(limit=100) | <5ms | Com índice |
| PrecedentSystem.search() | <20ms | FTS5 query |
| PrecedentSystem.record() | <5ms | Com ID hash |

### Tamanhos Típicos

| Componente | Tamanho após 1 mês | Crescimento |
|------------|-------------------|-------------|
| ActivityStore | ~50-100MB | ~2MB/dia |
| PrecedentSystem | ~1-5MB | ~100KB/dia |
| MemoryStore | ~1MB | (não em uso) |

### Cleanup Automático

```python
# ActivityStore - remover registros > 30 dias
store.cleanup(days_old=30)

# MemoryStore - remover importância < 0.2 e > 90 dias
store.cleanup(days_old=90, min_importance=0.2)
```

---

## Testes

```bash
# Todos os testes de memory
python -m pytest tests/test_activity_store.py tests/test_memory_store.py \
    tests/test_precedent_system.py tests/test_db_utils.py -v

# Testes específicos
python -m pytest tests/test_precedent_system.py::TestSearch -v
```

---

## Limitações Honestas

1. **FTS5** não é busca semântica verdadeira - é busca textual
2. **Não há replicação** - dados locais apenas
3. **SQLite** tem limites de concorrência (WAL ajuda, mas não resolve tudo)
4. **MemoryStore** não está integrado - futuro incerto

---

*Documentação atualizada em 2025-12-13*

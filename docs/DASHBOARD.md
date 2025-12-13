# DAIMON Dashboard

**Interface Web de Controle e Monitoramento**

---

## Visão Geral

O Dashboard é uma interface web para monitorar e controlar o DAIMON. Construído com FastAPI + Jinja2 + Tailwind CSS + Alpine.js.

### Funcionalidades

- 📊 Status de todos os componentes em tempo real
- 🔍 Busca em corpus, precedentes e memória
- 📝 Visualização e edição do CLAUDE.md
- 🔄 Trigger manual de reflexão
- 📦 Gerenciamento de backups
- 🧠 Estado cognitivo e estilo de comunicação

---

## Arquitetura

```
dashboard/
├── __init__.py          # Exports
├── app.py               # FastAPI application
├── helpers.py           # Funções auxiliares
├── models.py            # Pydantic models
├── routes/
│   ├── __init__.py
│   ├── status.py        # Endpoints de status
│   ├── corpus.py        # Endpoints de corpus
│   ├── memory.py        # Endpoints de memória
│   └── cognitive.py     # Endpoints cognitivos
└── templates/
    └── index.html       # UI principal
```

---

## Iniciar o Dashboard

### Via daimon_daemon

```bash
python daimon_daemon.py  # Inclui dashboard na porta 8003
```

### Standalone

```bash
# Desenvolvimento
python -m uvicorn dashboard.app:app --port 8003 --reload

# Produção
python -m uvicorn dashboard.app:app --port 8003 --workers 2
```

### Via Python

```python
from dashboard import run_dashboard
run_dashboard(host="0.0.0.0", port=8003)
```

---

## Endpoints API

### Status

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Interface HTML principal |
| `/api/status` | GET | Status de todos os serviços |
| `/api/preferences` | GET | Preferências do ReflectionEngine |
| `/api/reflect` | POST | Trigger reflexão manual |
| `/api/collectors` | GET | Status dos collectors |
| `/api/collectors/{name}/start` | POST | Iniciar collector |
| `/api/collectors/{name}/stop` | POST | Parar collector |

### CLAUDE.md

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/claude-md` | GET | Ler conteúdo atual |
| `/api/claude-md` | PUT | Atualizar conteúdo |
| `/api/backups` | GET | Listar backups |
| `/api/backups/restore` | POST | Restaurar backup |

### Corpus

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/corpus/stats` | GET | Estatísticas do corpus |
| `/api/corpus/tree` | GET | Estrutura de diretórios |
| `/api/corpus/search` | GET | Buscar textos (`?q=query`) |
| `/api/corpus/texts` | GET | Listar textos (`?category=`) |
| `/api/corpus/text/{id}` | GET | Obter texto específico |
| `/api/corpus/text` | POST | Adicionar texto |
| `/api/corpus/text/{id}` | DELETE | Remover texto |

### Memória

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/memory/stats` | GET | Estatísticas da memória |
| `/api/memory/search` | GET | Buscar memórias (`?q=query`) |
| `/api/precedents/stats` | GET | Estatísticas de precedentes |
| `/api/precedents/search` | GET | Buscar precedentes (`?q=query`) |

### Cognitivo

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/cognitive` | GET | Estado cognitivo atual |
| `/api/cognitive/event` | POST | Registrar evento de keystroke |
| `/api/style` | GET | Perfil de estilo de comunicação |
| `/api/metacognitive` | GET | Análise metacognitiva |
| `/api/metacognitive/insights` | GET | Histórico de insights |

### Atividade

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/activity/stats` | GET | Estatísticas de atividade |
| `/api/activity/recent` | GET | Atividade recente (`?watcher=&hours=`) |
| `/api/activity/summary` | GET | Sumário de atividade |

### Browser (experimental)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/browser/status` | GET | Status do browser watcher |

---

## Exemplos de Uso

### Obter Status Geral

```bash
curl http://localhost:8003/api/status
```

```json
{
  "noesis_consciousness": "healthy",
  "noesis_reflector": "healthy",
  "dashboard": "healthy",
  "shell_watcher": "running",
  "claude_watcher": "running",
  "reflection_engine": "running"
}
```

### Buscar no Corpus

```bash
curl "http://localhost:8003/api/corpus/search?q=wisdom"
```

```json
{
  "query": "wisdom",
  "results": [
    {"id": "marcus-meditations", "title": "Meditations", "score": 0.85},
    {"id": "socrates-apology", "title": "Apology", "score": 0.72}
  ],
  "total": 2
}
```

### Trigger Reflexão

```bash
curl -X POST http://localhost:8003/api/reflect
```

```json
{
  "status": "completed",
  "signals": 15,
  "insights": 3,
  "updated": true
}
```

### Restaurar Backup

```bash
curl -X POST http://localhost:8003/api/backups/restore \
  -H "Content-Type: application/json" \
  -d '{"backup": "CLAUDE.md.2025-12-13T10-30-00.bak"}'
```

---

## Interface HTML

### Seções

1. **Status** - Cards com status de cada componente
2. **Preferences** - Visualização do CLAUDE.md
3. **Activity** - Gráficos de atividade recente
4. **Corpus** - Busca e navegação de textos
5. **Memory** - Busca em memórias e precedentes
6. **Cognitive** - Estado cognitivo e estilo

### Tecnologias

- **Tailwind CSS** - Estilização
- **Alpine.js** - Interatividade
- **Chart.js** - Gráficos (se necessário)
- **Jinja2** - Templates

---

## Helpers

**Arquivo:** `dashboard/helpers.py`

```python
# Verificar serviço HTTP
async def check_service(url: str, timeout: float = 2.0) -> bool:
    """Verifica se serviço HTTP está respondendo."""

# Verificar socket Unix
def check_socket(socket_path: str = SOCKET_PATH) -> bool:
    """Verifica se socket do shell_watcher existe."""

# Verificar processo
def check_process(name: str) -> bool:
    """Verifica se processo está rodando via pgrep."""

# URLs dos serviços
NOESIS_URL = "http://localhost:8001"
REFLECTOR_URL = "http://localhost:8002"
SOCKET_PATH = Path.home() / ".daimon" / "daimon.sock"
```

---

## Models

**Arquivo:** `dashboard/models.py`

```python
class ClaudeMdUpdate(BaseModel):
    """Payload para atualizar CLAUDE.md."""
    content: str

class CorpusTextCreate(BaseModel):
    """Payload para criar texto no corpus."""
    author: str
    title: str
    category: str
    content: str
    themes: List[str]
    source: str
    relevance: float

class BackupRestore(BaseModel):
    """Payload para restaurar backup."""
    backup: str
```

---

## Configuração

### Variáveis de Ambiente

```bash
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8003
NOESIS_URL=http://localhost:8001
REFLECTOR_URL=http://localhost:8002
```

### CORS

Por padrão, CORS está habilitado para desenvolvimento:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Em produção**, restringir origins.

---

## Segurança

### O que NÃO tem

⚠️ **Não há autenticação** - Dashboard é para uso local apenas

### Recomendações

1. **Não expor** na internet sem autenticação
2. **Usar** apenas em `localhost` ou rede confiável
3. **Considerar** nginx com basic auth se precisar expor

---

## Testes

```bash
# Testes do dashboard
python -m pytest tests/test_real_dashboard.py -v

# Testar endpoints manualmente
curl http://localhost:8003/api/status
curl http://localhost:8003/api/corpus/stats
```

---

## Troubleshooting

### Dashboard não inicia

```bash
# Verificar porta
lsof -i :8003

# Verificar dependências
pip install fastapi uvicorn jinja2
```

### Endpoints retornam erro

```bash
# Verificar logs
python -m uvicorn dashboard.app:app --port 8003 --log-level debug

# Verificar se NOESIS está rodando (para endpoints que dependem)
curl http://localhost:8001/api/consciousness/state
```

### Templates não carregam

```bash
# Verificar estrutura
ls -la dashboard/templates/

# Deve ter index.html
```

---

## Limitações Honestas

1. **Sem autenticação** - Não usar em redes públicas
2. **UI básica** - Funcional mas não polida
3. **Sem WebSocket** - Updates via polling manual
4. **Single-user** - Não projetado para múltiplos usuários

---

*Documentação atualizada em 2025-12-13*

# DAIMON Integrations

**Integração com NOESIS e MCP Server**

---

## Visão Geral

O DAIMON integra com dois sistemas externos:

1. **NOESIS** - Motor de consciência (Kuramoto, ESGT, Tribunal)
2. **Claude Code** - Via MCP Server e Hooks

### Arquitetura de Integração

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLAUDE CODE                                 │
│                                                                          │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐        │
│  │  MCP Server    │    │    Hooks       │    │   Subagent     │        │
│  │  (5 tools)     │    │  (noesis_hook) │    │ (noesis-sage)  │        │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘        │
│          │                     │                     │                  │
└──────────┼─────────────────────┼─────────────────────┼──────────────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              NOESIS                                      │
│                                                                          │
│  ┌────────────────────────────┐    ┌────────────────────────────┐      │
│  │  maximus_core_service      │    │  metacognitive_reflector   │      │
│  │  (porta 8001)              │    │  (porta 8002)              │      │
│  │                            │    │                            │      │
│  │  • Kuramoto Oscillators    │    │  • Tribunal (3 juízes)     │      │
│  │  • ESGT Protocol           │    │  • Reflection API          │      │
│  │  • Quick Check             │    │  • Verdict API             │      │
│  │  • Consciousness API       │    │                            │      │
│  └────────────────────────────┘    └────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Arquitetura de Arquivos

```
integrations/
├── __init__.py
├── mcp_server.py              # Entry point para Claude Code
└── mcp_tools/
    ├── __init__.py
    ├── config.py              # URLs e configurações
    ├── http_utils.py          # HTTP com retry/backoff
    ├── server.py              # FastMCP server setup
    ├── noesis_tools.py        # Tools que chamam NOESIS
    └── corpus_tools.py        # Tools para corpus local
```

---

## MCP Server

**Arquivo:** `integrations/mcp_server.py`

O MCP Server expõe ferramentas para Claude Code usar via Model Context Protocol.

### Registrar no Claude Code

```bash
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py
```

### Tools Disponíveis

| Tool | Endpoint NOESIS | Descrição |
|------|-----------------|-----------|
| `noesis_consult` | POST /v1/consciousness/introspect | Questionamento maiêutico |
| `noesis_tribunal` | POST /v1/exocortex/confront | Julgamento ético (3 juízes) |
| `noesis_precedent` | POST /reflect/verdict | Buscar/criar precedentes |
| `noesis_confront` | POST /v1/exocortex/confront | Confrontação socrática |
| `noesis_health` | GET /api/consciousness/state | Health check |

### Exemplo de Uso

```python
# noesis_consult - Retorna PERGUNTAS, não respostas
result = await noesis_consult(
    query="Should I refactor the authentication system?",
    depth=2  # 1=shallow, 2=moderate, 3=deep
)
# → {
#     "questions": [
#         "What specific issues are you experiencing?",
#         "Have you identified the root cause?",
#         "What are the risks of not refactoring?"
#     ],
#     "meta_awareness_level": 0.75
# }

# noesis_tribunal - Julgamento ético
result = await noesis_tribunal(
    action="Delete all user records from production",
    context="Cleanup of test data"
)
# → {
#     "verdict": "FAIL",
#     "judges": {
#         "VERITAS": {"score": 0.3, "reason": "Data loss risk"},
#         "SOPHIA": {"score": 0.2, "reason": "Irreversible action"},
#         "DIKE": {"score": 0.4, "reason": "User rights concern"}
#     },
#     "consensus": "FAIL 100%"
# }
```

---

## HTTP Utils (Retry/Backoff)

**Arquivo:** `integrations/mcp_tools/http_utils.py`

Todas as chamadas HTTP usam exponential backoff para resiliência.

### Configuração

```python
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5  # segundos
MAX_BACKOFF = 4.0      # segundos
BACKOFF_MULTIPLIER = 2.0
REQUEST_TIMEOUT = 10.0  # segundos
```

### Comportamento

| Erro | Retry? | Notas |
|------|--------|-------|
| Timeout | ✅ | Até MAX_RETRIES |
| 5xx (Server Error) | ✅ | Até MAX_RETRIES |
| 4xx (Client Error) | ❌ | Retorna imediatamente |
| Connection Error | ✅ | Até MAX_RETRIES |

### API

```python
from integrations.mcp_tools.http_utils import http_post, http_get

# POST com retry
result = await http_post(
    url="http://localhost:8001/v1/consciousness/introspect",
    payload={"query": "...", "depth": 2},
    timeout=10.0
)
# → {"narrative": "...", "meta_awareness_level": 0.8}
# ou {"error": "timeout", "message": "..."}

# GET com retry
result = await http_get(
    url="http://localhost:8001/api/consciousness/state",
    timeout=5.0
)
```

---

## Hooks

**Arquivo:** `.claude/hooks/noesis_hook.py`

Hooks interceptam prompts do usuário e chamadas de ferramentas.

### Eventos Interceptados

| Evento | Handler | Ação |
|--------|---------|------|
| UserPromptSubmit | `handle_user_prompt_submit()` | Análise de risco |
| PreToolUse | `handle_pre_tool_use()` | Intercepta Bash destrutivo |

### Fluxo do Hook

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    USER PROMPT SUBMIT HOOK                              │
│                                                                          │
│  1. classify_risk(prompt)                                               │
│     ├── "high" → HIGH_RISK_KEYWORDS (delete, drop, rm -rf, etc)        │
│     ├── "medium" → MEDIUM_RISK_KEYWORDS (refactor, migrate, etc)       │
│     └── "low" → Nenhum keyword detectado                               │
│                                                                          │
│  2. Se risk != "low":                                                   │
│     └── quick_check(prompt) → POST /api/consciousness/quick-check      │
│         └── {salience: 0.8, should_emerge: true, mode: "emerge"}       │
│                                                                          │
│  3. Se should_emerge:                                                   │
│     └── Adiciona contexto para noesis-sage ativar                      │
│                                                                          │
│  Timeout: 1000ms (fail-safe)                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Keywords de Risco

```python
# endpoints/constants.py

HIGH_RISK_KEYWORDS = [
    "delete", "drop", "rm -rf", "truncate", "production",
    "deploy", "release", "push --force", "reset --hard",
    "format", "wipe", "destroy", "purge",
]

MEDIUM_RISK_KEYWORDS = [
    "refactor", "migrate", "architecture", "redesign",
    "auth", "security", "permission", "credential",
    "database", "schema", "api", "endpoint",
]
```

### Configuração

```json
// .claude/settings.json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "python .claude/hooks/noesis_hook.py user_prompt_submit",
        "timeout": 1000
      }
    ],
    "PreToolUse": [
      {
        "type": "command",
        "command": "python .claude/hooks/noesis_hook.py pre_tool_use",
        "timeout": 500
      }
    ]
  }
}
```

---

## Subagent (noesis-sage)

**Arquivo:** `.claude/agents/noesis-sage.md`

Subagente especializado que usa as ferramentas NOESIS.

### Workflow

```
1. Precedent Search → noesis_precedent("query")
   └── Busca decisões similares no histórico

2. Tribunal Judgment → noesis_tribunal(action, context)
   └── Julgamento ético pelos 3 juízes

3. Socratic Confrontation → noesis_confront(premise)
   └── Questiona premissas subjacentes

4. Maieutic Questions → noesis_consult(query, depth)
   └── Gera perguntas para reflexão

5. Output Formatado:
   NOESIS
   Precedents: #N encontrados
   Questions:
   1. ...
   2. ...
   Tribunal: PASS/REVIEW/FAIL
   [p] proceed [d] discuss [e] explore
```

### Quando Ativar

O subagent é ativado automaticamente quando:
1. Hook detecta risco alto/médio
2. Usuário menciona "noesis" ou "tribunal"
3. Ação envolve keywords críticos

---

## Endpoints NOESIS

### maximus_core_service (porta 8001)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/consciousness/quick-check` | POST | Análise heurística rápida |
| `/api/consciousness/state` | GET | Estado atual da consciência |
| `/v1/consciousness/introspect` | POST | Introspecção profunda |
| `/v1/exocortex/confront` | POST | Confrontação socrática |
| `/v1/exocortex/journal` | POST | Registro de eventos |

### metacognitive_reflector (porta 8002)

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/reflect/verdict` | POST | Julgamento do Tribunal |
| `/reflect/analyze` | POST | Análise metacognitiva |
| `/api/reflector/health` | GET | Health check |

---

## Feedback Loop com NOESIS

Quando NOESIS retorna um verdict, este alimenta o PreferenceLearner:

```python
# Em daimon_routes.py
def _feed_verdict_to_learner(verdict: Dict, request: SessionEndRequest) -> None:
    """
    Fecha o feedback loop: Tribunal → PreferenceLearner.
    """
    verdict_result = verdict.get("verdict", "unknown")

    if verdict_result in ("approved", "success"):
        signal_type = "approval"
    elif verdict_result in ("rejected", "failure"):
        signal_type = "rejection"
    else:
        return  # Skip neutral

    signal = PreferenceSignal(
        signal_type=signal_type,
        category="session_quality",
        context=f"Tribunal verdict: {verdict.get('reasoning', '')}",
        strength=verdict.get("confidence", 0.5),
        session_id=request.session_id,
    )

    learner.signals.append(signal)
    learner._update_counts(signal)
```

---

## Graceful Degradation

O DAIMON funciona mesmo quando NOESIS está indisponível:

| Componente | Com NOESIS | Sem NOESIS |
|------------|------------|------------|
| Hook quick_check | ✅ Análise completa | ⚠️ Apenas keywords |
| noesis_consult | ✅ Perguntas maiêuticas | ❌ Erro retornado |
| noesis_tribunal | ✅ Julgamento completo | ❌ Erro retornado |
| Precedentes | ✅ Via NOESIS | ✅ Local (PrecedentSystem) |
| Aprendizado | ✅ Com feedback do Tribunal | ✅ Apenas sinais locais |

---

## Verificar Integração

```bash
# Verificar se NOESIS está rodando
curl http://localhost:8001/api/consciousness/state
curl http://localhost:8002/api/reflector/health

# Testar MCP Server
python integrations/mcp_server.py --test

# Verificar hook
python .claude/hooks/noesis_hook.py --test "delete all files"
```

---

## Testes

```bash
# Testes de integração
python -m pytest tests/test_mcp_server.py tests/test_noesis_hook.py -v

# Testes com NOESIS real (requer NOESIS rodando)
python -m pytest tests/test_integration.py -v
```

---

## Troubleshooting

### MCP Server não registra

```bash
# Verificar registro
claude mcp list

# Re-registrar
claude mcp remove daimon-consciousness
claude mcp add daimon-consciousness -- python /media/juan/DATA/projetos/daimon/integrations/mcp_server.py
```

### Hook timeout

```bash
# Verificar timeout no settings.json
cat ~/.claude/settings.json | jq '.hooks'

# Aumentar timeout se necessário (máximo recomendado: 1000ms)
```

### NOESIS não responde

```bash
# Verificar portas
lsof -i :8001
lsof -i :8002

# Reiniciar NOESIS
cd /media/juan/DATA/projetos/Noesis/Daimon
./noesis wakeup
```

---

## Limitações Honestas

1. **Latência** - Chamadas NOESIS adicionam 50-200ms
2. **Dependência** - Funcionalidades avançadas requerem NOESIS
3. **Timeout Hook** - 1000ms pode não ser suficiente para análises profundas
4. **Sem cache** - Cada chamada é uma requisição HTTP nova

---

*Documentação atualizada em 2025-12-13*

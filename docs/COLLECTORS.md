# DAIMON Collectors

**Sistema de Observabilidade OS - Captura de Dados Comportamentais**

---

## Visão Geral

Os Collectors são responsáveis por capturar dados comportamentais do usuário de forma não-intrusiva. Seguem o padrão **Heartbeat** (inspirado no ActivityWatch): capturam estados que se fundem quando similares, reduzindo volume de dados.

### Princípios

1. **Privacidade Primeiro** - Capturamos INTENÇÃO, não CONTEÚDO
2. **Heartbeat Pattern** - Estados que se fundem, não eventos isolados
3. **Graceful Degradation** - Falhas não quebram o sistema
4. **Plugin Architecture** - Novos collectors via decorator `@register_collector`

---

## Arquitetura

```
collectors/
├── __init__.py          # Exports e auto-registro
├── base.py              # BaseWatcher + Heartbeat
├── registry.py          # CollectorRegistry (plugin system)
├── shell_watcher.py     # Comandos do terminal
├── claude_watcher.py    # Sessões Claude Code
├── window_watcher.py    # Janelas focadas (X11)
├── input_watcher.py     # Dinâmica de digitação
├── afk_watcher.py       # Detecção de ausência
└── browser_watcher.py   # Atividade web (requer extensão)
```

---

## Base Classes

### Heartbeat

```python
@dataclass
class Heartbeat:
    """Estado pontual capturado por um collector."""
    timestamp: datetime
    watcher_type: str
    data: Dict[str, Any]
```

### BaseWatcher

```python
class BaseWatcher(ABC):
    """Interface base para todos os watchers."""

    name: str = "base"           # Identificador único
    version: str = "1.0.0"       # Versão semântica
    batch_interval: float = 30.0 # Segundos entre flushes

    # Métodos abstratos (obrigatórios)
    async def collect(self) -> Optional[Heartbeat]: ...
    def get_config(self) -> Dict[str, Any]: ...

    # Métodos herdados
    async def start(self) -> None      # Inicia loop de coleta
    async def stop(self) -> None       # Para e faz flush final
    async def flush(self) -> List[Heartbeat]  # Envia dados acumulados
    def get_status(self) -> Dict[str, Any]    # Status atual
```

### CollectorRegistry

```python
# Registro via decorator
@register_collector
class MyWatcher(BaseWatcher):
    name = "my_watcher"
    ...

# Uso
CollectorRegistry.get_names()           # Lista nomes
CollectorRegistry.create_instance(name) # Cria/obtém instância
CollectorRegistry.get_status()          # Status do registry
```

---

## Collectors Disponíveis

### 1. Shell Watcher

**Arquivo:** `collectors/shell_watcher.py`
**Status:** ✅ Funcional

Captura comandos do terminal via hooks no `~/.zshrc`.

#### Como Funciona

```
~/.zshrc hook → Unix Socket (~/.daimon/daimon.sock) → HeartbeatAggregator → flush()
```

#### Dados Capturados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `command` | string | Comando executado |
| `pwd` | string | Diretório atual |
| `exit_code` | int | Código de saída |
| `duration` | float | Tempo de execução (segundos) |
| `git_branch` | string | Branch atual (se em repo git) |

#### Padrões Detectados

- **error_streak**: 3+ comandos consecutivos com erro
- **repetitive_command**: Mesmo comando 3+ vezes seguidas

#### Instalação

```bash
# Adicionar hooks ao .zshrc
python collectors/shell_watcher.py --zshrc >> ~/.zshrc
source ~/.zshrc

# Iniciar daemon
python collectors/shell_watcher.py --daemon
```

#### Integração

- **ActivityStore**: ✅ Via `flush()`
- **StyleLearner**: ✅ Via `add_shell_sample()`
- **NOESIS**: ✅ Via `POST /api/daimon/shell/batch`

---

### 2. Claude Watcher

**Arquivo:** `collectors/claude_watcher.py`
**Status:** ✅ Funcional

Monitora sessões Claude Code via polling de arquivos JSONL.

#### Como Funciona

```
~/.claude/projects/*/sessions/*.jsonl → SessionTracker.poll() → flush()
```

#### Dados Capturados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `intention` | string | Intenção detectada (create/fix/refactor/etc) |
| `files_touched` | list | Arquivos mencionados |
| `project` | string | Nome do projeto |
| `preference_signal` | string | approval/rejection/null |
| `preference_category` | string | Categoria inferida |

#### Intenções Detectáveis

- `create` - Criar algo novo
- `fix` - Corrigir bug
- `refactor` - Refatorar código
- `understand` - Entender código
- `delete` - Remover código
- `test` - Criar/rodar testes
- `deploy` - Deploy/release

#### Privacidade

**IMPORTANTE**: Capturamos INTENÇÃO, não o CONTEÚDO do prompt.

#### Integração

- **ActivityStore**: ✅ Via `flush()`
- **StyleLearner**: ✅ Via `add_claude_sample()`
- **NOESIS**: ✅ Via `POST /api/daimon/claude/event`

---

### 3. Window Watcher

**Arquivo:** `collectors/window_watcher.py`
**Status:** ✅ Funcional

Rastreia janelas focadas via X11 EWMH.

#### Requisitos

- Linux com X11 (não funciona em Wayland puro)
- Pacote `python-xlib`

#### Dados Capturados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `window_title` | string | Título da janela |
| `app_name` | string | Nome do aplicativo |
| `wm_class` | string | Classe X11 |
| `focus_start` | datetime | Início do foco |
| `focus_duration` | float | Duração em segundos |

#### Integração

- **ActivityStore**: ✅ Via `flush()`
- **StyleLearner**: ✅ Via `add_window_sample()`

---

### 4. Input Watcher

**Arquivo:** `collectors/input_watcher.py`
**Status:** ✅ Funcional

Captura dinâmica de digitação via `pynput`.

#### Requisitos

- Pacote `pynput`
- Permissões de input (pode requerer `sudo` ou grupo `input`)

#### Dados Capturados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `keystroke_count` | int | Total de teclas |
| `keystroke_dynamics` | dict | Métricas de digitação |
| `typing_speed_cpm` | float | Caracteres por minuto |
| `pause_patterns` | list | Padrões de pausa |

#### Privacidade

**IMPORTANTE**: NÃO capturamos as teclas em si, apenas métricas agregadas:
- Velocidade de digitação
- Padrões de pausa
- Taxa de backspace (correções)

#### Integração

- **ActivityStore**: ✅ Via `flush()`
- **StyleLearner**: ✅ Via `add_keystroke_sample()`
- **KeystrokeAnalyzer**: ✅ Via eventos real-time

---

### 5. AFK Watcher

**Arquivo:** `collectors/afk_watcher.py`
**Status:** ✅ Funcional

Detecta ausência do usuário via X11/proc.

#### Métodos de Detecção

1. **X11 Idle Time** - Tempo desde última interação
2. **Screen Lock** - Tela bloqueada
3. **Process Check** - Screensaver ativo

#### Dados Capturados

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `is_afk` | bool | Se está ausente |
| `afk_duration` | float | Tempo ausente (segundos) |
| `last_activity` | datetime | Última atividade |
| `reason` | string | Motivo (idle/locked/etc) |

#### Integração

- **ActivityStore**: ✅ Via `flush()`
- **StyleLearner**: ✅ Via `add_afk_sample()`

---

### 6. Browser Watcher

**Arquivo:** `collectors/browser_watcher.py`
**Status:** ⚠️ Parcial (requer extensão)

Rastreia atividade do navegador.

#### Requisitos

- Extensão de browser (não incluída)
- WebSocket ou Native Messaging

#### Dados Capturados (quando disponível)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `url` | string | URL atual (domínio apenas) |
| `title` | string | Título da página |
| `domain` | string | Domínio |
| `time_spent` | float | Tempo na página |

#### Integração

- **ActivityStore**: ✅ Via `flush()`

---

## Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          COLLECTORS                                      │
│                                                                          │
│  shell ──┐                                                               │
│  claude ─┤                                                               │
│  window ─┼──► Heartbeat ──► flush() ──┬──► ActivityStore                │
│  input ──┤                            ├──► StyleLearner                 │
│  afk ────┤                            └──► NOESIS (shell/claude only)   │
│  browser ┘                                                               │
│                                                                          │
│  input ──► KeystrokeAnalyzer (real-time, bypass flush)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Criando um Novo Collector

```python
from collectors import BaseWatcher, register_collector, Heartbeat

@register_collector
class MyWatcher(BaseWatcher):
    name = "my_watcher"
    version = "1.0.0"

    def __init__(self, batch_interval: float = 30.0):
        super().__init__(batch_interval)
        # Inicialização específica

    async def collect(self) -> Optional[Heartbeat]:
        """Coleta um heartbeat."""
        data = self._get_current_state()
        if not data:
            return None
        return Heartbeat(
            timestamp=datetime.now(),
            watcher_type=self.name,
            data=data,
        )

    def get_config(self) -> Dict[str, Any]:
        """Retorna configuração."""
        return {"batch_interval": self.batch_interval}

    async def flush(self) -> List[Heartbeat]:
        """Override para enviar dados específicos."""
        flushed = await super().flush()
        if flushed:
            # Enviar para ActivityStore
            store = get_activity_store()
            for hb in flushed:
                store.add(watcher_type=self.name, timestamp=hb.timestamp, data=hb.data)
            # Enviar para StyleLearner
            learner = get_style_learner()
            for hb in flushed:
                learner.add_custom_sample(hb.data)
        return flushed
```

---

## Configuração

### Variáveis de Ambiente

```bash
DAIMON_SOCKET_PATH=~/.daimon/daimon.sock  # Socket do shell_watcher
DAIMON_BATCH_INTERVAL=30                   # Intervalo de flush (segundos)
NOESIS_URL=http://localhost:8001          # URL do NOESIS
```

### Systemd

Os collectors são gerenciados pelo `daimon_daemon.py` ou individualmente:

```bash
# Via daemon unificado
python daimon_daemon.py --daemon

# Via systemd
systemctl --user start daimon
```

---

## Métricas

| Collector | Heartbeats/hora | Storage/dia | CPU |
|-----------|-----------------|-------------|-----|
| shell | ~50-200 | ~50KB | <1% |
| claude | ~10-50 | ~20KB | <1% |
| window | ~100-500 | ~100KB | <1% |
| input | ~1000-5000 | ~200KB | 1-2% |
| afk | ~50-100 | ~10KB | <1% |
| browser | ~50-200 | ~50KB | <1% |

---

## Troubleshooting

### Shell Watcher não captura comandos

```bash
# Verificar socket
ls -la ~/.daimon/daimon.sock

# Testar manualmente
echo '{"command":"test","pwd":"/tmp","exit_code":0}' | nc -U ~/.daimon/daimon.sock

# Verificar hooks no .zshrc
grep daimon ~/.zshrc
```

### Window Watcher não funciona

```bash
# Verificar X11
echo $DISPLAY

# Testar python-xlib
python -c "from Xlib import X, display; d = display.Display(); print(d)"
```

### Input Watcher sem permissão

```bash
# Adicionar usuário ao grupo input
sudo usermod -a -G input $USER
# Relogar

# Ou rodar com permissão elevada (não recomendado)
sudo python collectors/input_watcher.py --daemon
```

---

## Testes

```bash
# Todos os testes de collectors
python -m pytest tests/test_collectors_*.py tests/test_shell_watcher.py tests/test_claude_watcher.py -v

# Testes específicos
python -m pytest tests/test_collectors_base.py -v
python -m pytest tests/test_collectors_registry.py -v
```

---

*Documentação atualizada em 2025-12-13*

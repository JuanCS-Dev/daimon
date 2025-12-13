# Plano: Observabilidade Completa do Sistema para DAIMON

## Status: PRONTO PARA IMPLEMENTACAO

### Decisoes do Usuario:
- **Browser**: Extensao propria minimalista
- **Display**: X11 (Xorg)
- **Prioridade**: Todos watchers em paralelo

---

## Resumo Executivo

Adicionar observabilidade completa do OS ao DAIMON com foco em:
- **Privacy-preserving**: Captura metadados e padroes, NAO conteudo
- **Eficiencia**: Heartbeat pattern, batching, throttling (~2% CPU adicional)
- **Arquitetura modular**: Plugin system para collectors

---

## 1. Arquitetura Proposta

### 1.1 Plugin System para Collectors

```
collectors/
├── base.py          # Interface abstrata BaseWatcher
├── registry.py      # CollectorRegistry (auto-discovery)
├── shell_watcher.py # (existente)
├── claude_watcher.py# (existente)
├── window_watcher.py# NOVO - tracking de janelas
├── input_watcher.py # NOVO - keystroke dynamics
├── afk_watcher.py   # NOVO - deteccao inatividade
├── browser_watcher.py # NOVO - integracao browser
└── optimization.py  # Batching, throttling
```

### 1.2 Fluxo de Dados

```
Watchers (collectors/)
    ↓ Heartbeat Pattern (batch 30-60s)
ActivityStore (memory/)
    ↓ Dados agregados
StyleLearner (learners/)
    ↓ Insights de estilo
ReflectionEngine
    ↓ Atualiza
~/.claude/CLAUDE.md
```

---

## 2. Novos Collectors

### 2.1 Window Watcher
- **Backend X11**: python-xlib (atual)
- **Backend Wayland**: D-Bus via GNOME extension (futuro)
- **Dados**: app_name, window_title (sanitizado), tempo de foco
- **Intervalo**: 5s

### 2.2 Input Watcher (Keystroke Dynamics)
- **Biblioteca**: pynput
- **Captura**: Timing apenas (hold-time, seek-time) - NAO teclas
- **Metricas**: velocidade, ritmo, pausas, rajadas
- **Throttling**: 50ms minimo entre eventos
- **Intervalo**: 60s agregacao

### 2.3 AFK Watcher
- **Threshold**: 3 minutos de inatividade
- **Dados**: periodos AFK, padroes de trabalho

### 2.4 Browser Watcher
- **Abordagem**: Extensao minimalista propria (Chrome/Firefox)
- **Backend**: HTTP POST para localhost:8003
- **Dados**: dominio apenas, tempo gasto
- **Arquivos**: `browser_extension/manifest.json`, `browser_extension/background.js`

---

## 3. Style Learner

Novo learner que infere estilo de comunicacao:

```python
@dataclass
class CommunicationStyle:
    typing_pace: str       # "fast", "moderate", "deliberate"
    editing_frequency: str # "minimal", "moderate", "heavy"
    interaction_pattern: str # "burst", "steady", "sporadic"
    confidence: float
```

Gera sugestoes para CLAUDE.md baseado em padroes observados.

---

## 4. Estimativa de Recursos

| Componente | CPU (idle) | CPU (active) | RAM |
|------------|------------|--------------|-----|
| window_watcher | <0.1% | ~0.5% | ~5MB |
| input_watcher | <0.5% | ~1% | ~8MB |
| browser_watcher | 0% | ~0.2% | ~3MB |
| afk_watcher | <0.1% | ~0.2% | ~2MB |
| style_learner | 0% | ~0.3% | ~5MB |
| **Total Adicional** | **<1%** | **~2.2%** | **~23MB** |

**Consumo Total DAIMON:** ~56MB RAM, 4-5% CPU pico

---

## 5. Arquivos a Criar

| Arquivo | Descricao |
|---------|-----------|
| `collectors/base.py` | Interface abstrata |
| `collectors/registry.py` | Plugin registry |
| `collectors/window_watcher.py` | Window tracking |
| `collectors/input_watcher.py` | Keystroke dynamics |
| `collectors/afk_watcher.py` | Inatividade |
| `collectors/browser_watcher.py` | Browser |
| `collectors/optimization.py` | Batching |
| `learners/style_learner.py` | Estilo comunicacao |
| `memory/activity_store.py` | Storage atividades |

## 6. Arquivos a Modificar

| Arquivo | Modificacao |
|---------|-------------|
| `daimon_daemon.py` | Usar CollectorRegistry |
| `learners/reflection_engine.py` | Integrar StyleLearner |
| `dashboard/app.py` | Status novos collectors |
| `pyproject.toml` | Deps: python-xlib, pynput |

---

## 7. Privacidade

### NAO capturado:
- Conteudo de teclas
- URLs completas
- Screenshots
- Audio/video

### Capturado:
- Nome do app em foco
- Titulo sanitizado
- Metricas de timing
- Contagens agregadas
- Periodos de inatividade

### Salvaguardas:
- Dados 100% locais
- Limpeza automatica 7 dias
- Codigo auditavel

---

## 8. Sequencia de Implementacao (Paralelo)

### Fase 1: Fundacao (primeiro)
- `collectors/base.py` - Interface BaseWatcher
- `collectors/registry.py` - CollectorRegistry
- Modificar `daimon_daemon.py` para usar registry

### Fase 2: Todos Watchers (paralelo)
- `collectors/window_watcher.py` - X11 backend
- `collectors/input_watcher.py` - Keystroke dynamics
- `collectors/afk_watcher.py` - Inatividade
- `collectors/browser_watcher.py` + extensao
- `collectors/optimization.py` - Batching/throttling
- `memory/activity_store.py` - Storage

### Fase 3: Learner + Dashboard
- `learners/style_learner.py` - Analise de estilo
- Integrar com `reflection_engine.py`
- Atualizar `dashboard/app.py`

---

## Dependencias

```toml
python-xlib>=0.33    # X11 window tracking
pynput>=1.7.6        # Input capture
```

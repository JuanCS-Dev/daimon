# AUDITORIA BRUTAL DAIMON v2
## Contra CODE_CONSTITUTION.md

**Data:** 2025-12-13
**Auditor:** Claude Opus 4.5
**Metodologia:** Varredura completa + análise de fluxo de dados

---

## RESUMO EXECUTIVO

| Métrica | Atual | Requerido | Status |
|---------|-------|-----------|--------|
| Test Coverage | 64% | 80% | FALHA |
| Arquivos > 500 linhas | 16 | 0 | FALHA |
| TODOs/FIXMEs | 0 | 0 | OK |
| Pylint Score | 9.81/10 | 9.0/10 | OK |
| AIR GAPS | 7 | 0 | FALHA |

---

## 1. VIOLAÇÕES DO PADRÃO PAGANI (CRITICAL)

### 1.1 Arquivos > 500 Linhas (16 violações)

**Constituição diz: "Files > 500 lines FORBIDDEN"**

| Arquivo | Linhas | Excesso |
|---------|--------|---------|
| `audit_system.py` | 1021 | +521 |
| `dashboard/app.py` | 890 | +390 |
| `integrations/mcp_server.py` | 759 | +259 |
| `tests/test_mcp_server.py` | 741 | +241 |
| `tests/test_corpus_manager.py` | 665 | +165 |
| `corpus/manager.py` | 657 | +157 |
| `tests/test_precedent_system.py` | 643 | +143 |
| `learners/metacognitive_engine.py` | 641 | +141 |
| `learners/keystroke_analyzer.py` | 626 | +126 |
| `tests/test_preference_learner.py` | 623 | +123 |
| `tests/test_shell_watcher.py` | 589 | +89 |
| `learners/style_learner.py` | 576 | +76 |
| `tests/test_config_refiner.py` | 550 | +50 |
| `tests/test_claude_watcher.py` | 540 | +40 |
| `learners/preference_learner.py` | 518 | +18 |
| `endpoints/daimon_routes.py` | 514 | +14 |

**Ação:** Dividir cada arquivo em módulos menores.

### 1.2 Test Coverage < 80% (CRITICAL)

**Constituição diz: "Unit test coverage >= 80% REQUIRED"**

**Coverage atual: 64%** (16 pontos abaixo)

**Módulos com 0% coverage (CAPITAL OFFENSE):**
- `audit_system.py` - 0%
- `daimon_daemon.py` - 0%
- `dashboard/app.py` - 0%
- `corpus/loaders/*` - 0% (todos)

**Módulos com < 50% coverage:**
- `collectors/window_watcher.py` - 23%
- `collectors/input_watcher.py` - 29%
- `collectors/afk_watcher.py` - 25%
- `collectors/browser_watcher.py` - 31%
- `learners/keystroke_analyzer.py` - 25%
- `corpus/semantic_search.py` - 33%
- `learners/metacognitive_engine.py` - 48%
- `learners/style_learner.py` - 50%

---

## 2. AIR GAPS IDENTIFICADOS (7 CRÍTICOS)

### AIR GAP #1: Collectors não armazenam no ActivityStore

**Problema:** 4 collectors NÃO enviam dados para ActivityStore.

| Collector | Usa ActivityStore? |
|-----------|-------------------|
| shell_watcher | SIM |
| claude_watcher | SIM |
| window_watcher | **NÃO** |
| input_watcher | **NÃO** |
| afk_watcher | **NÃO** |
| browser_watcher | **NÃO** |

**Impacto:** Dados de window, input, afk e browser são perdidos para análise.

### AIR GAP #2: StyleLearner desconectado de collectors

**Problema:** StyleLearner tem métodos `add_*_sample()` mas apenas 2 collectors os chamam.

| Método | Chamado por |
|--------|-------------|
| add_input_sample() | NINGUÉM |
| add_window_sample() | NINGUÉM |
| add_afk_sample() | NINGUÉM |
| add_shell_sample() | shell_watcher |
| add_claude_sample() | claude_watcher |

**Impacto:** StyleLearner opera com apenas 40% dos dados disponíveis.

### AIR GAP #3: KeystrokeAnalyzer duplicado e desconectado

**Problema:** Existem DUAS implementações de análise de keystroke:
1. `learners/keystroke_analyzer.py` - KeystrokeAnalyzer (626 linhas)
2. `collectors/input_watcher.py` - KeystrokeDynamics (embutido)

**Impacto:**
- Código duplicado (violação DRY)
- KeystrokeAnalyzer NUNCA recebe dados do input_watcher
- Dashboard mostra "idle" sempre para cognitive state

### AIR GAP #4: SemanticCorpus não indexa automaticamente

**Problema:** `corpus/semantic_search.py` existe mas:
- `semantic_indexed: 0` no stats
- Não indexa textos automaticamente ao adicionar
- Depende de chamar `reindex_semantic()` manualmente

**Impacto:** Busca semântica não funciona até reindexar.

### AIR GAP #5: MetacognitiveEngine não mede efetividade

**Problema:** `measure_effectiveness()` existe mas NUNCA é chamado:
- Insights são logados mas efetividade nunca medida
- `effectiveness_score` é sempre `None`
- Recomendações baseadas em dados inexistentes

### AIR GAP #6: Browser Watcher sem servidor HTTP

**Problema:** `browser_watcher.py` espera receber eventos via HTTP, mas:
- Não inicia servidor próprio
- Dashboard/app.py tem endpoint `/api/browser/event` mas...
- Extensão do browser precisa saber a porta 8003
- Não há extensão de browser implementada!

### AIR GAP #7: Precedent System sem integração real com NOESIS

**Problema:** `memory/precedent_system.py` existe mas:
- `PrecedentTribunal` simula veredictos localmente
- Não consulta NOESIS `/reflect/verdict` endpoint
- Session end em `daimon_routes.py` cria precedentes fake

---

## 3. VIOLAÇÕES DE ERROR HANDLING

### 3.1 Silent Exception Handlers (24 ocorrências)

**Constituição diz: "Fail fast, fail loud"**

Encontrados 24 blocos `except ...: pass` que silenciam erros:

```python
# Exemplos de violações:
except (OSError, IOError):
    pass  # SILENCIANDO ERRO!

except Exception:
    pass  # BROAD EXCEPTION + SILENCIAMENTO!
```

**Arquivos afetados:**
- learners/reflection_engine.py
- collectors/window_watcher.py (4x)
- collectors/afk_watcher.py
- collectors/base.py
- actuators/config_refiner.py (2x)
- collectors/input_watcher.py (2x)
- daimon_daemon.py (3x)
- endpoints/daimon_routes.py (2x)
- dashboard/app.py

### 3.2 Broad Exception Catches (6 ocorrências)

```python
except Exception:  # MUITO GENÉRICO
```

**Arquivos:**
- daimon_daemon.py
- dashboard/app.py (2x)
- learners/preference_learner.py
- collectors/claude_watcher.py
- collectors/shell_watcher.py

---

## 4. VIOLAÇÕES MENORES

### 4.1 Import Errors (4)

```
corpus/semantic_search.py:303: E0401: Unable to import 'faiss'
corpus/semantic_search.py:347: E0401: Unable to import 'faiss'
tests/test_integration.py:220: E0401: Unable to import 'noesis_hook'
tests/test_noesis_hook.py:19: E0401: Unable to import 'noesis_hook'
```

### 4.2 Arquivos Warning (> 400 linhas)

9 arquivos entre 400-500 linhas (próximos de violar).

---

## 5. PLANO DE CORREÇÃO

### Prioridade 1: AIR GAPS (CRÍTICO)

1. **Conectar collectors ao ActivityStore:**
   - window_watcher.py
   - input_watcher.py
   - afk_watcher.py
   - browser_watcher.py

2. **Conectar collectors ao StyleLearner:**
   - Chamar `add_*_sample()` nos collectors faltantes

3. **Unificar KeystrokeAnalyzer:**
   - Remover KeystrokeDynamics de input_watcher
   - Usar learners/keystroke_analyzer.py
   - input_watcher alimenta KeystrokeAnalyzer

4. **Ativar indexação semântica:**
   - Indexar automaticamente ao adicionar texto
   - Remover lazy loading problemático

5. **Implementar medição de efetividade:**
   - Chamar `measure_effectiveness()` após período de observação
   - Comparar métricas before/after

### Prioridade 2: Test Coverage

1. **Criar testes para módulos 0%:**
   - dashboard/app.py
   - daimon_daemon.py
   - corpus/loaders/*

2. **Aumentar coverage para módulos < 50%**

### Prioridade 3: Refatorar Arquivos Grandes

1. Dividir `audit_system.py` (1021 linhas)
2. Dividir `dashboard/app.py` (890 linhas)
3. Dividir `integrations/mcp_server.py` (759 linhas)

### Prioridade 4: Error Handling

1. Substituir `pass` por logging apropriado
2. Especificar exceções ao invés de `except Exception:`

---

## 6. MÉTRICAS ALVO PÓS-CORREÇÃO

| Métrica | Atual | Alvo |
|---------|-------|------|
| Test Coverage | 64% | 85% |
| Arquivos > 500 linhas | 16 | 0 |
| AIR GAPS | 7 | 0 |
| Silent Exceptions | 24 | 0 |
| Broad Exceptions | 6 | 0 |
| Pylint Score | 9.81 | 9.9 |

---

## CONCLUSÃO

O sistema DAIMON tem **infraestrutura sólida** mas sofre de **fragmentação crítica**.
Os AIR GAPS identificados significam que apenas ~40% dos dados coletados fluem corretamente pelo sistema.

**O dashboard mostra dados, mas metade deles são vazios ou incorretos porque os componentes não estão conectados.**

A arquitetura está correta no papel, mas a implementação tem lacunas que precisam ser fechadas para o sistema funcionar como prometido.

---

*"The letter killeth, but the spirit giveth life."*

O código existe, mas o espírito da integração não foi implementado completamente.

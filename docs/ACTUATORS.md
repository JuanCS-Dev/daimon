# DAIMON Actuators

**Sistema de Atuação - Aplicação de Preferências Aprendidas**

---

## Visão Geral

Os Actuators são responsáveis por transformar insights em ações concretas. O principal actuator é o **ConfigRefiner**, que atualiza o arquivo `~/.claude/CLAUDE.md` com preferências aprendidas.

### Princípios

1. **Safe by Default** - Sempre fazer backup antes de modificar
2. **Preservar Manual** - Nunca sobrescrever conteúdo escrito pelo usuário
3. **Transparência** - Log de todas as alterações
4. **Reversibilidade** - Backups permitem restaurar qualquer versão

---

## Arquitetura

```
actuators/
├── __init__.py          # Exports
└── config_refiner.py    # Atualizador de CLAUDE.md
```

### Storage

```
~/.claude/
├── CLAUDE.md            # Arquivo de preferências (destino)
├── backups/
│   ├── CLAUDE.md.2025-12-13T10-30-00.bak  # Backup automático
│   ├── CLAUDE.md.2025-12-13T11-00-00.bak
│   └── ... (últimos 10)
└── update_log.jsonl     # Log de atualizações
```

---

## ConfigRefiner

**Arquivo:** `actuators/config_refiner.py`
**Status:** ✅ Funcional

### Formato do CLAUDE.md

```markdown
# Minhas Preferências Manuais

(conteúdo escrito pelo usuário - PRESERVADO)

<!-- DAIMON:AUTO:START -->
# Preferencias Aprendidas (DAIMON)
*Ultima atualizacao: 2025-12-13 15:30*

## Communication Style
- [Alta] User prefers concise responses
- [Media] Quick drafts welcome for iteration

## Testing
- [Alta] Continue generating tests proactively

## Workflow
- [Baixa] User shows fatigue - suggest breaks
<!-- DAIMON:AUTO:END -->

# Mais Conteúdo Manual

(também PRESERVADO)
```

### Seção DAIMON

A seção entre `<!-- DAIMON:AUTO:START -->` e `<!-- DAIMON:AUTO:END -->` é gerenciada automaticamente pelo DAIMON. Todo o resto é preservado.

### API

```python
from actuators.config_refiner import ConfigRefiner

refiner = ConfigRefiner()

# Atualizar preferências
insights = [
    {"category": "verbosity", "action": "reduce", "confidence": 0.85,
     "suggestion": "Preferir respostas concisas"},
    {"category": "testing", "action": "reinforce", "confidence": 0.9,
     "suggestion": "Continuar gerando testes proativamente"},
]
updated = refiner.update_preferences(insights)
# → True se atualizou, False se não houve mudanças

# Ler preferências atuais
current = refiner.get_current_preferences()
# → "## Communication Style\n- [Alta] ..."

# Ler conteúdo manual (fora da seção DAIMON)
manual = refiner.get_manual_content()
# → "# Minhas Preferências Manuais\n..."

# Listar backups
backups = refiner.get_backup_list()
# → ["CLAUDE.md.2025-12-13T10-30-00.bak", ...]

# Restaurar backup
refiner.restore_backup("CLAUDE.md.2025-12-13T10-30-00.bak")
```

### Fluxo de Atualização

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ConfigRefiner.update_preferences()                  │
│                                                                         │
│  1. _read_current()                                                    │
│     └── Lê ~/.claude/CLAUDE.md atual                                   │
│                                                                         │
│  2. _generate_section(insights)                                        │
│     └── Gera markdown com preferências                                 │
│         - Agrupa por categoria                                         │
│         - Marca confiança: [Alta], [Média], [Baixa]                   │
│         - Adiciona timestamp                                           │
│                                                                         │
│  3. _sections_equal(old, new)                                          │
│     └── Verifica se houve mudança real (ignora timestamp)             │
│                                                                         │
│  4. _create_backup()                                                   │
│     └── Copia atual para ~/.claude/backups/                           │
│         - Mantém últimos 10 backups                                    │
│         - Nome: CLAUDE.md.{ISO_TIMESTAMP}.bak                         │
│                                                                         │
│  5. _merge_content(current, new_section)                               │
│     └── Mescla preservando conteúdo manual                            │
│         - Se seção existe: substitui                                   │
│         - Se não existe: adiciona no final                            │
│                                                                         │
│  6. _write(updated_content)                                            │
│     └── Escreve ~/.claude/CLAUDE.md                                   │
│                                                                         │
│  7. _log_update(insights)                                              │
│     └── Registra em update_log.jsonl                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Níveis de Confiança

| Nível | Confiança | Descrição |
|-------|-----------|-----------|
| [Alta] | ≥ 0.7 | Padrão consistente, muitos sinais |
| [Média] | 0.4-0.7 | Tendência detectada |
| [Baixa] | < 0.4 | Sinal fraco, observação inicial |

### Log de Atualizações

Cada atualização é registrada em `~/.claude/update_log.jsonl`:

```json
{"timestamp": "2025-12-13T15:30:00", "insights_count": 3, "categories": ["verbosity", "testing"], "backup": "CLAUDE.md.2025-12-13T15-30-00.bak"}
{"timestamp": "2025-12-13T16:00:00", "insights_count": 2, "categories": ["workflow"], "backup": "CLAUDE.md.2025-12-13T16-00-00.bak"}
```

---

## Integração com ReflectionEngine

O ConfigRefiner é chamado automaticamente pelo ReflectionEngine:

```python
# Em reflection_engine.py:_apply_insights()
async def _apply_insights(self, insights: list[dict]) -> bool:
    if not insights:
        return False

    if not self.refiner:
        logger.warning("ConfigRefiner not available")
        return False

    try:
        updated = self.refiner.update_preferences(insights)
        if updated:
            self.stats.total_updates += 1
            await self._notify_update(insights)
        return updated
    except Exception as e:
        logger.error("Failed to update CLAUDE.md: %s", e)
        return False
```

---

## Restauração de Backup

### Via API

```python
refiner = ConfigRefiner()

# Listar backups disponíveis
backups = refiner.get_backup_list()
# → ["CLAUDE.md.2025-12-13T15-30-00.bak", "CLAUDE.md.2025-12-13T15-00-00.bak", ...]

# Restaurar específico
refiner.restore_backup("CLAUDE.md.2025-12-13T15-00-00.bak")
```

### Via Dashboard

```
POST /api/backups/restore
Body: {"backup": "CLAUDE.md.2025-12-13T15-00-00.bak"}
```

### Via CLI

```bash
# Listar
ls -la ~/.claude/backups/

# Restaurar manualmente
cp ~/.claude/backups/CLAUDE.md.2025-12-13T15-00-00.bak ~/.claude/CLAUDE.md
```

---

## Configuração

### Variáveis de Ambiente

```bash
CLAUDE_MD_PATH=~/.claude/CLAUDE.md     # Caminho do arquivo
CLAUDE_BACKUP_DIR=~/.claude/backups    # Diretório de backups
CLAUDE_MAX_BACKUPS=10                   # Máximo de backups mantidos
```

### Constantes

```python
# Em config_refiner.py
DAIMON_SECTION_START = "<!-- DAIMON:AUTO:START -->"
DAIMON_SECTION_END = "<!-- DAIMON:AUTO:END -->"
MAX_BACKUPS = 10
```

---

## Segurança

### Proteções Implementadas

1. **Backup Obrigatório** - Sempre cria backup antes de modificar
2. **Validação de Conteúdo** - Verifica se merge preservou manual
3. **Atomic Write** - Escreve em temp, depois renomeia
4. **Log Imutável** - update_log.jsonl é append-only

### O que NÃO Fazemos

1. **NÃO modificamos** conteúdo fora da seção DAIMON
2. **NÃO deletamos** o arquivo CLAUDE.md
3. **NÃO removemos** backups automaticamente (só os mais antigos)

---

## Testes

```bash
# Todos os testes de actuators
python -m pytest tests/test_config_refiner.py -v

# Testes específicos
python -m pytest tests/test_config_refiner.py::TestMergeContent -v
python -m pytest tests/test_config_refiner.py::TestCreateBackup -v
```

---

## Limitações Honestas

1. **Sem validação LLM** - Sugestões são aceitas diretamente (planejado para futuro)
2. **Dependência de regex** - Padrões de seção podem quebrar se editados manualmente de forma incorreta
3. **Sem rollback automático** - Se algo der errado, precisa restaurar manualmente

---

## Exemplo Completo

### Antes

```markdown
# Minhas Notas

Sempre usar TypeScript.
```

### Insights Recebidos

```python
[
    {"category": "verbosity", "confidence": 0.85, "suggestion": "Respostas concisas"},
    {"category": "testing", "confidence": 0.75, "suggestion": "Gerar testes proativamente"},
]
```

### Depois

```markdown
# Minhas Notas

Sempre usar TypeScript.

<!-- DAIMON:AUTO:START -->
# Preferencias Aprendidas (DAIMON)
*Ultima atualizacao: 2025-12-13 15:30*

## Verbosity
- [Alta] Respostas concisas

## Testing
- [Alta] Gerar testes proativamente
<!-- DAIMON:AUTO:END -->
```

---

*Documentação atualizada em 2025-12-13*

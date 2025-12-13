# DAIMON - Guia Rápido de Operação

**Versão 2.0 | Dezembro 2025**

---

## 1. Comandos Essenciais

### Iniciar/Parar

O DAIMON depende do NOESIS. Ao iniciar o DAIMON, o NOESIS inicia automaticamente.

```bash
# Iniciar (inicia NOESIS + DAIMON)
systemctl --user start daimon

# Parar tudo
systemctl --user stop daimon noesis

# Reiniciar
systemctl --user restart daimon

# Status de ambos
systemctl --user status daimon noesis
```

### Controle Individual

```bash
# Apenas NOESIS
systemctl --user start noesis
systemctl --user stop noesis

# Via CLI do NOESIS
cd /media/juan/DATA/projetos/Noesis/Daimon
./noesis status    # Ver status
./noesis wakeup    # Iniciar tudo
./noesis sleep     # Parar tudo
```

### Habilitar no Boot

```bash
# Ativar autostart (ambos serviços)
systemctl --user enable daimon noesis

# Desativar autostart
systemctl --user disable daimon noesis
```

### Aliases Úteis

Adicione ao seu `~/.zshrc` ou `~/.bashrc`:

```bash
# Abrir dashboard no navegador
alias daimon_dash='xdg-open http://localhost:8003'

# Status rápido
alias daimon_status='systemctl --user status daimon'

# Logs em tempo real
alias daimon_logs='journalctl --user -u daimon -f'
```

---

## 2. Acessos

| Recurso | Endereço |
|---------|----------|
| **Dashboard** | http://localhost:8003 |
| **NOESIS** | http://localhost:8001 |
| **Reflector** | http://localhost:8002 |

---

## 3. Logs

```bash
# Logs em tempo real (systemd)
journalctl --user -u daimon -f

# Arquivo de log
cat ~/.daimon/logs/daimon.log

# Últimas 50 linhas
tail -50 ~/.daimon/logs/daimon.log
```

---

## 4. Arquivos Importantes

| Arquivo | Descrição |
|---------|-----------|
| `~/.daimon/daimon.sock` | Socket do Shell Watcher |
| `~/.daimon/daimon.pid` | PID do daemon |
| `~/.daimon/state.json` | Estado atual |
| `~/.daimon/logs/daimon.log` | Log principal |
| `~/.daimon/memory/memories.db` | Banco de memórias |
| `~/.daimon/memory/precedents.db` | Banco de precedentes |
| `~/.daimon/corpus/` | Textos de sabedoria |
| `~/.claude/CLAUDE.md` | Preferências (auto-updated) |
| `~/.claude/backups/` | Backups do CLAUDE.md |

---

## 5. Componentes

| Componente | Função | Status |
|------------|--------|--------|
| **Shell Watcher** | Captura comandos shell | Socket ativo |
| **Claude Watcher** | Monitora sessões Claude | Polling 5s |
| **Dashboard** | Interface web | Porta 8003 |
| **Reflection Engine** | Aprende preferências | Loop 30min |

---

## 6. Dashboard - Funcionalidades

### Painel Principal
- Status de serviços (verde/vermelho)
- Preferências aprendidas com gráficos
- Controle de collectors (start/stop)

### Editor CLAUDE.md
- Visualizar e editar preferências
- Backups automáticos
- Restaurar versões anteriores

### Corpus de Sabedoria
- Árvore de categorias navegável
- Adicionar novos textos
- Busca full-text
- Preview de conteúdo

---

## 7. Triggers de Reflexão

| Trigger | Condição |
|---------|----------|
| **Temporal** | A cada 30 minutos |
| **Threshold** | >5 rejeições mesma categoria |
| **Threshold** | >10 aprovações mesma categoria |
| **Manual** | Botão "Refletir" no dashboard |

---

## 8. Categorias de Preferência

- `code_style` - Estilo de código
- `verbosity` - Nível de detalhe
- `testing` - Testes e cobertura
- `architecture` - Decisões arquiteturais
- `documentation` - Documentação
- `workflow` - Fluxo de trabalho
- `general` - Geral

---

## 9. Troubleshooting

### Dashboard não abre
```bash
# Verificar se está rodando
curl http://localhost:8003/api/status

# Reiniciar
systemctl --user restart daimon
```

### Shell Watcher não funciona
```bash
# Verificar socket
ls -la ~/.daimon/daimon.sock

# Verificar hooks no .zshrc
grep daimon ~/.zshrc
```

### Sem preferências detectadas
```bash
# Trigger manual via API
curl -X POST http://localhost:8003/api/reflect

# Ou via dashboard: botão "Refletir"
```

### Serviço não inicia no boot
```bash
# Verificar se está habilitado
systemctl --user is-enabled daimon

# Habilitar
systemctl --user enable daimon

# Verificar linger (necessário para user services)
loginctl enable-linger $USER
```

---

## 10. API Rápida

```bash
# Status geral
curl http://localhost:8003/api/status

# Preferências
curl http://localhost:8003/api/preferences

# Trigger reflexão
curl -X POST http://localhost:8003/api/reflect

# Buscar no corpus
curl "http://localhost:8003/api/corpus/search?q=virtue"

# Estatísticas corpus
curl http://localhost:8003/api/corpus/stats

# Estatísticas memória
curl http://localhost:8003/api/memory/stats
```

---

## 11. Backup Manual

```bash
# Backup do CLAUDE.md
cp ~/.claude/CLAUDE.md ~/.claude/CLAUDE.md.backup.$(date +%Y%m%d)

# Backup dos bancos
cp ~/.daimon/memory/*.db ~/backup/

# Backup do corpus
cp -r ~/.daimon/corpus ~/backup/
```

---

## 12. Estrutura do Projeto

```
/media/juan/DATA/projetos/daimon/
├── daimon_daemon.py      # Daemon principal
├── install.sh            # Script de instalação
├── collectors/           # Watchers (shell, claude)
├── dashboard/            # Interface web
├── endpoints/            # API routes
├── learners/             # Preference learning
├── actuators/            # Config updates
├── memory/               # SQLite + FTS5
├── corpus/               # Wisdom texts
└── integrations/         # MCP Server
```

---

## Contato

**Projeto**: DAIMON - Personal Exocortex
**Dashboard**: http://localhost:8003
**Logs**: `journalctl --user -u daimon -f`

---

*Impresso em: ___/___/______*

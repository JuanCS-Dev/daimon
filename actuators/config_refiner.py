"""
DAIMON Config Refiner.

Atualiza ~/.claude/CLAUDE.md baseado em preferencias aprendidas.
Mantem separacao clara entre instrucoes automaticas e manuais.

Regras de seguranca:
- NUNCA deletar conteudo existente
- Secao auto-gerada claramente marcada
- Backup antes de modificar
- Log de todas as mudancas
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
BACKUP_DIR = Path.home() / ".claude" / "backups"

# Marcadores para secao auto-gerada
DAIMON_SECTION_START = "<!-- DAIMON:AUTO:START -->"
DAIMON_SECTION_END = "<!-- DAIMON:AUTO:END -->"


class ConfigRefiner:
    """
    Atualiza CLAUDE.md com preferencias aprendidas pelo DAIMON.

    Mantem separacao entre:
    - Secao DAIMON (auto-gerada, entre marcadores)
    - Conteudo manual do usuario (preservado intacto)

    Usage:
        refiner = ConfigRefiner()
        if refiner.update_preferences(insights):
            print("CLAUDE.md atualizado!")
    """

    def __init__(
        self,
        claude_md_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ):
        self.claude_md = claude_md_path or CLAUDE_MD_PATH
        self.backup_dir = backup_dir or BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def update_preferences(self, insights: list[dict], force_timestamp: bool = False) -> bool:
        """
        Atualiza CLAUDE.md com novos insights.

        Args:
            insights: Lista de insights do PreferenceLearner
                      [{category, action, confidence, suggestion}, ...]
            force_timestamp: Se True, sempre atualiza timestamp mesmo sem mudanca de conteudo

        Returns:
            True se atualizou, False se nao havia mudancas
        """
        if not insights:
            return False

        # Ler conteudo atual
        current_content = self._read_current()

        # Gerar nova secao DAIMON
        new_daimon_section = self._generate_section(insights)

        # Verificar se houve mudanca real
        current_daimon = self.get_current_preferences()
        is_same = current_daimon and self._sections_equal(
            current_daimon, new_daimon_section
        )
        content_changed = not is_same

        # Se nao mudou e nao forcar, retorna
        if not content_changed and not force_timestamp:
            return False

        # Backup antes de qualquer modificacao
        self._create_backup()

        # Merge conteudo
        updated_content = self._merge_content(current_content, new_daimon_section)

        # Escrever
        self._write(updated_content)

        # Log da atualizacao
        self._log_update(insights)

        return True

    def _create_backup(self) -> Optional[Path]:
        """
        Cria backup timestamped do CLAUDE.md atual.

        Returns:
            Path do backup criado, ou None se arquivo nao existia
        """
        if not self.claude_md.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"CLAUDE.md.{timestamp}"
        shutil.copy(self.claude_md, backup_path)

        # Manter apenas ultimos 10 backups
        backups = sorted(self.backup_dir.glob("CLAUDE.md.*"))
        for old_backup in backups[:-10]:
            try:
                old_backup.unlink()
            except OSError:
                pass

        return backup_path

    def _read_current(self) -> str:
        """Le conteudo atual do CLAUDE.md."""
        if self.claude_md.exists():
            try:
                return self.claude_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return ""
        return ""

    def _generate_section(self, insights: list[dict]) -> str:
        """
        Gera secao DAIMON do CLAUDE.md.

        Formato:
        <!-- DAIMON:AUTO:START -->
        # Preferencias Aprendidas (DAIMON)
        *Ultima atualizacao: 2025-12-12 19:30*

        ## Code Style
        - [sugestao]

        ## Verbosity
        - [sugestao]
        <!-- DAIMON:AUTO:END -->
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            DAIMON_SECTION_START,
            "# Preferencias Aprendidas (DAIMON)",
            f"*Ultima atualizacao: {timestamp}*",
            "",
        ]

        # Agrupar por categoria
        by_category: dict[str, list[dict]] = {}
        for insight in insights:
            cat = insight.get("category", "general")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(insight)

        # Gerar instrucoes por categoria
        for category in sorted(by_category.keys()):
            cat_insights = by_category[category]

            # Titulo formatado
            title = category.replace("_", " ").title()
            lines.append(f"## {title}")

            for insight in cat_insights:
                confidence = insight.get("confidence", 0)
                suggestion = insight.get("suggestion", "")

                # Indicador visual de confianca
                if confidence >= 0.7:
                    indicator = "[Alta]"
                elif confidence >= 0.4:
                    indicator = "[Media]"
                else:
                    indicator = "[Baixa]"

                lines.append(f"- {indicator} {suggestion}")

            lines.append("")

        lines.append(DAIMON_SECTION_END)

        return "\n".join(lines)

    def _merge_content(self, current: str, new_section: str) -> str:
        """
        Merge nova secao DAIMON com conteudo existente.

        Preserva todo conteudo manual do usuario.
        """
        # Se ja existe secao DAIMON, substituir
        if DAIMON_SECTION_START in current and DAIMON_SECTION_END in current:
            pattern = (
                re.escape(DAIMON_SECTION_START)
                + r".*?"
                + re.escape(DAIMON_SECTION_END)
            )
            return re.sub(pattern, new_section, current, flags=re.DOTALL)

        # Se existe conteudo manual, adicionar secao DAIMON no inicio
        if current.strip():
            return f"{new_section}\n\n---\n\n{current}"

        # Arquivo vazio ou nao existe, criar com secao DAIMON
        return new_section

    def _sections_equal(self, section1: str, section2: str) -> bool:
        """Compara secoes ignorando timestamps."""
        # Remover linha de timestamp antes de comparar
        pattern = r"\*Ultima atualizacao:.*\*"
        s1 = re.sub(pattern, "", section1).strip()
        s2 = re.sub(pattern, "", section2).strip()
        return s1 == s2

    def _write(self, content: str) -> None:
        """Escreve novo conteudo no CLAUDE.md."""
        self.claude_md.parent.mkdir(parents=True, exist_ok=True)
        self.claude_md.write_text(content, encoding="utf-8")

    def _log_update(self, insights: list[dict]) -> None:
        """Loga atualizacao para auditoria."""
        log_path = self.backup_dir / "update_log.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "insights_count": len(insights),
            "categories": list(set(i.get("category", "general") for i in insights)),
            "actions": list(set(i.get("action", "unknown") for i in insights)),
        }

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except (OSError, IOError):
            pass

    def get_current_preferences(self) -> Optional[str]:
        """
        Retorna secao DAIMON atual do CLAUDE.md.

        Returns:
            Conteudo da secao DAIMON, ou None se nao existir
        """
        content = self._read_current()

        if DAIMON_SECTION_START in content and DAIMON_SECTION_END in content:
            start = content.index(DAIMON_SECTION_START)
            end = content.index(DAIMON_SECTION_END) + len(DAIMON_SECTION_END)
            return content[start:end]

        return None

    def get_manual_content(self) -> str:
        """
        Retorna conteudo manual (nao-DAIMON) do CLAUDE.md.

        Returns:
            Conteudo fora da secao DAIMON
        """
        content = self._read_current()

        if DAIMON_SECTION_START in content and DAIMON_SECTION_END in content:
            # Remover secao DAIMON
            pattern = (
                re.escape(DAIMON_SECTION_START)
                + r".*?"
                + re.escape(DAIMON_SECTION_END)
            )
            manual = re.sub(pattern, "", content, flags=re.DOTALL)
            # Limpar separadores extras
            manual = re.sub(r"\n---\n\n+", "\n", manual)
            return manual.strip()

        return content.strip()

    def get_backup_list(self) -> list[dict]:
        """
        Lista backups disponiveis.

        Returns:
            [{path, timestamp, size_bytes}, ...]
        """
        backups = []

        for backup_file in sorted(self.backup_dir.glob("CLAUDE.md.*"), reverse=True):
            try:
                stat = backup_file.stat()
                # Extrair timestamp do nome
                ts_str = backup_file.suffix.lstrip(".")
                backups.append({
                    "path": str(backup_file),
                    "filename": backup_file.name,
                    "timestamp": ts_str,
                    "size_bytes": stat.st_size,
                })
            except OSError:
                continue

        return backups

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restaura um backup especifico.

        Args:
            backup_path: Caminho completo do backup

        Returns:
            True se restaurou com sucesso
        """
        backup_file = Path(backup_path)

        if not backup_file.exists():
            return False

        # Criar backup do estado atual antes de restaurar
        self._create_backup()

        try:
            shutil.copy(backup_file, self.claude_md)
            return True
        except (OSError, IOError):
            return False


if __name__ == "__main__":
    # Teste standalone
    print("DAIMON Config Refiner - Teste")
    print("=" * 50)

    refiner = ConfigRefiner()

    # Mostrar estado atual
    current_prefs = refiner.get_current_preferences()
    if current_prefs:
        print("\nSecao DAIMON atual:")
        print(current_prefs[:200] + "..." if len(current_prefs) > 200 else current_prefs)
    else:
        print("\nNenhuma secao DAIMON existente")

    manual_content = refiner.get_manual_content()
    if manual_content:
        print(f"\nConteudo manual: {len(manual_content)} caracteres")

    # Testar com insights de exemplo
    test_insights = [
        {
            "category": "verbosity",
            "action": "reduce",
            "confidence": 0.8,
            "suggestion": "Preferir respostas concisas.",
        },
        {
            "category": "testing",
            "action": "reinforce",
            "confidence": 0.9,
            "suggestion": "Continuar gerando testes proativamente.",
        },
    ]

    print("\nTestando update com insights de exemplo...")
    if refiner.update_preferences(test_insights):
        print("CLAUDE.md atualizado!")
        print("\nNova secao DAIMON:")
        print(refiner.get_current_preferences())
    else:
        print("Nenhuma mudanca necessaria")

    # Listar backups
    backup_list = refiner.get_backup_list()
    print(f"\nBackups disponiveis: {len(backup_list)}")
    for bkp in backup_list[:3]:
        print(f"  - {bkp['filename']} ({bkp['size_bytes']} bytes)")

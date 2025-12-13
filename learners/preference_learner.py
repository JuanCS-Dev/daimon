"""
DAIMON Preference Learner.

Analisa sessoes Claude Code para detectar padroes de aprovacao/rejeicao.
Gera sinais que alimentam o Reflection Engine.

Campos criticos do JSONL (verificado):
- type: "user" | "assistant" | "system"
- message.content[]: array com text, tool_use, tool_result
- toolUseResult.status: "success" | "failed"
- uuid + parentUuid: encadeamento de conversa
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Padroes de aprovacao/rejeicao (portugues + ingles)
APPROVAL_PATTERNS = [
    r"\b(sim|yes|ok|perfeito|otimo|excelente|isso|gostei)\b",
    r"\b(aceito|aprovo|pode|manda|vai|bora|certo|correto)\b",
    r"^(s|y|ok|sim)$",
    r"(thumbs.?up|great|good|nice|awesome)",
]

REJECTION_PATTERNS = [
    r"\b(nao|no|nope|errado|ruim|feio|pare|espera)\b",
    r"\b(rejeito|recuso|para|cancela|volta|desfaz)\b",
    r"\b(menos|mais simples|muito|demais|longo)\b",
    r"(thumbs.?down|bad|wrong|incorrect)",
]

# Categorias inferidas pelo contexto
CATEGORY_KEYWORDS = {
    "code_style": ["formatacao", "estilo", "naming", "indent", "lint", "format", "style"],
    "verbosity": ["verboso", "longo", "curto", "resumo", "detalhado", "verbose", "brief"],
    "testing": ["teste", "test", "coverage", "mock", "assert", "spec", "unit"],
    "architecture": ["arquitetura", "estrutura", "pattern", "design", "refactor"],
    "documentation": ["doc", "comment", "readme", "docstring", "jsdoc"],
    "workflow": ["commit", "branch", "git", "deploy", "ci", "cd", "push"],
    "security": ["security", "auth", "password", "token", "secret", "vulnerability"],
    "performance": ["performance", "speed", "fast", "slow", "optimize", "cache"],
}


@dataclass
class PreferenceSignal:
    """Representa um sinal de preferencia detectado."""

    timestamp: str
    signal_type: str  # "approval" | "rejection" | "modification"
    context: str
    category: str
    strength: float  # 0.0 - 1.0
    session_id: str
    tool_involved: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class PreferenceLearner:
    """
    Detecta padroes de preferencia nas sessoes Claude Code.

    Analisa logs JSONL para identificar:
    - Aprovacoes explicitas (sim, ok, perfeito)
    - Rejeicoes explicitas (nao, errado, para)
    - Modificacoes implicitas (usuario corrige apos proposta)

    Usage:
        learner = PreferenceLearner()
        signals = learner.scan_sessions(since_hours=24)
        summary = learner.get_preference_summary()
        insights = learner.get_actionable_insights()
    """

    def __init__(self, projects_dir: Optional[Path] = None):
        self.projects_dir = projects_dir or CLAUDE_PROJECTS
        self.signals: list[PreferenceSignal] = []
        self.category_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"approvals": 0, "rejections": 0}
        )

    def scan_sessions(self, since_hours: int = 24) -> list[PreferenceSignal]:
        """
        Escaneia sessoes recentes por sinais de preferencia.

        Primary: Use ActivityStore (populated by claude_watcher)
        Fallback: Direct file scan if ActivityStore unavailable

        Args:
            since_hours: Quantas horas atras considerar

        Returns:
            Lista de sinais detectados
        """
        # Try ActivityStore first (populated by claude_watcher)
        activity_signals = self._scan_from_activity_store(since_hours)
        if activity_signals:
            for signal in activity_signals:
                self.signals.append(signal)
                self._update_counts(signal)
            return self.signals

        # Fallback: Direct file scan
        cutoff = datetime.now().timestamp() - (since_hours * 3600)

        for session_file in self._get_recent_sessions(cutoff):
            for signal in self._analyze_session(session_file):
                self.signals.append(signal)
                self._update_counts(signal)

        return self.signals

    def _scan_from_activity_store(self, since_hours: int) -> list[PreferenceSignal]:
        """
        Scan preference signals from ActivityStore.

        Uses pre-computed signals from claude_watcher.

        Args:
            since_hours: Hours to look back.

        Returns:
            List of PreferenceSignals or empty list if unavailable.
        """
        try:
            from memory.activity_store import get_activity_store

            store = get_activity_store()
            records = store.get_recent(watcher_type="claude", hours=since_hours, limit=1000)

            signals = []
            for record in records:
                data = record.data
                preference_signal = data.get("preference_signal")

                # Skip records without preference signals
                if not preference_signal:
                    continue

                signals.append(PreferenceSignal(
                    timestamp=record.timestamp.isoformat(),
                    signal_type=preference_signal,
                    context=f"Project: {data.get('project', 'unknown')}",
                    category=data.get("preference_category", "general"),
                    strength=float(data.get("signal_strength", 0.5)),
                    session_id=data.get("project", "unknown"),
                    tool_involved=None,
                ))

            return signals

        except ImportError:
            return []
        except Exception:  # pylint: disable=broad-except
            return []

    def _get_recent_sessions(self, cutoff: float) -> Generator[Path, None, None]:
        """Retorna sessoes modificadas apos cutoff."""
        if not self.projects_dir.exists():
            return

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Pular diretorios ocultos e especiais
            if project_dir.name.startswith("."):
                continue

            for jsonl in project_dir.glob("*.jsonl"):
                # Pular agent sidechains (focar nas sessoes principais)
                if jsonl.name.startswith("agent-"):
                    continue

                try:
                    if jsonl.stat().st_mtime > cutoff:
                        yield jsonl
                except OSError:
                    continue

    def _load_session_messages(self, session_file: Path) -> list[dict]:
        """Load messages from session JSONL file."""
        messages: list[dict] = []
        try:
            with open(session_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except (OSError, IOError):
            pass
        return messages

    def _get_signal_from_tool_result(self, msg: dict) -> Optional[str]:
        """Check tool_result.status for implicit rejection signals."""
        tool_result = msg.get("toolUseResult", {})
        if tool_result.get("status") == "failed" or tool_result.get("interrupted"):
            return "rejection"
        return None

    def _analyze_session(self, session_file: Path) -> Generator[PreferenceSignal, None, None]:
        """
        Analisa uma sessao por sinais de preferencia.

        Estrategia:
        1. Ler pares (assistant propoe, user responde)
        2. Detectar feedback no texto do user
        3. Inferir feedback de tool_result.status
        """
        session_id = session_file.stem
        messages = self._load_session_messages(session_file)

        for i, msg in enumerate(messages):
            if msg.get("type") != "user":
                continue

            content = self._extract_user_content(msg)
            if not content:
                continue

            context, tool_involved = self._get_previous_context(messages, i)
            signal_type = self._detect_signal_type(content)
            if not signal_type:
                signal_type = self._get_signal_from_tool_result(msg)

            if signal_type:
                yield PreferenceSignal(
                    timestamp=msg.get("timestamp", datetime.now().isoformat()),
                    signal_type=signal_type,
                    context=context[:500],
                    category=self._infer_category(content + " " + context),
                    strength=self._calculate_strength(content),
                    session_id=session_id,
                    tool_involved=tool_involved,
                )

    def _get_previous_context(
        self, messages: list[dict], index: int
    ) -> tuple[str, Optional[str]]:
        """Get context and tool name from previous assistant message."""
        if index > 0:
            prev = messages[index - 1]
            if prev.get("type") == "assistant":
                return (
                    self._extract_assistant_context(prev),
                    self._extract_tool_name(prev),
                )
        return "", None

    def _extract_user_content(self, msg: dict) -> str:
        """Extrai texto do conteudo do usuario."""
        message = msg.get("message", {})
        content = message.get("content", "")

        # Pode ser string ou array
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    # Pegar texto, ignorar tool_result
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts)

        return ""

    def _extract_assistant_context(self, msg: dict) -> str:
        """Extrai contexto da resposta do assistant."""
        message = msg.get("message", {})
        content = message.get("content", [])

        if isinstance(content, str):
            return content[:200]

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Priorizar texto sobre tool_use
                    if item.get("type") == "text":
                        return item.get("text", "")[:200]

        return ""

    def _extract_tool_name(self, msg: dict) -> Optional[str]:
        """Extrai nome da ferramenta usada pelo assistant."""
        message = msg.get("message", {})
        content = message.get("content", [])

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    return item.get("name")

        return None

    def _detect_signal_type(self, content: str) -> Optional[str]:
        """Detecta se conteudo indica aprovacao ou rejeicao."""
        content_lower = content.lower()

        # Verificar aprovacao primeiro
        for pattern in APPROVAL_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return "approval"

        # Verificar rejeicao
        for pattern in REJECTION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return "rejection"

        return None

    def _infer_category(self, text: str) -> str:
        """Infere categoria baseado em keywords no texto."""
        text_lower = text.lower()

        # Contar matches por categoria
        scores: dict[str, int] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            # Retornar categoria com mais matches
            return max(scores, key=scores.get)  # type: ignore

        return "general"

    def _calculate_strength(self, content: str) -> float:
        """
        Calcula forca do sinal (0.0 - 1.0).

        Sinais curtos e diretos sao mais fortes.
        """
        word_count = len(content.split())

        if word_count <= 3:
            return 0.9  # "sim", "ok", "perfeito"
        if word_count <= 10:
            return 0.7  # Resposta curta
        if word_count <= 30:
            return 0.5  # Resposta media
        return 0.3  # Resposta longa (feedback diluido)

    def _update_counts(self, signal: PreferenceSignal) -> None:
        """Atualiza contadores por categoria."""
        if signal.signal_type == "approval":
            self.category_counts[signal.category]["approvals"] += 1
        elif signal.signal_type == "rejection":
            self.category_counts[signal.category]["rejections"] += 1

    def get_preference_summary(self) -> dict[str, dict]:
        """
        Retorna resumo de preferencias por categoria.

        Returns:
            {category: {approval_rate, total_signals, trend}}
        """
        summary: dict[str, dict] = {}

        for category, counts in self.category_counts.items():
            total = counts["approvals"] + counts["rejections"]
            if total > 0:
                approval_rate = counts["approvals"] / total
                summary[category] = {
                    "approval_rate": round(approval_rate, 2),
                    "total_signals": total,
                    "approvals": counts["approvals"],
                    "rejections": counts["rejections"],
                    "trend": (
                        "positive" if approval_rate > 0.6
                        else "negative" if approval_rate < 0.4
                        else "neutral"
                    ),
                }

        return summary

    def get_actionable_insights(self, min_signals: int = 3) -> list[dict]:
        """
        Retorna insights acionaveis para atualizar CLAUDE.md.

        Args:
            min_signals: Minimo de sinais para gerar insight

        Returns:
            Lista de insights com sugestoes
        """
        insights: list[dict] = []

        for category, counts in self.category_counts.items():
            total = counts["approvals"] + counts["rejections"]

            if total < min_signals:
                continue

            rate = counts["approvals"] / total

            if rate < 0.3:  # Alta rejeicao (>70%)
                insights.append({
                    "category": category,
                    "action": "reduce",
                    "confidence": min(total / 10, 1.0),
                    "approval_rate": round(rate, 2),
                    "total_signals": total,
                    "suggestion": self._generate_suggestion(category, "reduce"),
                })
            elif rate > 0.8:  # Alta aprovacao (>80%)
                insights.append({
                    "category": category,
                    "action": "reinforce",
                    "confidence": min(total / 10, 1.0),
                    "approval_rate": round(rate, 2),
                    "total_signals": total,
                    "suggestion": self._generate_suggestion(category, "reinforce"),
                })

        # Ordenar por confianca
        return sorted(insights, key=lambda x: x["confidence"], reverse=True)

    def _generate_suggestion(self, category: str, action: str) -> str:
        """Gera sugestao textual para CLAUDE.md."""
        suggestions = {
            # Reducoes
            ("verbosity", "reduce"): (
                "Preferir respostas concisas. Evitar explicacoes longas."
            ),
            ("code_style", "reduce"): (
                "Perguntar antes de reformatar codigo. Nao aplicar auto."
            ),
            ("testing", "reduce"): "Perguntar antes de gerar testes. Nao criar automaticamente.",
            ("architecture", "reduce"): "Evitar sugestoes arquiteturais nao solicitadas.",
            ("documentation", "reduce"): "Reduzir documentacao automatica. Focar no codigo.",
            ("workflow", "reduce"): "Nao executar comandos git sem confirmacao explicita.",
            ("security", "reduce"): "Avisar sobre riscos, mas nao bloquear sem solicitacao.",
            ("performance", "reduce"): "Focar em funcionalidade antes de otimizacao.",

            # Reforcos
            ("verbosity", "reinforce"): "Manter nivel atual de detalhe nas respostas.",
            ("code_style", "reinforce"): "Continuar aplicando padroes de codigo consistentes.",
            ("testing", "reinforce"): "Continuar gerando testes proativamente.",
            ("architecture", "reinforce"): "Sugestoes arquiteturais sao bem recebidas.",
            ("documentation", "reinforce"): "Manter documentacao detalhada.",
            ("workflow", "reinforce"): "Continuar com fluxo git proativo.",
            ("security", "reinforce"): "Manter alertas de seguranca proativos.",
            ("performance", "reinforce"): "Continuar sugerindo otimizacoes.",
        }

        key = (category, action)
        return suggestions.get(
            key,
            f"{action.title()} comportamento em '{category}'."
        )

    def clear(self) -> None:
        """Limpa sinais e contadores."""
        self.signals.clear()
        self.category_counts.clear()

    def get_stats(self) -> dict:
        """Retorna estatisticas gerais."""
        total_approvals = sum(c["approvals"] for c in self.category_counts.values())
        total_rejections = sum(c["rejections"] for c in self.category_counts.values())
        total = total_approvals + total_rejections

        return {
            "total_signals": total,
            "total_approvals": total_approvals,
            "total_rejections": total_rejections,
            "overall_approval_rate": round(total_approvals / total, 2) if total > 0 else 0,
            "categories_analyzed": len(self.category_counts),
            "signals_by_category": dict(self.category_counts),
        }

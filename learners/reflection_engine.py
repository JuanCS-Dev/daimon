"""
DAIMON Reflection Engine.

Orquestra PreferenceLearner + ConfigRefiner com triggers automaticos.

Triggers:
- Temporal: A cada 30min de sessao ativa
- Threshold: >5 rejeicoes do mesmo tipo
- Manual: /daimon reflect

Usage:
    engine = get_engine()
    await engine.start()  # Inicia loop de reflexao

    # Ou reflexao manual
    await engine.reflect()
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from .preference_learner import PreferenceLearner, PreferenceSignal
from .style_learner import get_style_learner
from .metacognitive_engine import get_metacognitive_engine

# Importar ConfigRefiner com tratamento de erro
# Usar import absoluto para funcionar tanto como modulo quanto standalone
_config_refiner_class: Optional[type] = None  # pylint: disable=invalid-name
try:
    from actuators.config_refiner import ConfigRefiner as _ConfigRefinerImport
    _config_refiner_class = _ConfigRefinerImport  # pylint: disable=invalid-name
except ImportError:
    try:
        from ..actuators.config_refiner import ConfigRefiner as _ConfigRefinerImport
        _config_refiner_class = _ConfigRefinerImport  # pylint: disable=invalid-name
    except ImportError:
        pass  # _config_refiner_class permanece None

if TYPE_CHECKING:
    from actuators.config_refiner import ConfigRefiner as ConfigRefinerType

logger = logging.getLogger("daimon.reflection")


@dataclass
class ReflectionConfig:
    """Configuration for reflection engine."""

    interval_minutes: int = 30
    rejection_threshold: int = 5
    approval_threshold: int = 10
    scan_hours: int = 48


@dataclass
class ReflectionStats:
    """Statistics for reflection engine."""

    total_reflections: int = 0
    total_updates: int = 0
    last_reflection: Optional[datetime] = field(default=None)


class ReflectionEngine:  # pylint: disable=too-many-instance-attributes
    """
    Motor de reflexao do DAIMON.

    Orquestra:
    1. PreferenceLearner - detecta padroes
    2. ConfigRefiner - atualiza CLAUDE.md
    3. Triggers - quando executar reflexao

    Triggers disponiveis:
    - Temporal: A cada reflection_interval
    - Threshold: Quando rejection_threshold atingido
    - Manual: Via reflect()
    """

    def __init__(self, config: Optional[ReflectionConfig] = None):
        """
        Inicializa engine de reflexao.

        Args:
            config: Configuracao do engine. Se None, usa defaults.
        """
        self.config = config or ReflectionConfig()
        self.stats = ReflectionStats()
        self.learner = PreferenceLearner()
        self.refiner = _config_refiner_class() if _config_refiner_class else None
        self.running = False
        self._task: Optional[asyncio.Task[None]] = None
        # Notification debounce (consolidate within 30s window)
        self._last_notification: Optional[datetime] = None
        self._pending_categories: set[str] = set()
        self._notification_debounce_seconds = 30

    async def start(self) -> None:
        """Inicia loop de reflexao assincirono."""
        if self.running:
            logger.warning("Reflection engine already running")
            return

        self.running = True
        interval = timedelta(minutes=self.config.interval_minutes)
        logger.info("Starting reflection engine (interval: %s)", interval)

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Para loop de reflexao."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping reflection engine")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        """Loop principal de reflexao."""
        while self.running:
            try:
                await self._check_triggers()
                await asyncio.sleep(60)  # Check a cada minuto
            except asyncio.CancelledError:
                break
            except (OSError, IOError, ValueError, RuntimeError) as e:
                logger.error("Error in reflection loop: %s", e)
                await asyncio.sleep(60)  # Continuar mesmo com erro

    async def _check_triggers(self) -> None:
        """Verifica se deve disparar reflexao."""
        now = datetime.now()
        interval = timedelta(minutes=self.config.interval_minutes)

        # Trigger temporal
        last = self.stats.last_reflection
        if last is None or (now - last) > interval:
            logger.debug("Temporal trigger activated")
            await self.reflect()
            return

        # Trigger por threshold (scan rapido)
        self.learner.clear()
        signals = self.learner.scan_sessions(since_hours=1)

        if self._check_threshold(signals):
            logger.info("Threshold trigger activated")
            await self.reflect()

    def _check_threshold(self, signals: list[PreferenceSignal]) -> bool:
        """
        Verifica se atingiu threshold para trigger imediato.

        Returns:
            True se deve executar reflexao
        """
        rejections: Counter[str] = Counter()
        approvals: Counter[str] = Counter()

        for sig in signals:
            if sig.signal_type == "rejection":
                rejections[sig.category] += 1
            elif sig.signal_type == "approval":
                approvals[sig.category] += 1

        # Verificar threshold de rejeicoes
        if any(c >= self.config.rejection_threshold for c in rejections.values()):
            return True

        # Verificar threshold de aprovacoes
        if any(c >= self.config.approval_threshold for c in approvals.values()):
            return True

        return False

    async def reflect(self) -> dict:
        """
        Executa reflexao completa.

        Returns:
            Resultado da reflexao {signals, insights, updated}
        """
        start_time = datetime.now()
        logger.info("Starting reflection at %s", start_time.isoformat())

        # Limpar estado anterior
        self.learner.clear()

        # Scan completo
        signals = self.learner.scan_sessions(since_hours=self.config.scan_hours)
        logger.info("Scanned %d signals", len(signals))

        # Gerar insights do PreferenceLearner
        insights = self.learner.get_actionable_insights(min_signals=3)

        # Adicionar insights do StyleLearner
        style_insights = self._get_style_insights()
        insights.extend(style_insights)

        logger.info("Generated %d actionable insights", len(insights))

        # Atualizar CLAUDE.md se houver refiner e insights
        updated = await self._apply_insights(insights, force_timestamp=True)

        # Atualizar estatisticas
        self.stats.last_reflection = datetime.now()
        self.stats.total_reflections += 1

        elapsed = (self.stats.last_reflection - start_time).total_seconds()
        logger.info("Reflection completed in %.2fs", elapsed)

        return {
            "signals_count": len(signals),
            "insights_count": len(insights),
            "style_insights": len(style_insights),
            "updated": updated,
            "elapsed_seconds": elapsed,
            "timestamp": self.stats.last_reflection.isoformat(),
        }

    def _get_style_insights(self) -> list[dict]:
        """
        Get insights from StyleLearner based on activity patterns.

        Returns:
            List of insight dicts compatible with ConfigRefiner.
        """
        try:
            style_learner = get_style_learner()
            style = style_learner.compute_style()

            # Only generate insights if we have enough confidence
            if style.confidence < 0.3:
                return []

            insights = []
            suggestions = style.to_claude_suggestions()

            for suggestion in suggestions:
                insights.append({
                    "category": "communication_style",
                    "action": "reinforce",
                    "confidence": style.confidence,
                    "suggestion": suggestion,
                })

            return insights

        except (ValueError, AttributeError, TypeError, KeyError) as e:
            logger.warning("Failed to get style insights: %s", e)
            return []

    async def _apply_insights(self, insights: list[dict], force_timestamp: bool = False) -> bool:
        """Apply insights to CLAUDE.md if refiner available."""
        if not insights:
            return False

        # Log insights to metacognitive engine for effectiveness tracking
        try:
            metacog = get_metacognitive_engine()
            for insight in insights:
                metacog.log_insight(
                    insight,
                    was_applied=self.refiner is not None,
                    context={"scan_hours": self.config.scan_hours},
                )
        except (ValueError, TypeError, ImportError) as e:
            logger.debug("Failed to log to metacognitive engine: %s", e)

        if not self.refiner:
            return False

        try:
            updated = self.refiner.update_preferences(insights, force_timestamp=force_timestamp)
            if updated:
                logger.info("Updated ~/.claude/CLAUDE.md")
                self.stats.total_updates += 1
                await self._notify_update(insights)
            return updated
        except (OSError, IOError, ValueError) as e:
            logger.error("Failed to update CLAUDE.md: %s", e)
            return False

    async def _notify_update(self, insights: list[dict]) -> None:
        """
        Notifica usuario sobre atualizacao com debounce.

        Consolida multiplas notificacoes em uma janela de 30s.
        Usa notify-send no Linux se disponivel.
        """
        # Acumular categorias
        for insight in insights:
            self._pending_categories.add(insight.get("category", "general"))

        now = datetime.now()

        # Debounce: se notificou recentemente, apenas acumula
        if self._last_notification:
            elapsed = (now - self._last_notification).total_seconds()
            if elapsed < self._notification_debounce_seconds:
                logger.debug("Notification debounced, %d categories pending", len(self._pending_categories))
                return

        # Enviar notificacao consolidada
        if not self._pending_categories:
            return

        categories_str = ", ".join(sorted(self._pending_categories))
        count = len(self._pending_categories)
        message = f"Atualizou {count} categoria(s): {categories_str}"

        # Limpar pendentes e atualizar timestamp
        self._pending_categories.clear()
        self._last_notification = now

        # Desktop notification (Linux)
        try:
            proc = await asyncio.create_subprocess_shell(
                f'notify-send -i dialog-information "DAIMON" "{message}" 2>/dev/null',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, OSError, IOError):
            pass  # Ignorar erros de notificacao

    def get_status(self) -> dict:
        """Retorna status completo do engine."""
        interval = timedelta(minutes=self.config.interval_minutes)
        next_reflection = None
        last = self.stats.last_reflection
        if last:
            next_reflection = (last + interval).isoformat()

        return {
            "running": self.running,
            "last_reflection": last.isoformat() if last else None,
            "next_reflection": next_reflection,
            "reflection_interval_minutes": self.config.interval_minutes,
            "total_reflections": self.stats.total_reflections,
            "total_updates": self.stats.total_updates,
            "signals_in_memory": len(self.learner.signals),
            "current_preferences": self.learner.get_preference_summary(),
            "current_stats": self.learner.get_stats(),
            "thresholds": {
                "rejection": self.config.rejection_threshold,
                "approval": self.config.approval_threshold,
            },
        }

    def get_learner(self) -> PreferenceLearner:
        """Retorna instancia do learner."""
        return self.learner

    def get_refiner(self) -> Optional["ConfigRefinerType"]:
        """Retorna instancia do refiner."""
        return self.refiner

    def get_metacognitive_analysis(self) -> dict:
        """
        Get metacognitive analysis of reflection effectiveness.

        Returns:
            Dict with meta-analysis results.
        """
        try:
            metacog = get_metacognitive_engine()
            return metacog.reflect_on_reflection()
        except (ValueError, TypeError, ImportError) as e:
            logger.warning("Failed to get metacognitive analysis: %s", e)
            return {"error": str(e)}


# Singleton storage (avoids global statement)
_singleton: dict[str, Optional[ReflectionEngine]] = {"engine": None}


def get_engine() -> ReflectionEngine:
    """
    Retorna instancia singleton do engine.

    Usage:
        engine = get_engine()
        await engine.reflect()
    """
    if _singleton["engine"] is None:
        _singleton["engine"] = ReflectionEngine()
    return _singleton["engine"]


def reset_engine() -> None:
    """Reseta singleton (para testes)."""
    _singleton["engine"] = None


if __name__ == "__main__":
    # Teste standalone
    async def main() -> None:
        """Test standalone execution of the reflection engine."""
        print("DAIMON Reflection Engine - Teste")
        print("=" * 50)

        engine = get_engine()

        print("\nStatus inicial:")
        status = engine.get_status()
        print(f"  Running: {status['running']}")
        print(f"  Last reflection: {status['last_reflection']}")

        print("\nExecutando reflexao manual...")
        result = await engine.reflect()

        print("\nResultado:")
        print(f"  Sinais analisados: {result['signals_count']}")
        print(f"  Insights gerados: {result['insights_count']}")
        print(f"  CLAUDE.md atualizado: {result['updated']}")
        print(f"  Tempo: {result['elapsed_seconds']:.2f}s")

        print("\nStatus apos reflexao:")
        status = engine.get_status()
        print(f"  Total reflexoes: {status['total_reflections']}")
        print(f"  Total updates: {status['total_updates']}")

        if status["current_preferences"]:
            print("\nPreferencias detectadas:")
            for cat, data in status["current_preferences"].items():
                print(f"  {cat}: {data['approval_rate']:.0%} ({data['total_signals']} sinais)")

    asyncio.run(main())

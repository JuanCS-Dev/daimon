"""
DAIMON Preference Insights Generator
=====================================

Generates actionable insights from preference analysis.

Responsible for:
- Generating preference summaries
- Creating actionable insights for CLAUDE.md updates
- LLM-powered semantic insight generation

Architecture:
    InsightGenerator is the final stage of the preference pipeline.
    It receives CategoryStats and produces PreferenceInsight objects.

Usage:
    from learners.preference.insights import InsightGenerator
    
    generator = InsightGenerator()
    insights = generator.get_insights(category_stats)
    summary = generator.get_summary(category_stats)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .models import CategoryStats, PreferenceInsight

logger = logging.getLogger("daimon.preference.insights")

# Suggestion templates for CLAUDE.md
SUGGESTION_TEMPLATES: Dict[tuple, str] = {
    # Reductions (high rejection rate)
    ("verbosity", "reduce"): (
        "Preferir respostas concisas. Evitar explicacoes longas."
    ),
    ("code_style", "reduce"): (
        "Perguntar antes de reformatar codigo. Nao aplicar auto."
    ),
    ("testing", "reduce"): (
        "Perguntar antes de gerar testes. Nao criar automaticamente."
    ),
    ("architecture", "reduce"): (
        "Evitar sugestoes arquiteturais nao solicitadas."
    ),
    ("documentation", "reduce"): (
        "Reduzir documentacao automatica. Focar no codigo."
    ),
    ("workflow", "reduce"): (
        "Nao executar comandos git sem confirmacao explicita."
    ),
    ("security", "reduce"): (
        "Avisar sobre riscos, mas nao bloquear sem solicitacao."
    ),
    ("performance", "reduce"): (
        "Focar em funcionalidade antes de otimizacao."
    ),
    
    # Reinforcements (high approval rate)
    ("verbosity", "reinforce"): (
        "Manter nivel atual de detalhe nas respostas."
    ),
    ("code_style", "reinforce"): (
        "Continuar aplicando padroes de codigo consistentes."
    ),
    ("testing", "reinforce"): (
        "Continuar gerando testes proativamente."
    ),
    ("architecture", "reinforce"): (
        "Sugestoes arquiteturais sao bem recebidas."
    ),
    ("documentation", "reinforce"): (
        "Manter documentacao detalhada."
    ),
    ("workflow", "reinforce"): (
        "Continuar com fluxo git proativo."
    ),
    ("security", "reinforce"): (
        "Manter alertas de seguranca proativos."
    ),
    ("performance", "reinforce"): (
        "Continuar sugerindo otimizacoes."
    ),
}


class InsightGenerator:
    """
    Generates actionable insights from preference data.
    
    Analyzes category statistics to identify:
    - High rejection categories (need behavior reduction)
    - High approval categories (can reinforce behavior)
    
    Supports LLM-powered insight generation for richer recommendations.
    """
    
    def __init__(self, enable_llm: bool = True):
        """
        Initialize generator.
        
        Args:
            enable_llm: Whether to use LLM for insight generation
        """
        self.enable_llm = enable_llm
        self._llm_service = None
    
    def get_summary(
        self,
        category_stats: Dict[str, CategoryStats],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate preference summary for all categories.
        
        Args:
            category_stats: Statistics per category
            
        Returns:
            Dictionary with approval rates, totals, and trends
        """
        summary = {}
        
        for category, stats in category_stats.items():
            if stats.total == 0:
                continue
            
            summary[category] = {
                "approval_rate": round(stats.approval_rate, 2),
                "total_signals": stats.total,
                "approvals": stats.approvals,
                "rejections": stats.rejections,
                "trend": self._calculate_trend(stats),
            }
        
        return summary
    
    def get_insights(
        self,
        category_stats: Dict[str, CategoryStats],
        min_signals: int = 3,
    ) -> List[PreferenceInsight]:
        """
        Generate actionable insights from category statistics.
        
        Uses LLM when available, falls back to template-based generation.
        
        Args:
            category_stats: Statistics per category
            min_signals: Minimum signals to generate insight
            
        Returns:
            List of PreferenceInsight objects sorted by confidence
        """
        if self.enable_llm:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(
                        self._get_insights_llm(category_stats, min_signals)
                    )
            except RuntimeError:
                pass
        
        return self._get_insights_template(category_stats, min_signals)
    
    async def _get_insights_llm(
        self,
        category_stats: Dict[str, CategoryStats],
        min_signals: int,
    ) -> List[PreferenceInsight]:
        """Generate insights using LLM service."""
        try:
            if self._llm_service is None:
                from learners.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            
            # Prepare data for LLM
            data = {
                cat: stats.to_dict()
                for cat, stats in category_stats.items()
                if stats.total >= min_signals
            }
            
            if not data:
                return []
            
            result = await self._llm_service.extract_insights(data)
            
            # Convert LLM insights to PreferenceInsight objects
            insights = []
            for i, insight_text in enumerate(result.insights):
                suggestion = (
                    result.suggestions[i]
                    if i < len(result.suggestions)
                    else insight_text
                )
                
                insights.append(PreferenceInsight(
                    category="general",
                    action="adjust",
                    confidence=result.confidence,
                    approval_rate=0.5,
                    total_signals=sum(s.total for s in category_stats.values()),
                    suggestion=suggestion,
                    from_llm=True,
                ))
            
            return insights
            
        except Exception as e:
            logger.debug("LLM insight generation failed: %s", e)
            return self._get_insights_template(category_stats, min_signals)
    
    def _get_insights_template(
        self,
        category_stats: Dict[str, CategoryStats],
        min_signals: int,
    ) -> List[PreferenceInsight]:
        """Generate insights using templates."""
        insights = []
        
        for category, stats in category_stats.items():
            if stats.total < min_signals:
                continue
            
            rate = stats.approval_rate
            
            if rate < 0.3:  # High rejection (>70%)
                insights.append(PreferenceInsight(
                    category=category,
                    action="reduce",
                    confidence=min(stats.total / 10, 1.0),
                    approval_rate=round(rate, 2),
                    total_signals=stats.total,
                    suggestion=self._get_suggestion(category, "reduce"),
                    from_llm=False,
                ))
            elif rate > 0.8:  # High approval (>80%)
                insights.append(PreferenceInsight(
                    category=category,
                    action="reinforce",
                    confidence=min(stats.total / 10, 1.0),
                    approval_rate=round(rate, 2),
                    total_signals=stats.total,
                    suggestion=self._get_suggestion(category, "reinforce"),
                    from_llm=False,
                ))
        
        # Sort by confidence
        return sorted(insights, key=lambda x: x.confidence, reverse=True)
    
    def _get_suggestion(self, category: str, action: str) -> str:
        """Get suggestion template for category and action."""
        key = (category, action)
        return SUGGESTION_TEMPLATES.get(
            key,
            f"{action.title()} comportamento em '{category}'."
        )
    
    def _calculate_trend(self, stats: CategoryStats) -> str:
        """Calculate trend indicator for category."""
        if stats.total < 3:
            return "insufficient_data"
        
        rate = stats.approval_rate
        if rate > 0.7:
            return "positive"
        elif rate < 0.3:
            return "negative"
        else:
            return "neutral"

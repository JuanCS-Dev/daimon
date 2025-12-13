"""
MCP NOESIS Tools - Consciousness and Ethics.

Tools:
- noesis_consult: Maieutic questioning (returns questions, not answers)
- noesis_tribunal: Ethical judgment (3-judge verdict)
- noesis_precedent: Search past decisions for guidance
- noesis_confront: Socratic confrontation of premises
- noesis_health: Check NOESIS services status
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastmcp import Context
from pydantic import Field

from .config import NOESIS_CONSCIOUSNESS_URL, NOESIS_REFLECTOR_URL
from .http_utils import http_get, http_post
from .server import mcp


@mcp.tool(
    tags={"consultation", "maieutic", "thinking"},
    annotations={
        "title": "NOESIS Consultation",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def noesis_consult(
    question: Annotated[str, Field(
        description="What you want to think through",
        min_length=1,
        max_length=2000
    )],
    context: Annotated[Optional[str], Field(
        description="Additional context about the situation",
        max_length=5000
    )] = None,
    depth: Annotated[int, Field(
        description="Thinking depth level",
        ge=1,
        le=5
    )] = 2,
    ctx: Context | None = None,
) -> str:
    """
    Consult NOESIS for maieutic guidance. Returns QUESTIONS, not answers.

    Use this when facing:
    - Architectural decisions with multiple valid approaches
    - Unclear or ambiguous requirements
    - Tradeoffs between competing concerns
    - Decisions that could have long-term consequences
    """
    if ctx:
        await ctx.info(f"Consulting NOESIS: {question[:50]}...")

    # Build query with context if provided
    full_query = question
    if context:
        full_query = f"{question}\n\nContext: {context}"

    payload = {"query": full_query}

    result = await http_post(
        f"{NOESIS_CONSCIOUSNESS_URL}/v1/consciousness/introspect",
        payload
    )

    if "error" in result:
        msg = result.get('message', 'unknown error')
        if ctx:
            await ctx.warning(f"NOESIS unavailable: {msg}")
        return (
            f"[NOESIS unavailable: {msg}]\n\n"
            "Consider these questions yourself:\n"
            "1. What are the core tradeoffs?\n"
            "2. What could go wrong?\n"
            "3. What assumptions am I making?"
        )

    # Support both response formats (narrative or response field)
    narrative = result.get("narrative", "") or result.get("response", "")
    meta_level = result.get("meta_level", 0)
    qualia = result.get("qualia_desc", "")
    consciousness = result.get("consciousness_state", {})

    output = ["## NOESIS Consultation\n"]

    if narrative:
        output.append(narrative)

    if qualia:
        output.append(f"\n*Qualia: {qualia}*")

    if meta_level > 0:
        output.append(f"*Meta-level: {meta_level:.2f}*")

    # Include consciousness state coherence if available
    if isinstance(consciousness, dict) and "coherence" in consciousness:
        output.append(f"*Coherence: {consciousness['coherence']:.2f}*")

    return "\n".join(output)


@mcp.tool(
    tags={"tribunal", "ethics", "judgment", "safety"},
    annotations={
        "title": "NOESIS Tribunal",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def noesis_tribunal(
    action: Annotated[str, Field(
        description="The action to be judged by the tribunal",
        min_length=1,
        max_length=5000
    )],
    justification: Annotated[Optional[str], Field(
        description="Why you believe this action is appropriate",
        max_length=2000
    )] = None,
    context: Annotated[Optional[str], Field(
        description="Additional context about the situation",
        max_length=5000
    )] = None,
    ctx: Context | None = None,
) -> str:
    """
    Submit an action for ethical judgment by the Tribunal (3 judges).

    Judges:
    - VERITAS (Truth): Factual accuracy, honesty
    - SOPHIA (Wisdom): Strategic thinking, prudence
    - DIKE (Justice): Fairness, rights protection

    Use this for destructive operations, security-sensitive code,
    user data handling, and production deployments.
    """
    if ctx:
        await ctx.info(f"Tribunal judging: {action[:50]}...")

    payload = {
        "trace_id": str(uuid.uuid4()),
        "agent_id": "claude-code",
        "task": action[:100],
        "action": action,
        "outcome": justification or "pending",
        "reasoning_trace": context or ""
    }

    result = await http_post(
        f"{NOESIS_REFLECTOR_URL}/reflect/verdict",
        payload
    )

    if "error" in result:
        msg = result.get('message', 'unknown error')
        if ctx:
            await ctx.warning(f"Tribunal unavailable: {msg}")
        return (
            f"[Tribunal unavailable: {msg}]\n\n"
            "**CAUTION**: Proceed with extra care without ethical oversight."
        )

    # Format the verdict
    output = ["## Tribunal Verdict\n"]

    verdict_raw = result.get("verdict", "UNKNOWN")
    # Handle case where verdict is a dict (nested response)
    if isinstance(verdict_raw, dict):
        verdict = str(verdict_raw.get("verdict", verdict_raw.get("decision", "UNKNOWN")))
    else:
        verdict = str(verdict_raw)
    consensus = result.get("consensus_score", 0)

    # Verdict display
    verdict_display = {
        "PASS": "PASS",
        "REVIEW": "REVIEW",
        "FAIL": "FAIL",
        "CAPITAL": "CAPITAL"
    }.get(verdict.upper(), verdict.upper())

    output.append(f"**Decision**: {verdict_display}")
    output.append(f"**Consensus**: {consensus:.1%}\n")

    # Individual judge verdicts
    individual = result.get("individual_verdicts", {})
    if individual:
        output.append("### Judge Verdicts\n")
        for judge, jv in individual.items():
            vote = jv.get("vote", "?")
            confidence = jv.get("confidence", 0)
            reasoning = jv.get("reasoning", "")[:200]
            output.append(f"**{judge}**: {vote} ({confidence:.0%})")
            if reasoning:
                output.append(f"> {reasoning}\n")

    # Crimes detected
    crimes = result.get("crimes_detected", [])
    if crimes:
        output.append(f"\n**Concerns**: {', '.join(crimes)}")

    # Overall reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        output.append(f"\n### Reasoning\n{reasoning}")

    return "\n".join(output)


@mcp.tool(
    tags={"precedent", "history", "search", "learning"},
    annotations={
        "title": "NOESIS Precedent Search",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def noesis_precedent(
    situation: Annotated[str, Field(
        description="Description of the current situation to find precedents for",
        min_length=1,
        max_length=2000
    )],
    limit: Annotated[int, Field(
        description="Maximum number of precedents to return",
        ge=1,
        le=10
    )] = 3,
    ctx: Context | None = None,
) -> str:
    """
    Search for similar past decisions (precedents) for guidance.

    The precedent ledger records past tribunal decisions. Use this to
    learn from past mistakes, find guidance for recurring situations,
    and understand historical patterns.
    """
    if ctx:
        await ctx.info(f"Searching precedents for: {situation[:50]}...")

    payload = {
        "execution_log": {
            "content": situation,
            "task": "precedent_search",
            "result": "searching"
        },
        "search_precedents_only": True,
        "precedent_limit": min(max(limit, 1), 10)
    }

    result = await http_post(
        f"{NOESIS_REFLECTOR_URL}/reflect/verdict",
        payload
    )

    output = ["## Precedent Search\n"]
    output.append(f"**Situation**: {situation[:200]}...\n")

    # Extract precedents from result
    precedents = result.get("precedent_guidance", [])

    if precedents:
        output.append(f"### Found {len(precedents)} Relevant Precedent(s)\n")

        for i, prec in enumerate(precedents[:limit], 1):
            decision = prec.get("decision", "?")
            consensus = prec.get("consensus_score", 0)
            reasoning = prec.get("key_reasoning", "")[:300]
            timestamp = prec.get("timestamp", "")[:10]

            output.append(f"**{i}. {decision}** (consensus: {consensus:.0%})")
            if timestamp:
                output.append(f"   *Date: {timestamp}*")
            if reasoning:
                output.append(f"   > {reasoning}")
            output.append("")
    else:
        output.append("*No directly relevant precedents found.*\n")
        output.append("This may be a novel situation. Consider:")
        output.append("1. Use `noesis_tribunal` for ethical judgment")
        output.append("2. Use `noesis_consult` for maieutic guidance")
        output.append("3. Document this decision for future reference")

    return "\n".join(output)


@mcp.tool(
    tags={"confrontation", "socratic", "critical-thinking", "bias"},
    annotations={
        "title": "NOESIS Socratic Confrontation",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def noesis_confront(
    statement: Annotated[str, Field(
        description="The premise or statement to challenge socratically",
        min_length=1,
        max_length=2000
    )],
    shadow_pattern: Annotated[Optional[str], Field(
        description="Pattern detected (e.g., 'overconfidence', 'assumption', 'bias')",
        max_length=100
    )] = None,
    ctx: Context | None = None,
) -> str:
    """
    Challenge a premise or statement socratically.

    Use this when you notice high confidence in uncertain claims,
    unexamined assumptions, potential cognitive biases, or
    overconfident assertions.
    """
    if ctx:
        await ctx.info(f"Confronting: {statement[:50]}...")

    payload = {
        "trigger_event": statement,
        "violated_rule_id": None,
        "shadow_pattern": shadow_pattern or "unexamined_premise",
        "user_state": "ANALYTICAL"
    }

    result = await http_post(
        f"{NOESIS_CONSCIOUSNESS_URL}/v1/exocortex/confront",
        payload
    )

    if "error" in result:
        if ctx:
            await ctx.warning("NOESIS unavailable, using fallback questions")
        return f"""## Socratic Confrontation

**Premise**: {statement[:200]}

*[NOESIS unavailable - using fallback questions]*

### Challenge Questions

1. **Evidence**: What evidence supports this claim? Is it verifiable?

2. **Assumptions**: What unexamined assumptions underlie this statement?

3. **Alternatives**: What other explanations could account for this?

4. **Consequences**: If this is wrong, what would the consequences be?

5. **Origin**: Where did this belief come from? Training data? Heuristic?

*Reflect honestly before proceeding.*
"""

    output = ["## Socratic Confrontation\n"]
    output.append(f"**Premise**: {statement[:200]}\n")

    question = result.get("ai_question", "")
    style = result.get("style", "socratic")
    conf_id = result.get("id", "")

    if question:
        output.append(f"### {style.upper()} Challenge\n")
        output.append(question)
        output.append(f"\n*Confrontation ID: {conf_id}*")
    else:
        output.append("*No specific challenge generated.*")

    return "\n".join(output)


@mcp.tool(
    tags={"health", "status", "diagnostics"},
    annotations={
        "title": "NOESIS Health Check",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def noesis_health(
    ctx: Context | None = None,
) -> str:
    """
    Check the health of NOESIS services.

    Returns the status of the Consciousness service (port 8001)
    and the Metacognitive reflector (port 8002).
    """
    if ctx:
        await ctx.info("Checking NOESIS health...")

    output = ["## NOESIS Health Check\n"]

    # Check consciousness service
    consciousness_health = await http_get(f"{NOESIS_CONSCIOUSNESS_URL}/api/consciousness/state")
    if "error" not in consciousness_health:
        output.append("**Consciousness**: ONLINE")
    else:
        msg = consciousness_health.get('message', 'unknown')
        output.append(f"**Consciousness**: OFFLINE ({msg})")

    # Check reflector service
    reflector_health = await http_get(f"{NOESIS_REFLECTOR_URL}/health")
    if "error" not in reflector_health:
        output.append("**Tribunal**: ONLINE")
    else:
        msg = reflector_health.get('message', 'unknown')
        output.append(f"**Tribunal**: OFFLINE ({msg})")

    return "\n".join(output)

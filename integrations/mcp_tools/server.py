"""
MCP Server Instance.
"""

from fastmcp import FastMCP

# MCP Server instance - shared across tool modules
mcp = FastMCP(
    name="daimon-consciousness",
    version="1.1.0",
    instructions="""
    DAIMON is your wise co-architect. Use these tools for thoughtful decisions:

    ## NOESIS Tools (Consciousness & Ethics)

    - noesis_consult: Ask before deciding. Returns QUESTIONS, not answers.
      Use for: architectural decisions, unclear requirements, tradeoffs.

    - noesis_tribunal: Submit actions for ethical judgment by 3 judges.
      Use for: destructive operations, security-sensitive code, user data.

    - noesis_precedent: Search for similar past decisions and their outcomes.
      Use for: recurring patterns, learning from history, avoiding past mistakes.

    - noesis_confront: Challenge your premises socratically.
      Use for: high-confidence statements, assumptions, overconfident assertions.

    ## Corpus Tools (Personal Knowledge Base)

    - corpus_search: Semantic search over wisdom texts and personal knowledge.
      Use for: finding relevant quotes, philosophical guidance, reference material.

    - corpus_add: Add new knowledge entries to the corpus.
      Use for: preserving insights, documenting decisions, building knowledge.

    - corpus_stats: View corpus statistics and contents.
      Use for: understanding available knowledge, checking corpus health.

    Philosophy: Silence is gold. Only emerge when truly significant.
    """,
)

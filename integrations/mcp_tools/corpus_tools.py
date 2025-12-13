"""
MCP Corpus Tools - Personal Knowledge Base.

Tools:
- corpus_search: Semantic search over wisdom texts
- corpus_add: Add new knowledge entries
- corpus_stats: View corpus statistics
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from .config import logger
from .server import mcp


def _get_corpus_manager():
    """Get corpus manager instance (lazy import to avoid startup overhead)."""
    from corpus import CorpusManager  # pylint: disable=import-outside-toplevel
    return CorpusManager()


@mcp.tool(
    tags={"corpus", "search", "knowledge", "semantic"},
    annotations={
        "title": "Corpus Semantic Search",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def corpus_search(
    query: Annotated[str, Field(
        description="Search query (natural language)",
        min_length=1,
        max_length=500
    )],
    limit: Annotated[int, Field(
        description="Maximum results to return",
        ge=1,
        le=20
    )] = 5,
    search_type: Annotated[str, Field(
        description="Search type: 'semantic', 'keyword', or 'hybrid'",
    )] = "hybrid",
    ctx: Context | None = None,
) -> str:
    """
    Search the personal knowledge corpus using semantic similarity.

    The corpus contains curated wisdom texts, philosophical writings,
    and personal knowledge entries organized by category.
    """
    if ctx:
        await ctx.info(f"Searching corpus: {query[:50]}...")

    try:
        manager = _get_corpus_manager()

        if search_type == "semantic":
            results = manager.semantic_search(query, limit=limit)
        elif search_type == "keyword":
            results = manager.search(query, limit=limit)
        else:  # hybrid (default)
            results = manager.hybrid_search(query, limit=limit)

        output = ["## Corpus Search Results\n"]
        output.append(f"**Query**: {query}")
        output.append(f"**Type**: {search_type}\n")

        if results:
            output.append(f"### Found {len(results)} Result(s)\n")

            for i, text in enumerate(results, 1):
                output.append(f"**{i}. {text.title}** by *{text.author}*")
                output.append(f"   Category: {text.category}")
                if text.themes:
                    output.append(f"   Themes: {', '.join(text.themes[:5])}")
                # Show excerpt
                excerpt = text.content[:200].replace("\n", " ")
                output.append(f"   > {excerpt}...")
                output.append("")
        else:
            output.append("*No results found.*")
            output.append("\nTry a different query or use `corpus_add` to expand the knowledge base.")

        return "\n".join(output)

    except Exception as e:
        logger.error("Corpus search failed: %s", e)
        raise ToolError(f"Corpus search failed: {e}") from e


@mcp.tool(
    tags={"corpus", "add", "knowledge"},
    annotations={
        "title": "Add to Corpus",
        "readOnlyHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def corpus_add(
    content: Annotated[str, Field(
        description="Text content to add to the corpus",
        min_length=10,
        max_length=50000
    )],
    title: Annotated[str, Field(
        description="Title for the entry",
        min_length=1,
        max_length=200
    )],
    author: Annotated[str, Field(
        description="Author or source attribution",
        min_length=1,
        max_length=100
    )],
    category: Annotated[str, Field(
        description="Category: filosofia/gregos, filosofia/estoicos, ciencia, etica, etc.",
        min_length=1,
        max_length=50
    )] = "filosofia/modernos",
    themes: Annotated[Optional[str], Field(
        description="Comma-separated themes (e.g., 'virtue,wisdom,ethics')",
        max_length=200
    )] = None,
    ctx: Context | None = None,
) -> str:
    """
    Add new knowledge entry to the personal corpus.

    Available categories:
    - filosofia/gregos: Greek Philosophy
    - filosofia/estoicos: Stoic Philosophy
    - filosofia/modernos: Modern Philosophy
    - teologia: Theology and Religion
    - ciencia: Science and Method
    - logica: Logic and Reasoning
    - etica: Ethics and Morality
    - literatura: Literature and Poetry
    """
    if ctx:
        await ctx.info(f"Adding to corpus: {title[:50]}...")

    try:
        from corpus import CorpusManager, TextMetadata  # pylint: disable=import-outside-toplevel

        manager = CorpusManager()

        # Parse themes
        theme_list = []
        if themes:
            theme_list = [t.strip() for t in themes.split(",") if t.strip()]

        metadata = TextMetadata(themes=theme_list)

        text_id = manager.add_text(
            author=author,
            title=title,
            category=category,
            content=content,
            metadata=metadata,
        )

        return f"""## Added to Corpus

**ID**: {text_id}
**Title**: {title}
**Author**: {author}
**Category**: {category}
**Themes**: {', '.join(theme_list) if theme_list else 'none'}
**Length**: {len(content)} characters

The entry has been indexed for semantic search.
"""

    except Exception as e:
        logger.error("Failed to add to corpus: %s", e)
        raise ToolError(f"Failed to add to corpus: {e}") from e


@mcp.tool(
    tags={"corpus", "stats", "info"},
    annotations={
        "title": "Corpus Statistics",
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def corpus_stats(
    ctx: Context | None = None,
) -> str:
    """
    Get statistics about the personal knowledge corpus.

    Shows total texts, authors, themes, and category breakdown.
    """
    if ctx:
        await ctx.info("Getting corpus statistics...")

    try:
        manager = _get_corpus_manager()
        stats = manager.get_stats()

        output = ["## Corpus Statistics\n"]
        output.append(f"**Total Texts**: {stats['total_texts']}")
        output.append(f"**Total Authors**: {stats['total_authors']}")
        output.append(f"**Total Themes**: {stats['total_themes']}")

        if stats.get("semantic_enabled"):
            output.append(f"**Semantic Index**: {stats.get('semantic_indexed', 0)} documents")
        else:
            output.append("**Semantic Index**: Not available (install sentence-transformers)")

        output.append(f"\n**Location**: {stats['corpus_path']}")

        by_category = stats.get("by_category", {})
        if by_category:
            output.append("\n### By Category\n")
            for cat, count in sorted(by_category.items()):
                output.append(f"- {cat}: {count}")

        # List authors
        authors = manager.list_authors()
        if authors:
            output.append(f"\n### Authors ({len(authors)})\n")
            output.append(", ".join(sorted(authors)[:20]))
            if len(authors) > 20:
                output.append(f"... and {len(authors) - 20} more")

        return "\n".join(output)

    except Exception as e:
        logger.error("Failed to get corpus stats: %s", e)
        raise ToolError(f"Failed to get corpus stats: {e}") from e

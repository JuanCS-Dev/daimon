"""
Corpus Search Functions.

Contains keyword and hybrid search implementations.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import WisdomText
    from .semantic_search import SemanticCorpus


def keyword_search(
    texts: Dict[str, "WisdomText"],
    query: str,
    limit: int = 10,
) -> List["WisdomText"]:
    """
    Search texts by content using keyword matching.

    Simple substring search in content, title, and themes.

    Args:
        texts: Dictionary of text_id -> WisdomText.
        query: Search query.
        limit: Maximum results.

    Returns:
        List of matching texts sorted by relevance.
    """
    query_lower = query.lower()
    results = []

    for text in texts.values():
        score = 0
        if query_lower in text.content.lower():
            score += 2
        if query_lower in text.title.lower():
            score += 3
        if any(query_lower in t.lower() for t in text.themes):
            score += 2

        if score > 0:
            results.append((text, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:limit]]


def keyword_search_scored(
    texts: Dict[str, "WisdomText"],
    query: str,
    limit: int,
) -> List[tuple]:
    """
    Internal keyword search returning (text_id, score) tuples.

    Args:
        texts: Dictionary of text_id -> WisdomText.
        query: Search query.
        limit: Maximum results.

    Returns:
        List of (text_id, score) tuples.
    """
    query_lower = query.lower()
    results = []

    for text in texts.values():
        score = 0.0
        if query_lower in text.content.lower():
            score += 0.4
        if query_lower in text.title.lower():
            score += 0.6
        if any(query_lower in t.lower() for t in text.themes):
            score += 0.4

        if score > 0:
            results.append((text.id, min(score, 1.0)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def hybrid_search(
    texts: Dict[str, "WisdomText"],
    semantic: Optional["SemanticCorpus"],
    query: str,
    limit: int = 5,
    semantic_weight: float = 0.7,
) -> List["WisdomText"]:
    """
    Hybrid search combining semantic and keyword matching.

    Args:
        texts: Dictionary of text_id -> WisdomText.
        semantic: Optional SemanticCorpus instance.
        query: Search query.
        limit: Maximum results.
        semantic_weight: Weight for semantic results (0-1).

    Returns:
        Combined results list.
    """
    # Get semantic results
    semantic_results = []
    if semantic:
        sem_raw = semantic.search(query, k=limit * 2)
        semantic_results = [
            (r.doc_id, r.score * semantic_weight)
            for r in sem_raw
            if r.doc_id in texts
        ]

    # Get keyword results
    keyword_results = keyword_search_scored(texts, query, limit * 2)
    keyword_weight = 1 - semantic_weight
    keyword_results = [
        (tid, score * keyword_weight)
        for tid, score in keyword_results
    ]

    # Merge scores
    scores: Dict[str, float] = {}
    for tid, score in semantic_results + keyword_results:
        scores[tid] = scores.get(tid, 0) + score

    # Sort and return
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [texts[tid] for tid in sorted_ids[:limit] if tid in texts]

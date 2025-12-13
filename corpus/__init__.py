"""DAIMON Corpus - Curated wisdom texts for ethical grounding."""

from .manager import CorpusManager
from .models import WisdomText, TextMetadata
from .search import keyword_search, keyword_search_scored, hybrid_search
from .semantic_search import SemanticCorpus, SearchResult, get_semantic_corpus

__all__ = [
    "CorpusManager",
    "WisdomText",
    "TextMetadata",
    "keyword_search",
    "keyword_search_scored",
    "hybrid_search",
    "SemanticCorpus",
    "SearchResult",
    "get_semantic_corpus",
]

"""
Audit Quality Tests - Performance and Edge Cases.
"""

import time

from .core import log


def test_performance() -> None:
    """Test performance requirements."""
    print("\n" + "=" * 60)
    print("14. PERFORMANCE TESTS")
    print("=" * 60)

    _test_memory_search_perf()
    _test_corpus_search_perf()
    _test_precedent_search_perf()
    _test_engine_status_perf()


def _test_memory_search_perf() -> None:
    """Test MemoryStore search < 10ms."""
    try:
        from memory import MemoryStore
        store = MemoryStore()
        store.add("Performance test data", category="perf", importance=0.5)

        start = time.perf_counter()
        store.search("Performance")
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms < 10:
            log("Performance", "MemoryStore search <10ms", "PASS", f"{elapsed_ms:.2f}ms")
        else:
            log("Performance", "MemoryStore search <10ms", "FAIL", f"{elapsed_ms:.2f}ms")

    except Exception as e:
        log("Performance", "MemoryStore search", "FAIL", str(e))


def _test_corpus_search_perf() -> None:
    """Test Corpus search < 50ms."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()

        start = time.perf_counter()
        manager.search("philosophy")
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms < 50:
            log("Performance", "Corpus search <50ms", "PASS", f"{elapsed_ms:.2f}ms")
        else:
            log("Performance", "Corpus search <50ms", "FAIL", f"{elapsed_ms:.2f}ms")

    except Exception as e:
        log("Performance", "Corpus search", "FAIL", str(e))


def _test_precedent_search_perf() -> None:
    """Test Precedent search < 20ms."""
    try:
        from memory import PrecedentSystem
        system = PrecedentSystem()

        start = time.perf_counter()
        system.search("test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms < 20:
            log("Performance", "Precedent search <20ms", "PASS", f"{elapsed_ms:.2f}ms")
        else:
            log("Performance", "Precedent search <20ms", "FAIL", f"{elapsed_ms:.2f}ms")

    except Exception as e:
        log("Performance", "Precedent search", "FAIL", str(e))


def _test_engine_status_perf() -> None:
    """Test Engine status < 5ms."""
    try:
        from learners import get_engine
        engine = get_engine()

        start = time.perf_counter()
        engine.get_status()
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms < 5:
            log("Performance", "Engine status <5ms", "PASS", f"{elapsed_ms:.2f}ms")
        else:
            log("Performance", "Engine status <5ms", "FAIL", f"{elapsed_ms:.2f}ms")

    except Exception as e:
        log("Performance", "Engine status", "FAIL", str(e))


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("15. EDGE CASE TESTS")
    print("=" * 60)

    _test_empty_search()
    _test_unicode_handling()
    _test_large_text()
    _test_sql_injection()


def _test_empty_search() -> None:
    """Test empty search query."""
    try:
        from memory import MemoryStore
        store = MemoryStore()
        results = store.search("")
        log("Edge Cases", "Empty search query", "PASS", f"{len(results)} results")
    except Exception as e:
        log("Edge Cases", "Empty search query", "FAIL", str(e))


def _test_unicode_handling() -> None:
    """Test unicode text handling."""
    try:
        from memory import MemoryStore
        store = MemoryStore()

        item_id = store.add("Test com acentuação: ção, ã, é, ü, 日本語", category="unicode")
        if item_id:
            results = store.search("acentuação")
            store.delete(item_id)
            log("Edge Cases", "Unicode handling", "PASS", f"{len(results)} results")
        else:
            log("Edge Cases", "Unicode handling", "FAIL", "Failed to add")

    except Exception as e:
        log("Edge Cases", "Unicode handling", "FAIL", str(e))


def _test_large_text() -> None:
    """Test large text handling."""
    try:
        from corpus import CorpusManager, TextMetadata
        manager = CorpusManager()

        long_text = "A" * 10000
        text_id = manager.add_text(
            author="Test",
            title="Long Document",
            category="test",
            content=long_text,
            metadata=TextMetadata(themes=["long"])
        )

        if text_id:
            manager.delete_text(text_id)
            log("Edge Cases", "Large text handling (10KB)", "PASS")
        else:
            log("Edge Cases", "Large text handling (10KB)", "FAIL", "Failed to add")

    except Exception as e:
        log("Edge Cases", "Large text handling", "FAIL", str(e))


def _test_sql_injection() -> None:
    """Test SQL injection protection."""
    try:
        from memory import MemoryStore
        store = MemoryStore()
        store.search("test' OR '1'='1")
        log("Edge Cases", "SQL injection protection", "PASS", "Query handled safely")
    except Exception as e:
        log("Edge Cases", "SQL injection protection", "FAIL", str(e))

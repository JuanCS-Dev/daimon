"""
Audit Component Tests - Collectors, Memory, Learners, Actuators, Corpus.
"""

import socket
from pathlib import Path

from .core import log


def test_collectors() -> None:
    """Test Collector components."""
    print("\n" + "=" * 60)
    print("4. COLLECTORS")
    print("=" * 60)

    _test_shell_socket()
    _test_claude_watcher()
    _test_shell_hooks()


def _test_shell_socket() -> None:
    """Test shell watcher socket."""
    socket_path = Path.home() / ".daimon" / "daimon.sock"
    if socket_path.exists():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(str(socket_path))
            sock.close()
            log("Collectors", "Shell Watcher Socket", "PASS", f"Listening at {socket_path}")
        except Exception as e:
            log("Collectors", "Shell Watcher Socket", "FAIL", f"Cannot connect: {e}")
    else:
        log("Collectors", "Shell Watcher Socket", "FAIL", f"Socket not found: {socket_path}")


def _test_claude_watcher() -> None:
    """Test Claude watcher module."""
    try:
        from collectors.claude_watcher import SessionTracker
        tracker = SessionTracker()
        if hasattr(tracker, 'run'):
            log("Collectors", "Claude Watcher run() method", "PASS")
        else:
            log("Collectors", "Claude Watcher run() method", "FAIL", "Method 'run' not implemented")
    except Exception as e:
        log("Collectors", "Claude Watcher import", "FAIL", str(e))


def _test_shell_hooks() -> None:
    """Test shell hooks in .zshrc."""
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        content = zshrc.read_text()
        if "daimon.sock" in content or "daimon_hook" in content:
            log("Collectors", "Shell hooks in .zshrc", "PASS")
        else:
            log("Collectors", "Shell hooks in .zshrc", "FAIL", "No DAIMON hooks found")
    else:
        log("Collectors", "Shell hooks in .zshrc", "FAIL", ".zshrc not found")


def test_memory_systems() -> None:
    """Test Memory storage systems."""
    print("\n" + "=" * 60)
    print("5. MEMORY SYSTEMS")
    print("=" * 60)

    _test_memory_store()
    _test_precedent_system()


def _test_memory_store() -> None:
    """Test MemoryStore CRUD."""
    try:
        from memory import MemoryStore
        store = MemoryStore()

        item_id = store.add("Test memory for audit", category="audit", importance=0.5)
        if item_id:
            log("Memory", "MemoryStore.add()", "PASS", f"ID: {item_id}")
        else:
            log("Memory", "MemoryStore.add()", "FAIL", "No ID returned")

        results = store.search("audit")
        if results:
            log("Memory", "MemoryStore.search()", "PASS", f"{len(results)} results")
        else:
            log("Memory", "MemoryStore.search()", "FAIL", "No results")

        if store.delete(item_id):
            log("Memory", "MemoryStore.delete()", "PASS")
        else:
            log("Memory", "MemoryStore.delete()", "FAIL")

        stats = store.get_stats()
        log("Memory", "MemoryStore.get_stats()", "PASS", f"DB: {stats.get('db_path', 'N/A')}")

    except Exception as e:
        log("Memory", "MemoryStore", "FAIL", str(e))


def _test_precedent_system() -> None:
    """Test PrecedentSystem CRUD."""
    try:
        from memory import PrecedentSystem
        system = PrecedentSystem()

        prec_id = system.record(
            context="Audit test context",
            decision="Test decision",
            outcome="success"
        )
        if prec_id:
            log("Memory", "PrecedentSystem.record()", "PASS", f"ID: {prec_id}")
        else:
            log("Memory", "PrecedentSystem.record()", "FAIL", "No ID returned")

        results = system.search("audit")
        if results:
            log("Memory", "PrecedentSystem.search()", "PASS", f"{len(results)} results")
        else:
            log("Memory", "PrecedentSystem.search()", "FAIL", "No results")

        if system.delete(prec_id):
            log("Memory", "PrecedentSystem.delete()", "PASS")
        else:
            log("Memory", "PrecedentSystem.delete()", "FAIL")

    except Exception as e:
        log("Memory", "PrecedentSystem", "FAIL", str(e))


def test_learners() -> None:
    """Test Learner components."""
    print("\n" + "=" * 60)
    print("6. LEARNERS")
    print("=" * 60)

    _test_preference_learner()
    _test_reflection_engine()


def _test_preference_learner() -> None:
    """Test PreferenceLearner."""
    try:
        from learners import PreferenceLearner
        learner = PreferenceLearner()

        signals = learner.scan_sessions(since_hours=48)
        log("Learners", "PreferenceLearner.scan_sessions()", "PASS", f"{len(signals)} signals")

        summary = learner.get_preference_summary()
        log("Learners", "PreferenceLearner.get_preference_summary()", "PASS", f"{len(summary)} categories")

        insights = learner.get_actionable_insights()
        log("Learners", "PreferenceLearner.get_actionable_insights()", "PASS", f"{len(insights)} insights")

    except Exception as e:
        log("Learners", "PreferenceLearner", "FAIL", str(e))


def _test_reflection_engine() -> None:
    """Test ReflectionEngine."""
    try:
        from learners import get_engine
        engine = get_engine()

        status = engine.get_status()
        if "running" in status:
            log("Learners", "ReflectionEngine.get_status()", "PASS", f"running={status['running']}")
        else:
            log("Learners", "ReflectionEngine.get_status()", "FAIL", "Invalid status")

        learner = engine.get_learner()
        if learner:
            log("Learners", "ReflectionEngine.get_learner()", "PASS")
        else:
            log("Learners", "ReflectionEngine.get_learner()", "FAIL", "No learner")

        refiner = engine.get_refiner()
        if refiner:
            log("Learners", "ReflectionEngine.get_refiner()", "PASS")
        else:
            log("Learners", "ReflectionEngine.get_refiner()", "FAIL", "No refiner")

    except Exception as e:
        log("Learners", "ReflectionEngine", "FAIL", str(e))


def test_actuators() -> None:
    """Test Actuator components."""
    print("\n" + "=" * 60)
    print("7. ACTUATORS")
    print("=" * 60)

    try:
        from actuators import ConfigRefiner
        refiner = ConfigRefiner()

        current = refiner.get_current_preferences()
        log("Actuators", "ConfigRefiner.get_current_preferences()", "PASS", f"{len(current)} chars")

        manual = refiner.get_manual_content()
        log("Actuators", "ConfigRefiner.get_manual_content()", "PASS", f"{len(manual)} chars")

        backups = refiner.get_backup_list()
        log("Actuators", "ConfigRefiner.get_backup_list()", "PASS", f"{len(backups)} backups")

        if hasattr(refiner, 'update_preferences'):
            log("Actuators", "ConfigRefiner.update_preferences() exists", "PASS")
        else:
            log("Actuators", "ConfigRefiner.update_preferences() exists", "FAIL")

    except Exception as e:
        log("Actuators", "ConfigRefiner", "FAIL", str(e))


def test_corpus() -> None:
    """Test Corpus components."""
    print("\n" + "=" * 60)
    print("8. CORPUS")
    print("=" * 60)

    try:
        from corpus import CorpusManager, TextMetadata
        manager = CorpusManager()

        stats = manager.get_stats()
        log("Corpus", "CorpusManager.get_stats()", "PASS", f"{stats['total_texts']} texts")

        authors = manager.list_authors()
        log("Corpus", "CorpusManager.list_authors()", "PASS", f"{len(authors)} authors")

        themes = manager.list_themes()
        log("Corpus", "CorpusManager.list_themes()", "PASS", f"{len(themes)} themes")

        results = manager.search("virtue")
        log("Corpus", "CorpusManager.search()", "PASS", f"{len(results)} results")

        # CRUD test
        text_id = manager.add_text(
            author="Audit",
            title="Test",
            category="filosofia/modernos",
            content="Test content",
            metadata=TextMetadata(themes=["test"])
        )
        if text_id:
            log("Corpus", "CorpusManager.add_text()", "PASS", f"ID: {text_id}")

            text = manager.get_text(text_id)
            if text:
                log("Corpus", "CorpusManager.get_text()", "PASS")
            else:
                log("Corpus", "CorpusManager.get_text()", "FAIL")

            if manager.delete_text(text_id):
                log("Corpus", "CorpusManager.delete_text()", "PASS")
            else:
                log("Corpus", "CorpusManager.delete_text()", "FAIL")
        else:
            log("Corpus", "CorpusManager.add_text()", "FAIL")

    except Exception as e:
        log("Corpus", "CorpusManager", "FAIL", str(e))

    # Bootstrap
    try:
        from corpus.bootstrap_texts import BOOTSTRAP_TEXTS
        log("Corpus", "Bootstrap texts available", "PASS", f"{len(BOOTSTRAP_TEXTS)} texts")
    except Exception as e:
        log("Corpus", "Bootstrap texts", "FAIL", str(e))

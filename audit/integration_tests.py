"""
Audit Integration Tests - Hooks, MCP, Files, Directories.
"""

import json
from pathlib import Path

from .core import log


def test_hooks() -> None:
    """Test Hook integration."""
    print("\n" + "=" * 60)
    print("9. HOOKS")
    print("=" * 60)

    base = Path(__file__).parent.parent

    # Hook file exists
    hook_path = base / ".claude" / "hooks" / "noesis_hook.py"
    if hook_path.exists():
        log("Hooks", "noesis_hook.py exists", "PASS", str(hook_path))
    else:
        log("Hooks", "noesis_hook.py exists", "FAIL", f"Not found: {hook_path}")

    # Settings file
    settings_path = base / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            hooks = settings.get("hooks", {})
            if "UserPromptSubmit" in hooks:
                log("Hooks", "UserPromptSubmit hook configured", "PASS")
            else:
                log("Hooks", "UserPromptSubmit hook configured", "FAIL", "Not in settings")

            if "PreToolUse" in hooks:
                log("Hooks", "PreToolUse hook configured", "PASS")
            else:
                log("Hooks", "PreToolUse hook configured", "FAIL", "Not in settings")
        except Exception as e:
            log("Hooks", "settings.json parse", "FAIL", str(e))
    else:
        log("Hooks", "settings.json exists", "FAIL", f"Not found: {settings_path}")

    # Subagent
    agent_path = base / ".claude" / "agents" / "noesis-sage.md"
    if agent_path.exists():
        log("Hooks", "noesis-sage.md exists", "PASS")
    else:
        log("Hooks", "noesis-sage.md exists", "FAIL", f"Not found: {agent_path}")

    # User hooks installed
    user_hook = Path.home() / ".claude" / "hooks" / "noesis_hook.py"
    if user_hook.exists():
        log("Hooks", "Hook installed in ~/.claude", "PASS")
    else:
        log("Hooks", "Hook installed in ~/.claude", "FAIL", "Not copied")

    user_agent = Path.home() / ".claude" / "agents" / "noesis-sage.md"
    if user_agent.exists():
        log("Hooks", "Agent installed in ~/.claude", "PASS")
    else:
        log("Hooks", "Agent installed in ~/.claude", "FAIL", "Not copied")


def test_mcp_server() -> None:
    """Test MCP Server tools."""
    print("\n" + "=" * 60)
    print("10. MCP SERVER")
    print("=" * 60)

    try:
        mcp_path = Path(__file__).parent.parent / "integrations" / "mcp_server.py"
        if mcp_path.exists():
            content = mcp_path.read_text()

            tools = ["noesis_consult", "noesis_tribunal", "noesis_precedent", "noesis_confront", "noesis_health"]
            for tool in tools:
                if f"def {tool}" in content or f"async def {tool}" in content:
                    log("MCP Server", f"Tool: {tool}", "PASS")
                else:
                    log("MCP Server", f"Tool: {tool}", "FAIL", "Not found")
        else:
            log("MCP Server", "mcp_server.py exists", "FAIL", f"Not found: {mcp_path}")

    except Exception as e:
        log("MCP Server", "MCP Server check", "FAIL", str(e))


def test_files_structure() -> None:
    """Test file structure completeness."""
    print("\n" + "=" * 60)
    print("11. FILE STRUCTURE")
    print("=" * 60)

    base = Path(__file__).parent.parent

    required_files = [
        "daimon_daemon.py",
        "install.sh",
        "integrations/mcp_server.py",
        "collectors/shell_watcher.py",
        "collectors/claude_watcher.py",
        "endpoints/daimon_routes.py",
        "endpoints/quick_check.py",
        "endpoints/constants.py",
        "memory/optimized_store.py",
        "memory/precedent_system.py",
        "memory/precedent_models.py",
        "learners/preference_learner.py",
        "learners/reflection_engine.py",
        "actuators/config_refiner.py",
        "corpus/manager.py",
        "corpus/bootstrap_texts.py",
        "dashboard/app.py",
        "dashboard/templates/index.html",
        ".claude/hooks/noesis_hook.py",
        ".claude/agents/noesis-sage.md",
        ".claude/settings.json",
    ]

    for f in required_files:
        path = base / f
        if path.exists():
            log("Files", f, "PASS")
        else:
            log("Files", f, "FAIL", "Missing")


def test_data_directories() -> None:
    """Test data directories."""
    print("\n" + "=" * 60)
    print("12. DATA DIRECTORIES")
    print("=" * 60)

    dirs = [
        Path.home() / ".daimon",
        Path.home() / ".daimon" / "logs",
        Path.home() / ".daimon" / "memory",
        Path.home() / ".daimon" / "corpus",
        Path.home() / ".claude",
        Path.home() / ".claude" / "backups",
    ]

    for d in dirs:
        if d.exists():
            log("Directories", str(d), "PASS")
        else:
            log("Directories", str(d), "FAIL", "Missing")


def test_integration() -> None:
    """Test component integration and workflows."""
    print("\n" + "=" * 60)
    print("13. INTEGRATION TESTS")
    print("=" * 60)

    _test_engine_learner_chain()
    _test_engine_refiner_chain()
    _test_memory_corpus_search()
    _test_daemon_components()


def _test_engine_learner_chain() -> None:
    """Test Engine->Learner integration."""
    try:
        from learners import get_engine
        engine = get_engine()
        learner = engine.get_learner()

        if learner is None:
            log("Integration", "Engine->Learner chain", "FAIL", "Learner is None")
        else:
            signals = learner.scan_sessions(since_hours=1)
            log("Integration", "Engine->Learner->scan()", "PASS", f"{len(signals)} signals")

    except Exception as e:
        log("Integration", "Engine->Learner chain", "FAIL", str(e))


def _test_engine_refiner_chain() -> None:
    """Test Engine->Refiner integration."""
    try:
        from learners import get_engine
        engine = get_engine()
        refiner = engine.get_refiner()

        if refiner is not None:
            backups = refiner.get_backup_list()
            log("Integration", "Engine->Refiner chain", "PASS", f"{len(backups)} backups")
        else:
            log("Integration", "Engine->Refiner chain", "FAIL", "Refiner is None")

    except Exception as e:
        log("Integration", "Engine->Refiner chain", "FAIL", str(e))


def _test_memory_corpus_search() -> None:
    """Test Memory+Corpus search consistency."""
    try:
        from memory import MemoryStore
        from corpus import CorpusManager

        store = MemoryStore()
        corpus = CorpusManager()

        mem_results = store.search("test")
        corp_results = corpus.search("virtue")

        log("Integration", "Memory+Corpus search APIs", "PASS", f"M:{len(mem_results)}, C:{len(corp_results)}")

    except Exception as e:
        log("Integration", "Memory+Corpus search", "FAIL", str(e))


def _test_daemon_components() -> None:
    """Test daemon component instantiation."""
    try:
        from collectors.shell_watcher import HeartbeatAggregator, ShellHeartbeat
        from collectors.claude_watcher import SessionTracker

        aggregator = HeartbeatAggregator()
        claude = SessionTracker()

        hb = ShellHeartbeat.from_json({
            "timestamp": "2025-12-12T12:00:00",
            "command": "test",
            "pwd": "/tmp",
            "exit_code": 0
        })
        aggregator.add(hb)

        has_claude_run = hasattr(claude, 'run')
        has_aggregator_add = hasattr(aggregator, 'add')

        if has_claude_run and has_aggregator_add:
            log("Integration", "Daemon components", "PASS")
        else:
            log("Integration", "Daemon components", "FAIL", f"run={has_claude_run}, add={has_aggregator_add}")

    except Exception as e:
        log("Integration", "Daemon components", "FAIL", str(e))

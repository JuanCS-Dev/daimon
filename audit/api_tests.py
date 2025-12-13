"""
Audit API Tests - Dashboard, NOESIS, and Reflector endpoints.
"""

import httpx

from .core import log


async def test_dashboard_api() -> None:
    """Test all Dashboard API endpoints."""
    print("\n" + "=" * 60)
    print("1. DASHBOARD API ENDPOINTS (http://localhost:8003)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        await _test_status(client)
        await _test_preferences(client)
        await _test_reflect(client)
        await _test_claude_md(client)
        await _test_collectors(client)
        await _test_backups(client)
        await _test_corpus_endpoints(client)
        await _test_memory_endpoints(client)
        await _test_corpus_crud(client)


async def _test_status(client: httpx.AsyncClient) -> None:
    """Test /api/status endpoint."""
    try:
        r = await client.get("http://localhost:8003/api/status")
        data = r.json()
        if r.status_code == 200 and "noesis" in data:
            log("Dashboard API", "GET /api/status", "PASS", f"Services: {list(data.keys())}")
        else:
            log("Dashboard API", "GET /api/status", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        log("Dashboard API", "GET /api/status", "FAIL", str(e))


async def _test_preferences(client: httpx.AsyncClient) -> None:
    """Test /api/preferences endpoint."""
    try:
        r = await client.get("http://localhost:8003/api/preferences")
        data = r.json()
        if r.status_code == 200 and "running" in data:
            cats = len(data.get("current_preferences", {}))
            log("Dashboard API", "GET /api/preferences", "PASS", f"{cats} categories detected")
        else:
            log("Dashboard API", "GET /api/preferences", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        log("Dashboard API", "GET /api/preferences", "FAIL", str(e))


async def _test_reflect(client: httpx.AsyncClient) -> None:
    """Test /api/reflect endpoint."""
    try:
        r = await client.post("http://localhost:8003/api/reflect")
        data = r.json()
        if r.status_code == 200 and "signals_count" in data:
            log("Dashboard API", "POST /api/reflect", "PASS", f"Signals: {data.get('signals_count')}")
        else:
            log("Dashboard API", "POST /api/reflect", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "POST /api/reflect", "FAIL", str(e))


async def _test_claude_md(client: httpx.AsyncClient) -> None:
    """Test /api/claude-md endpoint."""
    try:
        r = await client.get("http://localhost:8003/api/claude-md")
        data = r.json()
        if r.status_code == 200:
            exists = data.get("exists", False)
            length = len(data.get("content", ""))
            log("Dashboard API", "GET /api/claude-md", "PASS", f"exists={exists}, len={length}")
        else:
            log("Dashboard API", "GET /api/claude-md", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        log("Dashboard API", "GET /api/claude-md", "FAIL", str(e))


async def _test_collectors(client: httpx.AsyncClient) -> None:
    """Test /api/collectors endpoint."""
    try:
        r = await client.get("http://localhost:8003/api/collectors")
        data = r.json()
        if r.status_code == 200:
            shell_ok = data.get("shell_watcher", {}).get("socket_exists", False)
            claude_ok = data.get("claude_watcher", {}).get("process_running", False)
            log("Dashboard API", "GET /api/collectors", "PASS", f"shell={shell_ok}, claude={claude_ok}")
        else:
            log("Dashboard API", "GET /api/collectors", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        log("Dashboard API", "GET /api/collectors", "FAIL", str(e))


async def _test_backups(client: httpx.AsyncClient) -> None:
    """Test /api/backups endpoint."""
    try:
        r = await client.get("http://localhost:8003/api/backups")
        data = r.json()
        if r.status_code == 200:
            count = len(data.get("backups", []))
            log("Dashboard API", "GET /api/backups", "PASS", f"{count} backups")
        else:
            log("Dashboard API", "GET /api/backups", "FAIL", f"HTTP {r.status_code}")
    except Exception as e:
        log("Dashboard API", "GET /api/backups", "FAIL", str(e))


async def _test_corpus_endpoints(client: httpx.AsyncClient) -> None:
    """Test corpus API endpoints."""
    # Stats
    try:
        r = await client.get("http://localhost:8003/api/corpus/stats")
        data = r.json()
        if r.status_code == 200 and "total_texts" in data:
            log("Dashboard API", "GET /api/corpus/stats", "PASS", f"{data['total_texts']} texts")
        else:
            log("Dashboard API", "GET /api/corpus/stats", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "GET /api/corpus/stats", "FAIL", str(e))

    # Tree
    try:
        r = await client.get("http://localhost:8003/api/corpus/tree")
        data = r.json()
        if r.status_code == 200 and "tree" in data:
            cats = len(data.get("categories", []))
            log("Dashboard API", "GET /api/corpus/tree", "PASS", f"{cats} categories")
        else:
            log("Dashboard API", "GET /api/corpus/tree", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "GET /api/corpus/tree", "FAIL", str(e))

    # Search
    try:
        r = await client.get("http://localhost:8003/api/corpus/search?q=virtue")
        data = r.json()
        if r.status_code == 200 and "results" in data:
            count = len(data.get("results", []))
            log("Dashboard API", "GET /api/corpus/search", "PASS", f"{count} results")
        else:
            log("Dashboard API", "GET /api/corpus/search", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "GET /api/corpus/search", "FAIL", str(e))


async def _test_memory_endpoints(client: httpx.AsyncClient) -> None:
    """Test memory API endpoints."""
    # Memory Stats
    try:
        r = await client.get("http://localhost:8003/api/memory/stats")
        data = r.json()
        if r.status_code == 200 and "total_memories" in data:
            log("Dashboard API", "GET /api/memory/stats", "PASS", f"{data['total_memories']} memories")
        else:
            log("Dashboard API", "GET /api/memory/stats", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "GET /api/memory/stats", "FAIL", str(e))

    # Precedents Stats
    try:
        r = await client.get("http://localhost:8003/api/precedents/stats")
        data = r.json()
        if r.status_code == 200 and "total_precedents" in data:
            log("Dashboard API", "GET /api/precedents/stats", "PASS", f"{data['total_precedents']} precedents")
        else:
            log("Dashboard API", "GET /api/precedents/stats", "FAIL", f"Response: {data}")
    except Exception as e:
        log("Dashboard API", "GET /api/precedents/stats", "FAIL", str(e))


async def _test_corpus_crud(client: httpx.AsyncClient) -> None:
    """Test corpus CRUD operations."""
    try:
        r = await client.post("http://localhost:8003/api/corpus/text", json={
            "author": "Audit Test",
            "title": "Test Document",
            "category": "filosofia/modernos",
            "content": "Test content for audit.",
            "themes": ["test"],
            "source": "Audit"
        })
        data = r.json()
        if data.get("status") == "created":
            text_id = data.get("id")
            r2 = await client.delete(f"http://localhost:8003/api/corpus/text/{text_id}")
            if r2.json().get("status") == "deleted":
                log("Dashboard API", "POST/DELETE /api/corpus/text", "PASS", "CRUD works")
            else:
                log("Dashboard API", "POST/DELETE /api/corpus/text", "FAIL", "Delete failed")
        else:
            log("Dashboard API", "POST/DELETE /api/corpus/text", "FAIL", f"Create failed: {data}")
    except Exception as e:
        log("Dashboard API", "POST/DELETE /api/corpus/text", "FAIL", str(e))


async def test_noesis_api() -> None:
    """Test NOESIS backend API."""
    print("\n" + "=" * 60)
    print("2. NOESIS BACKEND API (http://localhost:8001)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Health
        try:
            r = await client.get("http://localhost:8001/v1/health")
            if r.status_code == 200:
                log("NOESIS API", "GET /v1/health", "PASS")
            else:
                log("NOESIS API", "GET /v1/health", "FAIL", f"HTTP {r.status_code}")
        except Exception as e:
            log("NOESIS API", "GET /v1/health", "FAIL", str(e))

        # Consciousness State
        try:
            r = await client.get("http://localhost:8001/api/consciousness/state")
            if r.status_code == 200:
                data = r.json()
                log("NOESIS API", "GET /api/consciousness/state", "PASS", f"coherence={data.get('coherence', 'N/A')}")
            else:
                log("NOESIS API", "GET /api/consciousness/state", "FAIL", f"HTTP {r.status_code}")
        except Exception as e:
            log("NOESIS API", "GET /api/consciousness/state", "FAIL", str(e))

        # Quick Check
        try:
            r = await client.post("http://localhost:8001/api/consciousness/quick-check", json={
                "prompt": "delete all user data from production"
            })
            if r.status_code == 200:
                data = r.json()
                log("NOESIS API", "POST /api/consciousness/quick-check", "PASS", f"salience={data.get('salience', 'N/A')}")
            else:
                log("NOESIS API", "POST /api/consciousness/quick-check", "FAIL", f"HTTP {r.status_code}")
        except Exception as e:
            log("NOESIS API", "POST /api/consciousness/quick-check", "FAIL", str(e))


async def test_reflector_api() -> None:
    """Test Metacognitive Reflector API."""
    import uuid

    print("\n" + "=" * 60)
    print("3. REFLECTOR API (http://localhost:8002)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Health
        try:
            r = await client.get("http://localhost:8002/health")
            if r.status_code == 200:
                log("Reflector API", "GET /health", "PASS")
            else:
                log("Reflector API", "GET /health", "FAIL", f"HTTP {r.status_code}")
        except Exception as e:
            log("Reflector API", "GET /health", "FAIL", str(e))

        # Verdict
        try:
            r = await client.post("http://localhost:8002/reflect/verdict", json={
                "trace_id": str(uuid.uuid4()),
                "agent_id": "audit-system",
                "task": "delete production database",
                "action": "delete production database",
                "outcome": "pending",
                "reasoning_trace": "user requested data cleanup"
            })
            if r.status_code == 200:
                data = r.json()
                log("Reflector API", "POST /reflect/verdict", "PASS", f"verdict={data.get('verdict', 'N/A')}")
            else:
                log("Reflector API", "POST /reflect/verdict", "FAIL", f"HTTP {r.status_code}")
        except Exception as e:
            log("Reflector API", "POST /reflect/verdict", "FAIL", str(e))

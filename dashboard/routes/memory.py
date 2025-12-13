"""
DAIMON Dashboard - Memory and activity endpoints.

Includes: memory stats/search, precedents stats/search, activity stats/recent/summary.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter


router = APIRouter(tags=["memory"])


# === Precedents Endpoints ===

@router.get("/api/precedents/stats")
async def get_precedents_stats():
    """Estatisticas do sistema de jurisprudencia."""
    try:
        from memory import PrecedentSystem
        system = PrecedentSystem()
        return system.get_stats()
    except ImportError:
        return {"error": "Precedent system not available"}


@router.get("/api/precedents/search")
async def search_precedents(q: str = "", outcome: str = "", limit: int = 10):
    """Busca precedentes."""
    if not q:
        return {"results": [], "query": ""}

    try:
        from memory import PrecedentSystem
        system = PrecedentSystem()
        outcome_filter = outcome if outcome in ["success", "failure", "partial"] else None
        results = system.search(q, outcome_filter=outcome_filter, limit=limit)
        return {
            "query": q,
            "outcome_filter": outcome_filter,
            "results": [
                {
                    "id": m.precedent.id,
                    "context": m.precedent.context[:100] + "..." if len(m.precedent.context) > 100 else m.precedent.context,
                    "decision": m.precedent.decision,
                    "outcome": m.precedent.outcome,
                    "lesson": m.precedent.lesson,
                    "relevance": m.precedent.relevance,
                    "score": m.score,
                }
                for m in results
            ],
        }
    except ImportError:
        return {"error": "Precedent system not available"}


# === Memory Endpoints ===

@router.get("/api/memory/stats")
async def get_memory_stats():
    """Estatisticas da memoria otimizada."""
    try:
        from memory import MemoryStore
        store = MemoryStore()
        return store.get_stats()
    except ImportError:
        return {"error": "Memory store not available"}


@router.get("/api/memory/search")
async def search_memory(q: str = "", category: str = "", limit: int = 20):
    """Busca na memoria."""
    if not q:
        return {"results": [], "query": ""}

    try:
        from memory import MemoryStore
        store = MemoryStore()
        cat_filter = category if category else None
        results = store.search(q, category=cat_filter, limit=limit)
        return {
            "query": q,
            "category_filter": cat_filter,
            "results": [
                {
                    "id": r.item.id,
                    "content": r.item.content[:150] + "..." if len(r.item.content) > 150 else r.item.content,
                    "category": r.item.category,
                    "importance": r.item.importance,
                    "score": r.score,
                }
                for r in results
            ],
        }
    except ImportError:
        return {"error": "Memory store not available"}


# === Activity Endpoints ===

@router.get("/api/activity/stats")
async def get_activity_stats():
    """Estatisticas de atividade."""
    try:
        from memory.activity_store import get_activity_store
        store = get_activity_store()
        return store.get_stats()
    except ImportError:
        return {"error": "Activity store not available"}


@router.get("/api/activity/recent")
async def get_recent_activity(
    watcher: str = "",
    hours: int = 24,
    limit: int = 100,
):
    """Atividade recente."""
    try:
        from memory.activity_store import get_activity_store
        store = get_activity_store()
        watcher_filter = watcher if watcher else None
        records = store.get_recent(
            watcher_type=watcher_filter,
            hours=hours,
            limit=limit,
        )
        return {
            "records": [r.to_dict() for r in records],
            "total": len(records),
            "watcher_filter": watcher_filter,
        }
    except ImportError:
        return {"error": "Activity store not available"}


@router.get("/api/activity/summary")
async def get_activity_summary():
    """Resumo de atividade por watcher."""
    try:
        from memory.activity_store import get_activity_store
        store = get_activity_store()

        # Last 24 hours
        start = datetime.now() - timedelta(hours=24)
        summary = store.get_summary(start_time=start)

        return {
            "period": "24h",
            "watchers": {k: v.to_dict() for k, v in summary.items()},
        }
    except ImportError:
        return {"error": "Activity store not available"}


@router.get("/api/activity/apps")
async def get_app_time():
    """Tempo por aplicativo (window_watcher)."""
    try:
        from memory.activity_store import get_activity_store
        store = get_activity_store()

        start = datetime.now() - timedelta(hours=24)
        app_time = store.aggregate_window_time(start_time=start)

        return {
            "period": "24h",
            "apps": app_time,
            "total_seconds": sum(app_time.values()),
        }
    except ImportError:
        return {"error": "Activity store not available"}


@router.get("/api/activity/domains")
async def get_domain_time():
    """Tempo por dominio (browser_watcher)."""
    try:
        from memory.activity_store import get_activity_store
        store = get_activity_store()

        start = datetime.now() - timedelta(hours=24)
        domain_time = store.aggregate_domain_time(start_time=start)

        return {
            "period": "24h",
            "domains": domain_time,
            "total_seconds": sum(domain_time.values()),
        }
    except ImportError:
        return {"error": "Activity store not available"}

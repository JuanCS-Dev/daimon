from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from maximus_core_service.consciousness.persistence.chat_store import ChatStore

router = APIRouter(prefix="/chat", tags=["chat_management"])
store = ChatStore()

@router.get("/sessions")
async def list_sessions(limit: int = 20, offset: int = 0):
    """List chat sessions."""
    return store.list_sessions(limit=limit, offset=offset)

@router.post("/sessions")
async def create_session(title: Optional[str] = Body(None, embed=True)):
    """Create a new chat session."""
    session_id = store.create_session(title)
    return {"id": session_id}

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session history."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    store.delete_session(session_id)
    return {"ack": True}

@router.get("/search")
async def search_chat(q: str = Query(..., min_length=3)):
    """Full text search in chat history."""
    return store.search_messages(q)

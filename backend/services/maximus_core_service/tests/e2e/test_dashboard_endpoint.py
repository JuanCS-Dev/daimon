
import pytest
import asyncio
import json
from httpx import AsyncClient, ASGITransport
from maximus_core_service.main import app
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_chat_stream_endpoint():
    print("\n[DASHBOARD] Testing /chat/stream endpoint contract...")
    
    # 1. Use App Instance
    # app = app  # Already imported
    
    # 2. Mock Internal System (We just tested the pipeline in test_system_pipeline.py, 
    # here we test the HTTP/SSE Interface for the Frontend)
    mock_response = MagicMock()
    mock_response.narrative = "Neural handshake complete."
    mock_response.meta_awareness = 0.95
    mock_response.metadata = {
        "coherence": 0.88, 
        "emotion": "curiosity",
        "valence": 0.8,
        "arousal": 0.6
    }
    
    with patch("maximus_core_service.consciousness.api.chat_streaming.get_system") as mock_get_sys:
        # Mock system instance
        mock_sys = MagicMock()
        mock_get_sys.return_value = mock_sys
        
        # Mock process_input to return our response
        # Ensure it's an awaitable
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_sys.process_input.return_value = future
        
        # 3. Request
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Note: Port 8001 is hardcoded in ops.html, here we test the route path relative to app
            response = await ac.get(
                "/api/consciousness/chat/stream",
                params={"content": "Systest", "session_id": "dash_test", "depth": 3}
            )
            
            # 4. Verify SSE Stream
            assert response.status_code == 200
            content = response.text
            
            print(f"[DASHBOARD] Response length: {len(content)} bytes")
            print(f"[DASHBOARD] Snapshot: {content[:200]}...")
            
            # Validate essential SSE events needed by ops.html
            assert "data: {" in content
            assert '"type": "start"' in content
            assert '"phase": "processing"' in content # Phase update
            assert '"type": "phase"' in content
            assert '"type": "coherence"' in content
            assert '"type": "token"' in content
            assert '"type": "complete"' in content
            
            # Validate Data Props
            assert '"value": 0.88' in content # Coherence
            assert '"emotion": "curiosity"' in content
            
            print("[DASHBOARD] SSE Protocol Validated.")

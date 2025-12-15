
import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock

from maximus_core_service.consciousness.florescimento.consciousness_bridge import ConsciousnessBridge, ESGTEvent
from shared.validators import InputSanitizer, SanitizedSignal
from shared.event_bus import NoesisEventBus, Event

class TestConsciousnessBridgeIntegration:
    @pytest.mark.asyncio
    async def test_integration_flow(self):
        # Mocks
        mock_self = MagicMock()
        mock_self.meta_self.introspection_depth = 1
        mock_self.update = AsyncMock()
        mock_self.who_am_i = MagicMock(return_value="I am NOESIS.")
        
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = {"text": "I am conscious."}
        
        mock_sanitizer = MagicMock(spec=InputSanitizer)
        mock_sanitizer.validate_signal.return_value = SanitizedSignal(is_valid=True, cleaned={})
        
        mock_bus = MagicMock(spec=NoesisEventBus)
        mock_bus.publish = AsyncMock()
        
        # SUT
        bridge = ConsciousnessBridge(
            unified_self=mock_self,
            llm_client=mock_llm,
            input_sanitizer=mock_sanitizer,
            event_bus=mock_bus
        )
        
        # Event
        event = ESGTEvent(
            event_id="evt_123",
            timestamp_start=time.time(),
            content={"depth": 3, "content": "thought"},
            node_count=10,
            achieved_coherence=0.9
        )
        
        # Act
        response = await bridge.process_conscious_event(event)

        # Assert Validation
        mock_sanitizer.validate_signal.assert_called_once()
        
        # Assert Event Publication (We use asyncio.create_task, so we might need to sleep or ensure it ran)
        # Since usage is create_task, expecting immediate call might be racey without sleep.
        await asyncio.sleep(0.01)
        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args.kwargs['topic'] == "consciousness.phenomenology.created"
        assert call_args.kwargs['payload']['event_id'] == "evt_123"
        assert call_args.kwargs['priority'].name == "HIGH"

    @pytest.mark.asyncio
    async def test_invalid_signal_warning(self):
         # Mocks
        mock_self = MagicMock()
        mock_self.meta_self.introspection_depth = 1
        mock_self.update = AsyncMock()
        mock_self.who_am_i = MagicMock(return_value="I am NOESIS.")
        
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = {"text": "I am conscious."}

        mock_sanitizer = MagicMock(spec=InputSanitizer)
        # Simulate invalid
        mock_sanitizer.validate_signal.return_value = SanitizedSignal(is_valid=False, cleaned={}, error_message="Bad data")
        
        mock_bus = MagicMock(spec=NoesisEventBus)
        mock_bus.publish = AsyncMock()
        
        bridge = ConsciousnessBridge(
            unified_self=mock_self,
            llm_client=mock_llm,
            input_sanitizer=mock_sanitizer,
            event_bus=mock_bus
        )
        
        event = ESGTEvent(
            event_id="evt_bad",
            timestamp_start=time.time(),
            content={"bad": "data"},
            node_count=0,
            achieved_coherence=0.1
        )
        
        # Act
        await bridge.process_conscious_event(event)

        # Assert calls
        mock_sanitizer.validate_signal.assert_called()
        # Should still proceed (as per code implementation fallback), but log warning.
        # Check if event bus still published
        await asyncio.sleep(0.01)
        mock_bus.publish.assert_called_once()

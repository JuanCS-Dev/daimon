"""
Integration Test for Phase 4: Meta-Optimizer & Kuramoto Enhancement
====================================================================

Verifies:
1. Meta-Optimizer initialization in ConsciousnessSystem.
2. Coherence tracking loop.
3. Automatic parameter tuning (AKOrN pattern).
4. Adaptive synchronization in ESGTCoordinator.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from maximus_core_service.consciousness.system import ConsciousnessSystem, ConsciousnessConfig
from maximus_core_service.consciousness.esgt.coordinator import ESGTEvent, ESGTPhase
from maximus_core_service.consciousness.meta_optimizer.coherence_tracker import CoherenceSnapshot

# Common patches for all tests
@pytest.fixture
def mock_dependencies():
    with patch('maximus_core_service.consciousness.system.TIGFabric') as MockTIG, \
         patch('maximus_core_service.consciousness.system.ToMEngine') as MockToM, \
         patch('maximus_core_service.consciousness.system.PrefrontalCortex') as MockPFC, \
         patch('maximus_core_service.consciousness.system.MetacognitiveMonitor') as MockMeta, \
         patch('maximus_core_service.consciousness.system.ESGTCoordinator') as MockESGT:
        
        # Setup TIG
        msg_tig = MockTIG.return_value
        msg_tig.initialize_async = AsyncMock()
        msg_tig.stop = AsyncMock()
        msg_tig.exit_esgt_mode = AsyncMock()
        
        # Setup ToM
        msg_tom = MockToM.return_value
        msg_tom.initialize = AsyncMock()
        msg_tom.close = AsyncMock()
        
        # Setup ESGT
        msg_esgt = MockESGT.return_value
        msg_esgt.start = AsyncMock()
        msg_esgt.stop = AsyncMock()
        msg_esgt.initiate_esgt = AsyncMock()
        msg_esgt.use_adaptive_sync = True # Default attr
        
        yield {
            'tig': MockTIG,
            'tom': MockToM,
            'pfc': MockPFC,
            'meta': MockMeta,
            'esgt': MockESGT
        }

@pytest.mark.asyncio
async def test_meta_optimizer_initialization(mock_dependencies):
    """Verify Meta-Optimizer components are initialized."""
    config = ConsciousnessConfig(tig_node_count=1)
    system = ConsciousnessSystem(config)
    
    await system.start()
    
    assert system.coherence_tracker is not None
    assert system.config_tuner is not None
    # We can check if ESGT was init with flag
    mock_dependencies['esgt'].assert_called()
    call_kwargs = mock_dependencies['esgt'].call_args[1]
    assert call_kwargs.get('use_adaptive_sync') is True
    
    await system.stop()

@pytest.mark.asyncio
async def test_coherence_recording_loop(mock_dependencies):
    """Verify process_input records coherence metrics."""
    system = ConsciousnessSystem()
    
    # Setup Mock ESGT Event return
    mock_esgt_instance = mock_dependencies['esgt'].return_value
    mock_event = MagicMock(spec=ESGTEvent)
    mock_event.event_id = "test-event"
    mock_event.achieved_coherence = 0.85
    mock_event.total_duration_ms = 100.0
    mock_event.was_successful.return_value = True
    mock_event.current_phase = ESGTPhase.COMPLETE
    mock_esgt_instance.initiate_esgt.return_value = mock_event
    
    await system.start()
    
    # Mock Bridge manually since it's created in start too (or we can patch it)
    # Actually ConsciousnessBridge is created in start.
    with patch('maximus_core_service.consciousness.system.ConsciousnessBridge') as MockBridge:
        mock_bridge_instance = MockBridge.return_value
        # If created in start, we missed the patch context?
        # Check system.py: "self.consciousness_bridge = ConsciousnessBridge(...)"
        # Since we already started, we must patch the instance on system
        system.consciousness_bridge = AsyncMock()
        mock_response = MagicMock()
        mock_response.narrative = "Test response"
        mock_response.meta_awareness_level = 0.5
        system.consciousness_bridge.process_conscious_event.return_value = mock_response

        # Spy on tracker
        system.coherence_tracker.record = MagicMock(wraps=system.coherence_tracker.record)
        
        await system.process_input("Test input", depth=1)
        
        # Verify recording
        system.coherence_tracker.record.assert_called_once()
        args = system.coherence_tracker.record.call_args[1]
        assert args['coherence'] == 0.85
        assert args['was_successful'] is True
    
    await system.stop()

@pytest.mark.asyncio
async def test_auto_tuning_trigger(mock_dependencies):
    """Verify low coherence triggers parameter tuning."""
    system = ConsciousnessSystem()
    
    # Setup Mock ESGT to have config
    mock_esgt_instance = mock_dependencies['esgt'].return_value
    mock_esgt_instance.kuramoto_config = MagicMock()
    mock_esgt_instance.kuramoto_config.coupling_strength = 0.5
    
    # Mock low coherence event
    mock_event = MagicMock(spec=ESGTEvent)
    mock_event.event_id = "test-low-coherence"
    mock_event.achieved_coherence = 0.40
    mock_event.total_duration_ms = 100.0
    mock_event.was_successful.return_value = True
    mock_esgt_instance.initiate_esgt.return_value = mock_event
    
    await system.start()
    
    # Mock Bridge
    system.consciousness_bridge = AsyncMock()
    
    # Mock tracker to force trigger
    system.coherence_tracker.should_trigger_optimization = MagicMock(return_value=True)
    
    await system.process_input("Test input", depth=1)
    
    # Verify tuning logic
    # The system updates accessing `self.esgt_coordinator.kuramoto_config.coupling_strength`
    # Since `self.esgt_coordinator` is `mock_esgt_instance`, we check that property
    
    # We need to verify that it CHANGED. 
    # But wait, `mock_esgt_instance.kuramoto_config.coupling_strength` is a PropertyMock? Or just a value?
    # It was set to 0.5.
    # The code does: `self.esgt_coordinator.kuramoto_config.coupling_strength = suggestion.new_value`
    
    assert mock_esgt_instance.kuramoto_config.coupling_strength != 0.5
    
    await system.stop()

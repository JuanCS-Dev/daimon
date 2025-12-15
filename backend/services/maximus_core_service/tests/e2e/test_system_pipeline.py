"""
E2E Pipeline Test: Consciousness System
=======================================

Verifies the complete flow from Sensory Input -> Salience -> ESGT (Kuramoto) -> Bridge -> Output.
Tracks Phase 4 components (Meta-Optimizer, AKOrN).

Usage:
    pytest backend/services/maximus_core_service/tests/e2e/test_system_pipeline.py
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from maximus_core_service.consciousness.system import ConsciousnessSystem, ConsciousnessConfig
from maximus_core_service.consciousness.esgt.models import ESGTPhase

@pytest.mark.asyncio
async def test_full_consciousness_pipeline():
    """
    Simulate a full lifecycle event:
    1. System Start (Pre-heat TIG/Meta-Optimizer)
    2. Sensory Input Processing
    3. ESGT Ignition (Kuramoto)
    4. Introspective Output
    5. Feedback Loop (Auto-Tuning)
    """
    
    print("\n[E2E] Starting Consciousness System Pipeline Test...")
    
    # 1. Configuration (Minimal for speed, but functional)
    config = ConsciousnessConfig(
        tig_node_count=10, # Small fabric
        esgt_min_salience=0.0, # Guarantee ignition for test
        esgt_min_available_nodes=5, # Allow small fabric test
    )
    
    system = ConsciousnessSystem(config)
    
    # Mock LLM to avoid external API calls/costs, but verify interface usage
    # We patch get_nebius_client to return a Mock that behaves like the Nebius client
    mock_nebius = AsyncMock()
    mock_nebius.generate.return_value.text = "I am conscious of this test input."
    
    # Mock Physics to ensure high coherence (decouple from stochasticity)
    from maximus_core_service.consciousness.esgt.kuramoto_models import PhaseCoherence
    import time
    mock_coherence = PhaseCoherence(
        order_parameter=0.98,
        mean_phase=0.0,
        phase_variance=0.01,
        coherence_quality="deep",
        timestamp=time.time()
    )
    
    with patch('maximus_core_service.consciousness.system.get_nebius_client', return_value=lambda: mock_nebius), \
         patch('maximus_core_service.consciousness.esgt.kuramoto.KuramotoNetwork.get_coherence', return_value=mock_coherence):
             
         # Start System
         print("[E2E] Booting system...")
         await system.start()
         
         # Wait for TIG Fabric async init
         print("[E2E] Waiting for TIG Fabric initialization...")
         await asyncio.sleep(2.0)
         
         assert system.coherence_tracker is not None, "Meta-Optimizer not initialized"
         assert system.config_tuner is not None, "Config Tuner not initialized"
         assert system.esgt_coordinator.use_adaptive_sync is True, "AKOrN not enabled"
         
         # 2. Process Input
         user_query = "Hello Noesis, report status."
         print(f"[E2E] Sending Input: '{user_query}'")
         
         response = await system.process_input(
             content=user_query,
             depth=3, # Moderate depth
             source="e2e_test"
         )
         
         # 3. Validation
         print(f"[E2E] Response Received: {response.narrative[:50]}...")
         print(f"[E2E] Event ID: {response.event_id}")
         print(f"[E2E] Meta-Awareness: {response.meta_awareness_level}")
         
         # Verify Flow
         last_event = system.esgt_coordinator.event_history[-1]
         print(f"[E2E] Last Event Phase: {last_event.current_phase}")
         print(f"[E2E] Last Event Success: {last_event.was_successful()}")
         
         assert last_event.was_successful(), f"ESGT Ignition failed: {last_event.failure_reason}"
         assert last_event.current_phase == ESGTPhase.COMPLETE
         assert last_event.achieved_coherence > 0.0
         
         # Verify Meta-Optimizer
         history = system.coherence_tracker.get_recent_snapshots()
         print(f"[E2E] Tracker History Length: {len(history)}")
         assert len(history) >= 1, "Coherence Tracker missed the event"
         print(f"[E2E] Tracker History: {history[-1]}")
         
         # Verify Output Structure
         assert response.event_id == last_event.event_id
         assert response.meta_awareness_level > 0
         
         print("[E2E] Pipeline verified successfully.")
         
         await system.stop()

if __name__ == "__main__":
    asyncio.run(test_full_consciousness_pipeline())

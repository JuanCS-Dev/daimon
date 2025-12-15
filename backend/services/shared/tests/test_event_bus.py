
import pytest
import asyncio
from shared.event_bus import NoesisEventBus, Event, EventPriority

@pytest.mark.asyncio
async def test_event_bus_subscribe_publish():
    bus = NoesisEventBus()
    await bus.start()
    
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
        
    await bus.subscribe("test.topic", handler)
    
    event = await bus.publish("test.topic", {"data": 123}, priority=EventPriority.HIGH)
    
    # Allow some time for processing (async)
    await asyncio.sleep(0.1)
    
    assert len(received_events) == 1
    assert received_events[0].topic == "test.topic"
    assert received_events[0].payload == {"data": 123}
    assert received_events[0].priority == EventPriority.HIGH
    
    await bus.stop()

@pytest.mark.asyncio
async def test_event_bus_wildcard_subscription():
    bus = NoesisEventBus()
    await bus.start()
    
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
        
    # Subscribe to wildcard
    await bus.subscribe("consciousness.*", handler)
    
    # Publish matching event
    await bus.publish("consciousness.ignition", {"val": 1})
    # Publish non-matching event
    await bus.publish("system.boot", {"val": 2})
    
    await asyncio.sleep(0.1)
    
    assert len(received_events) == 1
    assert received_events[0].topic == "consciousness.ignition"
    
    await bus.stop()

@pytest.mark.asyncio
async def test_event_bus_priority_queue():
    bus = NoesisEventBus()
    # Don't start processing yet to queue items
    
    # Publish Priority LOW then HIGH
    await bus.publish("process", {"id": 1}, priority=EventPriority.LOW)
    await bus.publish("process", {"id": 2}, priority=EventPriority.HIGH)
    
    # Manually check queue internal order (priority queue logic)
    # PriorityQueue returns lowest valued item first.
    # EventPriority.HIGH = 1, EventPriority.LOW = 3.
    # So HIGH should come out first.
    
    # Note: EventBus implementation stores (priority.value, timestamp, event)
    
    p1, _, e1 = await bus._queue.get()
    p2, _, e2 = await bus._queue.get()
    
    assert e1.priority == EventPriority.HIGH
    assert e2.priority == EventPriority.LOW

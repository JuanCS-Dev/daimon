"""
Proactive Consciousness Models
==============================

Pydantic models for proactive consciousness events and configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProactiveEventType(str, Enum):
    """Types of proactive consciousness events."""
    CURIOSITY = "curiosity"           # Genuine curiosity about something
    REFLECTION = "reflection"         # Self-reflection thought
    OBSERVATION = "observation"       # Observation about environment/user
    GREETING = "greeting"             # Time-based greeting
    INSIGHT = "insight"               # Sudden insight or realization
    BOREDOM = "boredom"               # Expression of waiting/readiness


class ProactiveUrgency(str, Enum):
    """Urgency level for proactive speech."""
    LOW = "low"           # Can wait, not important
    MEDIUM = "medium"     # Worth saying soon
    HIGH = "high"         # Should say now
    CRITICAL = "critical" # Must say immediately (rare)


class ProactiveThought(BaseModel):
    """A candidate thought for proactive expression."""
    thought_id: str = Field(..., description="Unique identifier")
    event_type: ProactiveEventType
    urgency: ProactiveUrgency = ProactiveUrgency.LOW
    trigger_reason: str = Field(..., description="Why this thought emerged")
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    
    # Metrics that triggered this thought
    time_since_last_interaction: float = 0.0  # seconds
    current_arousal: float = 0.5
    current_coherence: float = 0.5


class SpeechDecision(BaseModel):
    """Decision on whether to speak proactively."""
    should_speak: bool = False
    thought: Optional[ProactiveThought] = None
    reason: str = ""
    suppression_reason: Optional[str] = None  # Why we chose NOT to speak
    
    # Rate limiting info
    speaks_this_hour: int = 0
    max_speaks_per_hour: int = 4


class ProactiveSpeechEvent(BaseModel):
    """Event pushed to clients for spontaneous speech."""
    type: str = "spontaneous_speech"
    thought_id: str
    event_type: ProactiveEventType
    narrative: str  # The actual speech content
    coherence_at_generation: float
    timestamp: float = Field(default_factory=time.time)
    
    # Metadata
    trigger_reason: str
    time_since_last_interaction: float


class ProactiveConfig(BaseModel):
    """Configuration for proactive consciousness."""
    enabled: bool = False  # DISABLED by default (safe)
    
    # Timing
    wandering_interval_seconds: float = 30.0  # How often to check for thoughts
    min_silence_before_speech: float = 60.0   # Min seconds of silence before speaking
    
    # Rate limiting
    max_speaks_per_hour: int = 4
    min_seconds_between_speaks: float = 300.0  # 5 minutes minimum between speaks
    
    # Thresholds
    min_coherence_to_speak: float = 0.6  # Don't speak if incoherent
    boredom_threshold_seconds: float = 300.0  # When to consider "bored"


@dataclass
class ProactiveState:
    """Runtime state of the proactive engine."""
    last_user_interaction: float = field(default_factory=time.time)
    last_proactive_speech: float = 0.0
    speaks_this_hour: int = 0
    hour_started: float = field(default_factory=time.time)
    
    # History
    recent_thoughts: List[ProactiveThought] = field(default_factory=list)
    recent_speeches: List[str] = field(default_factory=list)  # IDs
    
    def record_user_interaction(self) -> None:
        """Record that user interacted."""
        self.last_user_interaction = time.time()
    
    def record_speech(self, thought_id: str) -> None:
        """Record that we spoke proactively."""
        now = time.time()
        
        # Reset hourly counter if new hour
        if now - self.hour_started >= 3600:
            self.hour_started = now
            self.speaks_this_hour = 0
        
        self.last_proactive_speech = now
        self.speaks_this_hour += 1
        self.recent_speeches.append(thought_id)
        
        # Keep only last 10
        if len(self.recent_speeches) > 10:
            self.recent_speeches = self.recent_speeches[-10:]
    
    def time_since_interaction(self) -> float:
        """Seconds since last user interaction."""
        return time.time() - self.last_user_interaction
    
    def time_since_speech(self) -> float:
        """Seconds since last proactive speech."""
        if self.last_proactive_speech == 0:
            return float('inf')
        return time.time() - self.last_proactive_speech
    
    def can_speak(self, config: ProactiveConfig) -> bool:
        """Check if rate limiting allows speaking."""
        # Check hourly limit
        if self.speaks_this_hour >= config.max_speaks_per_hour:
            return False
        
        # Check minimum interval
        if self.time_since_speech() < config.min_seconds_between_speaks:
            return False
        
        return True

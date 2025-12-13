"""
DAIMON Preference Signal Detector
=================================

Detects approval/rejection signals in user-assistant conversations.

Responsible for:
- Analyzing message pairs for feedback signals
- Extracting user content from different message formats
- Detecting signal types via LLM or heuristics

Architecture:
    SignalDetector receives messages from SessionScanner and produces
    PreferenceSignal objects for the Categorizer to process.

Usage:
    from learners.preference.detector import SignalDetector
    
    detector = SignalDetector()
    signals = detector.analyze_session(messages, session_id)

Follows CODE_CONSTITUTION: Clarity Over Cleverness, Safety First.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from .models import PreferenceSignal, SignalType

logger = logging.getLogger("daimon.preference.detector")

# Approval patterns (Portuguese + English)
APPROVAL_PATTERNS = [
    r"\b(sim|yes|ok|perfeito|otimo|excelente|isso|gostei)\b",
    r"\b(aceito|aprovo|pode|manda|vai|bora|certo|correto)\b",
    r"^(s|y|ok|sim)$",
    r"(thumbs.?up|great|good|nice|awesome)",
]

# Rejection patterns (Portuguese + English)
REJECTION_PATTERNS = [
    r"\b(nao|no|nope|errado|ruim|feio|pare|espera)\b",
    r"\b(rejeito|recuso|para|cancela|volta|desfaz)\b",
    r"\b(menos|mais simples|muito|demais|longo)\b",
    r"(thumbs.?down|bad|wrong|incorrect)",
]


class SignalDetector:
    """
    Detects preference signals in conversation messages.
    
    Uses a two-stage detection approach:
    1. LLM-based detection (when available) for semantic understanding
    2. Heuristic/regex fallback for reliability
    
    The detector analyzes user responses following assistant proposals
    to identify approval, rejection, or modification signals.
    """
    
    def __init__(self, enable_llm: bool = True):
        """
        Initialize detector.
        
        Args:
            enable_llm: Whether to use LLM for detection (with fallback)
        """
        self.enable_llm = enable_llm
        self._llm_service = None
    
    def analyze_session(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
    ) -> Generator[PreferenceSignal, None, None]:
        """
        Analyze a session's messages for preference signals.
        
        Strategy:
        1. Read (assistant proposes, user responds) pairs
        2. Detect feedback in user text
        3. Infer feedback from tool_result.status
        
        Args:
            messages: List of message dictionaries from session
            session_id: Identifier for this session
            
        Yields:
            PreferenceSignal objects for each detected signal
        """
        for i, msg in enumerate(messages):
            msg_type = msg.get("type", "")
            
            # Check tool_result status for implicit rejections
            if msg_type == "tool_result":
                signal_type = self._check_tool_result(msg)
                if signal_type:
                    yield PreferenceSignal(
                        timestamp=datetime.now().isoformat(),
                        signal_type=signal_type,
                        context="Tool execution failed or interrupted",
                        category="general",
                        strength=0.6,
                        session_id=session_id,
                        tool_involved=msg.get("tool_name"),
                    )
            
            # Analyze user messages for explicit feedback
            if msg_type == "user":
                content = self.extract_user_content(msg)
                if not content:
                    continue
                
                context, tool_name = self._get_previous_context(messages, i)
                signal_type = self.detect_signal_type(content, context)
                
                if signal_type:
                    from .categorizer import PreferenceCategorizer
                    categorizer = PreferenceCategorizer()
                    
                    yield PreferenceSignal(
                        timestamp=datetime.now().isoformat(),
                        signal_type=signal_type,
                        context=context[:200] if context else "No context",
                        category=categorizer.infer_category(content + " " + context),
                        strength=categorizer.calculate_strength(content),
                        session_id=session_id,
                        tool_involved=tool_name,
                    )
    
    def detect_signal_type(
        self,
        content: str,
        context: str = "",
    ) -> Optional[str]:
        """
        Detect if content indicates approval or rejection.
        
        Uses LLM when available, falls back to heuristics.
        
        Args:
            content: User message text
            context: Optional context from assistant
            
        Returns:
            "approval", "rejection", or None if neutral
        """
        # Try async LLM detection
        if self.enable_llm:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in running loop, use heuristic
                    return self._detect_heuristic(content)
                else:
                    return loop.run_until_complete(
                        self._detect_with_llm(content, context)
                    )
            except RuntimeError:
                pass
        
        return self._detect_heuristic(content)
    
    async def _detect_with_llm(
        self,
        content: str,
        context: str,
    ) -> Optional[str]:
        """Detect signal type using LLM service."""
        try:
            if self._llm_service is None:
                from learners.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            
            result = await self._llm_service.classify(
                content,
                ["approval", "rejection", "neutral"],
                context=context,
            )
            
            if result.category == "neutral":
                return None
            
            return result.category
            
        except Exception as e:
            logger.debug("LLM detection failed: %s", e)
            return self._detect_heuristic(content)
    
    def _detect_heuristic(self, content: str) -> Optional[str]:
        """Detect signal type using regex patterns."""
        content_lower = content.lower()
        
        # Check approval patterns
        for pattern in APPROVAL_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return SignalType.APPROVAL.value
        
        # Check rejection patterns
        for pattern in REJECTION_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return SignalType.REJECTION.value
        
        return None
    
    def _check_tool_result(self, msg: Dict[str, Any]) -> Optional[str]:
        """Check tool_result for implicit rejection signals."""
        tool_result = msg.get("toolUseResult", {})
        
        if tool_result.get("status") == "failed":
            return SignalType.REJECTION.value
        
        if tool_result.get("interrupted"):
            return SignalType.REJECTION.value
        
        return None
    
    def extract_user_content(self, msg: Dict[str, Any]) -> str:
        """
        Extract text content from user message.
        
        Handles different message formats:
        - Direct text string
        - Content array with text blocks
        - Nested message structure
        
        Args:
            msg: User message dictionary
            
        Returns:
            Extracted text content
        """
        # Try message.content first
        content = msg.get("message", {}).get("content", [])
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return " ".join(text_parts)
        
        # Fallback to direct content field
        direct = msg.get("content")
        if isinstance(direct, str):
            return direct
        
        return ""
    
    def _get_previous_context(
        self,
        messages: List[Dict[str, Any]],
        current_index: int,
    ) -> Tuple[str, Optional[str]]:
        """
        Get context and tool name from previous assistant message.
        
        Args:
            messages: All messages in session
            current_index: Index of current user message
            
        Returns:
            Tuple of (context_text, tool_name)
        """
        if current_index == 0:
            return "", None
        
        prev = messages[current_index - 1]
        if prev.get("type") != "assistant":
            return "", None
        
        context = self._extract_assistant_context(prev)
        tool_name = self._extract_tool_name(prev)
        
        return context, tool_name
    
    def _extract_assistant_context(self, msg: Dict[str, Any]) -> str:
        """Extract context text from assistant message."""
        content = msg.get("message", {}).get("content", [])
        
        if isinstance(content, str):
            return content[:200]
        
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")[:200]
        
        return ""
    
    def _extract_tool_name(self, msg: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from assistant message."""
        content = msg.get("message", {}).get("content", [])
        
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return block.get("name")
        
        return None

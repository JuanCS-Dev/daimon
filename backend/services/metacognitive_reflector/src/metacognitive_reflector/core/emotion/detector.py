"""
Emotion Detector
=================

LLM-based emotion detection using VAD + 28 GoEmotions.

Uses inline LLM calls (Llama/Gemini) for emotion detection,
avoiding external dependencies while maintaining accuracy.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Awaitable, Callable, Dict

from metacognitive_reflector.core.emotion.schemas import (
    EmotionDetectionResult,
    VADScore,
)
from metacognitive_reflector.core.emotion.constants import GOEMOTION_LABELS

logger = logging.getLogger(__name__)


# Compact prompt for fast detection
EMOTION_DETECTION_PROMPT = '''Analyze emotional content.

TEXT: "{text}"

Evaluate on these dimensions:
1. PRIMARY: One of [{emotions}]
2. VALENCE: -1.0 (negative) to +1.0 (positive)
3. AROUSAL: 0.0 (calm) to 1.0 (excited)
4. DOMINANCE: 0.0 (submissive) to 1.0 (dominant)
5. CONFIDENCE: 0.0 to 1.0
6. TOP3: Three emotions with scores

Return ONLY valid JSON:
{{"primary_emotion":"...","valence":0.0,"arousal":0.5,"dominance":0.5,
"confidence":0.7,"emotion_scores":{{"emotion1":0.8,"emotion2":0.5}}}}'''


class EmotionDetector:
    """
    Detects emotions via LLM inline.

    Uses the same LLM as the main system (Llama/Gemini) to analyze
    user input and extract VAD dimensions + categorical emotions.
    """

    def __init__(
        self,
        llm_generate: Callable[[str], Awaitable[str]],
        emotional_level: float = 0.7,
        enabled: bool = True
    ) -> None:
        """
        Initialize emotion detector.

        Args:
            llm_generate: Async function to generate LLM response
            emotional_level: Sensitivity level (0.0-1.0)
            enabled: Whether detection is enabled
        """
        self.llm_generate = llm_generate
        self.emotional_level = emotional_level
        self.enabled = enabled
        self._cache: Dict[str, EmotionDetectionResult] = {}

    async def detect(self, text: str) -> EmotionDetectionResult:
        """
        Detect emotions in user text.

        Args:
            text: User input text

        Returns:
            EmotionDetectionResult with VAD and categorical emotions
        """
        if not self.enabled or not text.strip():
            return self._neutral_result(text)

        # Check cache for identical input
        cache_key = text[:200]  # Use first 200 chars as key
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Update timestamp
            cached.detected_at = datetime.now()
            return cached

        # Build prompt
        emotions_str = ", ".join(GOEMOTION_LABELS[:15])  # Top 15 for brevity
        prompt = EMOTION_DETECTION_PROMPT.format(
            text=text[:500],  # Truncate for efficiency
            emotions=emotions_str
        )

        try:
            response = await self.llm_generate(prompt)
            result = self._parse_response(response, text)

            # Cache result
            self._cache[cache_key] = result
            # Limit cache size
            if len(self._cache) > 100:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            logger.debug(
                "Emotion detected: %s (V=%.2f, A=%.2f, D=%.2f, conf=%.2f)",
                result.primary_emotion,
                result.vad.valence,
                result.vad.arousal,
                result.vad.dominance,
                result.confidence
            )

            return result

        except Exception as e:
            logger.warning("Emotion detection failed: %s", e)
            return self._fallback_detection(text)

    def _parse_response(self, response: str, original_text: str) -> EmotionDetectionResult:
        """Parse LLM response into EmotionDetectionResult."""
        try:
            # Strip markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                # Remove ```json or ``` prefix and ``` suffix
                lines = clean_response.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                clean_response = "\n".join(lines)

            # Extract JSON from response (handles nested objects)
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            primary = data.get("primary_emotion", "neutral")
            if primary not in GOEMOTION_LABELS:
                primary = "neutral"

            return EmotionDetectionResult(
                vad=VADScore(
                    valence=float(data.get("valence", 0.0)),
                    arousal=float(data.get("arousal", 0.5)),
                    dominance=float(data.get("dominance", 0.5))
                ),
                primary_emotion=primary,
                emotion_scores=data.get("emotion_scores", {primary: 1.0}),
                confidence=float(data.get("confidence", 0.5)),
                raw_text=original_text,
                detected_at=datetime.now()
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug("Parse error: %s, using fallback", e)
            return self._fallback_detection(original_text)

    def _fallback_detection(self, text: str) -> EmotionDetectionResult:
        """
        Fallback heuristic detection if LLM fails.

        Uses simple keyword matching for basic emotion detection.
        """
        text_lower = text.lower()

        # Simple keyword-based fallback
        positive_words = ["happy", "joy", "love", "great", "amazing", "feliz", "alegria", "amor"]
        negative_words = ["sad", "angry", "fear", "hate", "terrible", "triste", "raiva", "medo"]
        high_arousal = ["!", "excited", "amazing", "terrible", "urgent", "urgente"]

        valence = 0.0
        arousal = 0.5
        primary = "neutral"

        for word in positive_words:
            if word in text_lower:
                valence += 0.3
                primary = "joy" if valence > 0.5 else "optimism"

        for word in negative_words:
            if word in text_lower:
                valence -= 0.3
                primary = "sadness" if valence < -0.3 else "annoyance"

        for word in high_arousal:
            if word in text_lower:
                arousal += 0.2

        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        return EmotionDetectionResult(
            vad=VADScore(valence=valence, arousal=arousal, dominance=0.5),
            primary_emotion=primary,
            emotion_scores={primary: 0.6, "neutral": 0.4},
            confidence=0.3,  # Low confidence for fallback
            raw_text=text,
            detected_at=datetime.now()
        )

    def _neutral_result(self, text: str) -> EmotionDetectionResult:
        """Return neutral result for empty/disabled detection."""
        return EmotionDetectionResult(
            vad=VADScore(0.0, 0.3, 0.5),
            primary_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            confidence=1.0,
            raw_text=text,
            detected_at=datetime.now()
        )

    def clear_cache(self) -> None:
        """Clear detection cache."""
        self._cache.clear()


# Convenience function for quick detection
async def detect_emotion(
    text: str,
    llm_generate: Callable[[str], Awaitable[str]]
) -> EmotionDetectionResult:
    """
    Quick emotion detection without maintaining state.

    Args:
        text: Text to analyze
        llm_generate: LLM generation function

    Returns:
        EmotionDetectionResult
    """
    detector = EmotionDetector(llm_generate)
    return await detector.detect(text)

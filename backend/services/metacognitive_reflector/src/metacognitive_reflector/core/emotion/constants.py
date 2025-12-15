"""
Emotional Intelligence Constants
=================================

GoEmotions 28-class taxonomy and VAD mappings.

Based on:
- Demszky et al. (2020): GoEmotions dataset
- Russell (1980): Circumplex Model (VAD)
- Mehrabian & Russell (1974): PAD emotional state model
"""

from typing import Dict, Tuple

# GoEmotions - 28 emotion categories (27 + neutral)
# Source: https://arxiv.org/abs/2005.00547
GOEMOTION_LABELS = [
    "admiration",      # Respect and approval
    "amusement",       # Finding something funny
    "anger",           # Strong displeasure
    "annoyance",       # Mild irritation
    "approval",        # Endorsement, agreement
    "caring",          # Affection, desire to help
    "confusion",       # Lack of understanding
    "curiosity",       # Desire to learn/know
    "desire",          # Wanting something
    "disappointment",  # Sadness from unmet expectations
    "disapproval",     # Rejection, disagreement
    "disgust",         # Strong aversion
    "embarrassment",   # Self-conscious discomfort
    "excitement",      # Enthusiastic anticipation
    "fear",            # Anxiety about danger
    "gratitude",       # Thankfulness
    "grief",           # Deep sorrow, loss
    "joy",             # Happiness, delight
    "love",            # Deep affection
    "nervousness",     # Anxious anticipation
    "optimism",        # Hopeful expectation
    "pride",           # Satisfaction in achievement
    "realization",     # Sudden understanding
    "relief",          # Release from distress
    "remorse",         # Regret for wrongdoing
    "sadness",         # Unhappiness, sorrow
    "surprise",        # Unexpected reaction
    "neutral",         # No strong emotion
]

# VAD mappings: (Valence, Arousal, Dominance)
# Valence: -1 (negative) to +1 (positive)
# Arousal: 0 (calm) to 1 (excited/activated)
# Dominance: 0 (submissive) to 1 (dominant/in control)
EMOTION_TO_VAD: Dict[str, Tuple[float, float, float]] = {
    # Positive high-arousal emotions
    "joy": (0.9, 0.7, 0.7),
    "excitement": (0.8, 0.9, 0.6),
    "amusement": (0.8, 0.6, 0.6),
    "love": (0.9, 0.5, 0.5),
    "admiration": (0.8, 0.5, 0.4),
    "pride": (0.8, 0.6, 0.8),
    "gratitude": (0.9, 0.4, 0.4),

    # Positive low-arousal emotions
    "optimism": (0.7, 0.5, 0.6),
    "approval": (0.6, 0.3, 0.5),
    "caring": (0.7, 0.4, 0.5),
    "relief": (0.6, 0.2, 0.5),

    # Negative high-arousal emotions
    "anger": (-0.7, 0.8, 0.6),
    "fear": (-0.8, 0.8, -0.6),
    "disgust": (-0.6, 0.5, 0.4),
    "annoyance": (-0.4, 0.5, 0.4),
    "nervousness": (-0.5, 0.7, -0.4),

    # Negative low-arousal emotions
    "sadness": (-0.7, 0.2, -0.4),
    "grief": (-0.9, 0.4, -0.6),
    "disappointment": (-0.6, 0.3, -0.3),
    "disapproval": (-0.5, 0.4, 0.3),
    "remorse": (-0.6, 0.3, -0.4),
    "embarrassment": (-0.5, 0.5, -0.5),

    # Neutral/Cognitive emotions
    "surprise": (0.2, 0.8, 0.1),
    "curiosity": (0.5, 0.6, 0.4),
    "confusion": (-0.2, 0.5, -0.3),
    "realization": (0.3, 0.5, 0.4),
    "desire": (0.4, 0.6, 0.3),
    "neutral": (0.0, 0.3, 0.5),
}

# Response strategies based on user emotional state
# Maps quadrant of VAD space to response approach
RESPONSE_STRATEGIES: Dict[str, str] = {
    # High negative valence + High arousal (anger, fear, anxiety)
    "validar_acalmar": (
        "User in high-arousal negative state (anger, fear, anxiety). "
        "Validate feelings without mirroring. Use calm, grounded tone. "
        "Offer perspective without minimizing. Keep responses concise."
    ),

    # High negative valence + Low arousal (sadness, grief, disappointment)
    "empatia_presenca": (
        "User in low-arousal negative state (sadness, grief). "
        "Show genuine presence and empathy. Use gentle, welcoming tone. "
        "Don't try to 'fix' immediately. Acknowledge difficulty."
    ),

    # High positive valence + High arousal (joy, excitement, enthusiasm)
    "espelhar_expandir": (
        "User in high-arousal positive state (joy, excitement). "
        "Mirror enthusiasm appropriately. Expand conversation with curiosity. "
        "Maintain energetic tone. Celebrate together."
    ),

    # High positive valence + Low arousal (calm, content, relief)
    "manter_serenidade": (
        "User in calm positive state (content, relieved). "
        "Maintain serene tone. Respond reflectively. "
        "Don't add unnecessary energy. Good moment to deepen."
    ),

    # Neutral or mixed state
    "resposta_padrao": (
        "Neutral or mixed emotional state. "
        "Respond naturally. Stay attentive to emotional signals. "
        "Adapt as conversation evolves."
    ),
}

# Emotional salience thresholds
SALIENCE_THRESHOLDS = {
    "high": 0.7,      # Strong emotional signal
    "moderate": 0.4,  # Noticeable emotion
    "low": 0.2,       # Subtle emotion
}

# Emotion groupings for analysis
EMOTION_GROUPS = {
    "positive_active": ["joy", "excitement", "amusement", "pride", "love"],
    "positive_passive": ["gratitude", "relief", "optimism", "approval", "caring"],
    "negative_active": ["anger", "fear", "disgust", "annoyance", "nervousness"],
    "negative_passive": ["sadness", "grief", "disappointment", "remorse", "embarrassment"],
    "cognitive": ["curiosity", "confusion", "surprise", "realization", "desire"],
    "neutral": ["neutral", "approval", "disapproval"],
}

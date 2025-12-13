"""
Keystroke Biometric Calculations.

Pure functions for computing keystroke biometric features.
"""

import statistics
from typing import Dict, List, Tuple

from .keystroke_models import KeystrokeEvent, KeystrokeBiometrics

# Configuration
FATIGUE_WINDOW_SIZE = 60
FLOW_CONSISTENCY_THRESHOLD = 0.75
FLOW_SPEED_THRESHOLD = 0.15


def calculate_hold_times(
    presses: List[KeystrokeEvent],
    releases: List[KeystrokeEvent],
) -> List[float]:
    """Calculate hold times by matching press/release pairs."""
    hold_times = []

    # Create lookup of releases by key
    releases_by_key: Dict[str, List[KeystrokeEvent]] = {}
    for r in releases:
        if r.key not in releases_by_key:
            releases_by_key[r.key] = []
        releases_by_key[r.key].append(r)

    for press in presses:
        if press.key in releases_by_key:
            # Find first release after this press
            for release in releases_by_key[press.key]:
                if release.timestamp > press.timestamp:
                    hold_time = release.timestamp - press.timestamp
                    # Sanity check: hold time should be reasonable
                    if 0.01 < hold_time < 2.0:
                        hold_times.append(hold_time)
                    break

    return hold_times


def calculate_seek_times(events: List[KeystrokeEvent]) -> List[float]:
    """Calculate inter-key intervals (seek times)."""
    seek_times = []

    presses = sorted(
        [e for e in events if e.event_type == "press"],
        key=lambda e: e.timestamp,
    )

    for i in range(1, len(presses)):
        interval = presses[i].timestamp - presses[i - 1].timestamp
        # Sanity check: interval should be reasonable
        if 0.01 < interval < 5.0:
            seek_times.append(interval)

    return seek_times


def calculate_rhythm_consistency(seek_times: List[float]) -> float:
    """
    Calculate rhythm consistency (0-1, higher = more consistent).

    Uses coefficient of variation (CV) as inverse measure of consistency.
    """
    if len(seek_times) < 3:
        return 0.0

    try:
        mean_interval = statistics.mean(seek_times)
        std_interval = statistics.stdev(seek_times)

        if mean_interval == 0:
            return 0.0

        # Coefficient of variation
        cv = std_interval / mean_interval

        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0.0, 1.0 - cv)
        return min(1.0, consistency)

    except (statistics.StatisticsError, ZeroDivisionError):
        return 0.0


def calculate_fatigue_index(seek_times: List[float]) -> float:
    """
    Calculate fatigue index (0-1) by comparing recent vs older intervals.

    Higher values indicate slowing down (fatigue).
    """
    if len(seek_times) < FATIGUE_WINDOW_SIZE:
        return 0.0

    half = len(seek_times) // 2
    older_times = seek_times[:half]
    newer_times = seek_times[half:]

    try:
        older_mean = statistics.mean(older_times)
        newer_mean = statistics.mean(newer_times)

        if older_mean == 0:
            return 0.0

        slowdown_ratio = newer_mean / older_mean
        fatigue = min(1.0, max(0.0, (slowdown_ratio - 1.0)))
        return fatigue

    except (statistics.StatisticsError, ZeroDivisionError):
        return 0.0


def calculate_focus_score(seek_times: List[float]) -> float:
    """
    Calculate focus score (0-1) based on burst patterns.

    Higher scores indicate focused bursts of typing vs scattered.
    """
    if len(seek_times) < 5:
        return 0.0

    burst_threshold = 0.3  # 300ms between keys
    in_burst = False
    burst_lengths = []
    current_burst = 0

    for interval in seek_times:
        if interval < burst_threshold:
            if not in_burst:
                in_burst = True
                current_burst = 1
            else:
                current_burst += 1
        else:
            if in_burst:
                burst_lengths.append(current_burst)
                in_burst = False
                current_burst = 0

    if in_burst:
        burst_lengths.append(current_burst)

    if not burst_lengths:
        return 0.2

    avg_burst = statistics.mean(burst_lengths)
    focus = min(1.0, avg_burst / 10.0)
    return focus


def calculate_cognitive_load(
    hold_times: List[float],
    seek_times: List[float],
) -> float:
    """
    Calculate cognitive load (0-1) from hold times and intervals.

    Higher cognitive load is inferred from longer hold times and
    more variable intervals.
    """
    if not hold_times or not seek_times:
        return 0.0

    try:
        avg_hold = statistics.mean(hold_times)
        hold_factor = min(1.0, avg_hold / 0.2)

        interval_cv = 0.0
        if len(seek_times) >= 3:
            mean_int = statistics.mean(seek_times)
            std_int = statistics.stdev(seek_times)
            if mean_int > 0:
                interval_cv = std_int / mean_int

        variability_factor = min(1.0, interval_cv)
        cognitive_load = (hold_factor * 0.4 + variability_factor * 0.6)
        return cognitive_load

    except (statistics.StatisticsError, ZeroDivisionError):
        return 0.0


def calculate_typing_speed(presses: List[KeystrokeEvent]) -> float:
    """Calculate typing speed in keys per minute."""
    if len(presses) < 2:
        return 0.0

    sorted_presses = sorted(presses, key=lambda e: e.timestamp)
    time_span = sorted_presses[-1].timestamp - sorted_presses[0].timestamp

    if time_span <= 0:
        return 0.0

    keys_per_second = len(presses) / time_span
    return keys_per_second * 60


def calculate_error_rate(presses: List[KeystrokeEvent]) -> float:
    """
    Estimate error rate from backspace usage.

    Returns ratio of backspace keys to total keys.
    """
    if not presses:
        return 0.0

    backspace_keys = {"backspace", "delete", "BackSpace", "Delete"}
    backspace_count = sum(1 for p in presses if p.key in backspace_keys)

    return backspace_count / len(presses)


def infer_state(biometrics: KeystrokeBiometrics) -> Tuple[str, float]:
    """
    Infer cognitive state from biometrics.

    Returns (state_name, confidence).
    """
    # Flow state: consistent rhythm + fast typing + low errors
    if (
        biometrics.rhythm_consistency > FLOW_CONSISTENCY_THRESHOLD
        and biometrics.avg_seek_time < FLOW_SPEED_THRESHOLD
        and biometrics.error_rate < 0.05
    ):
        confidence = min(
            biometrics.rhythm_consistency,
            1.0 - (biometrics.avg_seek_time / FLOW_SPEED_THRESHOLD),
        )
        return ("flow", confidence)

    # Fatigued: high fatigue index
    if biometrics.fatigue_index > 0.4:
        return ("fatigued", biometrics.fatigue_index)

    # Stressed: fast but erratic with errors
    if (
        biometrics.typing_speed > 200
        and biometrics.rhythm_consistency < 0.5
        and biometrics.error_rate > 0.1
    ):
        confidence = min(biometrics.error_rate * 2, 1.0)
        return ("stressed", confidence)

    # Distracted: low consistency, low focus
    if biometrics.rhythm_consistency < 0.4 and biometrics.focus_score < 0.3:
        confidence = max(
            1.0 - biometrics.rhythm_consistency,
            1.0 - biometrics.focus_score,
        )
        return ("distracted", confidence * 0.7)

    # Focused: decent consistency and focus
    if biometrics.rhythm_consistency > 0.5 and biometrics.focus_score > 0.4:
        confidence = (biometrics.rhythm_consistency + biometrics.focus_score) / 2
        return ("focused", confidence)

    # Default: moderate state
    return ("focused", 0.5)

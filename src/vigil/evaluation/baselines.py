"""Section 13.1: baseline (B1) - per-channel fixed-threshold alarms.

This is exactly the "threshold monitor with an HR limit of 110" mentioned
in the Worked Example (Section 12): it looks at each reading in isolation
and fires whenever ANY channel crosses a fixed limit, with no state, no
debounce, and no suppression. It re-fires on every single reading that
still crosses the limit, which is the point — it's the alarm-fatigue
failure mode VIGIL is built to avoid.
"""
from vigil.models import Observation

# Fixed thresholds — chosen to match the "3" (most severe) NEWS2 bands,
# i.e. the limits a bedside monitor would typically be configured with.
THRESHOLDS = {
    "HR": {"low": 40, "high": 130},     # HR <=40 or >=131 on NEWS2 scale
    "RR": {"low": 8, "high": 24},       # RR <=8 or >=25
    "SpO2": {"low": 91, "high": None},  # SpO2 <=91
    "SBP": {"low": 90, "high": 219},    # SBP <=90 or >=220
}


def channel_breaches(obs: Observation) -> set:
    breached = set()
    for channel, limits in THRESHOLDS.items():
        value = getattr(obs, channel)
        if value is None:
            continue
        if limits["low"] is not None and value <= limits["low"]:
            breached.add(channel)
        if limits["high"] is not None and value >= limits["high"]:
            breached.add(channel)
    return breached


def run_b1(observations: list) -> list:
    """Returns a list of dicts, one per observation, with whether B1 fired
    and which channels breached. No state carried between calls."""
    results = []
    for obs in observations:
        breached = channel_breaches(obs)
        results.append({"timestamp": obs.timestamp, "alert": bool(breached), "channels": sorted(breached)})
    return results


# ----------------------------------------------------------------------
# B2: partial NEWS2 aggregate threshold, re-firing on every reading.
# ----------------------------------------------------------------------
"""Section 13.1: baseline (B2) - a partial NEWS2 aggregate threshold with
re-firing on every reading. This is a step up from B1 (it does look at all
four channels together, like Severity does), but it still has no state, no
suppression, and no distinction between a fresh escalation and a duplicate:
every reading at or above the threshold fires again, which is exactly the
"and again on every reading after 11:00" failure mode from the Worked
Example."""
from vigil.analysis.severity import partial_news2

B2_AGGREGATE_THRESHOLD = 5  # NEWS2-3.2's own "medium risk" cutoff


def run_b2(observations: list, profile: dict = None, threshold: int = B2_AGGREGATE_THRESHOLD) -> list:
    results = []
    for obs in observations:
        score = partial_news2(obs, profile)
        results.append({"timestamp": obs.timestamp, "alert": score >= threshold, "partial_news2": score})
    return results

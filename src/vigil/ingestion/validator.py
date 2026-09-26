"""Section 6.1: validation and artefact handling.

Flagged values never contribute to severity/momentum/coherence and can
never, on their own, cause an escalation — they only degrade a
per-channel data-quality signal and can produce a "sensor check" note.
"""
from vigil.config import PLAUSIBLE_RANGE, CHANNELS
from vigil.models import Observation


def validate(obs: Observation) -> Observation:
    """Mutates and returns obs.quality_flags with a reason per bad channel."""
    for c in CHANNELS:
        value = getattr(obs, c)
        if value is None:
            obs.quality_flags[c] = "missing"
            continue
        lo, hi = PLAUSIBLE_RANGE[c]
        if not (lo <= value <= hi):
            obs.quality_flags[c] = "implausible_range"
    return obs


def is_valid(obs: Observation, channel: str) -> bool:
    return channel not in obs.quality_flags and getattr(obs, channel) is not None

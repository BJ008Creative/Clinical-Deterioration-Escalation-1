"""Section 7.4: Coherence — do several signals agree on the same story?

A channel is "concordant" only if it is both adversely displaced AND
adversely moving. k_t (the concordant-channel count) is the gate used
downstream: a single loudly-abnormal channel is not, by itself,
coherent multi-parameter deterioration."""
from vigil.config import CHANNELS, TAU_Z, TAU_M
from vigil.analysis.deviation import deviation, norm_slope


def profile_weights(profile: dict) -> dict:
    """Deterministic profile modifiers (Section 4.2 / 7.4). Extend here as
    more profile-driven rules are added — keep them documented, not hidden
    inside an LLM prompt."""
    weights = {c: 1.0 for c in CHANNELS}
    meds = [m.lower() for m in profile.get("medications", [])]
    if any("bisoprolol" in m or "metoprolol" in m or "atenolol" in m for m in meds):
        weights["HR"] = 0.5  # beta-blockade blunts heart-rate response
    history = [h.lower() for h in profile.get("history", [])]
    if any("copd" in h for h in history):
        weights["SpO2"] = weights.get("SpO2", 1.0) * 1.2
        weights["RR"] = weights.get("RR", 1.0) * 1.2
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def concordant_channels(channel_states: dict, values: dict) -> set:
    concordant = set()
    for c in CHANNELS:
        if values.get(c) is None:
            continue
        z = deviation(c, values[c], channel_states[c])
        m = norm_slope(c, channel_states[c])
        if z > TAU_Z and m > TAU_M:
            concordant.add(c)
    return concordant


def coherence_score(channel_states: dict, values: dict, profile: dict) -> tuple:
    """Returns (C_t, k_t, concordant_channel_set)."""
    weights = profile_weights(profile)
    concordant = concordant_channels(channel_states, values)
    k = len(concordant)
    num = sum(weights[c] for c in concordant)
    denom = sum(weights.values())
    c_t = num / denom if denom else 0.0
    return c_t, k, concordant

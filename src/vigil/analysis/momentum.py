"""Section 7.3: Momentum M_t in [0, 1] — only adverse movement contributes,
so an improving patient scores close to zero."""
from vigil.config import CHANNELS, M_MAX
from vigil.analysis.deviation import norm_slope

# Equal weights by default; profile modifiers (e.g. beta-blocker blunting
# HR response) can override these per patient — see coherence.py for the
# same pattern applied there.
DEFAULT_WEIGHTS = {c: 1 / len(CHANNELS) for c in CHANNELS}


def momentum_score(channel_states: dict, weights: dict = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    total = 0.0
    for c in CHANNELS:
        m = norm_slope(c, channel_states[c])
        clipped = min(max(m, 0.0), M_MAX)
        total += weights.get(c, 0.0) * clipped
    return min(1.0, total / M_MAX)

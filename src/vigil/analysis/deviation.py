"""Section 7.1: normalised, direction-adjusted deviation z_c,t.

Direction-adjusting every channel means "worse" is always positive,
regardless of whether the adverse direction for that channel is up or
down — this is what lets severity/momentum/coherence sum channels
without sign errors."""
from vigil.config import ADVERSE_DIRECTION, ROBUST_SCALE
from vigil.state.patient_state import ChannelState


def deviation(channel: str, value: float, channel_state: ChannelState) -> float:
    d = ADVERSE_DIRECTION[channel]
    baseline = channel_state.blended_baseline()
    scale = ROBUST_SCALE[channel]
    return d * (value - baseline) / scale


def norm_slope(channel: str, channel_state: ChannelState) -> float:
    d = ADVERSE_DIRECTION[channel]
    scale = ROBUST_SCALE[channel]
    return d * channel_state.slope / scale

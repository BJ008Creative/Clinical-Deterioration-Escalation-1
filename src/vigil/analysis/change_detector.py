"""Section 7.1 (detector half): flag a meaningful change when at least one
channel exceeds tau_z, or evidence is already non-zero (so a decaying
episode keeps being tracked even if the very latest single reading looks
briefly normal)."""
from vigil.config import TAU_Z
from vigil.analysis.deviation import deviation


def detect_change(channel_states: dict, values: dict, current_evidence: float) -> tuple:
    """Returns (changed: bool, implicated_channels: set)."""
    implicated = set()
    for c, v in values.items():
        if v is None:
            continue
        if deviation(c, v, channel_states[c]) > TAU_Z:
            implicated.add(c)
    changed = bool(implicated) or current_evidence > 0
    return changed, implicated

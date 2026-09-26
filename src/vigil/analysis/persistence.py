"""Section 7.5: Persistence — fraction of the last K valid readings showing
any adverse displacement. Separates a transient blip from sustained change."""
from vigil.config import TAU_Z
from vigil.analysis.deviation import deviation


def any_adverse(channel_states: dict, values: dict) -> bool:
    for c, v in values.items():
        if v is None:
            continue
        if deviation(c, v, channel_states[c]) > TAU_Z:
            return True
    return False


def persistence_score(window) -> float:
    """window: deque of bools (any_adverse per past tick), already maxlen=K."""
    if not window:
        return 0.0
    return sum(window) / len(window)

"""Sections 8.1–8.2: the decaying evidence accumulator and the physiological
state machine (Figure 1). This is the module the whole "innovation" story
rests on — isolated readings decay away; sustained, corroborated evidence
survives and advances the state."""
from vigil.config import (
    LENS_WEIGHTS, EVIDENCE_DECAY, EVIDENCE_DRIFT,
    THETA_EMERGE, THETA_PERSIST, THETA_CONV, THETA_CAND,
    HYSTERESIS_MARGIN, K_MIN_COHERENT, P_MIN,
    RED_FLAG_NEWS2_MIN, RED_FLAG_SINGLE_SCORE, RED_FLAG_CONSEC_READINGS,
)
from vigil.models import PhysioState
from vigil.analysis.severity import per_channel_scores, partial_news2


def update_evidence(prev_evidence: float, severity: float, momentum: float,
                     coherence: float, persistence: float) -> float:
    g_t = (LENS_WEIGHTS["severity"] * severity
           + LENS_WEIGHTS["momentum"] * momentum
           + LENS_WEIGHTS["coherence"] * coherence
           + LENS_WEIGHTS["persistence"] * persistence)
    return max(0.0, EVIDENCE_DECAY * prev_evidence + g_t - EVIDENCE_DRIFT)


def check_red_flag(obs_scores: dict, patient_state) -> bool:
    """Persistent critical single-parameter score bypasses the coherence
    gate (Section 8.1, 'Red-flag path')."""
    news2_total = sum(obs_scores.values())
    if news2_total >= RED_FLAG_NEWS2_MIN:
        return True
    if any(s >= RED_FLAG_SINGLE_SCORE for s in obs_scores.values()):
        patient_state.consecutive_red_flag += 1
    else:
        patient_state.consecutive_red_flag = 0
    return patient_state.consecutive_red_flag >= RED_FLAG_CONSEC_READINGS


def next_physio_state(current: PhysioState, evidence: float, k: int,
                       persistence: float, red_flag: bool) -> PhysioState:
    """Forward transitions require accumulated evidence + gates; de-escalation
    uses lower (hysteresis) exit thresholds so the state doesn't flicker at a
    boundary."""
    order = [PhysioState.STABLE, PhysioState.EMERGING_CHANGE,
             PhysioState.PERSISTENT_CHANGE, PhysioState.CONVERGING_DETERIORATION,
             PhysioState.ESCALATION_CANDIDATE]
    idx = order.index(current)

    candidate_ok = evidence >= THETA_CAND and k >= K_MIN_COHERENT and persistence >= P_MIN
    if red_flag:
        candidate_ok = True  # bypasses the coherence/persistence gate

    if candidate_ok:
        return PhysioState.ESCALATION_CANDIDATE
    if evidence >= THETA_CONV and k >= K_MIN_COHERENT:
        target = PhysioState.CONVERGING_DETERIORATION
    elif evidence >= THETA_PERSIST:
        target = PhysioState.PERSISTENT_CHANGE
    elif evidence >= THETA_EMERGE:
        target = PhysioState.EMERGING_CHANGE
    else:
        target = PhysioState.STABLE

    target_idx = order.index(target)
    if target_idx >= idx:
        return target

    # De-escalating: only drop one level at a time, and only once evidence
    # has fallen below (entry_threshold - margin) for the level being left.
    exit_thresholds = {
        PhysioState.ESCALATION_CANDIDATE: THETA_CAND,
        PhysioState.CONVERGING_DETERIORATION: THETA_CONV,
        PhysioState.PERSISTENT_CHANGE: THETA_PERSIST,
        PhysioState.EMERGING_CHANGE: THETA_EMERGE,
    }
    exit_thr = exit_thresholds.get(current)
    if exit_thr is not None and evidence < exit_thr - HYSTERESIS_MARGIN:
        return order[idx - 1]
    return current

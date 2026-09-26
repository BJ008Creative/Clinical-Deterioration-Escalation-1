"""Section 8.4: cohort-level prioritisation. Suppressed patients stay
visible in the worklist at the correct rank instead of vanishing."""
from vigil.config import URGENCY_WEIGHTS, URGENCY_PENDING_BONUS, THETA_CAND
from vigil.models import AlertState


def urgency_score(severity: float, momentum: float, coherence: float,
                   evidence: float, alert_state: AlertState) -> float:
    e_hat = min(1.0, evidence / THETA_CAND)
    base = 100 * (
        URGENCY_WEIGHTS["severity"] * severity
        + URGENCY_WEIGHTS["momentum"] * momentum
        + URGENCY_WEIGHTS["coherence"] * coherence
        + URGENCY_WEIGHTS["evidence"] * e_hat
    )
    bonus = URGENCY_PENDING_BONUS if alert_state == AlertState.OPEN else 0.0
    return base + bonus


def rank_cohort(rows: list) -> list:
    """rows: list of dicts each with patient_id + the fields urgency_score needs.
    Returns rows sorted by descending urgency with a `urgency` field added."""
    for r in rows:
        r["urgency"] = urgency_score(r["severity"], r["momentum"], r["coherence"],
                                      r["evidence"], r["alert_state"])
    return sorted(rows, key=lambda r: r["urgency"], reverse=True)

"""Sections 8.2-8.3: the alert lifecycle state machine (Figure 3) and
novelty-aware suppression. A candidate is only ever emitted as a new alert
if it carries genuinely new evidence over the last one for that patient."""
from vigil.config import DELTA_NEW_EVIDENCE
from vigil.models import PhysioState, AlertState, news2_risk_band

PHYSIO_ORDER = [PhysioState.STABLE, PhysioState.EMERGING_CHANGE, PhysioState.PERSISTENT_CHANGE,
                PhysioState.CONVERGING_DETERIORATION, PhysioState.ESCALATION_CANDIDATE]


def _snoozed(patient_state, now: float) -> bool:
    """True while a deferral is still active (Section 8.3: 'Defer - alert
    snoozed for Td minutes; a higher severity band reopens it immediately')."""
    return patient_state.alert_state == AlertState.DEFERRED and now < patient_state.defer_until


def should_emit(patient_state, evidence: float, physio_state: PhysioState,
                 implicated_channels: set, red_flag: bool, now: float = None,
                 partial_news2: int = None) -> tuple:
    """Returns (emit: bool, reason: str)."""
    if physio_state != PhysioState.ESCALATION_CANDIDATE:
        return False, "not_candidate"

    last = patient_state.last_alert
    if last is None:
        return True, "first_escalation"

    # --- Deferral gate: while snoozed, ONLY a higher NEWS2 risk band or a
    # newly-active red flag can reopen the alert early; a generic evidence
    # increase is not enough (that's the whole point of a snooze). Compared
    # against the aggregate NEWS2 band rather than physio_state, because
    # physio_state is already ESCALATION_CANDIDATE (its ceiling) at the
    # moment a defer happens and so can never "rise" further on its own. ---
    if now is not None and _snoozed(patient_state, now):
        if red_flag and not last.get("red_flag"):
            return True, "red_flag_newly_active_during_snooze"
        current_band = news2_risk_band(partial_news2) if partial_news2 is not None else 0
        if current_band > patient_state.defer_news2_band:
            return True, "higher_severity_band_reopens_snooze"
        return False, "deferred_snooze_active"

    if red_flag and not last.get("red_flag"):
        return True, "red_flag_newly_active"

    higher_state = PHYSIO_ORDER.index(physio_state) > PHYSIO_ORDER.index(PhysioState(last["physio_state"]))
    if higher_state:
        return True, "higher_physiological_state"

    new_channels = implicated_channels - set(last.get("channels", []))
    if new_channels:
        return True, f"new_channel_implicated:{sorted(new_channels)}"

    bar = patient_state.evidence_at_last_alert + DELTA_NEW_EVIDENCE + patient_state.dismiss_bar
    if evidence >= bar:
        return True, "evidence_increase_exceeds_bar"

    return False, "duplicate_no_new_evidence"


def handle_snooze_expiry(patient_state, now: float):
    """Figure 3: 'snooze ends ... -> OPEN'. Once Td minutes pass with no
    early reopen, the alert becomes visible again on the worklist even
    without new evidence — it was never resolved, just snoozed."""
    if patient_state.alert_state == AlertState.DEFERRED and now >= patient_state.defer_until:
        patient_state.alert_state = AlertState.OPEN
        return True
    return False


def apply_clinician_action(patient_state, action: str, now: float, reason: str = "", audit_logger=None, event_bus=None):
    """Kept for backward compatibility with existing callers/tests. The real
    implementation is now Section 5.2's Command pattern (evidence/commands.py) —
    this just runs the command and discards the object. Call run_command()
    directly if you need to hold onto it for undo()."""
    from vigil.evidence.commands import run_command
    run_command(patient_state, action, now, reason, audit_logger=audit_logger, event_bus=event_bus)

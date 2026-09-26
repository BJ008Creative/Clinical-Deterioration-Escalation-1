"""Ties every module together for one observation (Figure 4's full loop,
including the investigation graph). This is the function the
simulator/dashboard calls once per incoming reading.

Full audit logging (R8): if an audit_logger is passed, EVERY observation is
logged (not just alerts) — features, quality flags, and physiological-state
transitions — plus every suppression, every retrieval, and every alert, so
the log can reconstruct any decision (Section 11)."""
from vigil.config import CHANNELS
from vigil.models import Observation, PhysioState, AlertState
from vigil.ingestion.validator import validate, is_valid
from vigil.analysis.severity import severity_score, per_channel_scores
from vigil.analysis.momentum import momentum_score
from vigil.analysis.coherence import coherence_score
from vigil.analysis.persistence import persistence_score, any_adverse
from vigil.analysis.change_detector import detect_change
from vigil.evidence.accumulator import update_evidence, next_physio_state, check_red_flag
from vigil.evidence.suppression import should_emit, handle_snooze_expiry
from vigil.agents.tools import ToolContext
from vigil.agents.graph import run_investigation_graph


def _obs_id(obs: Observation) -> str:
    return f"obs:{obs.patient_id}:{obs.timestamp}"


def _emit(event_type: str, audit_logger, event_bus, **fields):
    """Section 5.2's Observer/Pub-Sub pattern. If an event_bus is given,
    publish through it (any number of subscribers, audit logger included,
    can listen without pipeline.py knowing about them). Otherwise fall back
    to calling audit_logger directly, so existing callers/tests that only
    pass audit_logger keep working unchanged."""
    if event_bus is not None:
        event_bus.publish(event_type, **fields)
    elif audit_logger is not None:
        audit_logger.log(event_type, **fields)


def process_observation(patient_state, obs: Observation, guideline_index=None,
                         audit_logger=None, patient_store: dict = None, event_bus=None,
                         explanation_provider=None) -> dict:
    """Returns a tick summary dict: features, physio_state, alert (or None),
    suppression reason if any. Also updates patient_state in place.

    patient_store: the full {patient_id: PatientState} map, needed so the
    investigation graph's compare_cohort/get_alert_history tools can see
    every patient. If omitted (e.g. a single-patient unit test), a
    single-entry store is built so the tools still work, just with a
    cohort of one.
    """
    if patient_store is None:
        patient_store = {patient_state.patient_id: patient_state}

    validate(obs)
    values = {c: getattr(obs, c) for c in CHANNELS}
    patient_state.recent_obs_ids.append(_obs_id(obs))

    changed, implicated = detect_change(patient_state.channels, values, patient_state.evidence)

    severity = severity_score(obs, patient_state.profile)
    momentum = momentum_score(patient_state.channels)
    coherence, k, concordant = coherence_score(patient_state.channels, values, patient_state.profile)
    adverse_now = any_adverse(patient_state.channels, values)
    patient_state.window.append(adverse_now)
    persistence = persistence_score(patient_state.window)

    # Update rolling channel statistics AFTER computing this tick's deviation
    # features (so z_t compares the new value against the *pre*-update baseline).
    for c in CHANNELS:
        if is_valid(obs, c):
            patient_state.channels[c].update(getattr(obs, c))

    obs_scores = per_channel_scores(obs, patient_state.profile)
    red_flag = check_red_flag(obs_scores, patient_state)

    new_evidence = update_evidence(patient_state.evidence, severity, momentum, coherence, persistence)
    patient_state.evidence = new_evidence

    prev_physio_state = patient_state.physio_state
    new_state = next_physio_state(prev_physio_state, new_evidence, k, persistence, red_flag)
    patient_state.physio_state = new_state

    features = {"severity": severity, "momentum": momentum, "coherence": coherence,
                "persistence": persistence, "evidence": new_evidence, "k": k,
                "partial_news2": sum(obs_scores.values())}
    patient_state.last_features = features

    if audit_logger is not None or event_bus is not None:
        _emit("observation", audit_logger, event_bus, patient_id=obs.patient_id, obs_id=_obs_id(obs),
                          timestamp=obs.timestamp, values=values, quality_flags=dict(obs.quality_flags),
                          features=features, physiological_state=new_state.value,
                          alert_state=patient_state.alert_state.value)
        if new_state != prev_physio_state:
            _emit("state_transition", audit_logger, event_bus, patient_id=obs.patient_id, timestamp=obs.timestamp,
                              from_state=prev_physio_state.value, to_state=new_state.value,
                              evidence=new_evidence)

    result = {"patient_id": obs.patient_id, "timestamp": obs.timestamp, "features": features,
              "physio_state": new_state.value, "implicated_channels": sorted(concordant),
              "quality_flags": dict(obs.quality_flags), "alert": None, "suppressed_reason": None}

    # Snooze bookkeeping: if a deferred alert's Td has elapsed, it becomes
    # visible again on the worklist (Figure 3: "snooze ends -> OPEN").
    if handle_snooze_expiry(patient_state, obs.timestamp):
        if audit_logger is not None or event_bus is not None:
            _emit("alert_state_change", audit_logger, event_bus, patient_id=obs.patient_id, timestamp=obs.timestamp,
                              new_alert_state="OPEN", reason="snooze_expired")

    emit, reason = should_emit(patient_state, new_evidence, new_state, concordant, red_flag,
                                now=obs.timestamp, partial_news2=features["partial_news2"])
    if not emit:
        result["suppressed_reason"] = reason
        if (audit_logger is not None or event_bus is not None) and reason != "not_candidate":
            # "not_candidate" means the patient simply isn't at
            # ESCALATION_CANDIDATE this tick — that's normal monitoring, not
            # a suppressed alert, so it isn't logged as one (Section 8.3
            # only describes suppression for genuine repeat candidates).
            _emit("suppressed", audit_logger, event_bus, patient_id=obs.patient_id, timestamp=obs.timestamp,
                              reason=reason, evidence=new_evidence, physiological_state=new_state.value)
        return result

    # --- Escalation candidate that passed the suppression gate: run the
    # investigation graph (Section 9.1/9.2: tools -> explain -> verify). ---
    tools = ToolContext(patient_store=patient_store, guideline_index=guideline_index)
    graph_result = run_investigation_graph(obs.patient_id, obs, features, concordant, tools,
                                            explanation_provider=explanation_provider)

    if audit_logger is not None or event_bus is not None:
        for call in graph_result.tool_calls:
            _emit("tool_call", audit_logger, event_bus, patient_id=obs.patient_id, timestamp=obs.timestamp, **call)

    alert_id = f"A-{obs.patient_id}-{len(patient_state.alert_history) + 1:04d}"
    alert = {
        "alert_id": alert_id,
        "physio_state": new_state.value,
        "features": features,
        "channels": sorted(concordant),
        "red_flag": red_flag,
        "explanation": graph_result.explanation,
        "verifier_result": graph_result.verifier_result,
        "retrieved_context": [f"profile:{obs.patient_id}"] + [c["id"] for c in graph_result.package.get("retrieved", [])],
        "triggering_observations": list(patient_state.recent_obs_ids),
        "decision_reason": reason,
        "timestamp": obs.timestamp,
        "clinician_action": None,
        "previous_alert_id": patient_state.last_alert["alert_id"] if patient_state.last_alert else None,
    }
    patient_state.alert_state = AlertState.OPEN
    patient_state.evidence_at_last_alert = new_evidence
    patient_state.dismiss_bar = 0.0
    patient_state.last_alert = alert
    patient_state.alert_history.append(alert)
    result["alert"] = alert

    if audit_logger is not None or event_bus is not None:
        _emit("alert", audit_logger, event_bus, patient_id=obs.patient_id, **alert)

    return result

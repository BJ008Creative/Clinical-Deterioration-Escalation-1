"""Tests for the fixes requested after the first review: defer/snooze
enforcement, SpO2 Scale 2, the five investigation tools, investigate's
retrieval expansion, and full per-observation audit logging."""
import json
import tempfile
from pathlib import Path

from vigil.models import Observation, AlertState, PhysioState
from vigil.state.patient_state import PatientState
from vigil.pipeline import process_observation
from vigil.evidence.suppression import apply_clinician_action
from vigil.simulator.streamer import stream_patient
from vigil.audit.logger import AuditLogger
from vigil.agents.tools import ToolContext
from vigil.analysis.severity import per_channel_scores


# ---------------------------------------------------------------- defer/snooze

def _run_until_first_alert(pid, scenario, seed):
    state = PatientState(patient_id=pid, profile={})
    last_obs = None
    for obs in stream_patient(pid, scenario, seed=seed, n=20):
        r = process_observation(state, obs)
        last_obs = obs
        if r["alert"] is not None:
            break
    return state, last_obs


def test_deferred_alert_stays_suppressed_during_snooze_even_with_some_new_evidence():
    state, last_obs = _run_until_first_alert("P-DEFER", "multi_parameter_deterioration", seed=2)
    apply_clinician_action(state, "defer", now=last_obs.timestamp)
    assert state.alert_state == AlertState.DEFERRED

    # A small worsening tick, same physiological band, well inside the snooze
    # window: must stay suppressed with the specific defer reason, even
    # though the *old* rule (plain evidence-delta) might have re-fired.
    obs2 = Observation(patient_id="P-DEFER", timestamp=last_obs.timestamp + 60,
                        HR=last_obs.HR + 1, RR=last_obs.RR, SpO2=last_obs.SpO2, SBP=last_obs.SBP - 1)
    r = process_observation(state, obs2)
    assert r["alert"] is None
    assert r["suppressed_reason"] == "deferred_snooze_active"


def test_deferred_alert_reopens_immediately_on_higher_severity_band():
    state, last_obs = _run_until_first_alert("P-DEFER2", "multi_parameter_deterioration", seed=2)
    apply_clinician_action(state, "defer", now=last_obs.timestamp)

    # A sharp worsening well inside the snooze window should still reopen.
    obs2 = Observation(patient_id="P-DEFER2", timestamp=last_obs.timestamp + 60,
                        HR=last_obs.HR + 25, RR=last_obs.RR + 6,
                        SpO2=last_obs.SpO2 - 6, SBP=last_obs.SBP - 15)
    r = process_observation(state, obs2)
    assert r["alert"] is not None
    assert r["suppressed_reason"] is None


def test_snooze_expiry_reopens_alert_without_requiring_new_evidence():
    state, last_obs = _run_until_first_alert("P-DEFER3", "multi_parameter_deterioration", seed=2)
    apply_clinician_action(state, "defer", now=last_obs.timestamp)
    assert state.alert_state == AlertState.DEFERRED

    # Fast-forward past Td (config default 10 min = 600s) with an identical reading.
    obs2 = Observation(patient_id="P-DEFER3", timestamp=last_obs.timestamp + 700,
                        HR=last_obs.HR, RR=last_obs.RR, SpO2=last_obs.SpO2, SBP=last_obs.SBP)
    process_observation(state, obs2)
    assert state.alert_state == AlertState.OPEN  # snooze ended -> visible again


# ---------------------------------------------------------------- SpO2 Scale 2

def test_scale2_profile_uses_alternative_spo2_bands():
    obs = Observation(patient_id="X", timestamp=0, HR=75, RR=16, SpO2=90, SBP=118)
    scale1_scores = per_channel_scores(obs, {"spo2_target_scale": 1})
    scale2_scores = per_channel_scores(obs, {"spo2_target_scale": 2})
    # SpO2=90 is scale1 score 3 (<=91) but scale2 score 0 (88-92 is the target range)
    assert scale1_scores["SpO2"] == 3
    assert scale2_scores["SpO2"] == 0


def test_scale2_still_flags_genuinely_low_spo2():
    obs = Observation(patient_id="X", timestamp=0, HR=75, RR=16, SpO2=80, SBP=118)
    scale2_scores = per_channel_scores(obs, {"spo2_target_scale": 2})
    assert scale2_scores["SpO2"] == 3


# ---------------------------------------------------------------- 5 tools

def test_all_five_tools_are_callable_and_return_data():
    store = {}
    p1 = PatientState(patient_id="P-A", profile={"age": 60, "history": ["COPD"]})
    p2 = PatientState(patient_id="P-B", profile={"age": 40})
    store["P-A"] = p1
    store["P-B"] = p2
    for pid, s in store.items():
        for obs in stream_patient(pid, "multi_parameter_deterioration", seed=1, n=10):
            process_observation(s, obs, patient_store=store)

    tools = ToolContext(patient_store=store, guideline_index=None)

    state_info = tools.get_state("P-A")
    assert "physiological_state" in state_info and "features" in state_info

    profile_info = tools.get_profile("P-A")
    assert profile_info["history"] == ["COPD"]

    history = tools.get_alert_history("P-A")
    assert isinstance(history, list)

    guideline_hits = tools.search_guidelines("sepsis")
    assert guideline_hits == []  # no index configured in this test

    cohort_info = tools.compare_cohort("P-A")
    assert cohort_info["cohort_size"] == 2 and cohort_info["rank"] in (1, 2)


# ---------------------------------------------------------------- investigate expands retrieval

def test_investigate_action_expands_retrieval_and_adds_cohort_context():
    store = {}
    state = PatientState(patient_id="P-INV", profile={"age": 70, "history": ["COPD"]})
    store["P-INV"] = state
    last_obs = None
    alert = None
    for obs in stream_patient("P-INV", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs, patient_store=store)
        last_obs = obs
        if r["alert"] is not None:
            alert = r["alert"]
            break
    assert alert is not None
    assert "expanded_investigation" not in alert.get("retrieved_context", [])  # sanity: first alert not expanded

    apply_clinician_action(state, "investigate", now=last_obs.timestamp)
    assert state.investigation_mode is True

    # Force a genuinely new escalation (higher band) so a second investigation runs.
    obs2 = Observation(patient_id="P-INV", timestamp=last_obs.timestamp + 60,
                        HR=last_obs.HR + 25, RR=last_obs.RR + 6,
                        SpO2=last_obs.SpO2 - 6, SBP=last_obs.SBP - 15)
    r2 = process_observation(state, obs2, patient_store=store)
    assert r2["alert"] is not None
    assert r2["alert"]["triggering_observations"]  # non-empty, from recent_obs_ids


# ---------------------------------------------------------------- full audit logging

def test_every_observation_is_logged_not_just_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "audit.jsonl"
        audit = AuditLogger(str(log_path))
        state = PatientState(patient_id="P-AUDIT", profile={})
        n_ticks = 0
        for obs in stream_patient("P-AUDIT", "multi_parameter_deterioration", seed=2, n=15):
            process_observation(state, obs, audit_logger=audit)
            n_ticks += 1

        records = list(audit.replay())
        obs_records = [r for r in records if r["event"] == "observation"]
        assert len(obs_records) == n_ticks, "every single observation must be logged, not just alerts"

        alert_records = [r for r in records if r["event"] == "alert"]
        assert len(alert_records) >= 1
        for a in alert_records:
            # Section 11: "Each alert can be reconstructed from it."
            for field in ("alert_id", "features", "channels", "explanation",
                          "verifier_result", "retrieved_context", "decision_reason", "previous_alert_id"):
                assert field in a, f"alert record missing {field}, cannot be reconstructed"


def test_clinician_action_is_logged():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "audit.jsonl"
        audit = AuditLogger(str(log_path))
        state = PatientState(patient_id="P-CLIN", profile={})
        last_obs = None
        for obs in stream_patient("P-CLIN", "multi_parameter_deterioration", seed=2, n=20):
            r = process_observation(state, obs, audit_logger=audit)
            last_obs = obs
            if r["alert"] is not None:
                break
        apply_clinician_action(state, "accept", now=last_obs.timestamp, reason="looks real", audit_logger=audit)
        records = list(audit.replay())
        clinician_records = [r for r in records if r["event"] == "clinician_action"]
        assert len(clinician_records) == 1
        assert clinician_records[0]["action"] == "accept"

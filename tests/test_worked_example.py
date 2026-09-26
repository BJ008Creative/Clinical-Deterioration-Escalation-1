"""These tests check the *behavioural claims* of Section 12 of the midterm
report (transient noise is ignored, sustained coordinated change escalates,
repeats get suppressed, dismissal doesn't disable monitoring) rather than
its exact illustrative numbers — the report itself calls those numbers
"illustrative", and small NEWS2-band boundary effects mean two independent
implementations will not reproduce them digit-for-digit. Tuning the exact
thresholds/weights in config.py against a larger synthetic cohort is listed
as a Phase-2 task (see Section 13.1 of the report)."""
from vigil.models import Observation, AlertState
from vigil.state.patient_state import PatientState
from vigil.pipeline import process_observation
from vigil.evidence.suppression import apply_clinician_action
from vigil.simulator.streamer import stream_patient


def test_isolated_spike_does_not_reach_escalation():
    state = PatientState(patient_id="P-SPIKE", profile={})
    for obs in stream_patient("P-SPIKE", "transient_spike", seed=3, n=20):
        r = process_observation(state, obs)
        assert r["physio_state"] != "ESCALATION_CANDIDATE"
        assert r["alert"] is None


def test_sustained_multiparameter_change_escalates():
    state = PatientState(patient_id="P-DET", profile={})
    alerted = False
    for obs in stream_patient("P-DET", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs)
        if r["alert"] is not None:
            alerted = True
    assert alerted


def test_repeated_identical_readings_eventually_suppressed():
    state = PatientState(patient_id="P-DUP", profile={})
    last_obs = None
    for obs in stream_patient("P-DUP", "multi_parameter_deterioration", seed=2, n=20):
        process_observation(state, obs)
        last_obs = obs

    # Feed the same reading repeatedly: evidence should settle and the
    # suppression gate should start rejecting duplicates.
    suppressed_seen = False
    for j in range(8):
        obs2 = Observation(patient_id="P-DUP", timestamp=last_obs.timestamp + 60 * (j + 1),
                            HR=last_obs.HR, RR=last_obs.RR, SpO2=last_obs.SpO2, SBP=last_obs.SBP)
        r = process_observation(state, obs2)
        if r["suppressed_reason"] == "duplicate_no_new_evidence":
            suppressed_seen = True
    assert suppressed_seen


def test_dismissal_raises_bar_but_does_not_silence_patient():
    state = PatientState(patient_id="P-DISMISS", profile={})
    last_obs = None
    for obs in stream_patient("P-DISMISS", "multi_parameter_deterioration", seed=2, n=14):
        r = process_observation(state, obs)
        last_obs = obs
    assert state.last_alert is not None

    apply_clinician_action(state, "dismiss", now=last_obs.timestamp, reason="checked, looks ok")
    assert state.alert_state == AlertState.DISMISSED
    bar_after_dismiss = state.dismiss_bar
    assert bar_after_dismiss > 0

    # Further genuine worsening should still be able to clear the raised bar.
    reopened = False
    for step in range(1, 8):
        t = last_obs.timestamp + 60 * step
        obs2 = Observation(patient_id="P-DISMISS", timestamp=t,
                            HR=last_obs.HR + 8 * step, RR=last_obs.RR + 2 * step,
                            SpO2=max(80, last_obs.SpO2 - 1.5 * step), SBP=last_obs.SBP - 4 * step)
        r = process_observation(state, obs2)
        if r["alert"] is not None:
            reopened = True
            break
    assert reopened, "genuinely new deterioration must be able to overcome the dismissal bar"

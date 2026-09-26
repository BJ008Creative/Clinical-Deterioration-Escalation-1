from vigil.models import Observation
from vigil.analysis.severity import severity_score, partial_news2
from vigil.evidence.accumulator import update_evidence
from vigil.state.patient_state import PatientState
from vigil.pipeline import process_observation


def test_severity_zero_for_normal_vitals():
    obs = Observation(patient_id="X", timestamp=0, HR=75, RR=16, SpO2=98, SBP=118)
    assert partial_news2(obs) == 0
    assert severity_score(obs) == 0.0


def test_severity_high_for_critical_vitals():
    obs = Observation(patient_id="X", timestamp=0, HR=140, RR=5, SpO2=88, SBP=85)
    assert partial_news2(obs) == 12  # every channel maxes out


def test_evidence_decays_without_new_signal():
    e1 = update_evidence(0.0, severity=0.5, momentum=0.5, coherence=0.5, persistence=0.5)
    e2 = update_evidence(e1, severity=0.0, momentum=0.0, coherence=0.0, persistence=0.0)
    assert e2 < e1  # decay + no new drive should shrink it


def test_stable_scenario_never_escalates():
    from vigil.simulator.streamer import stream_patient
    state = PatientState(patient_id="P-STABLE", profile={})
    for obs in stream_patient("P-STABLE", "stable", seed=1, n=30):
        result = process_observation(state, obs)
        assert result["physio_state"] in ("STABLE", "EMERGING_CHANGE")
        assert result["alert"] is None


def test_multi_parameter_deterioration_eventually_escalates():
    from vigil.simulator.streamer import stream_patient
    state = PatientState(patient_id="P-DETERIORATE", profile={})
    alerted = False
    for obs in stream_patient("P-DETERIORATE", "multi_parameter_deterioration", seed=2, n=20):
        result = process_observation(state, obs)
        if result["alert"] is not None:
            alerted = True
    assert alerted

from vigil.models import AlertState
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation
from vigil.evidence.suppression import apply_clinician_action
from vigil.evidence.commands import run_command, DismissCommand


def _state_with_first_alert(pid="P-CMD", scenario="multi_parameter_deterioration", seed=2):
    state = PatientState(patient_id=pid, profile={})
    last_obs = None
    for obs in stream_patient(pid, scenario, seed=seed, n=20):
        r = process_observation(state, obs)
        last_obs = obs
        if r["alert"] is not None:
            break
    return state, last_obs


def test_apply_clinician_action_still_works_unchanged_for_existing_callers():
    """Backward compatibility: old call signature, old behaviour."""
    state, last_obs = _state_with_first_alert()
    apply_clinician_action(state, "dismiss", now=last_obs.timestamp, reason="checked")
    assert state.alert_state == AlertState.DISMISSED
    assert state.dismiss_bar > 0
    assert state.command_history[-1]["command"] == "dismiss"


def test_command_object_supports_undo():
    state, last_obs = _state_with_first_alert()
    command = run_command(state, "dismiss", now=last_obs.timestamp, reason="misclick")
    assert isinstance(command, DismissCommand)
    assert state.alert_state == AlertState.DISMISSED
    bar_after = state.dismiss_bar

    command.undo()
    assert state.alert_state == AlertState.OPEN  # restored to what it was before dismiss
    assert state.dismiss_bar < bar_after


def test_command_history_records_every_action_in_order():
    state, last_obs = _state_with_first_alert()
    run_command(state, "investigate", now=last_obs.timestamp)
    run_command(state, "accept", now=last_obs.timestamp + 60)
    assert [c["command"] for c in state.command_history] == ["investigate", "accept"]


def test_unknown_command_raises():
    state, last_obs = _state_with_first_alert()
    try:
        run_command(state, "not_a_real_action", now=last_obs.timestamp)
        assert False, "should have raised"
    except ValueError:
        pass

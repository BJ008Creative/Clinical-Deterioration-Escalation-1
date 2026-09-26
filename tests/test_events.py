import tempfile
from pathlib import Path

from vigil.agents.events import EventBus, audit_subscriber
from vigil.audit.sqlite_logger import SQLiteAuditLogger
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation


def test_event_bus_delivers_to_multiple_subscribers():
    bus = EventBus()
    received_a, received_b = [], []
    bus.subscribe("alert", lambda event_type, **f: received_a.append(f))
    bus.subscribe_all(lambda event_type, **f: received_b.append(event_type))

    state = PatientState(patient_id="P-EVT", profile={})
    for obs in stream_patient("P-EVT", "multi_parameter_deterioration", seed=2, n=15):
        process_observation(state, obs, event_bus=bus)

    assert len(received_a) >= 1  # at least one "alert" event reached its specific subscriber
    assert "observation" in received_b  # the catch-all subscriber saw everything, including ticks


def test_audit_subscriber_forwards_events_to_the_logger_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        audit = SQLiteAuditLogger(str(Path(tmp) / "audit.db"))
        bus = EventBus()
        bus.subscribe_all(audit_subscriber(audit))

        state = PatientState(patient_id="P-EVT2", profile={})
        n_ticks = 0
        for obs in stream_patient("P-EVT2", "multi_parameter_deterioration", seed=2, n=12):
            process_observation(state, obs, event_bus=bus)
            n_ticks += 1

        obs_records = list(audit.replay(event="observation"))
        assert len(obs_records) == n_ticks  # same completeness guarantee as the direct audit_logger path
        audit.close()


def test_event_bus_and_direct_audit_logger_are_mutually_exclusive_per_call():
    """When both are passed, event_bus wins (documented in _emit) -- this
    just confirms that behaviour rather than silently double-logging."""
    with tempfile.TemporaryDirectory() as tmp:
        audit = SQLiteAuditLogger(str(Path(tmp) / "audit.db"))
        bus = EventBus()
        bus_events = []
        bus.subscribe_all(lambda event_type, **f: bus_events.append(event_type))

        state = PatientState(patient_id="P-EVT3", profile={})
        for obs in stream_patient("P-EVT3", "stable", seed=1, n=5):
            process_observation(state, obs, audit_logger=audit, event_bus=bus)

        assert len(bus_events) == 5
        assert list(audit.replay()) == []  # audit_logger was NOT called directly since event_bus took priority
        audit.close()

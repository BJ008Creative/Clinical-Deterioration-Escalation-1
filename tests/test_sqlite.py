import tempfile
from pathlib import Path

from vigil.audit.sqlite_logger import SQLiteAuditLogger
from vigil.state.repository import SQLitePatientRepository
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation


def test_sqlite_audit_logger_logs_and_replays():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "audit.db")
        audit = SQLiteAuditLogger(db_path)
        state = PatientState(patient_id="P-SQL", profile={})
        n_ticks = 0
        for obs in stream_patient("P-SQL", "multi_parameter_deterioration", seed=2, n=15):
            process_observation(state, obs, audit_logger=audit)
            n_ticks += 1

        all_records = list(audit.replay())
        assert len(all_records) >= n_ticks  # at least one "observation" event per tick

        obs_records = list(audit.replay(patient_id="P-SQL", event="observation"))
        assert len(obs_records) == n_ticks

        alert_records = list(audit.replay(event="alert"))
        assert len(alert_records) >= 1
        audit.close()


def test_sqlite_audit_logger_jsonl_export_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "audit.db")
        jsonl_path = str(Path(tmp) / "export.jsonl")
        audit = SQLiteAuditLogger(db_path)
        state = PatientState(patient_id="P-EXPORT", profile={})
        for obs in stream_patient("P-EXPORT", "stable", seed=1, n=5):
            process_observation(state, obs, audit_logger=audit)
        audit.export_jsonl(jsonl_path)
        lines = Path(jsonl_path).read_text().strip().split("\n")
        assert len(lines) == 5
        audit.close()


def test_sqlite_state_repository_snapshot_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "state.db")
        repo = SQLitePatientRepository(db_path)
        state = PatientState(patient_id="P-REPO", profile={"age": 55})
        for obs in stream_patient("P-REPO", "multi_parameter_deterioration", seed=2, n=10):
            process_observation(state, obs)
            repo.save_snapshot(state, obs.timestamp)

        latest = repo.load_latest("P-REPO")
        assert latest is not None
        assert latest["patient_id"] == "P-REPO"
        assert latest["physiological_state"] == state.physio_state.value
        assert latest["profile"]["age"] == 55
        repo.close()


def test_sqlite_state_repository_load_all_latest_covers_full_cohort():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "state.db")
        repo = SQLitePatientRepository(db_path)
        for pid, scenario in [("P-1", "stable"), ("P-2", "multi_parameter_deterioration")]:
            state = PatientState(patient_id=pid, profile={})
            for obs in stream_patient(pid, scenario, seed=3, n=8):
                process_observation(state, obs)
                repo.save_snapshot(state, obs.timestamp)
        rows = repo.load_all_latest()
        assert {r["patient_id"] for r in rows} == {"P-1", "P-2"}
        repo.close()

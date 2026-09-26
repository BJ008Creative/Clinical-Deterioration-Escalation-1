import tempfile
from pathlib import Path

from vigil.audit.logger import AuditLogger
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation
from vigil.evaluation.pandas_analysis import (
    load_audit_log_as_dataframe, alerts_per_patient, suppression_reason_breakdown,
    evidence_trajectory_stats, cohort_summary_report,
)


def _build_log(tmp_dir):
    log_path = str(Path(tmp_dir) / "audit.jsonl")
    audit = AuditLogger(log_path)
    for pid, scenario, n in [("P-1", "multi_parameter_deterioration", 20),
                              ("P-2", "stable", 20),
                              ("P-3", "post_escalation_persistence", 26)]:
        state = PatientState(patient_id=pid, profile={})
        for obs in stream_patient(pid, scenario, seed=2, n=n):
            process_observation(state, obs, audit_logger=audit)
    return log_path


def test_load_audit_log_as_dataframe_flattens_nested_features():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = _build_log(tmp)
        df = load_audit_log_as_dataframe(log_path)
        assert not df.empty
        assert "features.evidence" in df.columns
        assert (df["event"] == "observation").sum() == 20 + 20 + 26


def test_alerts_per_patient_counts_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = _build_log(tmp)
        df = load_audit_log_as_dataframe(log_path)
        counts = alerts_per_patient(df)
        assert "P-2" not in counts.index  # stable patient never alerts
        assert counts.get("P-1", 0) > 0


def test_suppression_reason_breakdown_has_expected_reasons():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = _build_log(tmp)
        df = load_audit_log_as_dataframe(log_path)
        reasons = suppression_reason_breakdown(df)
        # post_escalation_persistence should generate real duplicate suppressions
        assert reasons.get("duplicate_no_new_evidence", 0) > 0


def test_evidence_trajectory_stats_are_numerically_sane():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = _build_log(tmp)
        df = load_audit_log_as_dataframe(log_path)
        stats = evidence_trajectory_stats(df, "P-1")
        assert stats["n_observations"] == 20
        assert stats["evidence_peak"] >= stats["evidence_mean"] >= 0
        assert stats["evidence_std"] >= 0

        stable_stats = evidence_trajectory_stats(df, "P-2")
        assert stable_stats["evidence_mean"] == 0.0  # genuinely stable patient never accrues evidence


def test_cohort_summary_report_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = _build_log(tmp)
        report = cohort_summary_report(log_path)
        assert report["n_patients"] == 3
        assert report["total_alerts"] > 0
        assert len(report["evidence_trajectories"]) == 3

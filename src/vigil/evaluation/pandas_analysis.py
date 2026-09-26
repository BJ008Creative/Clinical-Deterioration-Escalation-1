"""Section 5.3 lists NumPy/pandas in the tech stack, but nothing in the
codebase actually imported them (dataclasses + plain Python were enough for
the core pipeline). This module is real, tested usage: turning an audit log
into tabular summaries a clinician-facing report or the demo video could
show, using pandas for the grouping/aggregation and numpy for the numeric
work — exactly the kind of post-hoc analysis these libraries are for.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_audit_log_as_dataframe(jsonl_path: str) -> pd.DataFrame:
    """Reads a JSONL audit log (audit/logger.py's format) into a flat
    DataFrame, one row per event. Nested dicts (e.g. `features`) are
    flattened with pandas.json_normalize so severity/momentum/etc. become
    their own columns."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep=".")


def observation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Just the 'observation' events, with feature columns pulled to the top
    level for easy plotting/inspection."""
    obs = df[df["event"] == "observation"].copy()
    if obs.empty:
        return obs
    for col in ("features.severity", "features.momentum", "features.coherence",
                "features.persistence", "features.evidence", "features.partial_news2"):
        if col in obs.columns:
            obs[col.split(".")[1]] = obs[col]
    return obs


def alerts_per_patient(df: pd.DataFrame) -> pd.Series:
    """How many alerts each patient generated — the first thing a clinician
    lead would ask when reviewing a shift."""
    alerts = df[df["event"] == "alert"]
    if alerts.empty:
        return pd.Series(dtype=int)
    return alerts.groupby("patient_id").size().sort_values(ascending=False)


def evidence_trajectory_stats(df: pd.DataFrame, patient_id: str) -> dict:
    """NumPy-based summary of one patient's evidence trace: peak, mean,
    and the simulated-time-to-peak — the numbers behind a 'how did this
    patient's evidence evolve' chart."""
    obs = observation_dataframe(df)
    patient_obs = obs[obs["patient_id"] == patient_id].sort_values("timestamp")
    if patient_obs.empty or "evidence" not in patient_obs.columns:
        return {"patient_id": patient_id, "n_observations": 0}
    evidence = patient_obs["evidence"].to_numpy(dtype=float)
    timestamps = patient_obs["timestamp"].to_numpy(dtype=float)
    peak_idx = int(np.argmax(evidence))
    return {
        "patient_id": patient_id,
        "n_observations": len(evidence),
        "evidence_mean": float(np.mean(evidence)),
        "evidence_std": float(np.std(evidence)),
        "evidence_peak": float(evidence[peak_idx]),
        "time_to_peak_seconds": float(timestamps[peak_idx] - timestamps[0]),
    }


def suppression_reason_breakdown(df: pd.DataFrame) -> pd.Series:
    """Counts each suppression reason across the whole log — the evidence
    behind claims like '39% of candidate ticks were correctly suppressed as
    duplicates' in the evaluation report."""
    suppressed = df[df["event"] == "suppressed"]
    if suppressed.empty or "reason" not in suppressed.columns:
        return pd.Series(dtype=int)
    return suppressed["reason"].value_counts()


def cohort_summary_report(jsonl_path: str) -> dict:
    """Ties the above together into one dict — what scripts/run_demo.py or
    a notebook would print/save after a run."""
    df = load_audit_log_as_dataframe(jsonl_path)
    if df.empty:
        return {"n_events": 0}
    alerts = alerts_per_patient(df)
    suppression = suppression_reason_breakdown(df)
    patients = sorted(df["patient_id"].dropna().unique())
    trajectories = [evidence_trajectory_stats(df, pid) for pid in patients]
    return {
        "n_events": int(len(df)),
        "n_patients": len(patients),
        "alerts_per_patient": alerts.to_dict(),
        "total_alerts": int(alerts.sum()) if not alerts.empty else 0,
        "suppression_reason_counts": suppression.to_dict(),
        "evidence_trajectories": trajectories,
    }

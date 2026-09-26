#!/usr/bin/env python3
"""Section 13: runs the synthetic evaluation cohort through VIGIL and both
baselines, then reports every Section 13.2 metric. This is a real
measurement, not a mockup — every number printed comes from an actual run
of the code in src/vigil.

Usage:
    PYTHONPATH=src python3 evaluation/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation
from vigil.evidence.suppression import apply_clinician_action
from vigil.models import AlertState
from vigil.ranking.cohort_ranker import rank_cohort
from vigil.evaluation.synthetic_cohort import generate_cohort, generate_long_negative_cohort
from vigil.evaluation.baselines import run_b1, run_b2
from vigil.evaluation import metrics as M


def run_vigil_for_patient(entry):
    state = PatientState(patient_id=entry["patient_id"], profile=entry["profile"])
    obs_list = list(stream_patient(entry["patient_id"], entry["scenario"], seed=entry["seed"], n=entry["n_ticks"]))
    ticks = []
    alerts = []
    for i, obs in enumerate(obs_list):
        r = process_observation(state, obs)
        ticks.append({"tick": i, "alert": r["alert"] is not None, "suppressed_reason": r["suppressed_reason"]})
        if r["alert"] is not None:
            alerts.append(r["alert"])
    return obs_list, ticks, alerts, state


def run_baselines_for_patient(entry, obs_list):
    b1 = run_b1(obs_list)
    b2 = run_b2(obs_list, profile=entry["profile"])
    return b1, b2


def feedback_handling_cases():
    """A fixed set of clinician-action scenarios with an expected outcome
    each — the concrete, automatable form of Section 13.2's 'Feedback
    handling' metric."""
    results = []

    # accept -> ACKNOWLEDGED
    state = PatientState(patient_id="FH-1", profile={})
    for obs in stream_patient("FH-1", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs)
        if r["alert"]:
            break
    apply_clinician_action(state, "accept", now=obs.timestamp)
    results.append(state.alert_state == AlertState.ACKNOWLEDGED)

    # dismiss -> DISMISSED, and bar raised (dismiss_bar > 0)
    state = PatientState(patient_id="FH-2", profile={})
    for obs in stream_patient("FH-2", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs)
        if r["alert"]:
            break
    apply_clinician_action(state, "dismiss", now=obs.timestamp)
    results.append(state.alert_state == AlertState.DISMISSED and state.dismiss_bar > 0)

    # defer -> DEFERRED, and re-suppressed on a near-identical next reading
    state = PatientState(patient_id="FH-3", profile={})
    last_obs = None
    for obs in stream_patient("FH-3", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs)
        last_obs = obs
        if r["alert"]:
            break
    apply_clinician_action(state, "defer", now=last_obs.timestamp)
    from vigil.models import Observation
    obs2 = Observation(patient_id="FH-3", timestamp=last_obs.timestamp + 60,
                        HR=last_obs.HR, RR=last_obs.RR, SpO2=last_obs.SpO2, SBP=last_obs.SBP)
    r2 = process_observation(state, obs2)
    results.append(state.alert_state == AlertState.DEFERRED and r2["alert"] is None)

    # investigate -> investigation_mode True, stays OPEN
    state = PatientState(patient_id="FH-4", profile={})
    for obs in stream_patient("FH-4", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs)
        if r["alert"]:
            break
    apply_clinician_action(state, "investigate", now=obs.timestamp)
    results.append(state.investigation_mode is True and state.alert_state == AlertState.OPEN)

    return results


def main():
    cohort = generate_cohort(n_per_scenario=6)
    print(f"Evaluation cohort: {len(cohort)} synthetic patients across {len(set(e['scenario'] for e in cohort))} scenario types\n")

    patient_results = []
    all_vigil_ticks = []
    all_alerts = []
    b1_total_alerts = 0
    b2_total_alerts = 0
    vigil_total_alerts = 0

    for entry in cohort:
        obs_list, ticks, alerts, state = run_vigil_for_patient(entry)
        b1, b2 = run_baselines_for_patient(entry, obs_list)

        all_vigil_ticks.extend(ticks)
        all_alerts.extend(alerts)
        vigil_total_alerts += len(alerts)
        b1_total_alerts += sum(1 for r in b1 if r["alert"])
        b2_total_alerts += sum(1 for r in b2 if r["alert"])

        patient_results.append({
            "patient_id": entry["patient_id"], "scenario": entry["scenario"],
            "onset_tick": entry["onset_tick"], "n_ticks": entry["n_ticks"],
            "vigil_fired": [t["tick"] for t in ticks if t["alert"]],
            "b1_fired": [i for i, r in enumerate(b1) if r["alert"]],
            "b2_fired": [i for i, r in enumerate(b2) if r["alert"]],
        })

    # --- Prioritisation quality: rank a same-length subset of the cohort
    # concurrently at a mid-run snapshot tick. Stratified across scenario
    # types (not just the first N patients) so the subset actually contains
    # a mix of true-deteriorating and true-negative patients — otherwise
    # NDCG/precision are computed against an all-zero relevance vector and
    # come back as a meaningless 0.0. ---
    from collections import defaultdict
    by_scenario = defaultdict(list)
    for e in cohort:
        if e["n_ticks"] >= 20:
            by_scenario[e["scenario"]].append(e)
    ranking_subset = []
    for scenario_entries in by_scenario.values():
        ranking_subset.extend(scenario_entries[:3])
    states = {e["patient_id"]: PatientState(patient_id=e["patient_id"], profile=e["profile"]) for e in ranking_subset}
    streams = {e["patient_id"]: list(stream_patient(e["patient_id"], e["scenario"], seed=e["seed"], n=e["n_ticks"]))
               for e in ranking_subset}
    relevance = {e["patient_id"]: (1 if e["onset_tick"] is not None else 0) for e in ranking_subset}
    snapshot_ndcgs, snapshot_p5s = [], []
    for tick in range(15):
        rows = []
        for e in ranking_subset:
            pid = e["patient_id"]
            if tick < len(streams[pid]):
                process_observation(states[pid], streams[pid][tick])
            f = states[pid].last_features or {"severity": 0.0, "momentum": 0.0, "coherence": 0.0}
            rows.append({"patient_id": pid, "severity": f.get("severity", 0.0), "momentum": f.get("momentum", 0.0),
                         "coherence": f.get("coherence", 0.0), "evidence": states[pid].evidence,
                         "alert_state": states[pid].alert_state})
        ranked = rank_cohort(rows)
        ranked_pids = [r["patient_id"] for r in ranked]
        if tick >= 8:  # only score once deterioration scenarios have had time to show signal
            snapshot_ndcgs.append(M.ndcg_at_k(ranked_pids, relevance, k=5))
            snapshot_p5s.append(M.precision_at_k(ranked_pids, relevance, k=5))

    ndcg5 = sum(snapshot_ndcgs) / len(snapshot_ndcgs) if snapshot_ndcgs else None
    p5 = sum(snapshot_p5s) / len(snapshot_p5s) if snapshot_p5s else None

    # --- Realistic false-alert rate: a separate, actual-24h-long cohort of
    # stable patients with a few scattered transient spikes/artefacts,
    # rather than extrapolating from the ~20-minute scenarios above. ---
    long_negative_cohort = generate_long_negative_cohort(n_patients=8, n_ticks=1440)
    long_negative_results = []
    for entry in long_negative_cohort:
        obs_list, ticks, alerts, state = run_vigil_for_patient(entry)
        b1, b2 = run_baselines_for_patient(entry, obs_list)
        long_negative_results.append({
            "patient_id": entry["patient_id"], "onset_tick": None, "n_ticks": entry["n_ticks"],
            "vigil_fired": [t["tick"] for t in ticks if t["alert"]],
            "b1_fired": [i for i, r in enumerate(b1) if r["alert"]],
            "b2_fired": [i for i, r in enumerate(b2) if r["alert"]],
        })

    # --- Assemble and print every Section 13.2 metric ---
    report = {
        "cohort_size": len(cohort),
        "detection_sensitivity": {
            "VIGIL": M.detection_sensitivity(patient_results, "vigil_fired"),
            "B1": M.detection_sensitivity(patient_results, "b1_fired"),
            "B2": M.detection_sensitivity(patient_results, "b2_fired"),
        },
        "false_alert_rate_per_patient_per_24h": {
            "VIGIL": M.false_alert_rate_per_24h(patient_results, "vigil_fired"),
            "B1": M.false_alert_rate_per_24h(patient_results, "b1_fired"),
            "B2": M.false_alert_rate_per_24h(patient_results, "b2_fired"),
        },
        "false_alert_rate_per_patient_per_24h_CAVEAT": (
            "The numbers above are extrapolated from ~20-26-simulated-minute "
            "scenarios (~70x scale-up to 24h) and should not be read as a "
            "real clinical rate -- see false_alert_rate_realistic_24h below "
            "for a rate computed on an actual simulated 24h day."
        ),
        "false_alert_rate_realistic_24h": {
            "VIGIL": M.false_alert_rate_per_24h(long_negative_results, "vigil_fired"),
            "B1": M.false_alert_rate_per_24h(long_negative_results, "b1_fired"),
            "B2": M.false_alert_rate_per_24h(long_negative_results, "b2_fired"),
        },
        "total_alerts_fired": {"VIGIL": vigil_total_alerts, "B1": b1_total_alerts, "B2": b2_total_alerts},
        "alert_reduction_vs_B1": M.alert_reduction(b1_total_alerts, vigil_total_alerts),
        "alert_reduction_vs_B2": M.alert_reduction(b2_total_alerts, vigil_total_alerts),
        "duplicate_suppression_rate_VIGIL": M.duplicate_suppression_rate(all_vigil_ticks),
        "lead_time_seconds": {
            "VIGIL": M.lead_time_seconds(patient_results, "vigil_fired"),
            "B1": M.lead_time_seconds(patient_results, "b1_fired"),
            "B2": M.lead_time_seconds(patient_results, "b2_fired"),
        },
        "prioritisation_quality": {"NDCG@5": ndcg5, "precision@5": p5},
        "explanation_grounding_rate": M.explanation_grounding_rate(all_alerts),
        "audit_completeness_rate": M.audit_completeness_rate(all_alerts),
        "feedback_handling_rate": M.feedback_handling_rate(feedback_handling_cases()),
    }

    print(json.dumps(report, indent=2, default=str))

    out_path = Path("evaluation/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Section 13.2's ablation study. Each ablation monkeypatches the specific
module-level constants that mechanism actually reads (Python resolves
module globals at call time, so this works without reimporting anything),
runs a cohort through the real pipeline, measures sensitivity + false-alert
rate + duplicate-suppression rate, then restores the originals before the
next ablation. Every number here comes from an actual run — nothing is
estimated.

This directly investigates the open question from the main evaluation:
VIGIL's sensitivity (86.7%) trails B1's (96.7%) because some `recovery`
episodes resolve before enough evidence accumulates. The "faster_escalation"
and "no_persistence_gate" ablations below test whether loosening the
thresholds recovers that sensitivity, and at what false-alert cost.

Usage:
    PYTHONPATH=src python3 evaluation/ablations.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vigil.evidence.accumulator as accumulator
import vigil.evidence.suppression as suppression
import vigil.state.patient_state as patient_state_mod
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation
from vigil.evaluation.synthetic_cohort import generate_cohort, generate_long_negative_cohort
from vigil.evaluation import metrics as M

N_PER_SCENARIO = 4          # smaller than evaluate.py's 6, to keep ablations fast
N_NEGATIVE_PATIENTS = 4
N_NEGATIVE_TICKS = 600      # 10 simulated hours; enough to see the effect, faster than a full day


def run_cohort_once():
    """Runs the current (possibly monkeypatched) config over a fresh cohort
    and returns the two metrics we're tracking."""
    cohort = generate_cohort(n_per_scenario=N_PER_SCENARIO, base_seed=2000)
    patient_results = []
    all_ticks = []
    for entry in cohort:
        state = PatientState(patient_id=entry["patient_id"], profile=entry["profile"])
        fired = []
        for i, obs in enumerate(stream_patient(entry["patient_id"], entry["scenario"],
                                                 seed=entry["seed"], n=entry["n_ticks"])):
            r = process_observation(state, obs)
            all_ticks.append({"alert": r["alert"] is not None, "suppressed_reason": r["suppressed_reason"]})
            if r["alert"] is not None:
                fired.append(i)
        patient_results.append({"onset_tick": entry["onset_tick"], "n_ticks": entry["n_ticks"], "fired": fired})

    negatives = generate_long_negative_cohort(n_patients=N_NEGATIVE_PATIENTS, n_ticks=N_NEGATIVE_TICKS, base_seed=9000)
    negative_results = []
    for entry in negatives:
        state = PatientState(patient_id=entry["patient_id"], profile=entry["profile"])
        fired = []
        for i, obs in enumerate(stream_patient(entry["patient_id"], entry["scenario"],
                                                 seed=entry["seed"], n=entry["n_ticks"])):
            r = process_observation(state, obs)
            if r["alert"] is not None:
                fired.append(i)
        negative_results.append({"onset_tick": None, "n_ticks": entry["n_ticks"], "fired": fired})

    return {
        "sensitivity": M.detection_sensitivity(patient_results, "fired"),
        "false_alert_rate_per_24h": M.false_alert_rate_per_24h(negative_results, "fired"),
        "duplicate_suppression_rate": M.duplicate_suppression_rate(all_ticks),
        "total_alerts_on_deterioration_cohort": sum(len(p["fired"]) for p in patient_results),
    }


class patched:
    """Context manager: sets module.attr = value, restores the original on exit."""
    def __init__(self, module, attr, value):
        self.module, self.attr, self.value = module, attr, value

    def __enter__(self):
        self.original = getattr(self.module, self.attr)
        setattr(self.module, self.attr, self.value)

    def __exit__(self, *exc):
        setattr(self.module, self.attr, self.original)


def ablation_baseline():
    return run_cohort_once()


def ablation_no_coherence_gate():
    with patched(accumulator, "K_MIN_COHERENT", 1):
        return run_cohort_once()


def ablation_no_persistence_gate():
    with patched(accumulator, "P_MIN", 0.0):
        return run_cohort_once()


def ablation_no_suppression():
    # Trivially low bar: any non-negative evidence increase clears it, so
    # every candidate tick is emitted — isolates suppression's contribution.
    with patched(suppression, "DELTA_NEW_EVIDENCE", -999.0):
        return run_cohort_once()


def ablation_no_cold_start():
    # Huge N0 makes the personal-baseline weight ~0 forever: every patient
    # is scored against the population prior only, never their own history.
    with patched(patient_state_mod, "N0_COLD_START", 10 ** 9):
        return run_cohort_once()


def ablation_faster_escalation():
    """Investigates the sensitivity gap: lower THETA_CAND and a shorter
    persistence window so genuine-but-brief episodes (like `recovery`) have
    less time/evidence to accumulate before being recognised."""
    with patched(accumulator, "THETA_CAND", 0.7), \
         patched(patient_state_mod, "PERSISTENCE_WINDOW", 3):
        return run_cohort_once()


ABLATIONS = {
    "baseline (current config)": ablation_baseline,
    "no_coherence_gate (K_MIN_COHERENT=1)": ablation_no_coherence_gate,
    "no_persistence_gate (P_MIN=0)": ablation_no_persistence_gate,
    "no_suppression (bar disabled)": ablation_no_suppression,
    "no_cold_start (population prior only)": ablation_no_cold_start,
    "faster_escalation (lower THETA_CAND, shorter window)": ablation_faster_escalation,
}


def main():
    results = {}
    for name, fn in ABLATIONS.items():
        print(f"Running: {name} ...")
        results[name] = fn()

    print("\n" + json.dumps(results, indent=2, default=str))

    out_path = Path("evaluation/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

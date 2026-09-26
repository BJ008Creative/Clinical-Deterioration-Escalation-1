"""Section 13.1: 'seeded synthetic cohorts (e.g., 200 patients spanning all
scenarios ... with ground-truth deterioration onset times)'.

data/profiles/patients.json (5 patients) is the small demo cohort for
scripts/run_demo.py. This module generates a larger, purpose-built
evaluation cohort — every implemented scenario type repeated across many
seeds, each carrying its ground-truth onset tick — for evaluate.py and
ablations.py to run baselines and metrics against.
"""
import random

from vigil.simulator.scenarios import SCENARIOS, GROUND_TRUTH_ONSET_TICK

SCENARIO_LENGTHS = {
    "stable": 20, "transient_spike": 20, "motion_artefact": 20,
    "persistent_single_signal_drift": 24, "multi_parameter_deterioration": 20,
    "rapid_deterioration": 16, "recovery": 24,
    "post_escalation_persistence": 26, "post_escalation_worsening": 26,
}

PROFILE_VARIANTS = [
    {"age": 45, "sex": "M", "history": [], "medications": [], "spo2_target_scale": 1},
    {"age": 71, "sex": "F", "history": ["COPD"], "medications": ["bisoprolol"], "spo2_target_scale": 2},
    {"age": 66, "sex": "F", "history": ["hypertension"], "medications": ["amlodipine"], "spo2_target_scale": 1},
    {"age": 58, "sex": "M", "history": ["type 2 diabetes"], "medications": ["metformin"], "spo2_target_scale": 1},
]


def generate_cohort(n_per_scenario: int = 6, base_seed: int = 1000):
    """Returns a list of {patient_id, scenario, seed, profile, n_ticks,
    onset_tick, interval} dicts — everything evaluate.py/ablations.py need."""
    rng = random.Random(base_seed)
    cohort = []
    counter = 0
    for scenario in SCENARIOS:
        if scenario == "realistic_negative_day":
            continue  # too long to mix into the main cohort; see generate_long_negative_cohort
        for _ in range(n_per_scenario):
            counter += 1
            profile = dict(rng.choice(PROFILE_VARIANTS))
            seed = base_seed + counter
            cohort.append({
                "patient_id": f"E-{counter:04d}",
                "scenario": scenario,
                "seed": seed,
                "profile": profile,
                "n_ticks": SCENARIO_LENGTHS[scenario],
                "onset_tick": GROUND_TRUTH_ONSET_TICK[scenario],
                "interval": 60,
            })
    return cohort


def generate_long_negative_cohort(n_patients: int = 8, n_ticks: int = 1440, base_seed: int = 5000):
    """A separate, longer cohort (default 1440 ticks * 60s = a full
    simulated 24h) purely for computing a realistic false-alert-rate — the
    main generate_cohort() patients are only ~20-26 simulated minutes long,
    which makes 'alerts per 24h' computed on them a methodology artifact
    (extrapolating a 20-minute count by ~70x), not a real measurement."""
    rng = random.Random(base_seed)
    cohort = []
    for i in range(n_patients):
        profile = dict(rng.choice(PROFILE_VARIANTS))
        cohort.append({
            "patient_id": f"NEG-{i+1:03d}",
            "scenario": "realistic_negative_day",
            "seed": base_seed + i,
            "profile": profile,
            "n_ticks": n_ticks,
            "onset_tick": None,
            "interval": 60,
        })
    return cohort

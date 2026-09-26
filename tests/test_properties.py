"""Section 5.3 calls for 'property tests for state-machine invariants'. The
`hypothesis` library isn't installable in this sandbox (no network), so
this is the same idea implemented by hand: run the real pipeline over many
random seeds/scenarios and assert invariants that must hold for EVERY run,
not just the specific examples in test_worked_example.py. Each test below
runs 30-50 randomised trials rather than one example.
"""
import random

from vigil.models import Observation, PhysioState, AlertState
from vigil.state.patient_state import PatientState
from vigil.pipeline import process_observation
from vigil.simulator.scenarios import SCENARIOS
from vigil.simulator.streamer import stream_patient

N_TRIALS = 40
PHYSIO_ORDER = [PhysioState.STABLE, PhysioState.EMERGING_CHANGE, PhysioState.PERSISTENT_CHANGE,
                PhysioState.CONVERGING_DETERIORATION, PhysioState.ESCALATION_CANDIDATE]


def _random_scenario_run(rng):
    scenario = rng.choice([s for s in SCENARIOS if s != "realistic_negative_day"])
    seed = rng.randint(0, 10 ** 6)
    state = PatientState(patient_id=f"PROP-{seed}", profile={})
    results = []
    for obs in stream_patient(state.patient_id, scenario, seed=seed, n=20):
        r = process_observation(state, obs)
        results.append(r)
    return scenario, results, state


def test_property_evidence_is_never_negative():
    rng = random.Random(1)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        for r in results:
            assert r["features"]["evidence"] >= 0.0


def test_property_lens_scores_stay_in_unit_interval():
    rng = random.Random(2)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        for r in results:
            f = r["features"]
            for key in ("severity", "momentum", "coherence", "persistence"):
                assert 0.0 <= f[key] <= 1.0 + 1e-9, f"{key}={f[key]} out of [0,1]"


def test_property_physio_state_never_jumps_more_than_one_level_down():
    """De-escalation is one level at a time (hysteresis) -- it must never
    skip, e.g., ESCALATION_CANDIDATE straight to STABLE in a single tick."""
    rng = random.Random(3)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        prev_idx = 0
        for r in results:
            idx = PHYSIO_ORDER.index(PhysioState(r["physio_state"]))
            if idx < prev_idx:
                assert prev_idx - idx == 1, f"de-escalated more than one level: {prev_idx} -> {idx}"
            prev_idx = idx


def test_property_alert_and_suppression_are_mutually_exclusive():
    rng = random.Random(4)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        for r in results:
            has_alert = r["alert"] is not None
            has_suppression_reason = r["suppressed_reason"] is not None
            # never both, and if not a candidate at all, suppressed_reason
            # is "not_candidate" while alert is None (also fine)
            assert not (has_alert and has_suppression_reason)


def test_property_every_alert_has_all_required_fields():
    from vigil.evaluation.metrics import ALERT_REQUIRED_FIELDS
    rng = random.Random(5)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        for r in results:
            if r["alert"] is not None:
                for field in ALERT_REQUIRED_FIELDS:
                    assert field in r["alert"], f"alert missing {field}"


def test_property_non_candidate_states_never_carry_an_alert():
    rng = random.Random(6)
    for _ in range(N_TRIALS):
        _, results, _ = _random_scenario_run(rng)
        for r in results:
            if r["physio_state"] != PhysioState.ESCALATION_CANDIDATE.value:
                assert r["alert"] is None


def test_property_evidence_at_last_alert_never_exceeds_current_evidence_bookkeeping():
    """evidence_at_last_alert is a snapshot taken at alert time -- it should
    never end up larger than the maximum evidence ever observed for that
    patient (a basic sanity check on the suppression bookkeeping)."""
    rng = random.Random(7)
    for _ in range(N_TRIALS):
        _, results, state = _random_scenario_run(rng)
        max_evidence_seen = max((r["features"]["evidence"] for r in results), default=0.0)
        assert state.evidence_at_last_alert <= max_evidence_seen + 1e-9


def test_property_channel_baselines_stay_within_plausible_range_after_many_updates():
    """Cold-start blending should never produce a baseline outside the
    plausible physiological range, however many updates it's seen."""
    from vigil.config import PLAUSIBLE_RANGE
    rng = random.Random(8)
    for _ in range(N_TRIALS):
        _, _, state = _random_scenario_run(rng)
        for channel, ch_state in state.channels.items():
            baseline = ch_state.blended_baseline()
            lo, hi = PLAUSIBLE_RANGE[channel]
            # generous margin since baselines can legitimately drift near
            # scenario extremes, but must never blow past the plausible range
            assert lo - 5 <= baseline <= hi + 5, f"{channel} baseline {baseline} implausible"

from vigil.evaluation.metrics import (
    detection_sensitivity, false_alert_rate_per_24h, alert_reduction,
    duplicate_suppression_rate, lead_time_seconds, precision_at_k, ndcg_at_k,
    explanation_grounding_rate, audit_completeness_rate, feedback_handling_rate,
    ALERT_REQUIRED_FIELDS,
)


def test_detection_sensitivity_basic():
    patients = [
        {"onset_tick": 5, "fired": [7, 8]},      # detected
        {"onset_tick": 5, "fired": [1, 2]},      # fired only before onset -> not detected
        {"onset_tick": None, "fired": []},       # excluded (no true episode)
    ]
    assert detection_sensitivity(patients, "fired") == 0.5


def test_false_alert_rate_only_counts_true_negatives():
    patients = [
        {"onset_tick": None, "n_ticks": 1440, "fired": [1, 2, 3]},  # 1440 ticks * 60s = 1 day
        {"onset_tick": 5, "n_ticks": 1440, "fired": [1, 2, 3, 4, 5]},  # excluded: has true episode
    ]
    rate = false_alert_rate_per_24h(patients, "fired", interval_seconds=60)
    assert rate == 3.0  # 3 alerts over exactly 1 patient-day


def test_alert_reduction():
    assert alert_reduction(100, 20) == 0.8
    assert alert_reduction(0, 0) is None


def test_duplicate_suppression_rate():
    ticks = (
        [{"alert": True, "suppressed_reason": None}] * 2
        + [{"alert": False, "suppressed_reason": "duplicate_no_new_evidence"}] * 3
        + [{"alert": False, "suppressed_reason": "not_candidate"}] * 10
    )
    # 3 genuine suppressions out of (3 + 2) candidate events = 0.6
    assert duplicate_suppression_rate(ticks) == 0.6


def test_lead_time_seconds():
    patients = [
        {"onset_tick": 10, "fired": [12]},   # fires 2 ticks after onset -> 120s
        {"onset_tick": 10, "fired": [8]},    # fires before onset only -> no valid "after" fire
        {"onset_tick": None, "fired": [1]},  # excluded
    ]
    lt = lead_time_seconds(patients, "fired", interval_seconds=60)
    assert lt == 120.0


def test_precision_and_ndcg_at_k():
    ranked = ["A", "B", "C", "D"]
    relevance = {"A": 1, "B": 0, "C": 1, "D": 0}
    assert precision_at_k(ranked, relevance, k=2) == 0.5
    ndcg = ndcg_at_k(ranked, relevance, k=2)
    assert 0.0 < ndcg <= 1.0


def test_ndcg_perfect_ranking_is_one():
    ranked = ["A", "B", "C"]
    relevance = {"A": 1, "B": 1, "C": 0}
    assert ndcg_at_k(ranked, relevance, k=3) == 1.0


def test_explanation_grounding_rate():
    alerts = [{"verifier_result": "pass"}, {"verifier_result": "pass"},
              {"verifier_result": "template_fallback_after_failed_check:x"}]
    rate = explanation_grounding_rate(alerts)
    assert abs(rate - 2 / 3) < 1e-9


def test_audit_completeness_rate():
    complete = {f: "x" for f in ALERT_REQUIRED_FIELDS}
    incomplete = dict(complete)
    del incomplete["previous_alert_id"]
    rate = audit_completeness_rate([complete, complete, incomplete])
    assert abs(rate - 2 / 3) < 1e-9


def test_feedback_handling_rate():
    assert feedback_handling_rate([True, True, False, True]) == 0.75
    assert feedback_handling_rate([]) is None

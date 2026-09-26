"""Section 13.2 metrics, as pure functions over already-computed results.
Kept separate from evaluate.py's simulation loop so each metric can be unit
tested against small, hand-constructed inputs rather than only against a
full cohort run.

Definitions used here (the report gives the metric names, not exact
formulas, so these are our concrete, documented choices):

- detection sensitivity: fraction of patients with a true onset tick where
  the system fires at least once at/after that tick.
- false-alert rate: alerts per patient per 24h, counted only over patients
  whose scenario has NO true deterioration (onset_tick is None).
- alert reduction: (baseline_total - system_total) / baseline_total.
- duplicate-suppression rate: genuine suppressions / (genuine suppressions
  + emitted alerts), where "genuine" excludes ordinary non-candidate ticks.
- lead time: (first on/after-onset fire time - onset time), averaged over
  patients that were detected at all; negative would mean firing before the
  labelled onset tick (i.e. earlier than we call "true" onset).
- prioritisation quality: top-k precision and NDCG@k of the urgency ranking
  against a binary relevance label (1 = scenario has a true deterioration
  episode, 0 = it doesn't), averaged over several ranking snapshots.
- explanation grounding: fraction of alerts whose verifier_result == "pass".
- audit completeness: fraction of alert records containing every field
  needed to reconstruct the decision.
- feedback handling: fraction of a fixed set of clinician-action test cases
  that produced the expected state change.
"""
import math


def detection_sensitivity(patient_results: list, fired_key: str):
    with_onset = [p for p in patient_results if p["onset_tick"] is not None]
    if not with_onset:
        return None
    detected = sum(1 for p in with_onset if any(t >= p["onset_tick"] for t in p[fired_key]))
    return detected / len(with_onset)


def false_alert_rate_per_24h(patient_results: list, fired_key: str, interval_seconds: int = 60):
    negatives = [p for p in patient_results if p["onset_tick"] is None]
    if not negatives:
        return None
    total_alerts = sum(len(p[fired_key]) for p in negatives)
    total_patient_days = sum(p["n_ticks"] * interval_seconds for p in negatives) / 86400
    return total_alerts / total_patient_days if total_patient_days > 0 else None


def alert_reduction(baseline_total: int, system_total: int):
    if baseline_total == 0:
        return None
    return (baseline_total - system_total) / baseline_total


def duplicate_suppression_rate(vigil_ticks: list):
    genuine_suppressions = sum(1 for t in vigil_ticks if t["suppressed_reason"] not in (None, "not_candidate"))
    alerts = sum(1 for t in vigil_ticks if t["alert"])
    total_candidates = genuine_suppressions + alerts
    return genuine_suppressions / total_candidates if total_candidates > 0 else None


def lead_time_seconds(patient_results: list, fired_key: str, interval_seconds: int = 60):
    diffs = []
    for p in patient_results:
        if p["onset_tick"] is None:
            continue
        after = [t for t in p[fired_key] if t >= p["onset_tick"]]
        if after:
            diffs.append((min(after) - p["onset_tick"]) * interval_seconds)
    return sum(diffs) / len(diffs) if diffs else None


def precision_at_k(ranked_pids: list, relevance: dict, k: int):
    top = ranked_pids[:k]
    if not top:
        return None
    return sum(relevance.get(pid, 0) for pid in top) / len(top)


def ndcg_at_k(ranked_pids: list, relevance: dict, k: int):
    dcg = sum((2 ** relevance.get(pid, 0) - 1) / math.log2(i + 2) for i, pid in enumerate(ranked_pids[:k]))
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def explanation_grounding_rate(alert_records: list):
    if not alert_records:
        return None
    passed = sum(1 for a in alert_records if a.get("verifier_result") == "pass")
    return passed / len(alert_records)


ALERT_REQUIRED_FIELDS = ("alert_id", "features", "channels", "explanation",
                          "verifier_result", "retrieved_context", "decision_reason", "previous_alert_id")


def audit_completeness_rate(alert_records: list, required_fields=ALERT_REQUIRED_FIELDS):
    if not alert_records:
        return None
    complete = sum(1 for a in alert_records if all(f in a for f in required_fields))
    return complete / len(alert_records)


def feedback_handling_rate(pass_fail_list: list):
    if not pass_fail_list:
        return None
    return sum(1 for x in pass_fail_list if x) / len(pass_fail_list)

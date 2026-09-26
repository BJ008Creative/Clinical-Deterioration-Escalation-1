"""Section 9.2: the five investigation-agent tools.

The midterm report's tech-stack table explicitly allows "LangGraph (or an
equivalent explicit graph) with tool-calling nodes" — langgraph isn't
installable in this environment, so this is that equivalent: the same five
named tools, callable directly (and callable by graph.py's explicit nodes),
without an LLM required to invoke them for the deterministic escalation
path. An LLM-backed agent would call these exact functions as its tools —
swapping in real tool-calling only changes *who* calls them.
"""
from vigil.ranking.cohort_ranker import rank_cohort


class ToolContext:
    """Bundles what every tool needs: the full in-memory patient store (so
    compare_cohort can see everyone) and the guideline index."""

    def __init__(self, patient_store: dict, guideline_index=None):
        self.patient_store = patient_store
        self.guideline_index = guideline_index

    def get_state(self, patient_id: str) -> dict:
        """Tool 1: rolling window, features, evidence trace."""
        s = self.patient_store[patient_id]
        return {
            "patient_id": patient_id,
            "physiological_state": s.physio_state.value,
            "alert_state": s.alert_state.value,
            "evidence": s.evidence,
            "evidence_at_last_alert": s.evidence_at_last_alert,
            "features": dict(s.last_features),
            "recent_adverse_window": list(s.window),
            "recent_observation_ids": list(getattr(s, "recent_obs_ids", [])),
        }

    def get_profile(self, patient_id: str) -> dict:
        """Tool 2: static history, medications, labs, procedures."""
        return dict(self.patient_store[patient_id].profile)

    def get_alert_history(self, patient_id: str) -> list:
        """Tool 3: previous alerts and clinician decisions."""
        return [dict(a) for a in self.patient_store[patient_id].alert_history]

    def search_guidelines(self, query: str, top_k: int = 2) -> list:
        """Tool 4: retrieval over NEWS2 and protocol text."""
        if self.guideline_index is None:
            return []
        return self.guideline_index.search(query, top_k=top_k)

    def compare_cohort(self, patient_id: str) -> dict:
        """Tool 5: rank and neighbours in the worklist."""
        rows = []
        for pid, s in self.patient_store.items():
            f = s.last_features or {"severity": 0.0, "momentum": 0.0, "coherence": 0.0}
            rows.append({
                "patient_id": pid,
                "severity": f.get("severity", 0.0), "momentum": f.get("momentum", 0.0),
                "coherence": f.get("coherence", 0.0), "evidence": s.evidence,
                "alert_state": s.alert_state,
            })
        ranked = rank_cohort(rows)
        rank = next((i + 1 for i, r in enumerate(ranked) if r["patient_id"] == patient_id), None)
        return {"rank": rank, "cohort_size": len(ranked), "ranked": ranked}

"""
Minimal clinician dashboard. Run with:
    PYTHONPATH=src streamlit run src/vigil/ui/dashboard.py

This is intentionally thin: a ranked worklist + an alert feed + four
clinician action buttons, wired straight into the same `process_observation`
pipeline used by scripts/run_demo.py and the tests. If Streamlit isn't
available in your environment, `scripts/run_demo.py` demonstrates the exact
same pipeline from the terminal with zero extra dependencies.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation
from vigil.ranking.cohort_ranker import rank_cohort
from vigil.audit.logger import AuditLogger
from vigil.evidence.suppression import apply_clinician_action

st.set_page_config(page_title="VIGIL — Clinician Worklist", layout="wide")


@st.cache_resource
def load_guideline_index():
    try:
        from vigil.agents.retrieval import GuidelineIndex
        return GuidelineIndex("data/guidelines")
    except ImportError:
        return None


def _deterministic_seed(patient_id: str) -> int:
    """Python's built-in hash() is randomized per-process for strings
    (PYTHONHASHSEED), so hash(patient_id) would give a different seed every
    time the dashboard restarts — breaking the reproducibility guarantee
    every other part of VIGIL relies on (seeded scenarios, replayable audit
    log). This is deterministic across runs."""
    import zlib
    return zlib.crc32(patient_id.encode()) % 1000


def init_state():
    profiles = json.loads(Path("data/profiles/patients.json").read_text())
    st.session_state.states = {p["patient_id"]: PatientState(patient_id=p["patient_id"], profile=p) for p in profiles}
    st.session_state.streams = {
        p["patient_id"]: list(stream_patient(p["patient_id"], p["scenario"], seed=_deterministic_seed(p["patient_id"]), n=40))
        for p in profiles
    }
    st.session_state.tick = 0
    st.session_state.audit = AuditLogger("data/audit_log.jsonl")
    st.session_state.alerts_feed = []


if "states" not in st.session_state:
    init_state()

st.title("VIGIL — Clinician Worklist")
col_a, col_b = st.columns([1, 5])
with col_a:
    if st.button("Advance tick ▶"):
        guideline_index = load_guideline_index()
        tick = st.session_state.tick
        for pid, obs_list in st.session_state.streams.items():
            if tick >= len(obs_list):
                continue
            obs = obs_list[tick]
            state = st.session_state.states[pid]
            # patient_store=... lets get_alert_history/compare_cohort see the
            # WHOLE cohort, not just this one patient; audit_logger=... means
            # every observation/transition/tool-call/alert is logged, not
            # just alerts (matching what scripts/run_demo.py already does).
            result = process_observation(state, obs, guideline_index=guideline_index,
                                          audit_logger=st.session_state.audit,
                                          patient_store=st.session_state.states)
            if result["alert"] is not None:
                st.session_state.alerts_feed.insert(0, {"patient_id": pid, **result["alert"]})
        st.session_state.tick += 1
    if st.button("Reset"):
        init_state()
with col_b:
    st.caption(f"Simulated tick: {st.session_state.tick}")

rows = []
for pid, state in st.session_state.states.items():
    f = state.last_features or {"severity": 0.0, "momentum": 0.0, "coherence": 0.0}
    rows.append({"patient_id": pid, "severity": f["severity"], "momentum": f["momentum"],
                 "coherence": f["coherence"], "evidence": state.evidence, "alert_state": state.alert_state})
ranked = rank_cohort(rows)

st.subheader("Cohort worklist (by urgency)")
st.table([{"Patient": r["patient_id"], "Urgency": round(r["urgency"], 1),
           "Evidence": round(r["evidence"], 2), "Alert state": r["alert_state"].value}
          for r in ranked])

st.subheader("Alert feed")
for a in st.session_state.alerts_feed[:10]:
    with st.expander(f"{a['patient_id']} — {a['physio_state']} (verifier: {a['verifier_result']})"):
        st.write(a["explanation"])
        c1, c2, c3, c4 = st.columns(4)
        state = st.session_state.states[a["patient_id"]]
        if c1.button("Accept", key=f"accept-{a['patient_id']}-{a['timestamp']}"):
            apply_clinician_action(state, "accept", now=a["timestamp"], audit_logger=st.session_state.audit)
        if c2.button("Dismiss", key=f"dismiss-{a['patient_id']}-{a['timestamp']}"):
            apply_clinician_action(state, "dismiss", now=a["timestamp"], audit_logger=st.session_state.audit)
        if c3.button("Defer", key=f"defer-{a['patient_id']}-{a['timestamp']}"):
            apply_clinician_action(state, "defer", now=a["timestamp"], audit_logger=st.session_state.audit)
        if c4.button("Investigate", key=f"investigate-{a['patient_id']}-{a['timestamp']}"):
            apply_clinician_action(state, "investigate", now=a["timestamp"], audit_logger=st.session_state.audit)

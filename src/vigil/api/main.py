"""Section 5.3: FastAPI + WebSocket service layer over the existing
pipeline.

*** UNTESTED *** fastapi/uvicorn/websockets cannot be installed in this
sandbox (no network access). This file is syntactically checked
(`python3 -m py_compile`) but has never been imported or run. Treat it as a
scaffold to verify in a network-enabled environment, not working code.

Everything it calls (process_observation, ToolContext, rank_cohort,
apply_clinician_action) IS tested elsewhere — only this HTTP/WebSocket
wrapping is unverified.

To actually run and verify this:
    pip install fastapi uvicorn "websockets>=12" httpx
    uvicorn vigil.api.main:app --reload
    # then, separately, verify with FastAPI's TestClient (not just curl):
    #   from fastapi.testclient import TestClient
    #   client = TestClient(app); client.get("/cohort")
"""
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from vigil.state.patient_state import PatientState
from vigil.pipeline import process_observation
from vigil.ranking.cohort_ranker import rank_cohort
from vigil.evidence.commands import run_command
from vigil.models import Observation
from vigil.agents.events import EventBus
from vigil.audit.sqlite_logger import SQLiteAuditLogger

app = FastAPI(title="VIGIL API", version="0.1.0")

# Process-wide in-memory state. A real deployment would put this behind a
# proper session/store layer; for a hackathon demo a single shared cohort
# dict is enough, and it's exactly what scripts/run_demo.py already does.
PATIENT_STORE: dict = {}
AUDIT = SQLiteAuditLogger("data/api_audit.db")
BUS = EventBus()
BUS.subscribe_all(lambda event_type, **fields: AUDIT.log(event_type, **fields))

# WebSocket clients currently connected, to be pushed new events.
_WS_CLIENTS: list = []


class ObservationIn(BaseModel):
    patient_id: str
    timestamp: float
    HR: Optional[float] = None
    RR: Optional[float] = None
    SpO2: Optional[float] = None
    SBP: Optional[float] = None


class ClinicianActionIn(BaseModel):
    patient_id: str
    action: str  # accept | dismiss | defer | investigate
    now: float
    reason: str = ""


def _get_or_create_patient(patient_id: str, profile: dict = None) -> PatientState:
    if patient_id not in PATIENT_STORE:
        PATIENT_STORE[patient_id] = PatientState(patient_id=patient_id, profile=profile or {})
    return PATIENT_STORE[patient_id]


@app.post("/observations")
async def post_observation(payload: ObservationIn):
    """Ingest one observation for one patient and run it through the full
    pipeline (validation -> lenses -> evidence -> suppression -> possibly an
    alert). This is the same process_observation() that every test in
    tests/ already exercises — only the HTTP wrapping is new/unverified."""
    state = _get_or_create_patient(payload.patient_id)
    obs = Observation(patient_id=payload.patient_id, timestamp=payload.timestamp,
                       HR=payload.HR, RR=payload.RR, SpO2=payload.SpO2, SBP=payload.SBP)
    result = process_observation(state, obs, patient_store=PATIENT_STORE, event_bus=BUS)
    for client in list(_WS_CLIENTS):
        try:
            await client.send_json({"type": "tick", "patient_id": payload.patient_id, "result": result})
        except Exception:
            _WS_CLIENTS.remove(client)
    return result


@app.get("/cohort")
async def get_cohort():
    """The ranked worklist — the same rank_cohort() used by scripts/run_demo.py."""
    rows = []
    for pid, state in PATIENT_STORE.items():
        f = state.last_features or {"severity": 0.0, "momentum": 0.0, "coherence": 0.0}
        rows.append({"patient_id": pid, "severity": f.get("severity", 0.0), "momentum": f.get("momentum", 0.0),
                     "coherence": f.get("coherence", 0.0), "evidence": state.evidence,
                     "alert_state": state.alert_state})
    ranked = rank_cohort(rows)
    return [{"patient_id": r["patient_id"], "urgency": r["urgency"], "evidence": r["evidence"],
             "alert_state": r["alert_state"].value} for r in ranked]


@app.get("/patients/{patient_id}/alerts")
async def get_alert_history(patient_id: str):
    if patient_id not in PATIENT_STORE:
        return []
    return PATIENT_STORE[patient_id].alert_history


@app.post("/patients/{patient_id}/actions")
async def post_clinician_action(patient_id: str, payload: ClinicianActionIn):
    """Runs a real Command object (evidence/commands.py) - accept/dismiss/
    defer/investigate - against the patient's state."""
    if patient_id not in PATIENT_STORE:
        return {"error": "unknown patient_id"}
    state = PATIENT_STORE[patient_id]
    command = run_command(state, payload.action, now=payload.now, reason=payload.reason, event_bus=BUS)
    return {"action": command.name, "resulting_alert_state": state.alert_state.value}


@app.websocket("/ws/cohort")
async def websocket_cohort(websocket: WebSocket):
    """Pushes every tick/alert event to connected clients as it happens,
    matching Section 5.3's 'WebSocket for live cohort updates'."""
    await websocket.accept()
    _WS_CLIENTS.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep the connection open; clients don't need to send anything
    except WebSocketDisconnect:
        if websocket in _WS_CLIENTS:
            _WS_CLIENTS.remove(websocket)

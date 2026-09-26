"""Turns scenario rows into a generator of Observation objects. Kept
synchronous for the hackathon build (called from a Streamlit loop or a
pytest); Section 5.3 proposes an asyncio producer for the full system —
wrapping this generator in `async for` is a one-line change when the
FastAPI/WebSocket layer is added."""
from vigil.models import Observation
from vigil.simulator.scenarios import generate


def stream_patient(patient_id: str, scenario_name: str, seed: int = 0, **kwargs):
    for t, hr, rr, spo2, sbp in generate(scenario_name, seed=seed, **kwargs):
        yield Observation(patient_id=patient_id, timestamp=t, HR=hr, RR=rr, SpO2=spo2, SBP=sbp)

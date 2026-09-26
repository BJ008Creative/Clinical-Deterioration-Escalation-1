"""
Section 6 of the midterm report: the stateful patient model.

Everything here updates incrementally (O(1) per reading per patient) —
no module ever rescans the full history of a patient to process one
new observation.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from vigil.config import (
    CHANNELS, ROBUST_SCALE, POPULATION_PRIOR, N0_COLD_START,
    EWMA_ALPHA, PERSISTENCE_WINDOW,
)
from vigil.models import PhysioState, AlertState


@dataclass
class ChannelState:
    """Per-channel incremental statistics for one patient."""
    n_valid: int = 0
    baseline_pop: float = 0.0
    baseline_pat: Optional[float] = None   # personal EW mean, starts unset
    slope: float = 0.0                     # EW slope (momentum raw)
    last_value: Optional[float] = None

    def blended_baseline(self) -> float:
        if self.baseline_pat is None:
            return self.baseline_pop
        w = min(1.0, self.n_valid / N0_COLD_START)
        return (1 - w) * self.baseline_pop + w * self.baseline_pat

    def update(self, value: float):
        if self.baseline_pat is None:
            self.baseline_pat = value
        else:
            # simple EW mean for the personal baseline (slow-moving)
            self.baseline_pat = 0.1 * value + 0.9 * self.baseline_pat
        if self.last_value is not None:
            delta = value - self.last_value
            self.slope = EWMA_ALPHA * delta + (1 - EWMA_ALPHA) * self.slope
        self.last_value = value
        self.n_valid += 1


@dataclass
class PatientState:
    patient_id: str
    profile: dict = field(default_factory=dict)
    channels: dict = field(default_factory=lambda: {c: ChannelState(baseline_pop=POPULATION_PRIOR[c]) for c in CHANNELS})
    window: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE_WINDOW))  # recent per-channel adverse flags
    physio_state: PhysioState = PhysioState.STABLE
    alert_state: AlertState = AlertState.NO_ALERT
    evidence: float = 0.0
    evidence_at_last_alert: float = 0.0
    last_alert: Optional[dict] = None          # summary of last emitted alert
    consecutive_red_flag: int = 0
    dismiss_bar: float = 0.0                   # extra evidence bar after a dismissal
    defer_until: float = -1.0                  # simulated-seconds timestamp
    defer_band_state: Optional[PhysioState] = None  # physio_state snapshotted at the moment of deferral
    defer_news2_band: int = 0                  # NEWS2 risk band (0/1/2) snapshotted at deferral
    investigation_mode: bool = False           # set by "investigate"; widens retrieval + review window
    last_features: dict = field(default_factory=dict)  # most recent severity/momentum/coherence/etc.
    alert_history: list = field(default_factory=list)   # every alert ever raised for this patient (get_alert_history tool)
    command_history: list = field(default_factory=list)  # every clinician Command executed (Section 5.2 Command pattern)
    recent_obs_ids: deque = field(default_factory=lambda: deque(maxlen=PERSISTENCE_WINDOW))  # for triggering_observations in audit records
    history: list = field(default_factory=list)  # brief per-tick trace, for the dashboard/audit


def get_or_create(store: dict, patient_id: str, profile: dict) -> PatientState:
    if patient_id not in store:
        store[patient_id] = PatientState(patient_id=patient_id, profile=profile)
    return store[patient_id]

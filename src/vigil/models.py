"""Shared data structures. Kept dependency-free (plain dataclasses) so every
module — simulator, ingestion, state, analysis, evidence — can import from
here without a cycle. A Pydantic-schema version can wrap these for the API
layer later without changing the core logic."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class Observation:
    patient_id: str
    timestamp: float  # simulated seconds since stream start
    HR: Optional[float]
    RR: Optional[float]
    SpO2: Optional[float]
    SBP: Optional[float]
    quality_flags: dict = field(default_factory=dict)  # channel -> reason string


class PhysioState(str, Enum):
    STABLE = "STABLE"
    EMERGING_CHANGE = "EMERGING_CHANGE"
    PERSISTENT_CHANGE = "PERSISTENT_CHANGE"
    CONVERGING_DETERIORATION = "CONVERGING_DETERIORATION"
    ESCALATION_CANDIDATE = "ESCALATION_CANDIDATE"


class AlertState(str, Enum):
    NO_ALERT = "NO_ALERT"
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DEFERRED = "DEFERRED"
    DISMISSED = "DISMISSED"
    RESOLVED = "RESOLVED"


# NEWS2 partial scoring bands (Table 1 of the midterm report / RCP NEWS2).
# Each entry: (predicate(value) -> bool, score). Checked in order; first hit wins.
def _band(*rules):
    def score(value):
        if value is None:
            return None
        for pred, s in rules:
            if pred(value):
                return s
        return 0
    return score

NEWS2_RR = _band(
    (lambda v: v <= 8, 3),
    (lambda v: 9 <= v <= 11, 1),
    (lambda v: 12 <= v <= 20, 0),
    (lambda v: 21 <= v <= 24, 2),
    (lambda v: v >= 25, 3),
)
NEWS2_SPO2_SCALE1 = _band(
    (lambda v: v <= 91, 3),
    (lambda v: 92 <= v <= 93, 2),
    (lambda v: 94 <= v <= 95, 1),
    (lambda v: v >= 96, 0),
)
# Scale 2 (Table 1's note: "Patients with a target SpO2 scale of 2 (e.g.,
# hypercapnic respiratory failure) use the alternative scale"). Real NEWS2
# Scale 2 has separate rows for "on air" vs "on supplemental oxygen", but
# VIGIL never tracks supplemental-O2 status (Section 7.2: "not available,
# not imputed"), so this implements the room-air row only. Documented
# limitation: a Scale-2 patient who is actually on supplemental oxygen will
# be scored as if on room air.
NEWS2_SPO2_SCALE2_ROOM_AIR = _band(
    (lambda v: v <= 83, 3),
    (lambda v: 84 <= v <= 85, 2),
    (lambda v: 86 <= v <= 87, 1),
    (lambda v: 88 <= v <= 92, 0),
    (lambda v: 93 <= v <= 94, 1),
    (lambda v: 95 <= v <= 96, 2),
    (lambda v: v >= 97, 3),
)
NEWS2_SBP = _band(
    (lambda v: v <= 90, 3),
    (lambda v: 91 <= v <= 100, 2),
    (lambda v: 101 <= v <= 110, 1),
    (lambda v: 111 <= v <= 219, 0),
    (lambda v: v >= 220, 3),
)
NEWS2_HR = _band(
    (lambda v: v <= 40, 3),
    (lambda v: 41 <= v <= 50, 1),
    (lambda v: 51 <= v <= 90, 0),
    (lambda v: 91 <= v <= 110, 1),
    (lambda v: 111 <= v <= 130, 2),
    (lambda v: v >= 131, 3),
)

NEWS2_SCORERS = {"HR": NEWS2_HR, "RR": NEWS2_RR, "SpO2": NEWS2_SPO2_SCALE1, "SBP": NEWS2_SBP}
NEWS2_SCORERS_SCALE2 = {"HR": NEWS2_HR, "RR": NEWS2_RR, "SpO2": NEWS2_SPO2_SCALE2_ROOM_AIR, "SBP": NEWS2_SBP}


def news2_risk_band(partial_news2_total: int) -> int:
    """Section 9.3's own worked example cites the standard NEWS2 risk bands
    (low/medium/high, [NEWS2-3.1]-[NEWS2-3.3]). Used by the defer/snooze
    logic: "a higher severity band reopens it immediately" (Section 8.3) —
    this is the aggregate NEWS2 band, since physio_state itself saturates at
    ESCALATION_CANDIDATE the moment a defer happens and so can never rise
    further on its own."""
    if partial_news2_total >= 7:
        return 2  # high
    if partial_news2_total >= 5:
        return 1  # medium
    return 0  # low


def scorers_for_profile(profile: dict) -> dict:
    if (profile or {}).get("spo2_target_scale") == 2:
        return NEWS2_SCORERS_SCALE2
    return NEWS2_SCORERS

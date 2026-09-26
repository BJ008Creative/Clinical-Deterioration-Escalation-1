"""
Central configuration for VIGIL's deterministic core.

Every threshold used by the physiological lenses, the evidence accumulator
and the suppression gate lives here so the whole system can be re-tuned
from one place, and so ablation experiments (Section 13.1 of the midterm
report) can swap one number without touching module code.
"""
from dataclasses import dataclass, field

CHANNELS = ("HR", "RR", "SpO2", "SBP")

# +1 -> an increase is adverse (HR, RR). -1 -> a decrease is adverse (SpO2, SBP).
ADVERSE_DIRECTION = {"HR": 1, "RR": 1, "SpO2": -1, "SBP": -1}

# Plausible physiological ranges used by the ingestion validator.
PLAUSIBLE_RANGE = {
    "HR": (20, 250),
    "RR": (4, 60),
    "SpO2": (50, 100),
    "SBP": (40, 260),
}

# Robust per-channel scale (approx. scaled MAD), floored at a clinically
# meaningful minimum. Tune these against real/synthetic data later.
ROBUST_SCALE = {"HR": 10.0, "RR": 3.0, "SpO2": 2.0, "SBP": 10.0}

# Population priors used for cold-start baselines (very rough adult defaults;
# a real system would condition on age from the static profile).
POPULATION_PRIOR = {"HR": 75.0, "RR": 16.0, "SpO2": 97.0, "SBP": 120.0}
N0_COLD_START = 8          # readings needed before baseline is "reliable"

EWMA_ALPHA = 0.4            # momentum slope smoothing
PERSISTENCE_WINDOW = 3      # K: last-K-readings window for persistence

TAU_Z = 1.5                 # deviation threshold for "meaningfully abnormal"
TAU_M = 0.15                # slope threshold for "meaningfully moving"
M_MAX = 3.0                 # momentum clip ceiling

# Evidence accumulator
LENS_WEIGHTS = {"severity": 0.35, "momentum": 0.25, "coherence": 0.25, "persistence": 0.15}
EVIDENCE_DECAY = 0.85       # lambda
EVIDENCE_DRIFT = 0.05       # g0, small allowance so idle noise doesn't creep up

# Forward-transition thresholds on accumulated evidence E_t (ordering matters)
THETA_EMERGE = 0.15
THETA_PERSIST = 0.35
THETA_CONV = 0.6
THETA_CAND = 0.7

# Hysteresis: exit (de-escalation) thresholds are lower than entry thresholds
HYSTERESIS_MARGIN = 0.2

K_MIN_COHERENT = 2          # channels required for "coherent" multi-parameter change
P_MIN = 0.5                 # minimum persistence fraction for escalation candidacy

# Novelty-aware suppression
DELTA_NEW_EVIDENCE = 0.25       # Et must exceed Elast by this much to re-alert
DELTA_DISMISS_EXTRA = 0.35      # additional bar raised on clinician dismissal
DEFER_MINUTES = 10              # Td: snooze duration on "defer"

# Red-flag bypass (safety net against over-suppression)
RED_FLAG_NEWS2_MIN = 5          # partial NEWS2 aggregate that alone warrants review
RED_FLAG_SINGLE_SCORE = 3       # a single NEWS2 sub-score this high...
RED_FLAG_CONSEC_READINGS = 2    # ...for this many consecutive valid readings

# Cohort urgency weights (Section 8.4)
URGENCY_WEIGHTS = {"severity": 0.40, "momentum": 0.20, "coherence": 0.20, "evidence": 0.20}
URGENCY_PENDING_BONUS = 10.0

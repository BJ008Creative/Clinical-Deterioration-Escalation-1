"""Section 4.1: seeded synthetic per-patient time series with injected
scenarios. Each scenario returns a list of (t_seconds, HR, RR, SpO2, SBP)
tuples; None means a missing/dropped reading."""
import random


BASELINE = {"HR": 80, "RR": 16, "SpO2": 97, "SBP": 120}


def _noisy(value, sigma, rng):
    return value + rng.gauss(0, sigma)


def stable(rng, n=20, interval=60):
    return [(i * interval, _noisy(BASELINE["HR"], 3, rng), _noisy(BASELINE["RR"], 1, rng),
              min(100, _noisy(BASELINE["SpO2"], 0.5, rng)), _noisy(BASELINE["SBP"], 4, rng))
            for i in range(n)]


def transient_spike(rng, n=20, interval=60, spike_at=5):
    rows = stable(rng, n, interval)
    t, hr, rr, spo2, sbp = rows[spike_at]
    rows[spike_at] = (t, hr + 35, rr, spo2, sbp)  # single-channel spike, resolves next tick
    return rows


def motion_artefact(rng, n=20, interval=60, at=5, span=2):
    rows = stable(rng, n, interval)
    for i in range(at, min(at + span, n)):
        t, hr, rr, spo2, sbp = rows[i]
        rows[i] = (t, None, rr, 40.0, sbp)  # implausible flat SpO2 -> validator will flag it
    return rows


def multi_parameter_deterioration(rng, n=20, interval=60, onset=8):
    """Mirrors the worked example in Section 12: HR/RR up, SpO2/SBP down,
    accelerating after `onset`."""
    rows = []
    hr, rr, spo2, sbp = BASELINE["HR"], BASELINE["RR"], BASELINE["SpO2"], BASELINE["SBP"]
    for i in range(n):
        t = i * interval
        if i >= onset:
            step = i - onset
            hr = BASELINE["HR"] + 6 * step
            rr = BASELINE["RR"] + 1.6 * step
            spo2 = max(85, BASELINE["SpO2"] - 0.9 * step)
            sbp = BASELINE["SBP"] - 3.5 * step
        rows.append((t, _noisy(hr, 2, rng), _noisy(rr, 0.5, rng),
                     _noisy(spo2, 0.3, rng), _noisy(sbp, 2, rng)))
    return rows


def recovery(rng, n=24, interval=60, onset=6, peak=12):
    rows = multi_parameter_deterioration(rng, peak, interval, onset)
    tail = stable(rng, n - peak, interval)
    tail = [(peak * interval + t, hr, rr, spo2, sbp) for t, hr, rr, spo2, sbp in tail]
    return rows + tail


def persistent_single_signal_drift(rng, n=24, interval=60, onset=6):
    """One channel (RR) drifts and stays abnormal; everything else stays
    near baseline, so coherence never reaches k>=2 and the patient should
    rise to at most PERSISTENT_CHANGE, never ESCALATION_CANDIDATE."""
    rows = []
    rr = BASELINE["RR"]
    for i in range(n):
        t = i * interval
        if i >= onset:
            rr = min(24, BASELINE["RR"] + 1.2 * (i - onset))
        rows.append((t, _noisy(BASELINE["HR"], 3, rng), _noisy(rr, 0.5, rng),
                     min(100, _noisy(BASELINE["SpO2"], 0.5, rng)), _noisy(BASELINE["SBP"], 4, rng)))
    return rows


def rapid_deterioration(rng, n=16, interval=60, onset=4):
    """Same shape as multi_parameter_deterioration but much steeper, so
    escalation should be reached in far fewer ticks (tests time-to-escalation)."""
    rows = []
    hr, rr, spo2, sbp = BASELINE["HR"], BASELINE["RR"], BASELINE["SpO2"], BASELINE["SBP"]
    for i in range(n):
        t = i * interval
        if i >= onset:
            step = i - onset
            hr = BASELINE["HR"] + 14 * step
            rr = BASELINE["RR"] + 3.5 * step
            spo2 = max(80, BASELINE["SpO2"] - 2.0 * step)
            sbp = BASELINE["SBP"] - 7.0 * step
        rows.append((t, _noisy(hr, 2, rng), _noisy(rr, 0.5, rng),
                     _noisy(spo2, 0.3, rng), _noisy(sbp, 2, rng)))
    return rows


def post_escalation_persistence(rng, n=26, interval=60, onset=8, plateau_at=14):
    """Reaches escalation, then HOLDS at the same worsened level — duplicates
    should be suppressed (Section 4.1's 'Post-escalation persistence')."""
    rows = multi_parameter_deterioration(rng, plateau_at, interval, onset)
    t0, hr, rr, spo2, sbp = rows[-1]
    for i in range(plateau_at, n):
        t = i * interval
        rows.append((t, _noisy(hr, 1.5, rng), _noisy(rr, 0.4, rng),
                     _noisy(spo2, 0.3, rng), _noisy(sbp, 1.5, rng)))
    return rows


def post_escalation_worsening(rng, n=26, interval=60, onset=8, plateau_at=14):
    """Reaches escalation, then WORSENS further — should produce an updated
    escalation event rather than being suppressed as a duplicate."""
    rows = multi_parameter_deterioration(rng, plateau_at, interval, onset)
    for i in range(plateau_at, n):
        t = i * interval
        step = i - onset
        hr = BASELINE["HR"] + 6 * step
        rr = BASELINE["RR"] + 1.6 * step
        spo2 = max(75, BASELINE["SpO2"] - 0.9 * step)
        sbp = BASELINE["SBP"] - 3.5 * step
        rows.append((t, _noisy(hr, 2, rng), _noisy(rr, 0.5, rng),
                     _noisy(spo2, 0.3, rng), _noisy(sbp, 2, rng)))
    return rows


def realistic_negative_day(rng, n=1440, interval=60, n_spikes=3, n_artefacts=2):
    """A full simulated 24h (n=1440, interval=60s) of a genuinely stable
    patient, with a few transient spikes and motion artefacts scattered
    through the day — the actual point of the false-alert-rate metric is
    'how many times would this wake someone up over a real shift', which a
    20-minute window cannot answer. Still a true negative: no sustained,
    corroborated deterioration ever occurs."""
    rows = list(stable(rng, n, interval))
    for tick in rng.sample(range(10, n - 10), min(n_spikes, n - 20)):
        t, hr, rr, spo2, sbp = rows[tick]
        rows[tick] = (t, hr + 35, rr, spo2, sbp)
    for tick in rng.sample(range(10, n - 10), min(n_artefacts, n - 20)):
        t, hr, rr, spo2, sbp = rows[tick]
        rows[tick] = (t, None, rr, 40.0, sbp)
    return rows


SCENARIOS = {
    "stable": stable,
    "transient_spike": transient_spike,
    "motion_artefact": motion_artefact,
    "persistent_single_signal_drift": persistent_single_signal_drift,
    "multi_parameter_deterioration": multi_parameter_deterioration,
    "rapid_deterioration": rapid_deterioration,
    "recovery": recovery,
    "post_escalation_persistence": post_escalation_persistence,
    "post_escalation_worsening": post_escalation_worsening,
    "realistic_negative_day": realistic_negative_day,
    # "Clinician dismiss/defer/investigate" (Section 4.1's 10th scenario type)
    # is not a distinct vitals trajectory — it's this same deterioration
    # trajectory combined with clinician actions, and is exercised directly
    # in tests/test_fixes.py rather than as a separate generator here.
}

# Ground-truth deterioration onset (simulated seconds), used by the
# evaluation scripts to compute detection sensitivity / lead time. None
# means "no true deterioration episode" (a true negative scenario).
GROUND_TRUTH_ONSET_TICK = {
    "stable": None,
    "transient_spike": None,
    "motion_artefact": None,
    "persistent_single_signal_drift": None,  # single-signal only: never a true multi-parameter episode
    "multi_parameter_deterioration": 8,
    "rapid_deterioration": 4,
    "recovery": 6,
    "post_escalation_persistence": 8,
    "post_escalation_worsening": 8,
    "realistic_negative_day": None,
}


def generate(scenario_name: str, seed: int = 0, **kwargs):
    rng = random.Random(seed)
    return SCENARIOS[scenario_name](rng, **kwargs)

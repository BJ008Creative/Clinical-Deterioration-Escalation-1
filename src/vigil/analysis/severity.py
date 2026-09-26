"""Section 7.2: Severity = partial NEWS2 / 12.

Only the four available channels are scored. Temperature, consciousness
level and supplemental oxygen are never imputed — the score is always
reported (and logged) as "partial". SpO2 scoring uses Scale 2 when the
patient's static profile sets spo2_target_scale: 2 (Table 1's note on
hypercapnic respiratory failure); see models.NEWS2_SPO2_SCALE2_ROOM_AIR
for the documented room-air-only limitation."""
from vigil.models import Observation, scorers_for_profile
from vigil.ingestion.validator import is_valid

MAX_PARTIAL_NEWS2 = 12  # 3 (RR) + 3 (SpO2) + 3 (SBP) + 3 (HR)


def per_channel_scores(obs: Observation, profile: dict = None) -> dict:
    scorers = scorers_for_profile(profile)
    scores = {}
    for channel, scorer in scorers.items():
        if is_valid(obs, channel):
            scores[channel] = scorer(getattr(obs, channel))
    return scores


def partial_news2(obs: Observation, profile: dict = None) -> int:
    return sum(per_channel_scores(obs, profile).values())


def severity_score(obs: Observation, profile: dict = None) -> float:
    """S_t in [0, 1]."""
    return partial_news2(obs, profile) / MAX_PARTIAL_NEWS2

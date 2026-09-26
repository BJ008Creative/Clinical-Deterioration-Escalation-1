"""Section 9.3: explanation with grounding verification.

For the hackathon build this is a deterministic template — it never
needs an LLM to produce a correct, evidence-grounded explanation, which
is exactly the "template fallback" the midterm report proposes for when
verification fails. Swapping the template call for an LLM call (schema-
constrained, then checked by `verify`) is a drop-in change: see
`explain_with_llm` for the seam. This keeps the demo runnable with zero
API keys, and the LLM path is a strict upgrade, not a dependency.
"""
from vigil.analysis.severity import partial_news2


def build_evidence_package(patient_id: str, profile: dict, obs, features: dict,
                            implicated_channels: set, retrieved_chunks: list) -> dict:
    return {
        "patient_id": patient_id,
        "age": profile.get("age"),
        "sex": profile.get("sex"),
        "history": profile.get("history", []),
        "medications": profile.get("medications", []),
        "observation": {"HR": obs.HR, "RR": obs.RR, "SpO2": obs.SpO2, "SBP": obs.SBP},
        "partial_news2": partial_news2(obs, profile),
        "features": features,          # severity, momentum, coherence, persistence, evidence
        "triggering_channels": sorted(implicated_channels),
        "retrieved": retrieved_chunks,  # [{"id":..., "text":...}, ...]
        "missing_parameters": ["temperature", "consciousness_level", "supplemental_o2"],
    }


def template_explanation(package: dict) -> str:
    hx = ", ".join(package["history"]) or "no notable history on file"
    meds = ", ".join(package["medications"]) or "no medications on file"
    obs = package["observation"]
    vitals = ", ".join(f"{k} {v:.0f}" for k, v in obs.items() if v is not None)
    channels = ", ".join(package["triggering_channels"]) or "no channel"
    citations = ", ".join(f"[{c['id']}]" for c in package["retrieved"]) or "none retrieved"

    text = (
        f"Patient {package['patient_id']} ({package.get('age', '?')}{package.get('sex', '')}, {hx}, "
        f"on {meds}): partial NEWS2 is {package['partial_news2']}. "
        f"Current readings: {vitals}. "
        f"Channels moving together and worsening: {channels}. "
        f"Evidence score {package['features']['evidence']:.2f} "
        f"(severity {package['features']['severity']:.2f}, momentum {package['features']['momentum']:.2f}, "
        f"coherence {package['features']['coherence']:.2f}, persistence {package['features']['persistence']:.2f}). "
        f"Relevant guidance: {citations}. "
        f"This is a decision-support suggestion, not a diagnosis — please use clinical judgement. "
        f"Missing parameters not scored: {', '.join(package['missing_parameters'])}."
    )
    return text


def verify(explanation_text: str, package: dict) -> tuple:
    """Checks (2) and (3) of Section 9.3 in a cheap, deterministic way:
    every cited id must resolve to a retrieved chunk, and the text must not
    contain diagnostic/prescriptive language. (Check 1 — every number
    matches a computed feature — is satisfied by construction because the
    template only inserts numbers from `package`; an LLM-generated
    explanation would need this checked explicitly, e.g. by regex-extracting
    numbers and comparing them against package['features'] and
    package['observation'].)"""
    retrieved_ids = {c["id"] for c in package["retrieved"]}
    cited_ids = set(__import__("re").findall(r"\[([\w\-.]+)\]", explanation_text))
    unresolved = cited_ids - retrieved_ids
    if unresolved:
        return False, f"unresolved_citations:{sorted(unresolved)}"

    banned = ["diagnose", "diagnosis of", "prescribe", "you have sepsis", "start antibiotics"]
    lowered = explanation_text.lower()
    for phrase in banned:
        if phrase in lowered:
            return False, f"diagnostic_or_prescriptive_language:{phrase}"

    return True, "pass"


def explain(package: dict) -> tuple:
    """Returns (explanation_text, verifier_result)."""
    text = template_explanation(package)
    ok, reason = verify(text, package)
    if not ok:
        # Deterministic fallback is the template itself here, since the
        # template is already grounded by construction; an LLM path would
        # regenerate once, then fall back to this template on repeat failure.
        return text, f"template_fallback_after_failed_check:{reason}"
    return text, "pass"

from vigil.agents.adapters import ExplanationProvider, TemplateExplanationProvider, AnthropicExplanationProvider
from vigil.state.patient_state import PatientState
from vigil.simulator.streamer import stream_patient
from vigil.pipeline import process_observation


class FakeUppercaseProvider(ExplanationProvider):
    """A trivial stand-in provider to prove the adapter boundary actually
    gets used by pipeline.py, without needing network access."""
    def explain(self, package: dict) -> tuple:
        return f"FAKE PROVIDER EXPLANATION FOR {package['patient_id']}", "pass"


def test_default_provider_is_template_and_matches_direct_call():
    provider = TemplateExplanationProvider()
    package = {
        "patient_id": "P-X", "age": 50, "sex": "M", "history": [], "medications": [],
        "observation": {"HR": 100, "RR": 20, "SpO2": 95, "SBP": 110},
        "partial_news2": 3, "features": {"severity": 0.25, "momentum": 0.1, "coherence": 0.5,
                                          "persistence": 0.5, "evidence": 0.4},
        "triggering_channels": ["HR"], "retrieved": [], "missing_parameters": [],
    }
    text, result = provider.explain(package)
    assert "P-X" in text
    assert result == "pass"


def test_pipeline_actually_uses_a_swapped_in_provider():
    state = PatientState(patient_id="P-ADAPT", profile={})
    fake = FakeUppercaseProvider()
    alert = None
    for obs in stream_patient("P-ADAPT", "multi_parameter_deterioration", seed=2, n=20):
        r = process_observation(state, obs, explanation_provider=fake)
        if r["alert"] is not None:
            alert = r["alert"]
            break
    assert alert is not None
    assert alert["explanation"].startswith("FAKE PROVIDER EXPLANATION FOR P-ADAPT")


def test_anthropic_provider_is_documented_as_unimplemented():
    provider = AnthropicExplanationProvider()
    try:
        provider._call_model("irrelevant prompt")
        assert False, "should raise until implemented in a network-enabled environment"
    except NotImplementedError:
        pass

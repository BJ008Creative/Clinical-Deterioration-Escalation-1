"""Section 5.2's Adapter pattern: 'The LLM and vector store are accessed
through adapters so providers can be swapped.' Previously agents/explain.py
only had the template path with no adapter boundary at all. This defines
that boundary: ExplanationProvider is the interface, TemplateExplanationProvider
is the tested, zero-dependency implementation already used everywhere, and
AnthropicExplanationProvider is the real adapter shape for an LLM-backed
provider - written so a future chat with network/API-key access can finish
it without touching anything else in the codebase.

IMPORTANT: AnthropicExplanationProvider is UNTESTED. It cannot be exercised
in this sandbox (no network, no API key). Do not treat it as working code -
treat it as a scaffold with the right shape.
"""
from vigil.agents.explain import template_explanation, verify


class ExplanationProvider:
    """The adapter interface. Anything implementing explain(package) -> (text, verifier_result)
    can be dropped into pipeline.py's investigation graph unchanged."""

    def explain(self, package: dict) -> tuple:
        raise NotImplementedError


class TemplateExplanationProvider(ExplanationProvider):
    """The tested, dependency-free default (what graph.py currently uses).
    Grounded by construction: it only ever inserts numbers/citations that
    are already in `package`."""

    def explain(self, package: dict) -> tuple:
        text = template_explanation(package)
        ok, reason = verify(text, package)
        if not ok:
            return text, f"template_fallback_after_failed_check:{reason}"
        return text, "pass"


class AnthropicExplanationProvider(ExplanationProvider):
    """UNTESTED SCAFFOLD - requires network + ANTHROPIC_API_KEY, neither of
    which are available in this sandbox. Structure mirrors Section 9.3: ask
    the model for a JSON-schema-constrained explanation grounded in
    `package`, verify it exactly like the template path, and fall back to
    the template on repeated verification failure (never surface an
    unverified explanation to a clinician).

    To finish this in a network-enabled environment:
      pip install anthropic
      set ANTHROPIC_API_KEY
      implement _call_model() below and remove the NotImplementedError.
    """

    def __init__(self, model: str = "claude-sonnet-4-5", max_retries: int = 1):
        self.model = model
        self.max_retries = max_retries
        self.fallback = TemplateExplanationProvider()

    def _build_prompt(self, package: dict) -> str:
        return (
            "You are a clinical decision-support explanation generator. "
            "Using ONLY the facts in this JSON evidence package, write a short, "
            "plain-language explanation of why this patient is being escalated. "
            "Cite guideline chunks by their bracketed id, e.g. [NEWS2-3.2]. "
            "Never state a diagnosis or a treatment instruction. "
            f"Evidence package: {package}"
        )

    def _call_model(self, prompt: str) -> str:
        """Not implemented here - see class docstring. Should call the
        Anthropic Messages API and return the raw text response."""
        raise NotImplementedError(
            "AnthropicExplanationProvider is an untested scaffold - implement "
            "_call_model() in a network-enabled environment before using this."
        )

    def explain(self, package: dict) -> tuple:
        for attempt in range(self.max_retries + 1):
            text = self._call_model(self._build_prompt(package))
            ok, reason = verify(text, package)
            if ok:
                return text, "pass"
        # Never surface an unverified LLM explanation - fall back to the
        # template, which is grounded by construction.
        fallback_text, _ = self.fallback.explain(package)
        return fallback_text, f"llm_fallback_after_failed_verification:{reason}"

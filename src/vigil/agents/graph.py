"""Section 9.1's Figure 4 ('Retrieve profile + guideline -> Explain + verify')
implemented as an explicit node graph, since langgraph cannot be installed
in this environment. The report's own tech-stack table sanctions this:
"LangGraph (or an equivalent explicit graph) with tool-calling nodes" — the
node/edge shape and the five tools are identical; only the execution engine
differs. Each node is a plain function of a shared GraphState, run in a
fixed sequence (no LLM-driven branching is needed for the deterministic
explanation path); an LLM-backed version would add a routing node that
decides which tool to call next.
"""
from dataclasses import dataclass, field

from vigil.agents.explain import build_evidence_package
from vigil.agents.adapters import TemplateExplanationProvider
from vigil.agents.tools import ToolContext


@dataclass
class GraphState:
    patient_id: str
    obs: object
    features: dict
    implicated_channels: set
    tools: ToolContext
    package: dict = None
    explanation: str = None
    verifier_result: str = None
    tool_calls: list = field(default_factory=list)  # audit trail of which tools fired


def node_investigate(state: GraphState) -> GraphState:
    """Calls get_profile + search_guidelines always; calls get_alert_history
    + compare_cohort too when the patient is under active investigation
    (Section 8.3: 'Investigate ... the agent expands retrieval')."""
    patient_state = state.tools.patient_store[state.patient_id]
    profile = state.tools.get_profile(state.patient_id)
    state.tool_calls.append({"tool": "get_profile", "args": {"patient_id": state.patient_id}})

    expanded = patient_state.investigation_mode
    top_k = 5 if expanded else 2
    query_terms = " ".join(sorted(state.implicated_channels)) or "deterioration"
    query = f"{query_terms} NEWS2 escalation"
    retrieved = state.tools.search_guidelines(query, top_k=top_k)
    state.tool_calls.append({"tool": "search_guidelines", "args": {"query": query, "top_k": top_k},
                              "result_ids": [c["id"] for c in retrieved]})

    package = build_evidence_package(state.patient_id, profile, state.obs, state.features,
                                      state.implicated_channels, retrieved)

    if expanded:
        alert_history = state.tools.get_alert_history(state.patient_id)
        state.tool_calls.append({"tool": "get_alert_history", "args": {"patient_id": state.patient_id},
                                  "n_prior_alerts": len(alert_history)})
        cohort = state.tools.compare_cohort(state.patient_id)
        state.tool_calls.append({"tool": "compare_cohort", "args": {"patient_id": state.patient_id},
                                  "rank": cohort["rank"], "cohort_size": cohort["cohort_size"]})
        package["alert_history"] = alert_history
        package["cohort_context"] = {"rank": cohort["rank"], "cohort_size": cohort["cohort_size"]}
        package["expanded_investigation"] = True

    state.package = package
    return state


def node_explain(state: GraphState, provider=None) -> GraphState:
    """Section 5.2's Adapter pattern in action: `provider` defaults to the
    tested TemplateExplanationProvider, but any ExplanationProvider
    (agents/adapters.py) can be passed in instead - e.g. AnthropicExplanationProvider
    once that's implemented in a network-enabled environment."""
    provider = provider or TemplateExplanationProvider()
    text, verifier_result = provider.explain(state.package)
    state.explanation = text
    state.verifier_result = verifier_result
    return state


def run_investigation_graph(patient_id: str, obs, features: dict, implicated_channels: set,
                             tools: ToolContext, explanation_provider=None) -> GraphState:
    """The graph itself: fixed edge investigate -> explain, matching Figure 4."""
    state = GraphState(patient_id=patient_id, obs=obs, features=features,
                        implicated_channels=implicated_channels, tools=tools)
    state = node_investigate(state)
    state = node_explain(state, provider=explanation_provider)
    return state

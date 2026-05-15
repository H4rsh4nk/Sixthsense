# ADR-0002: "Agent" vocabulary — signal generators vs. decision agent

- **Status:** Accepted
- **Date:** 2026-05-14
- **Related:** ADR-0001, `GOAL_ARCHITECTURE.md` §4 (AI Agent Layer)

## Context

`GOAL_ARCHITECTURE.md` §4 enumerates four "AI agents": News, Market, Risk, Macro. The implementation does not have four agents — it has:

- Four / five **signal generators** (`InsiderSignal`, `NewsSignal`, `PoliticalSignal`, `PriceActionSignal`, `MacroSignal`) — these are deterministic pipelines that may *call* an LLM (e.g. `NewsSignal` runs Qwen for sentiment) but they are not agentic.
- One **decision agent** (`src/strategy/llm_agent.py::TradingAgent`) — a tool-using LLM (LiteLLM) invoked optionally by `SignalScorer` to rank candidates.
- One **risk manager** (`src/strategy/risk_manager.py::RiskManager`) — pure rules (drawdown caps, daily loss, circuit breaker). Not an LLM, not an agent.

Calling everything an "agent" muddles the architecture. New contributors and even the goal doc itself end up confused about which components do ReAct-style tool use and which are deterministic.

## Decision

Adopt the following vocabulary and use it consistently across code, docs, and ADRs:

- **Signal generator** — a deterministic-or-LLM-assisted module that emits `SignalResult` rows for the day. May internally call an LLM, but does not run a multi-turn tool-using loop.
- **Decision agent** — `TradingAgent`, a ReAct-style LLM with bounded tool calls. There is exactly **one** decision agent.
- **Risk engine** — `RiskManager`, pure deterministic rules. Not an agent.
- **Macro signal** — stays a signal generator (`MacroSignal`). It produces a regime tilt feature; it is **not** an agent.

Future LLM-driven components (e.g. an explicit "Risk reviewer" LLM that critiques the decision agent's output) may be promoted to "agent" status with their own ADR.

## Consequences

### Positive

- Code, docs, and ADRs use the same words for the same things.
- We can talk about "the agent" without ambiguity.
- The goal doc's "AI Agent Layer" becomes "Signal generators + decision agent + risk engine", which matches reality.

### Negative / accepted trade-offs

- The goal doc and several existing comments need rewriting. Done as part of this ADR's rollout.

### Neutral

- This is purely a naming decision; no runtime behavior changes.

## Alternatives considered

### A — Promote every signal generator to a full ReAct agent

Rejected. It would slow scoring by ~50× (each agent doing tool-use), inflate LLM costs, and add little signal quality over current generators. The bottleneck is not "how rich is the agent loop" — it is "how good is the underlying data."

### B — Keep "agent" loose

Rejected. It is the kind of vocabulary debt that compounds.

## References

- Code: `src/signals/`, `src/strategy/llm_agent.py`, `src/strategy/scorer.py`, `src/strategy/risk_manager.py`
- Docs: `GOAL_ARCHITECTURE.md` §4, `CURRENT_DESIGN_ARCHITECTURE.md` §§4–6

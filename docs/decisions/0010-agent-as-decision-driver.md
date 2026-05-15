# ADR-0010: Agent as the decision driver (dual-mode)

- **Status:** Accepted
- **Date:** 2026-05-15
- **Implementation:** Phase A (`src/agent/tool_registry.py`) + Phase C (`--dry-run` flag in `swing-trader trade`) landed 2026-05-15. Phase B (agent becomes the orchestrator) and Phase D (observability per ADR-0011) still in progress.
- **Related:** ADR-0002 (vocabulary), ADR-0006 (regression detection — needs follow-up note under agent mode), ADR-0011 (budgets / caching / observability)

## Context

Today's pipeline is **push-based with the agent as a passenger**:

1. Phase orchestrator fires every minute.
2. `pre_market_scan` / `market_open_entry` calls every enabled `SignalGenerator.generate_bulk()` regardless of relevance.
3. All `SignalResult` rows land in the DB.
4. `SignalScorer` aggregates them with weighted-sum rules; only *optionally*, if `agent.enabled=true`, the `TradingAgent` re-ranks the rules' output.
5. `OrderManager` enters trades.

In this layout the agent has almost no authority. It can re-order the rules' output but cannot decide which data sources matter today, cannot defer reading expensive sources when they're irrelevant, and cannot do cross-source reasoning ("the political event is bearish for energy, so ignore the bullish insider buys at XOM today"). The rules layer is the brain; the LLM is a cosmetic re-ranker.

We want to invert this. The LLM should be the **central decision-making component** — given a phase trigger, *it* decides what to look at, *it* asks for it, *it* reasons across it, and *it* proposes trades. The existing components (signal generators, risk manager, sizer, broker) become **tools** in a registry that the agent can invoke. This is the standard "agent + tools" pattern.

Two non-negotiables shape the design:

1. **Hard risk gates must stay deterministic.** Drawdown limits, position caps, circuit breakers, daily-loss limits — these are not LLM-overridable. The agent *proposes*; `RiskManager` *vetoes*.
2. **An escape hatch must exist.** If the LLM provider is down, slow, or producing degenerate output, the system must still trade safely (or stop cleanly). The existing rules path provides exactly this; we keep it as a fallback rather than ripping it out.

## Decision

Adopt **dual-mode operation**, where `agent.driver_mode` selects the active decision flow:

- **`agent`** *(new default once stable)* — the LLM agent runs at each phase as the driver. Signal generators are wrapped as tools and called only when the agent requests them.
- **`rules`** — current behavior. Bulk signal generation → `SignalScorer` rules path → trades. Agent is not invoked.
- **`dual`** *(initial default during rollout)* — agent runs first; on agent failure (timeout, parse error, empty proposal, missing required fields) or when `fallback_to_rules=true` and the agent's `decisions == []`, automatically run the rules path. Both attempts persist their trace to `decision_logs`.

### Tool surface (initial)

A `tool_registry` module (new) exposes the following tools to the agent. Each is a thin Python function with a JSON-schema signature suitable for LiteLLM's tool-use interface. All return JSON-serialisable dicts.

**Read tools (free):**

- `get_portfolio_state()` — equity, cash, open positions, exposure by sector, drawdown.
- `get_price_history(ticker, days)` — recent OHLCV from SQLite.
- `compute_indicators(ticker)` — RSI / MACD / volume Z-score on stored prices.
- `get_news_sentiment(ticker, since)` — cached LLM-scored headlines via `NewsSignal`.
- `get_insider_filings(ticker, days)` — cached `InsiderSignal` data.
- `get_political_events(sector, days)` — cached `PoliticalSignal` data.
- `get_macro_regime()` — current `MacroSignal` tilt.
- `query_regression_alerts(days)` — recent rows from `regression_alerts`.
- `get_signal_source_stats()` — per-source win rates (for Kelly mode awareness).
- `get_rules_ranking(as_of_date)` — *runs the rules scorer and returns its output.* The agent can use it as one opinion among many, or defer to it entirely. This is intentional: rules expertise is not discarded; it becomes a callable opinion.

**Write tool (gated):**

- `submit_paper_trade(ticker, direction, size_pct, stop_loss_pct, reasoning)` — proposes one trade. The function:
  1. Asks `RiskManager.can_open_position()` — if denied, return `{accepted: False, reason: …}` to the agent.
  2. Asks `PositionSizer.calculate()` — passes through ADR-0003 Kelly/fixed-risk logic.
  3. Submits via `OrderManager` (or in dry-run mode, logs the intent).
  4. Persists the proposal + outcome to `decision_logs` with `agent_trace`.

### Agent loop contract

Per phase, the agent receives a structured `PhaseContext`:

```python
PhaseContext {
    phase: str               # pre_market | market_open | intraday | post_market
    as_of: datetime
    universe: list[str]      # tickers currently in scope
    portfolio: dict          # snapshot from get_portfolio_state()
    recent_regression_alerts: list[dict]
    budget: { max_tool_calls: int, max_trades: int, max_tokens: int }
}
```

It then runs a bounded ReAct-style loop (existing `TradingAgent` infrastructure) and ends with one of:

- **`emit`** — a list of `submit_paper_trade` invocations (executed in order, each gated by `RiskManager`).
- **`abstain`** — no trades this phase, with a recorded reason.
- **`escalate`** — flag for human review (writes a `regression_alerts` row of type `agent_escalation` and, if Telegram alerts are on, sends a message). Future work: this hooks into ADR-0005's approval flow.

The loop **must terminate** when it hits the tool-call budget; on overrun, treat as `abstain` with `reason="budget_exceeded"`.

### What we keep deterministic

- `RiskManager` veto on every proposed trade (no override).
- `PositionSizer` math (ADR-0003) — agent can request a *size hint* via `size_pct`, but `PositionSizer` clamps to caps.
- Per-phase trade-count cap (`agent.max_trades_per_phase`, default 5) — hard limit, independent of the agent's plan.
- Exchange-calendar gating (ADR-0007) and persisted phase state (ADR-0008) still wrap the orchestrator.
- All side-effects flow through `OrderManager`; dry-run mode (forthcoming) short-circuits broker writes for safe end-to-end testing.

## Consequences

### Positive

- The LLM is genuinely the brain. Cross-source reasoning becomes possible; you can ask "why did the agent not trade today?" and get a coherent trace.
- Tool surface is a clean abstraction. Adding a new data source = adding a tool; the agent figures out when to use it.
- Old rules path stays as a graceful fallback — outages don't take the system down.
- Dry-run + the new tool registry make integration testing tractable (we can hand the agent scripted contexts and watch what it does).

### Negative / accepted trade-offs

- **Non-determinism.** Identical inputs may produce different decisions across runs. Mitigations: low `temperature` (already 0.2), seeded sampling where the provider supports it, full transcript persistence, and treating the agent transcript as a first-class debugging artefact.
- **Cost & latency.** Each phase invokes the LLM and (transitively) several tools. We address this in ADR-0011 (budgets + caching). Without those guardrails, the system is too expensive to run.
- **Backtest / live mismatch.** The backtest engine remains rules-driven for the foreseeable future; live trading is agent-driven. We accept this gap and document it. Building an LLM-replay cache to make backtest agent-aware is **out of scope for this ADR** — a separate ADR if/when we need it.
- **Regression detection loses per-source precision.** `signal_source_stats` win-rates assume each source is a separate pipeline. Under agent driving, the agent fuses sources, so "win rate by source" becomes a noisy proxy. ADR-0006 still works (Sharpe and drift checks are unaffected); a future addendum can tag trades with an agent-declared "primary theme" for theme-level win-rate.
- **The rules path will rot.** If we use `dual` mode for months, rules code keeps being tested in fallback. If we move to `agent` mode by default, rules code becomes shelfware. Acceptable.

### Neutral

- `signal_source_stats` keeps updating from closed trades regardless of which mode wrote them (ADR-0003 / ADR-0006 plumbing unchanged).
- `decision_logs` schema unchanged; `agent_trace` now becomes much richer.

## Alternatives considered

### A — Full agent-only, remove the rules path

Rejected. No escape hatch when the LLM is misbehaving or unreachable; harder to validate the system in isolation. The rules path is cheap to keep around.

### B — Status quo (agent as re-ranker)

Rejected. This is exactly the layout the user asked us to invert. Limits the agent's expressive power and makes it impossible to do source-selective reasoning.

### C — Multi-agent (one agent per signal type)

Rejected — see ADR-0002. The vocabulary keeps "decision agent" singular; multi-agent setups add coordination overhead, debugging complexity, and cost without obviously beating one well-toolkit'd agent at this task.

### D — Compile the LLM's reasoning into rules offline ("policy distillation")

Considered for the future. Today we don't have enough agent transcripts to distil from. Revisit after a few months of agent-driver mode.

## Implementation phases (proposed)

1. **Phase A — tool registry.** Wrap existing components (`InsiderSignal`, `NewsSignal`, etc., plus `RiskManager`, `PositionSizer`, `OrderManager`, regression queries) in a `tools/` package with explicit JSON schemas. **No behaviour change** — just refactor + add tests.
2. **Phase B — `TradingAgent` becomes the driver.** Extend the existing class with the `PhaseContext` input and the abstain/emit/escalate output. Add `agent.driver_mode` config. Wire into `pre_market_scan` and `market_open_entry`.
3. **Phase C — dry-run mode** for `swing-trader trade` (logs intended trades instead of submitting). Lets us watch agent behaviour end-to-end without paper-trade noise.
4. **Phase D — observability** per ADR-0011 (budgets, caching, tool-call log).

Each phase is its own PR / commit chain and is independently revertable.

## References

- Code (existing): `src/strategy/llm_agent.py`, `src/strategy/scorer.py`, `src/signals/*.py`, `src/strategy/risk_manager.py`, `src/strategy/position_sizer.py`
- Code (planned): `src/agent/tool_registry.py`, `src/agent/driver.py`, additions to `src/main.py`
- Docs: `CURRENT_DESIGN_ARCHITECTURE.md` §§4–6, `GOAL_ARCHITECTURE.md` §§4, 6, 8
- External: [Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents), [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)

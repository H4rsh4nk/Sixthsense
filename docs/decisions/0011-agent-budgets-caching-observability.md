# ADR-0011: Agent tool-call budgets, caching, and trace-based observability

- **Status:** Accepted
- **Date:** 2026-05-15
- **Implementation:** Direction accepted; schema-level work (table stubs) is implementation Phase α and lands alongside ADR-0010 Phase A. Phases β–δ (per-call logging, budgets in `TradingAgent`, heartbeat / dashboard surfacing) follow once ADR-0010 Phase B ships.
- **Related:** ADR-0010 (agent as driver), ADR-0006 (regression detection)

## Context

ADR-0010 makes the LLM the central decision driver. Without guardrails, that turns each phase tick into an open-ended LLM session: any number of tool calls, any volume of tokens, any latency. For a system that ticks every minute, this is a fast track to a melted GPU (local Ollama) or a six-figure monthly bill (hosted providers). It also masks regressions — if the agent silently starts making 30 tool calls per phase instead of 5, we should *see* that immediately, not discover it in the bill.

We need three things in place before agent-driver mode ships:

1. **Hard budgets per phase** so a malformed plan can't run away.
2. **Caching for slow tools** so the agent isn't paying RSS-fetch latency on every reasoning step.
3. **Trace-based observability** — every tool call, latency, token, and cost recorded so the dashboard and the heartbeat can show "what is the agent actually doing."

## Decision

### 1. Per-phase budgets

Add `agent.budget` config block:

```yaml
agent:
  driver_mode: dual          # ADR-0010; agent | rules | dual
  budget:
    max_tool_calls: 12       # hard cap; agent abstains if exceeded
    max_response_tokens: 4096
    max_trades_per_phase: 5  # independent of agent's plan
    deadline_seconds: 45     # wall-clock per phase
```

Enforcement points:

- The agent loop counts tool invocations; once `max_tool_calls` is reached, the next assistant turn must be a final answer or `abstain` is forced.
- Token caps are enforced at the LiteLLM call site (`max_tokens` param).
- `deadline_seconds` is enforced with a wall-clock check at the top of each agent iteration; if exceeded, treat as `abstain` with `reason="deadline_exceeded"`.
- `max_trades_per_phase` is enforced *after* the agent finishes, by `OrderManager` (truncates the proposal list rather than failing).

All budget exhaustion events are recorded as `regression_alerts` rows of type `agent_budget` (severity `warning`).

### 2. Caching policy

A new `src/agent/tool_cache.py` module — a small SQLite-backed key→value cache with per-key TTL. Tools that hit external APIs or do LLM work consult the cache first.

| Tool | TTL (live) | TTL (backtest) | Reason |
| --- | --- | --- | --- |
| `get_news_sentiment` | 30 min | per-day key | Headline sentiment is slow to compute (Qwen via Ollama, ~1–3s/headline) |
| `get_insider_filings` | 6 hours | per-day key | SEC EDGAR refresh is slow; data updates daily |
| `get_political_events` | 6 hours | per-day key | Federal Register publishes daily |
| `get_macro_regime` | 1 hour | per-day key | Cheap to compute but called often |
| `get_price_history` | 5 min | no cache | Cheap (SQLite); freshness matters intraday |
| `compute_indicators` | 5 min | no cache | Pure compute on cached prices |
| `get_portfolio_state` | **0** (no cache) | n/a | Must always be live |
| `query_regression_alerts` | 5 min | per-day key | Cheap; staleness OK |
| `get_signal_source_stats` | 15 min | per-day key | Updates only on trade close |
| `get_rules_ranking` | 0 (no cache) | n/a | Already deterministic given inputs |

Cache key = `(tool_name, sha1(canonical_json_args), as_of_date)`. Eviction is lazy: rows whose `expires_at < now()` are deleted on next read of the same key, with a periodic post-market sweep to keep the table small.

### 3. Tool-call log + heartbeat surfacing

New SQLite table:

```sql
CREATE TABLE IF NOT EXISTS tool_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase_id TEXT NOT NULL,          -- UUID per phase invocation; ties calls together
    phase_name TEXT NOT NULL,        -- pre_market | market_open | intraday | post_market
    call_idx INTEGER NOT NULL,       -- 0-based order within the phase
    tool_name TEXT NOT NULL,
    args_hash TEXT,                  -- sha1 of canonicalised args; full args go to JSON column
    args_json TEXT,
    result_summary TEXT,             -- truncated stringification of the result
    cache_hit INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tool_call_phase ON tool_call_log(phase_id);
CREATE INDEX IF NOT EXISTS idx_tool_call_time ON tool_call_log(created_at);
```

Every tool invocation writes one row, regardless of cache hit/miss. `decision_logs.agent_trace` still carries the full ReAct transcript (LLM reasoning + tool sequence) but `tool_call_log` is the structured / queryable view used by:

- The Streamlit dashboard ("Agent activity" tab — calls per phase, top tools, p95 latency, cumulative daily cost).
- The 15-minute heartbeat — adds a one-line summary: `agent: 3 phases, 27 calls, $0.12, p95 1.4s`.
- ADR-0006 regression detection — a new check class `agent_volume` watches the rolling 7-day call-count distribution and alerts on outliers (e.g., a phase that did 11 tool calls when the daily mean is 4 ± 1).

### 4. LiteLLM cost callback

LiteLLM exposes `success_callback` / `failure_callback` hooks that surface per-call cost in USD. We register a callback that pushes `cost_usd`, `tokens_in`, `tokens_out`, `latency_ms` into a thread-local context which the tool-registry layer reads when writing the `tool_call_log` row. For local providers (Ollama) cost is 0 but tokens and latency are still recorded.

## Consequences

### Positive

- The agent cannot quietly become expensive — budget caps make cost predictable and visible.
- Caching keeps slow tools from dominating phase latency without forcing the agent to "remember" what it already fetched.
- `tool_call_log` is the structured observability surface we'd otherwise have to reverse-engineer from `agent_trace` JSON blobs.
- Heartbeat-level cost visibility means "is the agent OK today" is answerable in one glance.

### Negative / accepted trade-offs

- Cache staleness can hurt agent quality on fast-moving headlines. We size TTLs conservatively for live mode (30 min on news) — short enough to track the same trading day, long enough to be useful.
- One more SQLite table to maintain. `tool_call_log` will grow at roughly (phases × tools-per-phase) per day; with `dual` mode at 4 phases × 12 calls = ~48 rows/day, well under any concern. We add a rolling-30-day cleanup to the post-market sweep.
- The cost callback couples us a little more tightly to LiteLLM's API. If we ever swap providers, the callback signature has to be re-implemented. Acceptable — LiteLLM is our chosen provider abstraction.

### Neutral

- Cost in Ollama mode is structurally zero; the column is still useful when we switch to a hosted provider or A/B with one.

## Alternatives considered

### A — No budgets; rely on the provider's rate-limits

Rejected. Provider rate limits are global, not per-phase; they protect the provider, not us. Without our own caps the agent can spend its full daily budget on a single phase.

### B — Heavy caching with long TTLs (e.g., 4 hours on news)

Rejected for live mode. Headlines that move stocks are usually <2 hours old. We pick 30 min as a compromise.

### C — No tool_call_log; just dump everything into `decision_logs.agent_trace`

Rejected. The JSON blob is fine for human reading after the fact but useless for aggregation queries ("how many calls did `get_news_sentiment` get this week?"). We want both — the trace for fidelity, the log for analytics.

### D — Build cost tracking only when we move off Ollama

Rejected. Building observability after a regression hits is too late. The hooks are cheap to add now and the column starts collecting useful latency / token data from day one even if cost is 0.

## Implementation phases

1. **Phase α — schema + cache module.** Add `tool_call_log`, write the cache helper, no behaviour change yet.
2. **Phase β — wire tools to log every call** (still on the rules path; tools just become a wrapper layer around existing components per ADR-0010 Phase A).
3. **Phase γ — budgets in `TradingAgent`** — added when ADR-0010 Phase B lands.
4. **Phase δ — heartbeat + dashboard panel** for agent activity.

## References

- Code (planned): `src/agent/tool_cache.py`, `src/agent/tool_registry.py`, `src/database.py` (new table), `src/strategy/llm_agent.py`, `src/monitoring/dashboard.py`
- Docs: ADR-0010, `CURRENT_DESIGN_ARCHITECTURE.md` §8 (observability)
- External: [LiteLLM callbacks documentation](https://docs.litellm.ai/docs/observability/callbacks), Anthropic *Building effective agents* — section on observability

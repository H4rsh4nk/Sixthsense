# Sixthsense — Goal Architecture

The target shape of the system. Items are tagged so the gap between this doc and `CURRENT_DESIGN_ARCHITECTURE.md` is explicit.

- **NEW** — does not exist in code today.
- **IMPROVED** — exists but needs substantial redesign.
- **DEFERRED** — explicitly out of scope for the foreseeable future; preserved here so we don't re-litigate it.

Architectural decisions referenced below live in [`docs/decisions/`](docs/decisions/README.md). Specifically:

- [ADR-0001](docs/decisions/0001-realistic-fast-path-sla.md) — fast-path SLA
- [ADR-0002](docs/decisions/0002-agent-vocabulary.md) — agent vocabulary
- [ADR-0003](docs/decisions/0003-fractional-kelly-sizing.md) — Kelly sizing
- [ADR-0004](docs/decisions/0004-sticky-broker-failover.md) — sticky failover
- [ADR-0005](docs/decisions/0005-human-approval-flow.md) — Telegram approval
- [ADR-0006](docs/decisions/0006-performance-regression-not-retraining.md) — regression detection vs retraining
- [ADR-0007](docs/decisions/0007-exchange-calendar-awareness.md) — calendar awareness
- [ADR-0008](docs/decisions/0008-idempotent-phase-state.md) — persisted phase state

## System Diagram

```
                         ┌────────────────────────────────────────────────────┐
                         │                    Data Sources                     │
                         │  Market · News/Macro · Enterprise · Alt (NEW)       │
                         │                + External Tools                     │
                         └──────────────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │           Ingestion & Validation (IMPROVED)              │
                    │   Schema · Anomaly · Dedup · Latency tag · Cost track    │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                  ┌────────────┴────────────┐
                                  ▼                         ▼
                    ┌────────────────────────┐   ┌────────────────────────────┐
                    │ Event-driven path (NEW)│   │ Scheduled batch path        │
                    │ websocket; <5 s SLA    │   │ pre-market / EOD / overnight│
                    │ (ADR-0001)             │   │                             │
                    └────────────┬───────────┘   └─────────────┬──────────────┘
                                 │                             │
                                 └──────────────┬──────────────┘
                                                ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │       Signal generators + Decision agent (ADR-0002)      │
                    │  News · Insider · Political · Price action · Macro       │
                    │           Single tool-using LLM decision agent           │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │      Signal aggregation & Conflict Resolution (IMPROVED) │
                    │  Weighted ensemble · Conflict gate · Confidence gate     │
                    │  Per-source accuracy priors (closed-trade feedback)      │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │                 Decision AI (IMPROVED)                    │
                    │  LLM + rules · context · sizing (fractional Kelly NEW)   │
                    │  Pre-trade gates · Human approval (NEW, ADR-0005)        │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │                Execution Layer (IMPROVED)                 │
                    │  Order manager · Primary broker                          │
                    │  Sticky failover w/ recovery reconcile (ADR-0004)        │
                    │  Second-broker (IBKR/Tradier) DEFERRED                   │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │            Monitoring & Governance (IMPROVED)             │
                    │  P&L · Signal quality · Audit · Drift · Cost tracking    │
                    │  Performance regression detection (ADR-0006)             │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               └── Operator-reviewed recalibration ──► Signals
```

## 1) Data Sources

- **Market data**: Price, volume; option flow and L2 order book where available. Daily bars suffice for the batch path; intraday quotes via Alpaca websocket feed the event-driven path.
- **News & macro**: RSS, social, government, geopolitical. Today via Google News + FinViz + Federal Register. Future: a breaking-news firehose feeds the event-driven path (ADR-0001).
- **Enterprise**: Earnings, SEC filings, reports.
- **Alt data (DEFERRED)**: Options flow, dark pool, 13F filings — explicitly deferred until the rest of the loop is stable.
- **External tools**: Charts, search, web APIs; LiteLLM-backed local Ollama for sentiment.

## 2) Ingestion & Validation (IMPROVED)

### Data quality gate

- Schema validation (per source).
- Anomaly detection (today: per-ticker single-day-return drop; future: rolling z-score for intraday).
- Deduplication (today: unique index on `(ticker, published_date, headline)`).
- **Latency tagging (NEW)** — record the gap between event publication time and ingestion time. Surfaced in `decision_logs.latency_ms` for downstream queries.
- **Cost tracking (NEW)** — record per-LLM-call token usage and dollar cost in `decision_logs.model_cost_usd` and aggregated per day.

## 3) Processing Paths

- **Event-driven path (NEW, <5 s SLA — ADR-0001)** — websocket trade/quote stream + breaking-news firehose feeding a small asyncio reactor. Pre-compiled rules (keyword match, z-score spike). **No LLM in the hot path.** Today's Python scheduler can host this; latency below 50 ms is **deferred** indefinitely.
- **Scheduled batch path (IMPROVED — ADR-0007, ADR-0008)** — minute-tick phase orchestrator, calendar-aware (skip non-trading days, respect early closes), idempotent across restarts via persisted `phase_state`.

## 4) Signal generators + Decision agent (vocabulary: ADR-0002)

- **Signal generators** (deterministic pipelines, may call an LLM but don't run a multi-turn loop):
  - **News** — Qwen 3.5 9B sentiment via Ollama / LiteLLM; NER + event detection.
  - **Insider** — SEC Form 4 clusters.
  - **Political** — Federal Register + sector-keyword mapping. Congress.gov ingest mentioned in code comments is **NEW** (not yet wired).
  - **Price action** — RSI, MACD, volume spike.
  - **Macro** — benchmark momentum regime tilt. **Not** promoted to "agent" status.
- **Decision agent** — single LLM with bounded tool calls (`TradingAgent`).
- **Risk engine** — `RiskManager`, deterministic rules. Not an agent.

## 5) Signal Aggregation & Conflict Resolution (IMPROVED)

- **Weighted ensemble** — outputs combined by configured weight × accuracy multiplier (`signal_source_stats`).
- **Conflict gate** — abstain when long vs short weighted masses are similarly large.
- **Confidence gate** — drop candidates below a configured mean-confidence floor.
- **Future** — apply both gates uniformly to the decision-agent path (today the gates are rules-path-only).

## 6) Decision AI (IMPROVED)

**LLM + rule-based hybrid** that maintains live portfolio state.

- **Portfolio context store** — live positions, P&L, exposure, sector limits.
- **Trade sizing** — today: fixed `risk_per_trade_pct`. **Planned (ADR-0003)**: opt-in fractional-Kelly mode (≤0.25×) gated by `kelly_min_trades_per_source` and capped by `kelly_max_position_pct`.
- **Entry / exit logic** — target price, stop loss, time horizon, partial fills.
- **Pre-trade risk check** — position limits, drawdown gates, correlated exposure.
- **Explainability log** — every decision row persisted to `decision_logs` (selected and rejected), with optional `latency_ms` and `model_cost_usd`.
- **Human override (NEW — ADR-0005)** — Telegram approve/deny gate for trades above a notional and/or portfolio-% threshold; fail-safe (no approval → no trade).

## 7) Execution Layer (IMPROVED)

- **Order manager** — paper-trade mode; smart routing, TWAP/VWAP intelligent slicing are **DEFERRED** until volume requires it.
- **Primary broker API** — Alpaca live execution, order status, fill confirmation.
- **Sticky failover (NEW — ADR-0004)** — three-state circuit breaker (`closed` / `open` / `half_open`) wrapping the secondary Alpaca credentials. Recovery transition triggers `OrderManager.reconcile()`.
- **Second broker (DEFERRED)** — IBKR or Tradier integration would change the failover from "same-broker keys" to "true multi-broker." Separate, larger workstream.

## 8) Monitoring & Governance (IMPROVED)

### Continuous observability

- Real-time P&L tracking, equity snapshots, drawdown.
- Signal accuracy metrics — already implemented via `signal_source_stats`.
- Audit trail — `decision_logs` covers selected + rejected with reasoning.
- **Performance regression detection (NEW — ADR-0006)** — rolling per-source win rate, Sharpe, and distribution drift (KL divergence). Alerts log + Telegram. **Replaces the older "retraining pipeline" framing.**
- Sharpe / Sortino — exist in backtest analytics; **NEW** to surface in the live loop heartbeat.
- Alerts and incident management (Telegram + log channels).
- Regulatory compliance log — included implicitly in `decision_logs` for single-operator use.

### Feedback loop

```
Closed trades → signal_source_stats → accuracy priors  ─┐
                                                        ├─► Operator-reviewed
Regression detection (win rate / Sharpe / drift KL)  ───┤   recalibration via PR
                                                        ├─► (no auto-apply; ADR-0006)
LLM cost & latency tracking (NEW)                    ───┘
```

The loop is explicitly **operator-in-the-loop**. Auto-apply of recalibrated parameters is rejected per ADR-0006.

## Out-of-scope / explicitly deferred

These items appeared in earlier iterations of this doc but are not pursued today, and the reasoning is captured for future reference.

- **Literal sub-50 ms fast path** — ADR-0001.
- **Multi-user SOC2 approval workflow** — too heavy for a single-operator system; ADR-0005 right-sizes this.
- **ML retraining pipeline** — no models to retrain in current stack; ADR-0006.
- **Multi-broker integration beyond Alpaca credentials** — ADR-0004 leaves the door open.
- **Smart-order-routing (TWAP/VWAP, dark pools)** — only relevant at much higher volumes than this strategy generates.
- **Alt data (options flow, dark pool, 13F)** — listed in §1 to acknowledge the ambition; we focus on the existing data sources first.

## Open follow-ups

When the items below are ready, write a new ADR rather than editing existing ones:

1. Concrete drift-detection metric definition (KL on what windows / which distributions?). Will be ADR-0009.
2. Async / parallel candidate processing for `market_open_entry` (needed once human-approval gate ships).
3. Cost-tracking middleware (LiteLLM has callbacks; `decision_logs.model_cost_usd` is the destination column).
4. Heartbeat-level Sharpe / Sortino exposure.

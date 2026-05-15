# Sixthsense — Current Architecture (Implemented)

Sixthsense is a scheduler-driven trading service: **Alpaca** for execution & quotes, **SQLite** for prices and artifacts, multiple **signal generators** → **SignalScorer** (rules and/or **decision agent**) → **OrderManager** with **RiskManager** / **PositionSizer** / **ExitManager**. The live loop runs a one-minute phase orchestrator in **US/Eastern**, gated by NYSE calendar awareness (skip holidays, honour early closes). All architectural choices with non-obvious trade-offs are recorded as ADRs in [`docs/decisions/`](docs/decisions/README.md).

Aspirational layers and labels are described in [GOAL_ARCHITECTURE.md](GOAL_ARCHITECTURE.md).

## System Diagram

```
                         ┌────────────────────────────────────────────────────┐
                         │                    Data Sources                     │
                         │ Yahoo OHLCV · News RSS · SEC insider · Federal Reg. │
                         │ Political/policy feed · Alpaca quotes & execution   │
                         └──────────────────────────┬─────────────────────────┘
                                                    │
                                                    ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │           Ingestion & persistence (current)              │
                    │   SQLite (WAL) · RSS/API fetch · OHLC anomaly drop       │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                  ┌────────────┴────────────┐
                                  ▼                         ▼
                    ┌────────────────────────┐   ┌────────────────────────────┐
                    │ Scheduled trading pass │   │ Post-market / overnight    │
                    │ (pre-market, open,     │   │ refresh (news LLM scores,  │
                    │ intraday risk/exits)   │   │ political event pulls)     │
                    └────────────┬───────────┘   └─────────────┬──────────────┘
                                 │                             │
                                 └──────────────┬──────────────┘
                                                ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │              Signal generators (current)                  │
                    │  Insider · News (Qwen) · Political · Price action · Macro │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │         Signal aggregation & gating (current)             │
                    │   Weighted scores · thresholds · optional decision agent  │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │              Decision & risk (LLM + rules)               │
                    │  TradingAgent (tools) · sizing · limits · exit rules     │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │                  Execution layer (current)                │
                    │  OrderManager · Alpaca · BrokerCascade (sticky failover)  │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │            Monitoring & observability (current)           │
                    │  Heartbeat · logs · decision_logs · equity snapshots      │
                    │            Streamlit dashboard · Telegram (optional)      │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                                               └── config & manual review ──► Signals
```

## Legend

- **Current**: Behavior and components that exist in this repository today.
- **Goal doc**: Broader target architecture (event-driven fast path, fractional Kelly sizing, second-broker integration, human-approval flow, etc.) lives in [GOAL_ARCHITECTURE.md](GOAL_ARCHITECTURE.md).
- **ADR**: Architectural choices with trade-offs are recorded in [`docs/decisions/`](docs/decisions/README.md) and cross-linked below.

## 1) Data Sources

- **Market data**: Historical OHLCV ingested into SQLite (Yahoo-style pipeline); live quotes from **Alpaca** during the trading loop.
- **News**: **Google News** and **FinViz** RSS; headlines scored with **Qwen** via **LiteLLM** (e.g. Ollama).
- **Insider / regulatory**: **SEC**-sourced insider activity (Form 4–style signals in code).
- **Political / policy**: **Federal Register** events (the file mentions Congress.gov but only Federal Register is wired in).
- **External tools**: LLM provider (local Ollama or other **LiteLLM** backends), optional **Telegram** alerts.

## 2) Ingestion & persistence

- **SQLite** database (WAL) for prices, signals, trades, news articles, decision traces, and persisted scheduler `phase_state` (see ADR-0008).
- Fetch-and-store patterns on signal classes (e.g. `fetch_and_score`, `fetch_and_store_events`) during post-market refresh windows.
- **Price ingest validation** (`data.ingestion` in settings): OHLC sanity checks plus per-ticker anomaly drop when single-day \|return\| exceeds `max_single_day_return_pct`; applied during `download_price_history`.
- **News dedup**: unique index on `(ticker, published_date, headline)` so `INSERT OR IGNORE` skips duplicates.

## 3) Processing & scheduling

- **Phase orchestrator** (`src/main.py::phase_orchestrator`) runs every minute and routes work to `pre_market_scan`, `market_open_entry`, `intraday_check`, `post_market_review`, `post_market_refresh`. Internally there are exactly three phases: `pre_market`, `in_market`, `post_market`.
- **Calendar awareness** (ADR-0007): the orchestrator uses `src/market_calendar.py` (`exchange_calendars` / XNYS) to skip non-trading days and shorten the in-market window on early-close days. Toggle: `broker.calendar_aware` (default `true`); falls back to weekday-only check if `exchange_calendars` is not installed.
- **Idempotent phase state** (ADR-0008): the in-memory `phase_state` is loaded from a SQLite `phase_state` table on startup and persisted after each phase runs. A process restart mid-day no longer double-runs pre-market or market-open entry. CLI: `swing-trader trade --force-phase <name>` clears a single phase row so it re-runs on the next tick.
- **Note on `broker.entry_delay_minutes` / `exit_before_close_minutes`**: these YAML knobs exist but are not currently read by runtime code. Effective entry timing comes from `scheduler.market_open_entry` (e.g. `09:35`), and the exit window is bounded by `effective_close_time()` which combines `scheduler.market_close_exit` with the NYSE early-close calendar.

## 4) Signal generators (terminology: ADR-0002)

- **News**: Headline sentiment via configured LLM (**Qwen 3.5 9B** typical) over Google News + FinViz RSS; consensus and thresholds from `signals.news`.
- **Insider**: Clustered insider transactions versus size and lookback windows.
- **Political**: Federal Register events mapped to ticker impact heuristics by sector.
- **Price action**: Indicators such as RSI, MACD, volume spike per `signals.price_action`.
- **Macro** (optional, `signals.macro`): Benchmark momentum regime tilt (default **SPY**); pipeline pulls the benchmark when macro is enabled. Stays a **signal**, not an "agent" (ADR-0002).

## 5) Signal aggregation & gating

- **SignalScorer** merges per-ticker **SignalResult** rows with configurable **weights** per source (including macro when enabled).
- **Rules path**: weighted strength; **conflict gate** (`scoring.conflict_balance_ratio`): abstains when opposing long vs short masses are similarly large; optional **confidence gate** (`min_aggregate_confidence`); thresholds **min_combined_score** / **min_signals_agreeing**.
- **Accuracy priors**: optional **`accuracy_weight_adjustment`** scales weights using **`signal_source_stats`** updated when trades close (win/loss per comma-separated signal source). This is the feedback loop's working part — see ADR-0006.
- **Agent path**: optional **TradingAgent** (LLM tool use); **`fallback_to_rules`** if agent init fails (agent path does not yet apply the same conflict/confidence gates as rules).
- **`last_decision_trace`** persisted to **`decision_logs`** with optional `latency_ms` and `model_cost_usd` columns reserved for observability roll-out.

## 6) Decision AI (LLM + rules)

- **TradingAgent** when `agent.enabled`: portfolio-aware reasoning with bounded tool calls — the single "decision agent" per ADR-0002.
- **RiskManager** (deterministic rules, not an agent): circuit breaker, daily loss limits, drawdown tracking, concurrency caps from `trading`.
- **PositionSizer** / **ExitManager**: position sizing against risk config and timed or stop-based exits (`exit_rules`).
- **Sizing modes (ADR-0003)**: `trading.sizing_mode` is `fixed_risk` (default) or `fractional_kelly`. Kelly mode pulls per-source win-rate and avg win/loss from closed trades (`Database.get_signal_kelly_inputs`), computes per-source `f* = (p·b − (1−p))/b`, averages across the candidate's sources, multiplies by `kelly_fraction`, and clamps to `min(kelly_max_position_pct, max_single_position_pct)`. Any contributing source with fewer than `kelly_min_trades_per_source` closed trades downgrades the candidate to fixed-risk for that entry. `PositionSize` records `sizing_mode_used`, `kelly_target_pct`, `per_source_kelly`, and any `kelly_fallback_reason`; the entry log line surfaces them.
- **Planned (ADR-0005)**: Telegram human-approval gate for trades above a configurable notional / portfolio-pct threshold. Not yet implemented.

## 7) Execution layer

- **OrderManager** orchestrates entries and exits. `reconcile()` compares broker positions vs DB-open trades and warns on drift; it is also wired as the recovery hook for the broker circuit breaker.
- **AlpacaBroker** is the single-broker path. **BrokerCascade** is used when `broker.fallback_enabled` is true and provides a **sticky failover circuit breaker** (ADR-0004): primary failures past `broker.failover_failure_threshold` trip the breaker to `open`, subsequent calls go straight to fallback for `broker.failover_cooldown_seconds`, then `half_open` probes primary; success closes the breaker and triggers `OrderManager.reconcile()`.
- **Deferred**: a *non-Alpaca* fallback broker (IBKR, Tradier, etc.) is **not** implemented. The current fallback is a second set of Alpaca credentials.

## 8) Monitoring & observability

- **Heartbeat** on `heartbeat_interval_minutes` (equity, drawdown, circuit breaker) plus a compact **signal row counts** digest from the **`signals`** table (recent window).
- **decision_logs**: selected/rejected candidates, reasoning, rejection causes, signal metadata, LLM trace when agent mode runs. New columns `latency_ms` and `model_cost_usd` are present (migration-applied) for future population.
- **Streamlit** dashboard and optional **Telegram** notifications (`alerts.telegram_enabled`).
- **Post-market** equity snapshots for history.
- **Performance regression detection (ADR-0006)**: `src/monitoring/regression.py` runs from `post_market_review` (after the equity snapshot is written). Two checks today: (a) per-signal-source 30-trade win rate vs 90-trade baseline (alerts when below `monitoring.regression_win_rate_floor` *and* trailing baseline by `regression_win_rate_delta`); (b) portfolio 30-day annualised Sharpe vs `regression_sharpe_floor`. Findings persist to the `regression_alerts` table and ring Telegram when enabled. Drift KL (distribution-level regression) is **deferred to ADR-0009**.

## Feedback loop

**Signal-level priors**: closed trades update **`signal_source_stats`** (used when **`accuracy_weight_adjustment`** is on). Per ADR-0006, the rest of the loop (regression detection + parameter recalibration) is an explicit operator-in-the-loop workflow, not an automated retraining pipeline.

## Decision Log

Architectural decisions live in [`docs/decisions/`](docs/decisions/README.md). Each ADR captures one choice's context, what we picked, what we rejected, and the trade-offs. Use the table below as a quick map from this document to the rationale behind it.

| Section | ADR | What the ADR decides |
| --- | --- | --- |
| §3 (scheduling) | [ADR-0007](docs/decisions/0007-exchange-calendar-awareness.md) | NYSE calendar awareness via `exchange_calendars`; rules for early-close days |
| §3 (scheduling) | [ADR-0008](docs/decisions/0008-idempotent-phase-state.md) | Persist `phase_state` to SQLite so restarts don't double-run phases |
| §4 (terminology) | [ADR-0002](docs/decisions/0002-agent-vocabulary.md) | "Signal generator" vs "decision agent" vs "risk engine" |
| §6 (sizing) | [ADR-0003](docs/decisions/0003-fractional-kelly-sizing.md) | Fractional-Kelly sizing, guarded by per-source closed-trade count |
| §6 (human gate, planned) | [ADR-0005](docs/decisions/0005-human-approval-flow.md) | Telegram approve/deny for trades above a $ / % threshold |
| §7 (execution) | [ADR-0004](docs/decisions/0004-sticky-broker-failover.md) | Circuit-breaker sticky failover with cooldown and recovery reconciliation |
| §8 (monitoring) | [ADR-0006](docs/decisions/0006-performance-regression-not-retraining.md) | Win-rate / Sharpe regression checks; drift deferred to ADR-0009 |
| Goal-doc-wide | [ADR-0001](docs/decisions/0001-realistic-fast-path-sla.md) | Reject literal <50 ms fast path; redefine as event-driven, <5 s SLA |

New ADRs are added to `docs/decisions/` with sequential numbering. Each section above links the ADR(s) that explain *why* the design choice exists.

## In-progress improvements (recent)

This section summarises code changes landed since the last revision of this doc so the diff between "what the doc says" and "what the code does" is auditable.

- **Calendar awareness** (`src/market_calendar.py`, `src/main.py`) — ADR-0007.
- **Sticky broker failover** (`src/execution/broker.py::BrokerCascade`) — ADR-0004. Adds `broker.failover_failure_threshold` and `broker.failover_cooldown_seconds` settings and a `set_recovery_callback` hook used by `main.py` to drive `OrderManager.reconcile()` on recovery.
- **Persisted phase_state** (`src/database.py`, `src/main.py`) — ADR-0008. Adds `phase_state` table, `load_phase_state` / `set_phase_state` / `clear_phase_state` helpers, and a `--force-phase` CLI flag.
- **decision_logs observability columns** (`src/database.py`) — `latency_ms` and `model_cost_usd` columns added via migration; populated by future work.
- **Fractional-Kelly sizing** (`src/strategy/position_sizer.py`, `src/database.py::get_signal_kelly_inputs`, `src/execution/order_manager.py`) — ADR-0003. Adds `trading.sizing_mode`, `kelly_fraction`, `kelly_min_trades_per_source`, `kelly_max_position_pct` config knobs. `fractional_kelly` mode falls back transparently to `fixed_risk` per-candidate when a contributing source has too few closed trades.
- **Performance regression detection** (`src/monitoring/regression.py`, `regression_alerts` table, `monitoring.*` config block) — ADR-0006. Runs from `post_market_review`; logs warnings, persists to `regression_alerts`, and emits Telegram alerts when enabled.
- **Dependency**: `exchange_calendars >= 4.5` added to `requirements.txt` and `pyproject.toml`.

Future improvements (not yet landed): ADR-0005 (Telegram approval flow); ADR-0009 (drift-KL definition for distribution regression).

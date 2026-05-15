# Testing playbook

A practical guide for verifying Sixthsense works end-to-end **before** wiring real
money or your live Telegram channel. Layered from quickest / safest to slowest /
most invasive.

> Why this exists: ADR-0010 makes the LLM agent the central decision-maker. Each
> piece of the system (signals, sizer, risk, broker, regression detector, agent
> tools) must therefore be observable on its own *and* in concert.

---

## 1. Automated smoke test (90 seconds)

The single command that verifies the whole stack:

```bash
python -m scripts.smoketest --quiet
```

What it covers, in order:

| Check                          | What it proves                                                 | Hits network? |
| ------------------------------ | -------------------------------------------------------------- | ------------- |
| Calendar awareness             | NYSE holidays / early closes resolve (ADR-0007)                | no            |
| Database schema                | All 14 required tables exist incl. `phase_state`, `tool_call_log` | no            |
| Universe seed                  | `insert_universe` round-trips                                  | no            |
| PositionSizer                  | fixed_risk and fractional_kelly modes + Kelly fallback (ADR-0003) | no         |
| RiskManager                    | Circuit breaker trips at `max_drawdown_pct`                    | no            |
| Phase state round-trip         | Idempotent scheduler state (ADR-0008)                          | no            |
| Regression detector            | Synthetic 0% win rate triggers a critical alert (ADR-0006)     | no            |
| BrokerCascade failover         | OPEN -> HALF_OPEN -> CLOSED + recovery callback (ADR-0004)     | no            |
| Price ingest (yfinance)        | OHLCV download + validation persists rows                      | **yes** (Yahoo) |
| Tool registry (Phase A)        | Every read-only tool resolves against seeded DB (ADR-0010)     | no            |

**Default behaviour:** uses a throwaway SQLite DB inside a `tempfile.TemporaryDirectory`.
Your `data/swing_trader.db` is never touched.

**Exit codes:** `0` on clean (skips allowed), `1` if any required check failed.

### Opt-in (slower / requires extra setup)

```bash
# Read-only Alpaca call (needs config/secrets.yaml populated)
python -m scripts.smoketest --include-live-broker

# NewsSignal real fetch + Ollama scoring (needs Ollama running on its port)
python -m scripts.smoketest --include-news

# Federal Register fetch for political events
python -m scripts.smoketest --include-political
```

### Skipping a single check

```bash
python -m scripts.smoketest --skip get_political_events
```

---

## 2. Dry-run a real trading cycle (5 minutes)

Phase C of ADR-0010 added `--dry-run` to `swing-trader trade`. Run a full
intraday cycle without placing a single order:

```bash
swing-trader trade --now --dry-run --force-phase market_open
```

You should see:

- Signal generators fetch real data (news, insider, macro, price action).
- The scorer ranks candidates as usual.
- For each candidate, `RiskManager.can_open_position` and `PositionSizer.calculate`
  run end-to-end.
- A log line per would-be entry, **without** broker submission or `trades` row:
  ```
  DRY-RUN ENTRY: LONG 13 x AAPL @ $190.42 | stop=$181.45 | score=0.71 | sizing=fixed_risk | reason=...
  ```
- The scheduler still respects calendar awareness, persists `phase_state`, and
  evaluates the post-market regression check.

When you're confident the picks look sane, drop `--dry-run` and the same flow
submits to the broker.

---

## 3. Manual checks (the dashboard + alerts)

These can't be smoke-tested cleanly; eyeball them once after big changes.

### Streamlit monitoring dashboard

```bash
swing-trader monitor
```

Verify these widgets render and refresh:

1. **Equity curve** — needs `equity_snapshots` rows. Run `swing-trader trade --now`
   in dry-run mode at least once to seed.
2. **Decision logs** — each scored ticker shows decision-mode (agent / rules),
   reasoning, latency, and model cost.
3. **Regression alerts** — surfaces rows from `regression_alerts`.
4. **Phase state** — last-run timestamp per phase (pre-market, market-open, ...).
5. **Signal source stats** — per-source win/loss + Beta-prior multiplier.

### Telegram alerts (opt-in)

If you set `secrets.telegram_bot_token` and `secrets.telegram_chat_id`, run:

```bash
swing-trader trade --now --dry-run --force-phase post_market
```

You should receive a heartbeat ping plus any regression-alert messages from the
synthetic seed (or real data). If nothing arrives, check `data/sixthsense.log`
for `AlertManager` errors.

---

## 4. Backtests as a regression harness

`swing-trader backtest --start 2024-01-01 --end 2024-12-31` runs the rules-only
pipeline against historical data. Use this as a coarse-grained regression test
after touching signals, scoring, or sizing. ADR-0010 notes that the agent path
is **not yet** backtestable (LLM calls aren't deterministic); we accept that gap
for now.

---

## 5. Adding a new check

The smoketest is intentionally additive — write a small `check_xxx` function
in `scripts/smoketest.py` and call it from `main()`. Use `_timed(...)` to get
PASS / FAIL / SKIP wrapping for free, and raise `_SkipCheck("reason")` when a
dependency isn't available rather than failing.

If a new piece of the system surfaces through the tool registry, add it to
`build_default_registry()` in `src/agent/tool_registry.py` and the smoketest's
`check_tool_registry` will pick it up automatically.

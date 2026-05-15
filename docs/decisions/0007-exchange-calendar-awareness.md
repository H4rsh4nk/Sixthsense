# ADR-0007: Exchange-calendar awareness via `exchange_calendars`

- **Status:** Accepted
- **Date:** 2026-05-14
- **Related:** ADR-0008

## Context

The phase orchestrator (`src/main.py::phase_orchestrator`) currently skips Saturday and Sunday via `now_local.weekday() < 5`, but treats every weekday identically. It therefore:

- Runs `pre_market_scan` and `market_open_entry` on **US market holidays** (Independence Day, Thanksgiving, MLK Day, etc.).
- Does not adjust the **market_close_exit** time on **early-close days** (e.g. day after Thanksgiving closes at 13:00 ET, not 16:00).
- Cannot tell `post_market_review` apart from "scheduler just started up on a holiday."

This produces noise in the logs, wastes LLM tokens, and on early-close days can have the system holding positions past actual market close.

`pandas_market_calendars` and `exchange_calendars` are both maintained libraries. `exchange_calendars` is lighter, pinned to the NYSE calendar, and used widely in quant codebases (Zipline-reloaded, Backtrader, etc.).

## Decision

Add a small helper module `src/market_calendar.py` wrapping `exchange_calendars` (XNYS) with:

```python
def is_trading_day(d: date) -> bool
def is_early_close(d: date) -> bool
def session_close_time(d: date) -> time | None  # local tz; None on non-trading days
```

Wire into `phase_orchestrator`:

- Skip all trading-related jobs on non-trading days (pre-market, market-open entry, intraday checks). Allow post-market data-refresh jobs to still run, since their inputs (Federal Register, RSS) do not require an open market.
- On early-close days, use `session_close_time(today)` instead of the static `sched.market_close_exit`, so the `in_market` window ends correctly and exits / post-market review run on time.

Add a daily startup log line that prints whether today is a trading day, and if so its scheduled close.

Add `exchange_calendars` to `requirements.txt` (>= 4.5).

## Consequences

### Positive

- Removes a real correctness gap. We currently log "PRE-MARKET SCAN" on Christmas Day.
- Pure addition — no behavior change on regular trading days.
- Reuses an actively maintained calendar source rather than hard-coding holiday lists.

### Negative / accepted trade-offs

- One new dependency. `exchange_calendars` is pure-Python with a small dependency footprint (`numpy`, `pandas`, which we already have, plus `pytz` which we already have).
- The library updates its holiday list as the exchanges publish new ones; we rely on keeping the package version reasonably current.

### Neutral

- Backtest engine is unaffected — it iterates over actual price rows, which already exclude holidays.

## Alternatives considered

### A — Hard-code a US holiday list

Rejected. Maintenance burden (early-close rules and non-standard holidays). Errors are silent.

### B — Call Alpaca `get_clock()` / `get_calendar()`

Considered. Authoritative, but requires a live API call every scheduler tick, fails closed if Alpaca is unreachable, and adds rate-limit pressure. The library is local and free.

### C — Calendar-aware *only* for the new code, leave `phase_orchestrator` alone

Rejected. The whole point is to fix the orchestrator.

## References

- Code: `src/market_calendar.py` (new), `src/main.py::phase_orchestrator`
- Docs: `CURRENT_DESIGN_ARCHITECTURE.md` §3
- External: [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars), NYSE calendar

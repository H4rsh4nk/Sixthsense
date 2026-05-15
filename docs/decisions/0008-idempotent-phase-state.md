# ADR-0008: Idempotent persisted `phase_state`

- **Status:** Accepted
- **Date:** 2026-05-14
- **Related:** ADR-0007

## Context

`phase_orchestrator` keeps an in-memory dict `phase_state` to mark which phase jobs have already run today:

```python
phase_state = {
    "pre_market_date": None,
    "market_open_entry_date": None,
    "post_market_review_date": None,
    "post_market_refresh_date": None,
    "last_intraday_check": None,
}
```

If the process restarts at 10:00 ET, the dict resets and `market_open_entry()` runs again — placing duplicate entry orders for the same candidates. This is a real correctness bug, not just a logging issue.

## Decision

Persist `phase_state` to a new SQLite table and load it on startup:

```sql
CREATE TABLE IF NOT EXISTS phase_state (
    phase_key TEXT PRIMARY KEY,
    last_run_date TEXT,        -- ISO date, e.g. '2026-05-14'
    last_run_at TEXT,          -- ISO datetime
    detail TEXT                -- optional, e.g. last_intraday_check ISO datetime
);
```

Helpers on `Database`:

- `get_phase_state(phase_key) -> dict | None`
- `set_phase_state(phase_key, last_run_date, last_run_at, detail=None)`

`phase_orchestrator` becomes:

1. Read all phase state rows once into a local dict at startup.
2. After each phase action runs successfully, **persist** the updated phase row before returning from the tick.
3. The check "did pre-market already run today?" reads from the persisted dict, which we keep in sync.

Idempotency contract: each phase function is wrapped so that the DB row is only written **on success**. If a phase function partially runs and raises, the next tick will retry. This is the right behavior — better to attempt again than to silently skip after a crash.

## Consequences

### Positive

- Closes a real duplication bug. A restart during `in_market` no longer triggers a second `market_open_entry`.
- Operator can audit `phase_state` and see exactly when each phase last ran today / yesterday.
- Combined with ADR-0007 (calendar awareness), today's "did pre-market run today" semantics are now date-correct *and* persistent.

### Negative / accepted trade-offs

- One more table. Trivial.
- A *manual* re-run of a phase needs an explicit way to clear the corresponding row. We add a `--force-phase <name>` CLI flag to support this.
- "On success" semantics mean a phase that succeeds 99 % then crashes will rerun fully next tick — could occasionally produce duplicate trades if the crash was *after* an order submission. Mitigation: `OrderManager.reconcile()` already exists; we will call it at the start of `market_open_entry` and skip candidates whose ticker is already open.

### Neutral

- Backtest engine doesn't use the orchestrator; unaffected.

## Alternatives considered

### A — File-based persistence (JSON in `data/phase_state.json`)

Rejected. Adds another consistency surface; we already have SQLite WAL with crash-safe semantics.

### B — Distributed lock (Redis / file lock)

Rejected. Overkill for a single-process scheduler.

### C — Make every phase function idempotent at the action level

Considered but harder. `market_open_entry` would have to track exact order IDs submitted per day. The DB phase guard is the right place to do this once.

## References

- Code: `src/database.py` (new `phase_state` table and helpers), `src/main.py::phase_orchestrator`
- Docs: `CURRENT_DESIGN_ARCHITECTURE.md` §3

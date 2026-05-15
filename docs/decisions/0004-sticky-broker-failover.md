# ADR-0004: Sticky broker failover with cooldown

- **Status:** Accepted
- **Date:** 2026-05-14
- **Related:** `GOAL_ARCHITECTURE.md` §7 (Execution Layer)

## Context

`GOAL_ARCHITECTURE.md` §7 says "Fallback broker (NEW): Circuit breaker auto-switches on primary API failure or timeout." Current `BrokerCascade` (`src/execution/broker.py`) does per-call try-then-fallback: every call hits primary first, and only switches to fallback on that specific call's exception. If primary is down for 10 minutes, we make hundreds of doomed primary calls — each one adds latency (typically a timeout) and noise to the logs.

Standard circuit-breaker pattern (Nygard, *Release It!* Ch. 5):

1. **Closed state** — calls go to primary.
2. After N consecutive failures within a window, **open** the breaker — all calls skip primary and go straight to fallback.
3. After a cooldown, transition to **half-open** — let one probe call hit primary; success closes the breaker, failure re-opens it.

Per-call retry (today) is **not** this pattern — it's at-most-once retry per request and provides no protection against an extended primary outage.

## Decision

Add a circuit-breaker state machine to `BrokerCascade` with three states (`closed`, `open`, `half_open`) and the following config knobs:

```yaml
broker:
  failover_failure_threshold: 3       # consecutive failures to trip
  failover_cooldown_seconds: 300      # how long to stay 'open' before probing
  failover_probe_method: get_account  # the method used for half-open probe
```

State transitions:

- `closed` → `open` after `failover_failure_threshold` consecutive primary failures.
- `open` → `half_open` after `failover_cooldown_seconds`.
- `half_open` → `closed` on a successful primary call.
- `half_open` → `open` on any primary failure (resets cooldown).

While `open`, **all read and write calls go to fallback only** — primary is not touched. We log a single warning at the transition, not one per call.

Position-reconciliation hook: every time the breaker transitions back to `closed`, we run `OrderManager.reconcile()` to detect any drift between what primary thinks we hold and what fallback executed while primary was down.

## Consequences

### Positive

- Dead primary stops being hammered — log volume and tail latency drop dramatically during outages.
- Recovery is explicit and audited via state-transition log lines.
- Reconciliation on recovery prevents silent position drift between brokers.

### Negative / accepted trade-offs

- More state to maintain inside `BrokerCascade` (small).
- During `open`, *reads* (`get_account`, `get_positions`) come from fallback only — if the user only configured fallback for execution emergencies, those reads will look different. Acceptable: both endpoints are Alpaca-paper today; the secondary-broker case is deferred.
- Requires the fallback credentials to actually work. We surface a startup warning when `broker.fallback_enabled=true` but no usable fallback secret is set (existing behaviour).

### Neutral

- For the single-broker case (`fallback_enabled=false`), behavior is unchanged.

## Alternatives considered

### A — Status quo (per-call retry)

Rejected. Documented above.

### B — Exponential-backoff retry on primary, no fallback switch

Rejected. Backoff helps with transient failures but does not help when primary is genuinely down. Doesn't satisfy the "auto-switch on failure" requirement.

### C — Adopt `pybreaker` library

Considered. Adds a dependency for ~120 lines of state machine. Decided to inline it — small enough, no dynamic policy needs, and avoids version-pinning a niche library.

## References

- Code: `src/execution/broker.py::BrokerCascade`, `src/execution/order_manager.py::reconcile`
- Docs: `GOAL_ARCHITECTURE.md` §7, `CURRENT_DESIGN_ARCHITECTURE.md` §7
- External: Michael Nygard, *Release It!* (Pragmatic Bookshelf, 2nd ed.) — Circuit Breaker pattern.

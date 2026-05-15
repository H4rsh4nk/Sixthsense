# ADR-0005: Telegram human-approval gate for large trades

- **Status:** Proposed
- **Date:** 2026-05-14
- **Related:** ADR-0002, `GOAL_ARCHITECTURE.md` §6

## Context

`GOAL_ARCHITECTURE.md` §6 lists "Human override (NEW): Trades above size threshold require manual approval." Reading this literally implies a multi-user approval workflow (request → reviewer → audit), which is overkill for a single-operator paper-trade system.

The realistic version: when the system wants to enter a position that exceeds a dollar threshold, send a Telegram message with the trade details, wait for an ACK from the operator, and only submit on ACK. If no ACK arrives before a timeout, skip the trade and log it as `rejection_reason="approval_timeout"`.

## Decision

Add `trading.human_approval` config block:

```yaml
trading:
  human_approval:
    enabled: false
    notional_threshold_usd: 2500     # absolute $ above which approval is required
    portfolio_pct_threshold: 0.05    # OR % of equity above which approval is required (whichever trips first)
    timeout_seconds: 90              # how long to wait for ACK
    require_explicit_yes: true       # message must include "yes" / "approve"
```

Flow in `OrderManager.enter_trade`:

1. Compute the candidate position's notional and % of equity.
2. If approval is enabled **and** either threshold is exceeded, send a Telegram message that includes ticker, direction, shares, notional, score, reasoning, contributing signals, and a request to reply `yes` to approve.
3. Poll for a response (Telegram long-poll or webhook) up to `timeout_seconds`.
4. On `yes`: proceed with normal entry; mark the decision log with `approval=manual`.
5. On `no` or timeout: skip the entry; write a decision_log row with `selected=0`, `rejection_reason="manual_rejected"` or `"approval_timeout"`.

Failures of the Telegram channel (network down, bot token bad) are **fail-safe**: skip the trade and log a clear error. We do *not* fall through to auto-approval.

## Consequences

### Positive

- Catches surprising large trades from bug/over-confident agent runs before they execute.
- Auditable: every "manual_rejected" / "approval_timeout" lives in `decision_logs`.
- Uses the Telegram channel that already exists in config.

### Negative / accepted trade-offs

- Adds blocking time inside `enter_trade` (up to `timeout_seconds`). For a 10-candidate market-open run, sequential approvals could push entries past the desirable execution window. Mitigation: thresholds default high so the gate only fires on outliers; future improvement could run candidates in parallel asyncio tasks.
- Operator must actually be reachable. If they aren't, no large trades happen — that's the *intent*.

### Neutral

- Off by default. Existing flows are unchanged when `enabled=false`.

## Alternatives considered

### A — Email approval

Rejected. Email round-trip is too slow for trading, and the Telegram channel is already wired into alerts.

### B — Web dashboard approval

Considered. More work and not obviously safer than Telegram. Could be a follow-up ADR if we add a real ops UI.

### C — Hard cap with no override

Rejected. Then we lose the benefit of the override: the operator can never opt in to a high-confidence outlier trade.

## References

- Code (future): `src/execution/order_manager.py`, `src/monitoring/alerts.py` (will need a `wait_for_reply` helper)
- Docs: `GOAL_ARCHITECTURE.md` §6, `CURRENT_DESIGN_ARCHITECTURE.md` §6

# Architecture Decision Records (ADRs)

This directory holds **Architecture Decision Records** for Sixthsense. Each ADR captures one architectural choice: the context that forced it, the decision itself, the alternatives we rejected, and the consequences (positive and negative) we accepted.

ADRs are append-only. If a decision is reversed, write a new ADR that supersedes the old one and update the **Status** of the original.

## Format

Use `0000-template.md` as the starting point. Filenames are `NNNN-kebab-case-title.md` with `NNNN` zero-padded sequential.

Each ADR has:

- **Status** — Proposed | Accepted | Superseded by ADR-XXXX | Deprecated
- **Context** — what problem / forces motivated the decision
- **Decision** — what we are doing
- **Consequences** — trade-offs and downstream impact
- **Alternatives Considered** — what we rejected and why
- **References** — links to code, related ADRs, external sources

## Index

| #    | Title                                          | Status   | Touches                                |
| ---- | ---------------------------------------------- | -------- | -------------------------------------- |
| 0001 | Realistic fast-path SLA (reject literal <50ms) | Accepted | `GOAL_ARCHITECTURE.md` §3              |
| 0002 | Agent vs signal-generator vocabulary           | Accepted | `src/signals/`, `src/strategy/`        |
| 0003 | Fractional Kelly sizing (≤0.25×, guarded)      | Accepted | `src/strategy/position_sizer.py`, `src/database.py::get_signal_kelly_inputs` |
| 0004 | Sticky broker failover with cooldown           | Accepted | `src/execution/broker.py`              |
| 0005 | Telegram human-approval flow for large trades  | Proposed | `src/execution/order_manager.py`       |
| 0006 | Performance regression detection (not ML retraining) | Accepted | `src/monitoring/regression.py`, `regression_alerts` table |
| 0007 | Exchange-calendar awareness (`exchange_calendars`) | Accepted | `src/main.py`, new `src/market_calendar.py` |
| 0008 | Idempotent persisted `phase_state`             | Accepted | `src/database.py`, `src/main.py`       |

Statuses:

- **Accepted** — implemented or actively being implemented.
- **Proposed** — agreed direction, not yet built.
- **Superseded** — replaced by a newer ADR.

## How `CURRENT_DESIGN_ARCHITECTURE.md` references this

`CURRENT_DESIGN_ARCHITECTURE.md` describes **what exists in code today**. Whenever a section mentions a design choice with non-obvious trade-offs, it links to the relevant ADR(s) here. That keeps "what" (architecture doc) and "why" (ADRs) cleanly separated.

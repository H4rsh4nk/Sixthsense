# ADR-0003: Fractional Kelly sizing, guarded and capped

- **Status:** Accepted
- **Date:** 2026-05-14
- **Implemented:** 2026-05-14 (`src/strategy/position_sizer.py`, `src/execution/order_manager.py`, `src/database.py::get_signal_kelly_inputs`)
- **Related:** ADR-0006 (signal stats feedback loop), `GOAL_ARCHITECTURE.md` §6

## Context

`GOAL_ARCHITECTURE.md` §6 says "Trade sizing: Kelly criterion with hard max-risk cap per trade." Current `PositionSizer` uses a fixed `risk_per_trade_pct` (default 0.5%) and ignores edge estimates entirely.

Pure Kelly maximises geometric growth under perfect knowledge of edge `e` and win probability `p`. In practice:

- Edge estimates from `signal_source_stats` are noisy until N closed trades exist per source.
- Kelly is famously sensitive to over-estimation of edge — being off by 20 % on win-rate produces wildly larger bets.
- Empirical practice (Thorp; Edwards, McLean & Thorp 2010) uses **fractional Kelly** (¼ to ½) to trade growth rate for variance reduction.

## Decision

Introduce an opt-in fractional-Kelly mode in `PositionSizer`:

```yaml
trading:
  sizing_mode: fixed_risk         # fixed_risk | fractional_kelly
  kelly_fraction: 0.25            # 0 < x ≤ 0.5, fractional multiplier on full Kelly
  kelly_min_trades_per_source: 30 # below this, fall back to fixed_risk
  kelly_max_position_pct: 0.10    # hard cap, never exceed this % of equity
```

Behaviour:

1. If `sizing_mode == "fixed_risk"`, behave exactly as today.
2. If `sizing_mode == "fractional_kelly"`:
   - Pull win-rate `p` and average win/loss ratio `b` from `Database.get_signal_kelly_inputs()` (computed live from closed trades) for each signal source on this candidate.
   - If any contributing source has fewer than `kelly_min_trades_per_source` closed trades, **fall back to fixed_risk** and log the reason.
   - Compute per-source `f* = (p · b − (1 − p)) / b`, clamp to `[0, ∞)`.
   - Aggregate by **simple mean** across sources (intentionally conservative for narrow ensembles), multiply by `kelly_fraction`, then clamp to `min(kelly_max_position_pct, trading.max_single_position_pct)`.
   - The shared per-trade caps (cash, max position pct) still constrain the final share count.

The `PositionSize` dataclass records the chosen sizing path (`sizing_mode_used`), the computed target pct (`kelly_target_pct`), the per-source `f*` breakdown (`per_source_kelly`), and any fallback reason — surfaced in the trade-entry log line and available for future persistence into a `sizing_logs` table.

## Consequences

### Positive

- Closed-loop sizing — better signals get bigger bets, weak ones get less, *automatically*.
- Safe by default: gated by a minimum-sample-size threshold and a fixed cap.
- Backwards compatible — fixed-risk mode stays the default.

### Negative / accepted trade-offs

- Win/loss ratio `b` needs to be tracked, not just wins/losses. Requires extending `signal_source_stats` with average win % and average loss %, or recomputing from `trades`.
- More moving parts for users to misconfigure.

### Neutral

- This is a sizing change only — no impact on signal generation, scoring, or execution timing.

## Alternatives considered

### A — Full (non-fractional) Kelly

Rejected. Over-bets on noisy edges; standard quant practice is fractional Kelly.

### B — Equal-weight sizing

Rejected. Ignores the information already in `signal_source_stats`; loses the benefit of the existing closed-trade feedback loop.

### C — Risk parity (volatility-scaled)

Considered. Could be a parallel sizing mode in a future ADR. Not in scope here because we already pay for accuracy stats via the trade-close loop; adding vol estimation is a separate dataset.

## References

- Code: `src/strategy/position_sizer.py`, `src/database.py::get_signal_accuracy_multipliers`
- Docs: `GOAL_ARCHITECTURE.md` §6, `CURRENT_DESIGN_ARCHITECTURE.md` §6
- External: Thorp (1969), "Optimal Gambling Systems"; Edwards et al. (2010), "Good and Bad Properties of the Kelly Criterion."

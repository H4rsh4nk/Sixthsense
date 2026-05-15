# ADR-0006: Performance regression detection replaces "retraining pipeline"

- **Status:** Accepted
- **Date:** 2026-05-14
- **Implemented:** 2026-05-14 — win-rate + Sharpe checks (`src/monitoring/regression.py`, scheduled from `main.py::post_market_review`). Drift KL deferred to ADR-0009.
- **Related:** ADR-0002, ADR-0003, `GOAL_ARCHITECTURE.md` §8 (Monitoring & Governance), Feedback Loop

## Context

`GOAL_ARCHITECTURE.md` §8 lists "Retraining pipeline trigger" and the feedback loop ends with "Agent retraining → Decision AI recalibration."

The system currently uses:

- **Qwen 3.5 9B via Ollama** for news sentiment — a pre-trained foundation model we run locally. We do not fine-tune it.
- **Rule-based signals** (insider clusters, RSI/MACD, political keywords, macro tilt) — no parameters to learn.
- **`SignalScorer`** — weighted sum with optional accuracy-prior adjustment from `signal_source_stats`. The "training" here is a Beta-prior win-rate update on each closed trade.

There is nothing to retrain in the ML sense. "Retraining pipeline" is the wrong vocabulary. The thing we *actually* need is detection of when the system's measured performance degrades, plus tools to recalibrate weights and thresholds.

## Decision

Replace the "Retraining pipeline" item in the goal architecture with **Performance Regression Detection (PRD)** and **Parameter Recalibration**:

### Performance Regression Detection

A daily monitoring job that computes:

- Per-signal-source rolling win rate over (a) last 30 closed trades and (b) last 90 closed trades.
- Portfolio Sharpe ratio over the last 30 / 90 trading days.
- Drift metric: KL divergence of *today's* signal-strength distribution vs the trailing 90-day distribution per source. Per ADR (to come) `0009-drift-detection-definition.md`.

It alerts (logs + Telegram) when:

- Any signal's 30-day win rate drops below a configurable floor *and* falls > X percentage points below its 90-day baseline.
- Sharpe drops below a floor.
- Drift KL > threshold for any source.

### Parameter Recalibration

A scripted (not auto) recalibration:

- Re-run backtest parameter sweep over the regression window.
- Propose new `signals.<src>.weight`, scoring thresholds, exit rules.
- Operator reviews the diff and applies via PR.

No auto-apply. Quant rule: a system that automatically retunes its risk thresholds can mask a structural regression with a noisier set of parameters.

## Consequences

### Positive

- Vocabulary matches the code.
- The feedback loop becomes concrete: closed trades → `signal_source_stats` → alerts → operator-driven recalibration.
- Removes pressure to build a "retraining" subsystem that has no model to retrain.

### Negative / accepted trade-offs

- Manual loop. We are explicitly trading "fully automated" for "auditable and stable."
- If Qwen ever gets fine-tuned later (e.g. on a news-sentiment dataset), this ADR will need a successor.

### Neutral

- All existing closed-trade → signal-stats logic stays exactly as is.

## Alternatives considered

### A — Build a real retraining pipeline

Rejected on the grounds that there is nothing to retrain today. Cost vs value is wrong.

### B — Auto-apply parameter changes when PRD trips

Rejected. Too risky — masks regressions; couples model drift to silent config changes.

## References

- Code: `src/database.py::get_signal_accuracy_multipliers`, `src/strategy/scorer.py` (accuracy weights), `signal_source_stats` table
- Docs: `GOAL_ARCHITECTURE.md` §8, Feedback Loop; `CURRENT_DESIGN_ARCHITECTURE.md` §§8, Feedback loop

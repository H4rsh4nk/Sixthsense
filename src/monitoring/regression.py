"""Performance regression detection (ADR-0006).

Daily post-market job that detects when the live system's measured performance
degrades against a recent baseline. This is intentionally *not* an ML
retraining pipeline — see `docs/decisions/0006-performance-regression-not-retraining.md`
for the framing.

The job evaluates three checks:

1. **Per-source win-rate regression.** For each signal source with at least
   ``regression_min_trades`` closed trades, compare the 30-trade win rate vs
   the 90-trade baseline. Alert when the 30-trade rate drops below
   ``regression_win_rate_floor`` **and** trails the 90-trade baseline by at
   least ``regression_win_rate_delta``.
2. **Portfolio Sharpe floor.** Compute the rolling 30-day Sharpe from
   ``equity_snapshots`` and alert when below ``regression_sharpe_floor``.
3. **Drift detection (planned).** Reserved for ADR-0009 — KL divergence on
   signal-strength distributions. Returns an empty list today.

Each finding is persisted to the ``regression_alerts`` table and, when
``alerts.telegram_enabled`` is on, emitted through ``AlertManager``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.config import AppConfig
from src.database import Database
from src.monitoring.alerts import AlertManager

logger = logging.getLogger(__name__)


# Annualisation factor for daily Sharpe (252 trading days/year).
_TRADING_DAYS_PER_YEAR = 252
_WIN_RATE_RECENT_WINDOW = 30
_WIN_RATE_BASELINE_WINDOW = 90
_SHARPE_WINDOW_DAYS = 30


@dataclass
class RegressionFinding:
    """One regression alert, ready to persist + emit."""

    alert_type: str           # win_rate | sharpe | drift
    severity: str             # info | warning | critical
    metric_value: float
    threshold: float
    detail: str
    signal_type: str | None = None

    def to_row(self, when: datetime) -> dict[str, Any]:
        return {
            "check_date": when.date().isoformat(),
            "check_time": when.isoformat(),
            "alert_type": self.alert_type,
            "signal_type": self.signal_type,
            "severity": self.severity,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


# ---- metric helpers --------------------------------------------------------


def _win_rate(trades: list[dict]) -> float:
    """Win rate over a list of closed trade dicts (each must have ``pnl_pct``)."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if (t.get("pnl_pct") or 0.0) > 0)
    return wins / len(trades)


def _trades_for_source(trades: list[dict], source: str) -> list[dict]:
    """Filter closed trades to those whose ``signal_type`` list contains ``source``."""
    out = []
    for t in trades:
        sources = [s.strip() for s in (t.get("signal_type") or "").split(",") if s.strip()]
        if source in sources:
            out.append(t)
    return out


def _annualised_sharpe(returns: list[float]) -> float | None:
    """Annualised Sharpe ratio from a daily-return series. None if undefined."""
    if len(returns) < 2:
        return None
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _daily_returns_from_snapshots(snapshots: list[dict]) -> list[float]:
    """Compute daily return series from chronological equity snapshots."""
    returns: list[float] = []
    for prev, cur in zip(snapshots, snapshots[1:]):
        e_prev = float(prev.get("total_equity") or 0.0)
        e_cur = float(cur.get("total_equity") or 0.0)
        if e_prev > 0:
            returns.append((e_cur - e_prev) / e_prev)
    return returns


# ---- evaluation ------------------------------------------------------------


def _signal_sources_from_trades(trades: list[dict]) -> set[str]:
    """Collect the set of signal-source names appearing in ``trades.signal_type``."""
    sources: set[str] = set()
    for t in trades:
        for s in (t.get("signal_type") or "").split(","):
            s = s.strip()
            if s:
                sources.add(s)
    return sources


def evaluate(db: Database, config: AppConfig) -> list[RegressionFinding]:
    """Run all configured regression checks; return findings (may be empty)."""
    cfg = config.monitoring
    if not cfg.regression_check_enabled:
        return []

    findings: list[RegressionFinding] = []

    # ---- win-rate regression per signal source ---------------------------
    recent_trades = db.get_recent_closed_trades(limit=_WIN_RATE_BASELINE_WINDOW * 2)
    if recent_trades:
        for source in _signal_sources_from_trades(recent_trades):
            src_trades = _trades_for_source(recent_trades, source)
            if len(src_trades) < cfg.regression_min_trades:
                continue
            recent = src_trades[:_WIN_RATE_RECENT_WINDOW]
            baseline = src_trades[:_WIN_RATE_BASELINE_WINDOW]
            recent_wr = _win_rate(recent)
            baseline_wr = _win_rate(baseline)

            below_floor = recent_wr < cfg.regression_win_rate_floor
            below_baseline = (baseline_wr - recent_wr) >= cfg.regression_win_rate_delta
            if below_floor and below_baseline:
                severity = "critical" if recent_wr < cfg.regression_win_rate_floor / 2 else "warning"
                findings.append(
                    RegressionFinding(
                        alert_type="win_rate",
                        signal_type=source,
                        severity=severity,
                        metric_value=recent_wr,
                        threshold=cfg.regression_win_rate_floor,
                        detail=(
                            f"30-trade win rate {recent_wr:.2%} vs 90-trade baseline "
                            f"{baseline_wr:.2%} (Δ={baseline_wr - recent_wr:.2%}) for "
                            f"source '{source}'"
                        ),
                    )
                )

    # ---- portfolio Sharpe floor ------------------------------------------
    snapshots = db.get_recent_equity_snapshots(days=max(_SHARPE_WINDOW_DAYS * 2, 60))
    if len(snapshots) >= cfg.regression_min_equity_days:
        window = snapshots[-(_SHARPE_WINDOW_DAYS + 1):]
        daily = _daily_returns_from_snapshots(window)
        sharpe = _annualised_sharpe(daily)
        if sharpe is not None and sharpe < cfg.regression_sharpe_floor:
            severity = "critical" if sharpe < cfg.regression_sharpe_floor - 1.0 else "warning"
            findings.append(
                RegressionFinding(
                    alert_type="sharpe",
                    signal_type=None,
                    severity=severity,
                    metric_value=sharpe,
                    threshold=cfg.regression_sharpe_floor,
                    detail=(
                        f"30-day annualised Sharpe {sharpe:.2f} below floor "
                        f"{cfg.regression_sharpe_floor:.2f}"
                    ),
                )
            )

    # ---- drift (deferred to ADR-0009) ------------------------------------
    # placeholder so callers can iterate over the full return shape.

    return findings


def run_check(
    db: Database,
    config: AppConfig,
    alerts: AlertManager | None = None,
    *,
    when: datetime | date | None = None,
) -> list[RegressionFinding]:
    """Evaluate, persist, and (optionally) Telegram-emit regression findings.

    Safe to call from `post_market_review`. Emits an INFO log even when no
    findings trigger, so the operator can see the check ran.
    """
    now = (
        when
        if isinstance(when, datetime)
        else datetime.combine(when or date.today(), datetime.min.time())
    )

    findings = evaluate(db, config)
    if not findings:
        logger.info("Regression check ran; no findings.")
        return []

    rows = [f.to_row(now) for f in findings]
    db.insert_regression_alerts(rows)

    for f in findings:
        level_fn = logger.error if f.severity == "critical" else logger.warning
        prefix = "REGRESSION CRITICAL" if f.severity == "critical" else "REGRESSION WARNING"
        scope = f.signal_type or "portfolio"
        level_fn("%s [%s/%s] %s", prefix, f.alert_type, scope, f.detail)

        if alerts is not None and alerts.enabled:
            emoji = "🚨" if f.severity == "critical" else "⚠️"
            try:
                alerts.send(
                    f"{emoji} *Regression {f.severity.upper()}*\n"
                    f"Type: `{f.alert_type}`\n"
                    f"Scope: `{scope}`\n"
                    f"Metric: `{f.metric_value:.4f}` (threshold `{f.threshold:.4f}`)\n"
                    f"{f.detail}"
                )
            except Exception as e:  # noqa: BLE001 - alerting is best-effort
                logger.error("Failed to send regression alert: %s", e)

    return findings

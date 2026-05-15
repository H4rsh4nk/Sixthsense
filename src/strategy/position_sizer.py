"""Position sizing — determines how many shares to buy per trade.

Two modes (see `docs/decisions/0003-fractional-kelly-sizing.md`):

- ``fixed_risk`` (default): legacy formula
  ``shares = (equity * risk_per_trade_pct) / (entry - stop)``.
- ``fractional_kelly``: derives a target position fraction from per-source
  win-rate and win/loss ratios in `signal_source_stats` (computed live from
  closed trades), multiplied by ``kelly_fraction`` and capped at
  ``kelly_max_position_pct``. Falls back to ``fixed_risk`` if any contributing
  source has fewer than ``kelly_min_trades_per_source`` closed trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Computed position size and its rationale."""

    shares: int
    entry_price: float
    stop_loss_price: float
    position_value: float
    risk_amount: float
    risk_pct: float
    sizing_mode_used: str = "fixed_risk"        # fixed_risk | fractional_kelly
    kelly_target_pct: float | None = None       # final f after fraction + cap
    kelly_fallback_reason: str | None = None    # populated when Kelly downgraded
    per_source_kelly: dict[str, float] = field(default_factory=dict)


class PositionSizer:
    """Position sizer supporting fixed-risk and fractional-Kelly modes."""

    def __init__(self, config: AppConfig):
        t = config.trading
        self.risk_pct = t.risk_per_trade_pct
        self.max_position_pct = t.max_single_position_pct
        self.sizing_mode = t.sizing_mode
        self.kelly_fraction = max(0.0, min(0.5, float(t.kelly_fraction)))
        self.kelly_min_trades = max(1, int(t.kelly_min_trades_per_source))
        self.kelly_max_position_pct = max(0.0, float(t.kelly_max_position_pct))

    def calculate(
        self,
        account_equity: float,
        available_cash: float,
        entry_price: float,
        stop_loss_pct: float,
        *,
        signal_sources: list[str] | None = None,
        kelly_inputs: dict[str, dict] | None = None,
    ) -> PositionSize | None:
        """Calculate position size.

        Args:
            account_equity: Total account value (cash + positions).
            available_cash: Cash available for new positions.
            entry_price: Expected entry price.
            stop_loss_pct: Stop-loss distance as a decimal (e.g., 0.05 = 5%).
            signal_sources: Signal types contributing to the candidate. Used in
                fractional-Kelly mode to look up per-source priors.
            kelly_inputs: Output of `Database.get_signal_kelly_inputs()`. When
                None or the sizing mode is fixed-risk, ignored.

        Returns:
            PositionSize or None if the trade is not viable.
        """

        if entry_price <= 0 or stop_loss_pct <= 0:
            return None

        stop_loss_price = entry_price * (1 - stop_loss_pct)
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return None

        # Pick a desired share count by sizing intent (fixed-risk vs Kelly).
        # Shared constraints below shrink it: max-position-pct cap, cash cap.
        sizing_mode_used = "fixed_risk"
        kelly_target_pct: float | None = None
        per_source_kelly: dict[str, float] = {}
        fallback_reason: str | None = None

        kelly_pct: float | None = None
        if self.sizing_mode == "fractional_kelly":
            kelly_pct, per_source_kelly, fallback_reason = self._fractional_kelly_pct(
                signal_sources, kelly_inputs
            )
            if kelly_pct is None and fallback_reason:
                logger.info(
                    "Kelly sizing downgraded to fixed_risk: %s", fallback_reason
                )

        if kelly_pct is not None:
            sizing_mode_used = "fractional_kelly"
            kelly_target_pct = kelly_pct
            target_position_value = account_equity * kelly_pct
            shares_by_intent = int(target_position_value / entry_price)
        else:
            # Fixed-risk: shares sized so stop-loss caps loss at risk_amount.
            risk_amount = account_equity * self.risk_pct
            shares_by_intent = int(risk_amount / risk_per_share)

        # Shared constraints — apply to both modes.
        shares_by_max = int((account_equity * self.max_position_pct) / entry_price)
        shares_by_cash = int(available_cash / entry_price)

        shares = min(shares_by_intent, shares_by_max, shares_by_cash)
        if shares <= 0:
            return None

        position_value = shares * entry_price
        actual_risk = shares * risk_per_share

        return PositionSize(
            shares=shares,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_pct=actual_risk / account_equity if account_equity > 0 else 0,
            sizing_mode_used=sizing_mode_used,
            kelly_target_pct=kelly_target_pct,
            kelly_fallback_reason=fallback_reason,
            per_source_kelly=per_source_kelly,
        )

    # ---- mode helpers -----------------------------------------------------

    def _fractional_kelly_pct(
        self,
        signal_sources: list[str] | None,
        kelly_inputs: dict[str, dict] | None,
    ) -> tuple[float | None, dict[str, float], str | None]:
        """Compute the target position % of equity via fractional Kelly.

        Returns ``(target_pct, per_source_breakdown, fallback_reason)``. If
        ``target_pct`` is None the caller falls back to fixed-risk mode.
        """

        if not signal_sources:
            return None, {}, "no signal_sources on candidate"
        if not kelly_inputs:
            return None, {}, "no kelly_inputs available"

        per_source: dict[str, float] = {}
        for src in signal_sources:
            stats = kelly_inputs.get(src)
            if not stats or stats["n"] < self.kelly_min_trades:
                return (
                    None,
                    {},
                    f"insufficient closed trades for source '{src}' "
                    f"(have {stats['n'] if stats else 0}, need {self.kelly_min_trades})",
                )
            p = stats["wins"] / stats["n"] if stats["n"] else 0.0
            avg_loss = stats["avg_loss_pct"]
            avg_win = stats["avg_win_pct"]
            if avg_loss <= 0 or avg_win <= 0:
                return None, {}, f"source '{src}' lacks both wins and losses"
            b = avg_win / avg_loss
            f_star = (p * b - (1 - p)) / b  # full Kelly fraction
            f_star = max(0.0, f_star)
            per_source[src] = f_star

        if not per_source:
            return None, {}, "no usable per-source Kelly fractions"

        # Aggregate sources by simple mean; tighter ensembles tend to be
        # over-confident, so this is intentionally conservative.
        avg_full_kelly = sum(per_source.values()) / len(per_source)
        target_pct = avg_full_kelly * self.kelly_fraction
        target_pct = min(target_pct, self.kelly_max_position_pct)
        target_pct = min(target_pct, self.max_position_pct)
        if target_pct <= 0:
            return None, per_source, "computed Kelly fraction non-positive"

        return target_pct, per_source, None

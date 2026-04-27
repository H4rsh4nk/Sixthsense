"""Position sizing — determines how many shares to buy per trade."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Computed position size."""
    shares: int
    entry_price: float
    stop_loss_price: float
    position_value: float
    risk_amount: float
    risk_pct: float


class PositionSizer:
    """Fixed-percentage risk position sizing.

    Formula: shares = (account_equity * risk_pct) / (entry_price - stop_price)

    Constraints:
      - Never exceed max_single_position_pct of account in one trade
      - Never buy more than available cash
      - Always buy at least 1 share
    """

    def __init__(self, config: AppConfig):
        self.risk_pct = config.trading.risk_per_trade_pct
        self.max_position_pct = config.trading.max_single_position_pct

    def calculate(
        self,
        account_equity: float,
        available_cash: float,
        entry_price: float,
        stop_loss_pct: float,
    ) -> PositionSize | None:
        """Calculate position size.

        Args:
            account_equity: Total account value (cash + positions).
            available_cash: Cash available for new positions.
            entry_price: Expected entry price.
            stop_loss_pct: Stop-loss distance as a decimal (e.g., 0.05 = 5%).

        Returns:
            PositionSize or None if trade is not viable.
        """
        if entry_price <= 0 or stop_loss_pct <= 0:
            return None

        stop_loss_price = entry_price * (1 - stop_loss_pct)
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0:
            return None

        # shares = risk_amount / risk_per_share
        risk_amount = account_equity * self.risk_pct
        shares_by_risk = int(risk_amount / risk_per_share)

        # Cap by max position size
        max_value = account_equity * self.max_position_pct
        shares_by_max = int(max_value / entry_price)

        # Cap by available cash
        shares_by_cash = int(available_cash / entry_price)

        shares = min(shares_by_risk, shares_by_max, shares_by_cash)

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
        )

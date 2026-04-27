"""Exit manager — time-based exits and stop-loss management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from src.config import AppConfig
from src.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """A signal to exit a position."""
    trade_id: int
    ticker: str
    reason: str  # time_exit, stop_loss, trailing_stop, circuit_breaker
    target_price: float | None = None


class ExitManager:
    """Manages trade exits: time-based, stop-loss, and trailing stops."""

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.hold_days = config.exit_rules.default_hold_days
        self.stop_loss_pct = config.exit_rules.stop_loss_pct
        self.trailing_stop = config.exit_rules.trailing_stop
        self.trailing_stop_pct = config.exit_rules.trailing_stop_pct

        # Track highest price since entry for trailing stops
        self._high_water_marks: dict[int, float] = {}  # trade_id → highest price

    def compute_target_exit_date(self, entry_date: date) -> date:
        """Compute the target exit date (entry + N trading days)."""
        current = entry_date
        days_added = 0
        while days_added < self.hold_days:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                days_added += 1
        return current

    def compute_stop_loss(self, entry_price: float, direction: str = "long") -> float:
        """Compute initial stop-loss price."""
        if direction == "long":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def register_trade(self, trade_id: int, entry_price: float) -> None:
        """Register a new trade for trailing stop tracking."""
        self._high_water_marks[trade_id] = entry_price

    def check_exits(self, current_date: date, current_prices: dict[str, float]) -> list[ExitSignal]:
        """Check all open trades for exit conditions.

        Args:
            current_date: Today's date.
            current_prices: Dict of ticker → current price.

        Returns:
            List of ExitSignal for positions that should be closed.
        """
        open_trades = self.db.get_open_trades()
        exits = []

        for trade in open_trades:
            trade_id = trade["id"]
            ticker = trade["ticker"]
            stop_loss_price = trade["stop_loss_price"]
            target_exit = date.fromisoformat(trade["target_exit_date"])

            if ticker not in current_prices:
                continue

            current_price = current_prices[ticker]

            # Check stop-loss
            if trade["direction"] == "long" and current_price <= stop_loss_price:
                exits.append(ExitSignal(
                    trade_id=trade_id,
                    ticker=ticker,
                    reason="stop_loss",
                    target_price=stop_loss_price,
                ))
                continue

            # Check time-based exit
            if current_date >= target_exit:
                exits.append(ExitSignal(
                    trade_id=trade_id,
                    ticker=ticker,
                    reason="time_exit",
                    target_price=None,  # Will exit at market
                ))
                continue

            # Update trailing stop if enabled
            if self.trailing_stop:
                high = self._high_water_marks.get(trade_id, trade["entry_price"])
                if current_price > high:
                    self._high_water_marks[trade_id] = current_price
                    high = current_price

                trailing_stop_price = high * (1 - self.trailing_stop_pct)
                if current_price <= trailing_stop_price and trailing_stop_price > stop_loss_price:
                    exits.append(ExitSignal(
                        trade_id=trade_id,
                        ticker=ticker,
                        reason="trailing_stop",
                        target_price=trailing_stop_price,
                    ))

        return exits

    def close_all_positions(self, reason: str = "circuit_breaker") -> list[ExitSignal]:
        """Generate exit signals for all open positions."""
        open_trades = self.db.get_open_trades()
        return [
            ExitSignal(
                trade_id=trade["id"],
                ticker=trade["ticker"],
                reason=reason,
            )
            for trade in open_trades
        ]

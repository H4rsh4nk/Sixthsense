"""Risk manager — drawdown tracking, circuit breaker, and position limits."""

from __future__ import annotations

import logging
from datetime import date

from src.config import AppConfig
from src.database import Database

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk limits: max drawdown, daily loss, sector concentration."""

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.max_drawdown_pct = config.trading.max_drawdown_pct
        self.daily_loss_limit_pct = config.trading.daily_loss_limit_pct
        self.max_concurrent = config.trading.max_concurrent_positions
        self.max_sector = config.trading.max_sector_positions
        self.initial_capital = config.trading.capital

        self._peak_equity = config.trading.capital
        self._circuit_breaker_active = False
        self._daily_starting_equity = config.trading.capital

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker after review."""
        self._circuit_breaker_active = False
        logger.info("Circuit breaker manually reset.")

    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking and check circuit breaker."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        if drawdown >= self.max_drawdown_pct and not self._circuit_breaker_active:
            self._circuit_breaker_active = True
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: drawdown={drawdown:.2%} "
                f"(peak=${self._peak_equity:,.2f}, current=${current_equity:,.2f})"
            )

    def start_new_day(self, current_equity: float) -> None:
        """Reset daily tracking at start of trading day."""
        self._daily_starting_equity = current_equity

    def check_daily_loss(self, current_equity: float) -> bool:
        """Check if daily loss limit has been hit. Returns True if limit exceeded."""
        if self._daily_starting_equity <= 0:
            return False
        daily_loss = (self._daily_starting_equity - current_equity) / self._daily_starting_equity
        return daily_loss >= self.daily_loss_limit_pct

    def can_open_position(self, ticker: str, sector: str = "") -> tuple[bool, str]:
        """Check if a new position is allowed under risk rules.

        Returns (allowed: bool, reason: str).
        """
        if self._circuit_breaker_active:
            return False, "Circuit breaker is active — no new trades allowed"

        # Check concurrent position limit
        open_trades = self.db.get_open_trades()
        if len(open_trades) >= self.max_concurrent:
            return False, f"Max concurrent positions ({self.max_concurrent}) reached"

        # Check if already holding this ticker
        held_tickers = {t["ticker"] for t in open_trades}
        if ticker in held_tickers:
            return False, f"Already holding {ticker}"

        # Check sector concentration
        if sector:
            sector_count = sum(1 for t in open_trades if self._get_sector(t["ticker"]) == sector)
            if sector_count >= self.max_sector:
                return False, f"Max sector positions ({self.max_sector}) reached for {sector}"

        return True, "OK"

    def get_current_drawdown(self, current_equity: float) -> float:
        """Get current drawdown percentage."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - current_equity) / self._peak_equity

    def get_status(self, current_equity: float) -> dict:
        """Get a summary of risk status."""
        drawdown = self.get_current_drawdown(current_equity)
        open_trades = self.db.get_open_trades()

        return {
            "current_equity": current_equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "circuit_breaker_active": self._circuit_breaker_active,
            "open_positions": len(open_trades),
            "max_positions": self.max_concurrent,
            "daily_starting_equity": self._daily_starting_equity,
        }

    def _get_sector(self, ticker: str) -> str:
        """Look up sector for a ticker."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT sector FROM universe WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            return row["sector"] if row else ""

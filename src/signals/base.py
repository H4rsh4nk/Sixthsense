"""Abstract base class for all signal generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class SignalResult:
    """Result from a signal generator."""
    ticker: str
    signal_date: date
    signal_type: str
    strength: float  # -1.0 (strong sell) to 1.0 (strong buy)
    direction: str  # "long" or "short"
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.strength > 0 and self.direction == "long"

    @property
    def is_sell(self) -> bool:
        return self.strength < 0 or self.direction == "short"


class Signal(ABC):
    """Base class for all signal generators."""

    signal_type: str = "base"

    @abstractmethod
    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Generate a signal for a ticker as of a specific date.

        Returns None if no signal is detected.
        Only uses data available on or before as_of_date (no look-ahead).
        """
        ...

    @abstractmethod
    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        """Generate signals for multiple tickers. Returns only tickers with signals."""
        ...

    @abstractmethod
    def backfill(self, start_date: date, end_date: date) -> None:
        """Download and store historical data needed for backtesting."""
        ...

"""Signal scorer and ranker — combines multiple signals into trade candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class TradeCandidate:
    """A scored trade candidate ready for position sizing."""
    ticker: str
    combined_score: float
    direction: str  # long | short
    signal_sources: list[str]
    num_signals: int
    signals: list[SignalResult]
    sector: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalScorer:
    """Combines signals from multiple generators into ranked trade candidates."""

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.weight_map = {
            "insider": config.signals.insider.weight,
            "news": config.signals.news.weight,
            "political": config.signals.political.weight,
            "price_action": config.signals.price_action.weight,
        }

    def score(
        self, signals: list[SignalResult], as_of_date: date
    ) -> list[TradeCandidate]:
        """Score and rank signal results into trade candidates.

        Groups signals by ticker, computes weighted combined score,
        and filters by minimum thresholds.
        """
        # Group by ticker
        by_ticker: dict[str, list[SignalResult]] = {}
        for sig in signals:
            by_ticker.setdefault(sig.ticker, []).append(sig)

        candidates = []
        for ticker, sigs in by_ticker.items():
            # Weighted combination
            weighted_sum = 0.0
            total_weight = 0.0
            direction_votes = {"long": 0, "short": 0}

            for sig in sigs:
                w = self.weight_map.get(sig.signal_type, 0.25)
                weighted_sum += sig.strength * w
                total_weight += w
                direction_votes[sig.direction] += w

            if total_weight == 0:
                continue

            combined = weighted_sum / total_weight
            direction = "long" if direction_votes["long"] >= direction_votes["short"] else "short"

            # Apply minimum thresholds
            if abs(combined) < self.config.scoring.min_combined_score:
                continue
            if len(sigs) < self.config.scoring.min_signals_agreeing:
                continue

            # Get sector info
            sector = self._get_sector(ticker)

            candidates.append(TradeCandidate(
                ticker=ticker,
                combined_score=abs(combined),
                direction=direction,
                signal_sources=[s.signal_type for s in sigs],
                num_signals=len(sigs),
                signals=sigs,
                sector=sector,
                metadata={
                    "individual_scores": {
                        s.signal_type: {"strength": s.strength, "confidence": s.confidence}
                        for s in sigs
                    },
                },
            ))

        # Rank by combined score (highest first)
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates

    def _get_sector(self, ticker: str) -> str:
        """Look up ticker's sector from universe table."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT sector FROM universe WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            return row["sector"] if row else ""

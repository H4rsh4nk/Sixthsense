"""Broad market regime signal from benchmark momentum (macro-style prior)."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


class MacroSignal(Signal):
    """Consensus risk-on/off tilt inferred from benchmark total return."""

    signal_type = "macro"

    def __init__(self, config: AppConfig, db: Database):
        self.cfg = config.signals.macro
        self.db = db
        self.app_config = config

    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Per-ticker entry point; prefers generate_bulk which computes regime once."""

        tilt = self._regime_snapshot(as_of_date)
        return self._to_result(ticker, as_of_date, tilt) if tilt is not None else None

    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        if not self.cfg.enabled:
            return []
        tilt = self._regime_snapshot(as_of_date)
        if tilt is None:
            return []
        return [
            res
            for t in tickers
            if (res := self._to_result(t, as_of_date, tilt)) is not None
        ]

    def _to_result(
        self, ticker: str, as_of_date: date, tilt: dict[str, Any]
    ) -> SignalResult:
        strength = tilt["tilt_strength"]
        direction = tilt["direction"]
        return SignalResult(
            ticker=ticker,
            signal_date=as_of_date,
            signal_type=self.signal_type,
            strength=strength if direction == "long" else -strength,
            direction=direction,
            confidence=tilt["confidence"],
            metadata=tilt,
        )

    def _regime_snapshot(self, as_of_date: date) -> dict[str, Any] | None:
        start = as_of_date - timedelta(days=max(self.cfg.lookback_sessions * 2, 21))
        start_s = start.isoformat()
        end_s = as_of_date.isoformat()
        rows = self.db.get_prices(self.cfg.benchmark_ticker, start_s, end_s)
        if len(rows) < self.cfg.lookback_sessions + 1:
            logger.debug(
                "Macro signal skipped — %s history short (%s rows)",
                self.cfg.benchmark_ticker,
                len(rows),
            )
            return None

        closes = [float(r["close"]) for r in rows[-(self.cfg.lookback_sessions + 1) :]]
        cumulative = closes[-1] / closes[0] - 1.0

        if cumulative >= self.cfg.risk_on_return:
            direction = "long"
            denom = abs(self.cfg.risk_on_return) if self.cfg.risk_on_return else 1e-6
            mag = min(1.0, cumulative / denom)
            conf = min(1.0, 0.5 + mag * 0.25)
        elif cumulative <= self.cfg.risk_off_return:
            direction = "short"
            denom = abs(self.cfg.risk_off_return) if self.cfg.risk_off_return else 1e-6
            mag = min(1.0, abs(cumulative) / denom)
            conf = min(1.0, 0.5 + mag * 0.25)
        else:
            return None

        strength = round(self.cfg.tilt_strength * mag, 4)
        return {
            "regime_direction": direction,
            "benchmark": self.cfg.benchmark_ticker,
            "cumulative_return_window": round(cumulative * 100, 4),
            "lookback_sessions": self.cfg.lookback_sessions,
            "direction": direction,
            "tilt_strength": strength,
            "confidence": round(conf, 4),
        }

    def backfill(self, start_date: date, end_date: date) -> None:
        """Uses shared price pipeline; macro has no standalone fetch."""

        _ = start_date
        _ = end_date

"""Price action / technical analysis signal generator.

Uses RSI, MACD, and volume spikes to confirm or generate signals.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import ta

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


class PriceActionSignal(Signal):
    """Generate signals from technical indicators."""

    signal_type = "price_action"

    def __init__(self, config: AppConfig, db: Database):
        self.config = config.signals.price_action
        self.app_config = config
        self.db = db

    def _get_price_df(self, ticker: str, as_of_date: date, lookback_days: int = 60) -> pd.DataFrame:
        """Get price data as a DataFrame with technical indicators computed."""
        start = (as_of_date - timedelta(days=lookback_days)).isoformat()
        end = as_of_date.isoformat()

        prices = self.db.get_prices(ticker, start, end)
        if len(prices) < 20:  # Need at least 20 bars for indicators
            return pd.DataFrame()

        df = pd.DataFrame(prices)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # RSI (14-period)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Volume spike (vs 20-day average)
        df["volume_sma20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma20"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"] = bb.bollinger_pband()

        return df

    def _analyze_rsi(self, df: pd.DataFrame) -> dict:
        """Analyze RSI for oversold/overbought conditions."""
        if df.empty or "rsi" not in df.columns:
            return {"signal": 0, "value": None}

        current_rsi = df["rsi"].iloc[-1]

        if pd.isna(current_rsi):
            return {"signal": 0, "value": None}

        if current_rsi <= self.config.rsi_oversold:
            # Oversold → bullish signal
            strength = (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold
            return {"signal": min(strength, 1.0), "value": current_rsi}
        elif current_rsi >= self.config.rsi_overbought:
            # Overbought → bearish signal
            strength = (current_rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
            return {"signal": -min(strength, 1.0), "value": current_rsi}

        return {"signal": 0, "value": current_rsi}

    def _analyze_macd(self, df: pd.DataFrame) -> dict:
        """Analyze MACD for crossover signals."""
        if df.empty or "macd_diff" not in df.columns or len(df) < 2:
            return {"signal": 0, "crossover": None}

        current_diff = df["macd_diff"].iloc[-1]
        prev_diff = df["macd_diff"].iloc[-2]

        if pd.isna(current_diff) or pd.isna(prev_diff):
            return {"signal": 0, "crossover": None}

        # Bullish crossover: MACD crosses above signal line
        if prev_diff <= 0 and current_diff > 0:
            return {"signal": 0.7, "crossover": "bullish"}
        # Bearish crossover: MACD crosses below signal line
        elif prev_diff >= 0 and current_diff < 0:
            return {"signal": -0.7, "crossover": "bearish"}

        return {"signal": 0, "crossover": None}

    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        """Analyze volume for unusual spikes."""
        if df.empty or "volume_ratio" not in df.columns:
            return {"signal": 0, "ratio": None}

        ratio = df["volume_ratio"].iloc[-1]

        if pd.isna(ratio):
            return {"signal": 0, "ratio": None}

        if ratio >= self.config.volume_spike_multiplier:
            # Volume spike — direction depends on price movement
            price_change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
            if price_change > 0:
                return {"signal": min(ratio / 4.0, 1.0), "ratio": ratio}
            else:
                return {"signal": -min(ratio / 4.0, 1.0), "ratio": ratio}

        return {"signal": 0, "ratio": ratio}

    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Generate price action signal combining multiple indicators."""
        df = self._get_price_df(ticker, as_of_date)
        if df.empty:
            return None

        # Filter to data up to as_of_date (no look-ahead)
        as_of_ts = pd.Timestamp(as_of_date)
        df = df[df.index <= as_of_ts]
        if df.empty:
            return None

        rsi_result = self._analyze_rsi(df)
        macd_result = self._analyze_macd(df)
        volume_result = self._analyze_volume(df)

        # Count agreeing signals
        signals = [rsi_result["signal"], macd_result["signal"], volume_result["signal"]]
        active_signals = [s for s in signals if s != 0]

        if not active_signals:
            return None

        # Weighted average of active signals
        avg_signal = sum(active_signals) / len(active_signals)

        # Require at least some signal strength
        if abs(avg_signal) < 0.2:
            return None

        direction = "long" if avg_signal > 0 else "short"

        return SignalResult(
            ticker=ticker,
            signal_date=as_of_date,
            signal_type=self.signal_type,
            strength=max(min(avg_signal, 1.0), -1.0),
            direction=direction,
            confidence=len(active_signals) / 3.0,
            metadata={
                "rsi": rsi_result,
                "macd": macd_result,
                "volume": volume_result,
                "active_indicators": len(active_signals),
                "current_price": float(df["close"].iloc[-1]),
            },
        )

    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        """Generate price action signals for all tickers."""
        results = []
        for ticker in tickers:
            signal = self.generate(ticker, as_of_date)
            if signal:
                results.append(signal)
        return results

    def backfill(self, start_date: date, end_date: date) -> None:
        """No separate backfill needed — uses price data already in DB."""
        logger.info("Price action signal uses existing price data — no backfill needed.")

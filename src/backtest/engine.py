"""Event-driven backtesting engine.

Simulates the full trading pipeline: signal generation → scoring → position sizing →
entry → stop-loss monitoring → time-based exit. No look-ahead bias.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """A position held during backtesting."""
    trade_id: int
    ticker: str
    direction: str
    entry_date: date
    entry_price: float
    shares: float
    stop_loss_price: float
    target_exit_date: date
    signal_type: str
    signal_score: float
    cost_basis: float = 0.0

    def __post_init__(self):
        self.cost_basis = self.entry_price * self.shares


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    initial_capital: float = 10000
    final_equity: float = 10000
    total_return_pct: float = 0.0
    num_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    parameters: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db

    def run(
        self,
        signals: list[Signal],
        start_date: date,
        end_date: date,
        hold_days: int | None = None,
        stop_loss_pct: float | None = None,
        signal_weights: dict[str, float] | None = None,
        min_combined_score: float | None = None,
    ) -> BacktestResult:
        """Run a backtest over a date range.

        Args:
            signals: List of Signal generators to use.
            start_date: Backtest start date.
            end_date: Backtest end date.
            hold_days: Override default hold period.
            stop_loss_pct: Override default stop-loss percentage.
            signal_weights: Override signal weights.
            min_combined_score: Override minimum combined score to trade.
        """
        hold_days = hold_days or self.config.exit_rules.default_hold_days
        stop_loss_pct = stop_loss_pct or self.config.exit_rules.stop_loss_pct
        min_score = min_combined_score or self.config.scoring.min_combined_score
        slippage = self.config.backtest.slippage_pct

        # Get trading universe
        tickers = self.db.get_all_tickers()
        if not tickers:
            logger.error("No tickers in universe. Run data pipeline first.")
            return BacktestResult()

        # Build weight map
        weights = signal_weights or {
            s.signal_type: getattr(self.config.signals, s.signal_type, None)
            for s in signals
        }
        weight_map = {}
        for s in signals:
            cfg = getattr(self.config.signals, s.signal_type, None)
            weight_map[s.signal_type] = cfg.weight if cfg else 0.25

        if signal_weights:
            weight_map = signal_weights

        # Initialize state
        cash = self.config.backtest.initial_capital
        positions: list[BacktestPosition] = []
        closed_trades: list[dict] = []
        equity_curve: list[dict] = []
        peak_equity = cash
        trade_counter = 0

        # Get all trading dates
        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info(
            f"Backtesting {len(signals)} signals over {len(trading_dates)} trading days "
            f"(hold={hold_days}d, stop={stop_loss_pct:.1%})"
        )

        for current_date in trading_dates:
            date_str = current_date.isoformat()

            # Get today's prices for all tickers (open, high, low, close)
            day_prices = self._get_day_prices(current_date)
            if not day_prices:
                continue

            # Step 1: Check stop-losses and time exits on open positions
            positions_to_close = []
            for pos in positions:
                if pos.ticker not in day_prices:
                    continue

                prices = day_prices[pos.ticker]

                # Check stop-loss (assume it triggers at stop price if low <= stop)
                if prices["low"] <= pos.stop_loss_price:
                    exit_price = pos.stop_loss_price * (1 - slippage)
                    positions_to_close.append((pos, exit_price, "stop_loss", current_date))
                    continue

                # Check time-based exit
                if current_date >= pos.target_exit_date:
                    exit_price = prices["open"] * (1 - slippage)
                    positions_to_close.append((pos, exit_price, "time_exit", current_date))

            for pos, exit_price, reason, exit_date in positions_to_close:
                pnl = (exit_price - pos.entry_price) * pos.shares
                if pos.direction == "short":
                    pnl = -pnl
                pnl_pct = pnl / pos.cost_basis
                hold_actual = (exit_date - pos.entry_date).days

                closed_trades.append({
                    "trade_id": pos.trade_id,
                    "ticker": pos.ticker,
                    "direction": pos.direction,
                    "signal_type": pos.signal_type,
                    "signal_score": pos.signal_score,
                    "entry_date": pos.entry_date.isoformat(),
                    "entry_price": pos.entry_price,
                    "shares": pos.shares,
                    "exit_date": exit_date.isoformat(),
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "hold_days": hold_actual,
                })
                cash += pos.cost_basis + pnl
                positions.remove(pos)

            # Step 2: Generate signals (only if we have capacity for new positions)
            max_positions = self.config.trading.max_concurrent_positions
            if len(positions) < max_positions:
                candidates = []
                for signal_gen in signals:
                    try:
                        results = signal_gen.generate_bulk(tickers, current_date)
                        candidates.extend(results)
                    except Exception as e:
                        logger.debug(f"Signal gen failed on {current_date}: {e}")

                # Score and rank candidates
                scored = self._score_candidates(candidates, weight_map)
                scored = [s for s in scored if s["combined_score"] >= min_score]
                scored.sort(key=lambda x: x["combined_score"], reverse=True)

                # Filter out tickers we already hold
                held_tickers = {p.ticker for p in positions}
                scored = [s for s in scored if s["ticker"] not in held_tickers]

                # Enter new positions (up to capacity)
                slots = max_positions - len(positions)
                for entry in scored[:slots]:
                    ticker = entry["ticker"]
                    if ticker not in day_prices:
                        continue

                    # Enter at next day's open (simulated as today's close + slippage)
                    entry_price = day_prices[ticker]["close"] * (1 + slippage)
                    stop_price = entry_price * (1 - stop_loss_pct)

                    # Position sizing: risk-based
                    risk_per_share = entry_price - stop_price
                    if risk_per_share <= 0:
                        continue

                    risk_amount = cash * self.config.trading.risk_per_trade_pct
                    shares = int(risk_amount / risk_per_share)

                    # Cap at max single position size
                    max_position_value = cash * self.config.trading.max_single_position_pct
                    max_shares_by_value = int(max_position_value / entry_price)
                    shares = min(shares, max_shares_by_value)

                    if shares <= 0:
                        continue

                    cost = entry_price * shares
                    if cost > cash:
                        shares = int(cash / entry_price)
                        if shares <= 0:
                            continue
                        cost = entry_price * shares

                    trade_counter += 1
                    target_exit = self._add_trading_days(current_date, hold_days)

                    pos = BacktestPosition(
                        trade_id=trade_counter,
                        ticker=ticker,
                        direction=entry["direction"],
                        entry_date=current_date,
                        entry_price=entry_price,
                        shares=shares,
                        stop_loss_price=stop_price,
                        target_exit_date=target_exit,
                        signal_type=entry["signal_type"],
                        signal_score=entry["combined_score"],
                    )
                    positions.append(pos)
                    cash -= cost

            # Step 3: Record daily equity
            positions_value = 0
            for pos in positions:
                if pos.ticker in day_prices:
                    positions_value += day_prices[pos.ticker]["close"] * pos.shares

            total_equity = cash + positions_value
            peak_equity = max(peak_equity, total_equity)
            drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0

            equity_curve.append({
                "date": date_str,
                "cash": cash,
                "positions_value": positions_value,
                "total_equity": total_equity,
                "drawdown_pct": drawdown,
                "open_positions": len(positions),
            })

            # Circuit breaker check
            if drawdown >= self.config.trading.max_drawdown_pct:
                logger.info(f"Circuit breaker hit on {date_str}: drawdown={drawdown:.2%}")
                # Close all positions at market
                for pos in list(positions):
                    if pos.ticker in day_prices:
                        exit_price = day_prices[pos.ticker]["close"] * (1 - slippage)
                        pnl = (exit_price - pos.entry_price) * pos.shares
                        pnl_pct = pnl / pos.cost_basis
                        closed_trades.append({
                            "trade_id": pos.trade_id,
                            "ticker": pos.ticker,
                            "direction": pos.direction,
                            "signal_type": pos.signal_type,
                            "signal_score": pos.signal_score,
                            "entry_date": pos.entry_date.isoformat(),
                            "entry_price": pos.entry_price,
                            "shares": pos.shares,
                            "exit_date": date_str,
                            "exit_price": exit_price,
                            "exit_reason": "circuit_breaker",
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "hold_days": (current_date - pos.entry_date).days,
                        })
                        cash += pos.cost_basis + pnl
                positions.clear()
                break  # Stop trading after circuit breaker

        # Force-close any remaining positions at last available price
        if positions and equity_curve:
            last_date_str = equity_curve[-1]["date"]
            last_prices = self._get_day_prices(date.fromisoformat(last_date_str))
            for pos in positions:
                if pos.ticker in (last_prices or {}):
                    exit_price = last_prices[pos.ticker]["close"]
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    pnl_pct = pnl / pos.cost_basis
                    closed_trades.append({
                        "trade_id": pos.trade_id,
                        "ticker": pos.ticker,
                        "direction": pos.direction,
                        "signal_type": pos.signal_type,
                        "signal_score": pos.signal_score,
                        "entry_date": pos.entry_date.isoformat(),
                        "entry_price": pos.entry_price,
                        "shares": pos.shares,
                        "exit_date": last_date_str,
                        "exit_price": exit_price,
                        "exit_reason": "backtest_end",
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "hold_days": (date.fromisoformat(last_date_str) - pos.entry_date).days,
                    })

        # Calculate summary stats
        result = self._compute_results(
            closed_trades, equity_curve, self.config.backtest.initial_capital,
        )
        result.parameters = {
            "hold_days": hold_days,
            "stop_loss_pct": stop_loss_pct,
            "signal_types": [s.signal_type for s in signals],
            "min_combined_score": min_score,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        return result

    def _score_candidates(
        self, candidates: list[SignalResult], weight_map: dict[str, float]
    ) -> list[dict]:
        """Score and combine signals for the same ticker."""
        # Group by ticker
        by_ticker: dict[str, list[SignalResult]] = {}
        for c in candidates:
            by_ticker.setdefault(c.ticker, []).append(c)

        scored = []
        for ticker, sigs in by_ticker.items():
            total_weight = 0
            weighted_sum = 0
            direction_votes = {"long": 0, "short": 0}

            for sig in sigs:
                w = weight_map.get(sig.signal_type, 0.25)
                weighted_sum += sig.strength * w
                total_weight += w
                direction_votes[sig.direction] += 1

            if total_weight == 0:
                continue

            combined = weighted_sum / total_weight
            direction = "long" if direction_votes["long"] >= direction_votes["short"] else "short"

            scored.append({
                "ticker": ticker,
                "combined_score": abs(combined),
                "direction": direction,
                "signal_type": ",".join(s.signal_type for s in sigs),
                "num_signals": len(sigs),
            })

        return scored

    def _get_trading_dates(self, start_date: date, end_date: date) -> list[date]:
        """Get list of trading dates (business days with price data)."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """SELECT DISTINCT date FROM prices
                   WHERE date >= ? AND date <= ?
                   ORDER BY date""",
                (start_date.isoformat(), end_date.isoformat()),
            )
            return [date.fromisoformat(row["date"]) for row in cursor.fetchall()]

    def _get_day_prices(self, current_date: date) -> dict[str, dict]:
        """Get OHLCV for all tickers on a given date."""
        date_str = current_date.isoformat()
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM prices WHERE date = ?", (date_str,)
            )
            result = {}
            for row in cursor.fetchall():
                result[row["ticker"]] = {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            return result

    def _add_trading_days(self, start: date, days: int) -> date:
        """Add N trading days to a date (approximate: skip weekends)."""
        current = start
        added = 0
        while added < days:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                added += 1
        return current

    def _compute_results(
        self,
        trades: list[dict],
        equity_curve: list[dict],
        initial_capital: float,
    ) -> BacktestResult:
        """Compute summary statistics from backtest trades."""
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
        )

        if not trades:
            result.final_equity = initial_capital
            return result

        # Basic stats
        result.num_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades) if trades else 0

        if wins:
            result.avg_win_pct = sum(t["pnl_pct"] for t in wins) / len(wins)
        if losses:
            result.avg_loss_pct = sum(t["pnl_pct"] for t in losses) / len(losses)

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Hold days
        result.avg_hold_days = sum(t["hold_days"] for t in trades) / len(trades)

        # Equity curve stats
        if equity_curve:
            result.final_equity = equity_curve[-1]["total_equity"]
            result.total_return_pct = (
                (result.final_equity - initial_capital) / initial_capital
            )
            result.max_drawdown_pct = max(e["drawdown_pct"] for e in equity_curve)

            # Sharpe and Sortino from daily returns
            equities = pd.Series([e["total_equity"] for e in equity_curve])
            daily_returns = equities.pct_change().dropna()

            if len(daily_returns) > 1 and daily_returns.std() > 0:
                result.sharpe_ratio = (
                    daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
                )

                downside = daily_returns[daily_returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    result.sortino_ratio = (
                        daily_returns.mean() / downside.std() * (252 ** 0.5)
                    )

        return result

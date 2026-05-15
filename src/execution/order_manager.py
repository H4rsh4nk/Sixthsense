"""Order manager — handles order lifecycle, persistence, and reconciliation."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from src.config import AppConfig
from src.database import Database
from src.execution.broker import AlpacaBroker, BrokerCascade
from src.strategy.exit_manager import ExitManager, ExitSignal
from src.strategy.position_sizer import PositionSizer, PositionSize
from src.strategy.risk_manager import RiskManager
from src.strategy.scorer import TradeCandidate

logger = logging.getLogger(__name__)


class OrderManager:
    """Orchestrates trade entries and exits through the broker.

    Set ``dry_run=True`` to exercise the full sizing / risk-check / decision-log
    pipeline without submitting orders to the broker — used by the
    ``swing-trader trade --dry-run`` flow for end-to-end testing.
    """

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        broker: AlpacaBroker | BrokerCascade,
        risk_manager: RiskManager,
        exit_manager: ExitManager,
        position_sizer: PositionSizer,
        dry_run: bool = False,
    ):
        self.config = config
        self.db = db
        self.broker = broker
        self.risk_manager = risk_manager
        self.exit_manager = exit_manager
        self.position_sizer = position_sizer
        self.dry_run = bool(dry_run)

        # Track broker order IDs → trade IDs for reconciliation
        self._order_map: dict[str, int] = {}  # broker_order_id → db_trade_id
        # self._stop_orders removed in favor of OTO orders and ticker-based cancelation

    def enter_trade(self, candidate: TradeCandidate) -> int | None:
        """Execute a trade entry for a candidate.

        Returns the trade ID if successful, None otherwise.
        """
        # Risk check
        allowed, reason = self.risk_manager.can_open_position(
            candidate.ticker, candidate.sector
        )
        if not allowed:
            logger.info(f"Trade blocked for {candidate.ticker}: {reason}")
            return None

        # Get current price
        price = self.broker.get_latest_price(candidate.ticker)
        if price is None:
            logger.warning(f"Cannot get price for {candidate.ticker}")
            return None

        # Size the position. In fractional_kelly mode we pass per-source priors
        # computed from closed trades; see ADR-0003.
        account = self.broker.get_account()
        kelly_inputs = None
        if self.config.trading.sizing_mode == "fractional_kelly":
            kelly_inputs = self.db.get_signal_kelly_inputs()
        size = self.position_sizer.calculate(
            account_equity=account.equity,
            available_cash=account.cash,
            entry_price=price,
            stop_loss_pct=self.config.exit_rules.stop_loss_pct,
            signal_sources=candidate.signal_sources,
            kelly_inputs=kelly_inputs,
        )
        if size is None:
            logger.info(f"Position size too small for {candidate.ticker} @ ${price:.2f}")
            return None

        today = date.today()

        # Dry-run mode: log the would-be trade and stop here. No broker call,
        # no DB writes — keeps the test database clean. See ADR-0010 Phase C.
        if self.dry_run:
            sizing_tag = size.sizing_mode_used
            if size.sizing_mode_used == "fractional_kelly" and size.kelly_target_pct is not None:
                sizing_tag += f"@{size.kelly_target_pct:.4f}"
            logger.info(
                "DRY-RUN ENTRY: %s %s x %s @ $%.2f | stop=$%.2f | score=%.2f | "
                "sizing=%s | reason=%s",
                candidate.direction.upper(),
                size.shares,
                candidate.ticker,
                price,
                size.stop_loss_price,
                candidate.combined_score,
                sizing_tag,
                candidate.metadata.get("reasoning", "n/a"),
            )
            return None

        # Place market order
        order = self.broker.place_market_order(
            ticker=candidate.ticker,
            qty=size.shares,
            side="buy" if candidate.direction == "long" else "sell",
            stop_loss_price=size.stop_loss_price,
        )

        if "error" in order.status:
            logger.error(f"Entry order failed: {order.status}")
            return None

        # Compute target exit date
        target_exit = self.exit_manager.compute_target_exit_date(today)

        # Record trade in database
        trade_id = self.db.insert_trade({
            "ticker": candidate.ticker,
            "direction": candidate.direction,
            "signal_type": ",".join(candidate.signal_sources),
            "signal_score": candidate.combined_score,
            "entry_reason": candidate.metadata.get("reasoning", ""),
            "entry_date": today.isoformat(),
            "entry_price": price,
            "shares": size.shares,
            "stop_loss_price": size.stop_loss_price,
            "target_exit_date": target_exit.isoformat(),
            "status": "open",
        })

        self._order_map[order.order_id] = trade_id

        # Stop-loss order is now handled via OTO on the entry order.

        self.exit_manager.register_trade(trade_id, price)

        sizing_detail = f"sizing={size.sizing_mode_used}"
        if size.sizing_mode_used == "fractional_kelly" and size.kelly_target_pct is not None:
            sizing_detail += f" kelly_target_pct={size.kelly_target_pct:.4f}"
        elif size.kelly_fallback_reason:
            sizing_detail += f" kelly_fallback={size.kelly_fallback_reason}"
        logger.info(
            f"ENTERED: {candidate.direction.upper()} {size.shares} x {candidate.ticker} "
            f"@ ${price:.2f} | Stop: ${size.stop_loss_price:.2f} | "
            f"Exit by: {target_exit} | Score: {candidate.combined_score:.2f} | "
            f"{sizing_detail} | "
            f"Reason: {candidate.metadata.get('reasoning', 'n/a')}"
        )
        return trade_id

    def process_exits(self, current_prices: dict[str, float]) -> list[int]:
        """Check for and execute exits. Returns list of closed trade IDs."""
        today = date.today()
        exit_signals = self.exit_manager.check_exits(today, current_prices)
        closed = []

        if self.dry_run and exit_signals:
            logger.info("DRY-RUN EXITS: would close %d positions", len(exit_signals))
            for sig in exit_signals:
                logger.info("  DRY-RUN EXIT: %s reason=%s", sig.ticker, sig.reason)
            return []

        for exit_sig in exit_signals:
            trade_id = exit_sig.trade_id
            ticker = exit_sig.ticker
            price = exit_sig.target_price or current_prices.get(ticker)

            if price is None:
                continue

            # Cancel all open orders (like attached stop losses) for this ticker
            if exit_sig.reason != "stop_loss":
                self.broker.cancel_open_orders(ticker)

            # Get open trade details
            open_trades = self.db.get_open_trades()
            trade = next((t for t in open_trades if t["id"] == trade_id), None)
            if not trade:
                continue

            # Place market exit order
            side = "sell" if trade["direction"] == "long" else "buy"
            order = self.broker.place_market_order(
                ticker=ticker,
                qty=int(trade["shares"]),
                side=side,
            )

            # Update database
            self.db.close_trade(
                trade_id=trade_id,
                exit_date=today.isoformat(),
                exit_price=price,
                exit_reason=exit_sig.reason,
            )

            logger.info(
                f"EXITED: {ticker} | Reason: {exit_sig.reason} | "
                f"Price: ${price:.2f}"
            )
            closed.append(trade_id)

        return closed

    def close_all(self, reason: str = "circuit_breaker") -> list[int]:
        """Close all open positions immediately."""
        exits = self.exit_manager.close_all_positions(reason)
        closed = []

        if self.dry_run and exits:
            logger.info("DRY-RUN CLOSE_ALL: would close %d positions (%s)", len(exits), reason)
            return []

        for exit_sig in exits:
            price = self.broker.get_latest_price(exit_sig.ticker)
            if price is None:
                continue

            trade = next(
                (t for t in self.db.get_open_trades() if t["id"] == exit_sig.trade_id),
                None,
            )
            if not trade:
                continue

            side = "sell" if trade["direction"] == "long" else "buy"
            self.broker.place_market_order(
                ticker=exit_sig.ticker,
                qty=int(trade["shares"]),
                side=side,
            )

            self.db.close_trade(
                trade_id=exit_sig.trade_id,
                exit_date=date.today().isoformat(),
                exit_price=price,
                exit_reason=reason,
            )
            closed.append(exit_sig.trade_id)

        # Cancel all open orders for the closed positions
        for exit_sig in exits:
            self.broker.cancel_open_orders(exit_sig.ticker)

        return closed

    def reconcile(self) -> None:
        """Reconcile local state with broker positions."""
        broker_positions = self.broker.get_positions()
        db_trades = self.db.get_open_trades()

        broker_tickers = {p["ticker"] for p in broker_positions}
        db_tickers = {t["ticker"] for t in db_trades}

        # Positions in broker but not in DB (unexpected)
        orphaned = broker_tickers - db_tickers
        if orphaned:
            logger.warning(f"Orphaned broker positions (not in DB): {orphaned}")

        # Positions in DB but not in broker (filled stop-loss or manual close)
        missing = db_tickers - broker_tickers
        if missing:
            logger.warning(f"DB trades not found in broker (may have been stopped out): {missing}")

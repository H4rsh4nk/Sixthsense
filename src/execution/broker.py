"""Alpaca broker abstraction — paper and live trading via a single interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    status: str


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    side: str
    qty: int
    order_type: str
    status: str
    filled_price: float | None = None
    filled_at: str | None = None


class AlpacaBroker:
    """Alpaca API wrapper supporting both paper and live trading."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._api = None

    @property
    def api(self):
        """Lazy-load Alpaca API client."""
        if self._api is None:
            import alpaca_trade_api as tradeapi

            base_url = (
                self.config.broker.base_url_paper
                if self.config.broker.paper
                else self.config.broker.base_url_live
            )
            self._api = tradeapi.REST(
                key_id=self.config.secrets.alpaca_api_key,
                secret_key=self.config.secrets.alpaca_secret_key,
                base_url=base_url,
                api_version="v2",
            )
            mode = "PAPER" if self.config.broker.paper else "LIVE"
            logger.info(f"Alpaca API initialized in {mode} mode")

        return self._api

    def get_account(self) -> AccountInfo:
        """Get current account information."""
        account = self.api.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            status=account.status,
        )

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.api.list_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def place_market_order(
        self, ticker: str, qty: int, side: str = "buy", stop_loss_price: float | None = None
    ) -> OrderResult:
        """Place a market order, optionally as an OTO order with a stop loss attached."""
        logger.info(f"Placing {side.upper()} market order: {qty} x {ticker}")
        try:
            kwargs = {
                "symbol": ticker,
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
            if stop_loss_price is not None:
                kwargs["order_class"] = "oto"
                kwargs["stop_loss"] = {"stop_price": str(round(stop_loss_price, 2))}
                logger.info(f"Attached stop loss at ${stop_loss_price:.2f}")

            order = self.api.submit_order(**kwargs)
            return OrderResult(
                order_id=order.id,
                ticker=ticker,
                side=side,
                qty=qty,
                order_type="market",
                status=order.status,
            )
        except Exception as e:
            logger.error(f"Order failed: {ticker} {side} {qty}: {e}")
            return OrderResult(
                order_id="",
                ticker=ticker,
                side=side,
                qty=qty,
                order_type="market",
                status=f"error: {e}",
            )

    def place_stop_order(
        self, ticker: str, qty: int, stop_price: float, side: str = "sell"
    ) -> OrderResult:
        """Place a stop (stop-loss) order."""
        logger.info(f"Placing {side.upper()} stop order: {qty} x {ticker} @ ${stop_price:.2f}")
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type="stop",
                time_in_force="gtc",
                stop_price=str(stop_price),
            )
            return OrderResult(
                order_id=order.id,
                ticker=ticker,
                side=side,
                qty=qty,
                order_type="stop",
                status=order.status,
            )
        except Exception as e:
            logger.error(f"Stop order failed: {ticker} {side} {qty} @ {stop_price}: {e}")
            return OrderResult(
                order_id="",
                ticker=ticker,
                side=side,
                qty=qty,
                order_type="stop",
                status=f"error: {e}",
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel order failed: {order_id}: {e}")
            return False

    def cancel_open_orders(self, ticker: str) -> bool:
        """Cancel all open orders for a ticker."""
        try:
            orders = self.api.list_orders(status="open", symbols=[ticker])
            for order in orders:
                self.api.cancel_order(order.id)
            return True
        except Exception as e:
            logger.error(f"Cancel open orders failed for {ticker}: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderResult | None:
        """Get the current status of an order."""
        try:
            order = self.api.get_order(order_id)
            return OrderResult(
                order_id=order.id,
                ticker=order.symbol,
                side=order.side,
                qty=int(order.qty),
                order_type=order.type,
                status=order.status,
                filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                filled_at=str(order.filled_at) if order.filled_at else None,
            )
        except Exception as e:
            logger.error(f"Get order status failed: {order_id}: {e}")
            return None

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the latest trade price for a ticker."""
        try:
            trade = self.api.get_latest_trade(ticker)
            return float(trade.price)
        except Exception as e:
            logger.error(f"Get latest price failed for {ticker}: {e}")
            return None

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get latest prices for multiple tickers."""
        prices = {}
        for ticker in tickers:
            price = self.get_latest_price(ticker)
            if price is not None:
                prices[ticker] = price
        return prices

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Clock check failed: {e}")
            return False

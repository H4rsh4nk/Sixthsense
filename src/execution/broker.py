"""Alpaca broker abstraction — paper and live trading via a single interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

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

    def __init__(
        self,
        config: AppConfig,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
        base_url: str | None = None,
    ):
        self.config = config
        self._api = None
        self._override_key = (api_key or "").strip()
        self._override_secret = (secret_key or "").strip()
        self._override_paper = paper
        self._override_base_url = (base_url or "").strip() or None

    @property
    def api(self):
        """Lazy-load Alpaca API client."""
        if self._api is None:
            import alpaca_trade_api as tradeapi

            use_paper = (
                self.config.broker.paper if self._override_paper is None else self._override_paper
            )
            base_url = self._override_base_url or (
                self.config.broker.base_url_paper
                if use_paper
                else self.config.broker.base_url_live
            )

            kid = (
                self._override_key
                if self._override_key
                else self.config.secrets.alpaca_api_key
            )
            sec = (
                self._override_secret
                if self._override_secret
                else self.config.secrets.alpaca_secret_key
            )

            self._api = tradeapi.REST(
                key_id=kid,
                secret_key=sec,
                base_url=base_url,
                api_version="v2",
            )
            mode = "PAPER" if use_paper else "LIVE"
            tag = " (fallback credentials)" if self._override_key else ""
            logger.info(f"Alpaca API initialized in {mode} mode{tag}")

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


class BrokerCascade:
    """Primary Alpaca client with sticky failover to a second credential set.

    Implements a three-state circuit breaker (closed | open | half_open):

    - **closed**  — calls go to primary; failures bump a counter.
    - **open**    — primary is skipped entirely, calls go to fallback only.
                    Triggered when `failover_failure_threshold` consecutive
                    primary failures occur. Stays open for `failover_cooldown_seconds`.
    - **half_open** — after cooldown, the next primary call probes recovery:
                    success → closed; failure → open with reset cooldown.

    On transition back to **closed** we fire an optional recovery callback so
    `OrderManager.reconcile()` can detect positions executed by the fallback
    while primary was down. See `docs/decisions/0004-sticky-broker-failover.md`.
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, config: AppConfig):
        self.config = config
        self.primary = AlpacaBroker(config)
        fk = (config.secrets.alpaca_fallback_api_key or "").strip()
        sk = (
            (
                config.secrets.alpaca_fallback_secret_key
                or config.secrets.alpaca_secret_key
                or ""
            )
        ).strip()
        b = config.broker
        use_fb_paper = b.fallback_paper
        fb_base = (b.fallback_base_url_paper if use_fb_paper else b.fallback_base_url_live) or ""

        self.fallback = AlpacaBroker(
            config,
            api_key=fk,
            secret_key=sk,
            paper=use_fb_paper,
            base_url=fb_base.strip() or None,
        )

        # Circuit-breaker state
        self._state: str = self.STATE_CLOSED
        self._opened_at: datetime | None = None
        self._consecutive_failures: int = 0
        self._failure_threshold = max(1, int(b.failover_failure_threshold))
        self._cooldown = timedelta(seconds=max(1, int(b.failover_cooldown_seconds)))
        self._recovery_cb: Callable[[], None] | None = None

    # ---- public hooks ------------------------------------------------------

    def set_recovery_callback(self, callback: Callable[[], None]) -> None:
        """Called once when the breaker transitions back to closed."""
        self._recovery_cb = callback

    @property
    def circuit_state(self) -> str:
        """Current breaker state — exposed for monitoring/heartbeat logging."""
        return self._state

    # ---- state-machine internals ------------------------------------------

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _transition(self, new_state: str, reason: str) -> None:
        if new_state == self._state:
            return
        logger.warning(
            "Broker circuit breaker: %s -> %s (%s)",
            self._state,
            new_state,
            reason,
        )
        prev = self._state
        self._state = new_state
        if new_state == self.STATE_OPEN:
            self._opened_at = self._now()
        elif new_state == self.STATE_CLOSED:
            self._opened_at = None
            self._consecutive_failures = 0
            if prev in (self.STATE_OPEN, self.STATE_HALF_OPEN) and self._recovery_cb:
                try:
                    self._recovery_cb()
                except Exception as e:  # noqa: BLE001 — recovery hook is best-effort
                    logger.error("Broker recovery callback failed: %s", e)

    def _maybe_probe(self) -> None:
        """If we're in OPEN past the cooldown, flip to HALF_OPEN for one probe."""
        if self._state == self.STATE_OPEN and self._opened_at is not None:
            if self._now() - self._opened_at >= self._cooldown:
                self._transition(self.STATE_HALF_OPEN, "cooldown elapsed; probing primary")

    def _record_primary_success(self) -> None:
        self._consecutive_failures = 0
        if self._state == self.STATE_HALF_OPEN:
            self._transition(self.STATE_CLOSED, "primary probe succeeded")

    def _record_primary_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        if self._state == self.STATE_HALF_OPEN:
            self._transition(self.STATE_OPEN, f"primary probe failed: {exc}")
        elif (
            self._state == self.STATE_CLOSED
            and self._consecutive_failures >= self._failure_threshold
            and self.fallback
        ):
            self._transition(
                self.STATE_OPEN,
                f"{self._consecutive_failures} consecutive primary failures",
            )

    def _call(self, method: str, *args: Any, _allow_fallback: bool = True, **kwargs: Any) -> Any:
        """Route a method call through the breaker, with optional fallback.

        Returns the broker's return value, or re-raises the last exception if
        both legs fail (and no fallback is configured).
        """

        self._maybe_probe()

        # If breaker is OPEN and we have a fallback, skip primary entirely.
        if self._state == self.STATE_OPEN and self.fallback and _allow_fallback:
            return getattr(self.fallback, method)(*args, **kwargs)

        # CLOSED or HALF_OPEN: try primary first.
        try:
            result = getattr(self.primary, method)(*args, **kwargs)
            self._record_primary_success()
            return result
        except Exception as primary_exc:  # noqa: BLE001 — broker SDK throws various
            self._record_primary_failure(primary_exc)
            if not (self.fallback and _allow_fallback):
                raise
            logger.warning(
                "Primary broker %s failed (%s); using fallback", method, primary_exc
            )
            return getattr(self.fallback, method)(*args, **kwargs)

    # ---- account / positions ----------------------------------------------

    def get_account(self) -> AccountInfo:
        return self._call("get_account")

    def get_positions(self) -> list[dict]:
        return self._call("get_positions")

    # ---- orders -----------------------------------------------------------

    def place_market_order(
        self, ticker: str, qty: int, side: str = "buy", stop_loss_price: float | None = None
    ) -> OrderResult:
        return self._call("place_market_order", ticker, qty, side, stop_loss_price)

    def place_stop_order(
        self, ticker: str, qty: int, stop_price: float, side: str = "sell"
    ) -> OrderResult:
        return self._call("place_stop_order", ticker, qty, stop_price, side)

    def cancel_order(self, order_id: str) -> bool:
        # Try every leg; cancellation is idempotent so we don't gate by state.
        for broker in self._chain_for_idempotent():
            try:
                if broker.cancel_order(order_id):
                    return True
            except Exception as e:  # noqa: BLE001
                logger.debug("cancel_order leg failed: %s", e)
        return False

    def cancel_open_orders(self, ticker: str) -> bool:
        ok = False
        for broker in self._chain_for_idempotent():
            try:
                ok |= broker.cancel_open_orders(ticker)
            except Exception as e:  # noqa: BLE001
                logger.debug("cancel_open_orders leg failed: %s", e)
        return ok

    def get_order_status(self, order_id: str) -> OrderResult | None:
        for broker in self._chain_for_idempotent():
            try:
                res = broker.get_order_status(order_id)
                if res:
                    return res
            except Exception as e:  # noqa: BLE001
                logger.debug("get_order_status leg failed: %s", e)
        return None

    # ---- prices -----------------------------------------------------------

    def get_latest_price(self, ticker: str) -> float | None:
        """Get latest price respecting the breaker state for the primary leg."""
        self._maybe_probe()

        if self._state == self.STATE_OPEN and self.fallback:
            return self.fallback.get_latest_price(ticker)

        # Primary first
        try:
            p = self.primary.get_latest_price(ticker)
        except Exception as exc:  # noqa: BLE001
            self._record_primary_failure(exc)
            if self.fallback:
                logger.warning("Primary price fetch raised for %s (%s); using fallback", ticker, exc)
                return self.fallback.get_latest_price(ticker)
            return None

        if p is not None:
            self._record_primary_success()
            return p

        # No exception but `None` price — count as a soft failure and try fallback.
        self._record_primary_failure(RuntimeError(f"no price for {ticker}"))
        if self.fallback:
            return self.fallback.get_latest_price(ticker)
        return None

    def get_latest_prices(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for t in tickers:
            px = self.get_latest_price(t)
            if px is not None:
                prices[t] = px
        return prices

    def is_market_open(self) -> bool:
        """Return True if any reachable leg reports the market open."""
        try:
            return bool(self._call("is_market_open"))
        except Exception:
            return False

    # ---- helpers ----------------------------------------------------------

    def _chain_for_idempotent(self) -> list[AlpacaBroker]:
        """Order legs by current breaker state for idempotent reads."""
        if self._state == self.STATE_OPEN and self.fallback:
            return [self.fallback, self.primary]
        if self.fallback:
            return [self.primary, self.fallback]
        return [self.primary]


def create_execution_broker(config: AppConfig) -> AlpacaBroker | BrokerCascade:
    if config.broker.fallback_enabled:
        fk = (config.secrets.alpaca_fallback_api_key or "").strip()
        fs = (
            config.secrets.alpaca_fallback_secret_key
            or config.secrets.alpaca_secret_key
            or ""
        ).strip()
        if fk and fs:
            return BrokerCascade(config)
        logger.warning(
            "broker.fallback_enabled without usable fallback Alpaca credentials — primary only"
        )
    return AlpacaBroker(config)

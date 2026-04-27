"""Telegram alert system for trade notifications."""

from __future__ import annotations

import logging
from datetime import date

from src.config import AppConfig

logger = logging.getLogger(__name__)


class AlertManager:
    """Sends notifications via Telegram."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.enabled = config.alerts.telegram_enabled
        self.chat_id = config.alerts.telegram_chat_id
        self.bot_token = config.secrets.telegram_bot_token
        self._bot = None

    @property
    def bot(self):
        if self._bot is None and self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.bot_token)
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
        return self._bot

    def send(self, message: str) -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            logger.info(f"[ALERT] {message}")
            return False

        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode="Markdown")
            )
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            logger.info(f"[ALERT] {message}")
            return False

    def signal_detected(self, ticker: str, signal_type: str, score: float, direction: str):
        """Alert: new signal detected."""
        self.send(
            f"📊 *Signal Detected*\n"
            f"Ticker: `{ticker}`\n"
            f"Type: {signal_type}\n"
            f"Score: {score:.2f}\n"
            f"Direction: {direction.upper()}"
        )

    def trade_entered(
        self, ticker: str, shares: int, price: float, stop_loss: float, signal_type: str
    ):
        """Alert: trade entered."""
        self.send(
            f"✅ *Trade Entered*\n"
            f"Ticker: `{ticker}`\n"
            f"Shares: {shares}\n"
            f"Entry: ${price:.2f}\n"
            f"Stop: ${stop_loss:.2f}\n"
            f"Signal: {signal_type}"
        )

    def trade_exited(
        self, ticker: str, entry_price: float, exit_price: float, pnl: float, reason: str
    ):
        """Alert: trade exited."""
        emoji = "🟢" if pnl > 0 else "🔴"
        self.send(
            f"{emoji} *Trade Exited*\n"
            f"Ticker: `{ticker}`\n"
            f"Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"P&L: ${pnl:+,.2f}\n"
            f"Reason: {reason}"
        )

    def daily_summary(
        self, equity: float, daily_pnl: float, drawdown_pct: float, open_positions: int
    ):
        """Alert: end-of-day summary."""
        emoji = "🟢" if daily_pnl >= 0 else "🔴"
        self.send(
            f"📋 *Daily Summary* ({date.today()})\n"
            f"Equity: ${equity:,.2f}\n"
            f"Daily P&L: {emoji} ${daily_pnl:+,.2f}\n"
            f"Drawdown: {drawdown_pct:.2%}\n"
            f"Open Positions: {open_positions}"
        )

    def circuit_breaker(self, equity: float, drawdown_pct: float):
        """Alert: circuit breaker triggered."""
        self.send(
            f"🚨 *CIRCUIT BREAKER TRIGGERED*\n"
            f"Equity: ${equity:,.2f}\n"
            f"Drawdown: {drawdown_pct:.2%}\n"
            f"All positions closed. No new trades until manual reset."
        )

    def system_error(self, error: str):
        """Alert: system error."""
        self.send(f"⚠️ *System Error*\n```\n{error[:500]}\n```")

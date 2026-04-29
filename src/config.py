"""Configuration loader with Pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"


class TradingConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    capital: float = 10000
    risk_per_trade_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    max_concurrent_positions: int = 3
    max_single_position_pct: float = 0.30
    max_sector_positions: int = 2
    daily_loss_limit_pct: float = 0.03


class ExitRulesConfig(BaseModel):
    default_hold_days: int = 5
    stop_loss_pct: float = 0.05
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.03
    sweep_hold_days: list[int] = Field(default_factory=lambda: [3, 5, 10, 15])
    sweep_stop_loss_pct: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.07])


class InsiderSignalConfig(BaseModel):
    enabled: bool = True
    lookback_days: int = 2
    min_transaction_value: int = 100000
    cluster_window_days: int = 7
    weight: float = 0.35


class NewsSignalConfig(BaseModel):
    enabled: bool = True
    sentiment_threshold: float = 0.6
    min_articles: int = 2
    weight: float = 0.25


class PoliticalSignalConfig(BaseModel):
    enabled: bool = True
    event_types: list[str] = Field(
        default_factory=lambda: ["tariff", "executive_order", "fed_meeting", "sanctions", "trade_deal"]
    )
    weight: float = 0.20


class PriceActionSignalConfig(BaseModel):
    enabled: bool = True
    indicators: list[str] = Field(default_factory=lambda: ["rsi", "macd", "volume_spike"])
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    volume_spike_multiplier: float = 2.0
    weight: float = 0.20


class SignalsConfig(BaseModel):
    insider: InsiderSignalConfig = Field(default_factory=InsiderSignalConfig)
    news: NewsSignalConfig = Field(default_factory=NewsSignalConfig)
    political: PoliticalSignalConfig = Field(default_factory=PoliticalSignalConfig)
    price_action: PriceActionSignalConfig = Field(default_factory=PriceActionSignalConfig)


class ScoringConfig(BaseModel):
    min_combined_score: float = 0.3
    min_signals_agreeing: int = 1


class BacktestConfig(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    slippage_pct: float = 0.001
    spread_pct: float = 0.0005
    initial_capital: float = 10000
    walk_forward_train_years: int = 2
    walk_forward_test_years: int = 1


class BrokerConfig(BaseModel):
    name: str = "alpaca"
    paper: bool = True
    base_url_paper: str = "https://paper-api.alpaca.markets"
    base_url_live: str = "https://api.alpaca.markets"
    entry_delay_minutes: int = 5
    exit_before_close_minutes: int = 5


class SchedulerConfig(BaseModel):
    pre_market: str = "08:00"
    market_open_entry: str = "09:35"
    intraday_check_interval_minutes: int = 30
    heartbeat_interval_minutes: int = 15
    market_close_exit: str = "15:55"
    post_market: str = "17:00"
    overnight_scan: str = "23:00"
    timezone: str = "US/Eastern"


class AlertsConfig(BaseModel):
    telegram_enabled: bool = False
    telegram_chat_id: str = ""


class DataConfig(BaseModel):
    universe: Literal["sp500", "custom"] = "sp500"
    custom_tickers: list[str] = Field(default_factory=list)
    db_path: str = "data/swing_trader.db"
    price_history_years: int = 3


class AgentConfig(BaseModel):
    enabled: bool = False
    provider: str = "gemini"  # gemini, openai, anthropic, ollama, together, etc.
    model: str = "gemini/gemini-2.0-flash"  # LiteLLM format: provider/model
    temperature: float = 0.2
    max_tool_calls: int = 5
    fallback_to_rules: bool = True
    api_base: str = ""  # Custom API base URL (for Ollama, vLLM, etc.)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/swing_trader.log"


class SecretsConfig(BaseModel):
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    telegram_bot_token: str = ""
    sec_edgar_user_agent: str = ""
    llm_api_key: str = ""  # API key for the LLM provider (Gemini/OpenAI/Anthropic/etc.)


class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    exit_rules: ExitRulesConfig = Field(default_factory=ExitRulesConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)


def load_config(settings_path: Path | None = None, secrets_path: Path | None = None) -> AppConfig:
    """Load config from YAML files and environment variables."""
    settings_path = settings_path or CONFIG_DIR / "settings.yaml"
    secrets_path = secrets_path or CONFIG_DIR / "secrets.yaml"

    settings_data = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings_data = yaml.safe_load(f) or {}

    secrets_data = {}
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets_data = yaml.safe_load(f) or {}

    # Flatten secrets into the config
    secrets = SecretsConfig(
        alpaca_api_key=os.getenv(
            "ALPACA_API_KEY", secrets_data.get("alpaca", {}).get("api_key", "")
        ),
        alpaca_secret_key=os.getenv(
            "ALPACA_SECRET_KEY", secrets_data.get("alpaca", {}).get("secret_key", "")
        ),
        telegram_bot_token=os.getenv(
            "TELEGRAM_BOT_TOKEN", secrets_data.get("telegram", {}).get("bot_token", "")
        ),
        sec_edgar_user_agent=os.getenv(
            "SEC_EDGAR_USER_AGENT", secrets_data.get("sec_edgar", {}).get("user_agent", "")
        ),
        llm_api_key=os.getenv(
            "LLM_API_KEY",
            secrets_data.get("llm", {}).get("api_key",
                # Fallback: check provider-specific keys
                secrets_data.get("gemini", {}).get("api_key",
                    secrets_data.get("openai", {}).get("api_key",
                        secrets_data.get("anthropic", {}).get("api_key", "")
                    )
                )
            )
        ),
    )

    settings_data["secrets"] = secrets.model_dump()
    return AppConfig(**settings_data)

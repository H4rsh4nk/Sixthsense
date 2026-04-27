"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from src.config import AppConfig, load_config
from src.database import Database


@pytest.fixture
def tmp_config(tmp_path) -> AppConfig:
    """Create a config with a temporary database."""
    settings = {
        "trading": {
            "mode": "paper",
            "capital": 10000,
            "risk_per_trade_pct": 0.02,
            "max_drawdown_pct": 0.10,
            "max_concurrent_positions": 3,
            "max_single_position_pct": 0.30,
            "max_sector_positions": 2,
            "daily_loss_limit_pct": 0.03,
        },
        "exit_rules": {
            "default_hold_days": 5,
            "stop_loss_pct": 0.05,
            "trailing_stop": False,
        },
        "signals": {
            "insider": {"enabled": True, "weight": 0.35},
            "news": {"enabled": True, "weight": 0.25},
            "political": {"enabled": True, "weight": 0.20},
            "price_action": {"enabled": True, "weight": 0.20},
        },
        "scoring": {
            "min_combined_score": 0.3,
            "min_signals_agreeing": 1,
        },
        "backtest": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "slippage_pct": 0.001,
            "initial_capital": 10000,
        },
        "data": {
            "universe": "custom",
            "custom_tickers": ["AAPL", "MSFT", "GOOGL"],
            "db_path": str(tmp_path / "test.db"),
        },
    }

    settings_path = tmp_path / "settings.yaml"
    with open(settings_path, "w") as f:
        yaml.dump(settings, f)

    return load_config(settings_path=settings_path, secrets_path=tmp_path / "nosecrets.yaml")


@pytest.fixture
def tmp_db(tmp_config) -> Database:
    """Create a temporary database."""
    return Database(tmp_config)


@pytest.fixture
def seeded_db(tmp_db) -> Database:
    """Database with sample universe and price data."""
    # Insert universe
    tmp_db.insert_universe([
        {"ticker": "AAPL", "company_name": "Apple", "sector": "Technology",
         "industry": "Consumer Electronics", "market_cap": 3e12},
        {"ticker": "MSFT", "company_name": "Microsoft", "sector": "Technology",
         "industry": "Software", "market_cap": 2.8e12},
        {"ticker": "GOOGL", "company_name": "Alphabet", "sector": "Communication Services",
         "industry": "Internet", "market_cap": 1.8e12},
        {"ticker": "JPM", "company_name": "JPMorgan", "sector": "Financials",
         "industry": "Banking", "market_cap": 5e11},
    ])

    # Insert sample price data (30 days)
    import random
    random.seed(42)

    base_prices = {"AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0, "JPM": 190.0}
    price_rows = []

    from datetime import timedelta
    start = date(2024, 1, 2)
    for day_offset in range(250):  # ~1 year of trading days
        d = start + timedelta(days=day_offset)
        if d.weekday() >= 5:  # skip weekends
            continue

        for ticker, base in base_prices.items():
            # Random walk
            change = random.uniform(-0.03, 0.03)
            base_prices[ticker] = base * (1 + change)
            p = base_prices[ticker]

            price_rows.append({
                "ticker": ticker,
                "date": d.isoformat(),
                "open": p * random.uniform(0.99, 1.01),
                "high": p * random.uniform(1.0, 1.02),
                "low": p * random.uniform(0.98, 1.0),
                "close": p,
                "adj_close": p,
                "volume": random.randint(10_000_000, 100_000_000),
            })

    tmp_db.insert_prices(price_rows)
    return tmp_db

"""Tests for database operations."""

from datetime import date

from src.database import Database


def test_insert_and_get_prices(tmp_db):
    rows = [
        {"ticker": "AAPL", "date": "2024-01-02", "open": 180.0, "high": 182.0,
         "low": 179.0, "close": 181.0, "adj_close": 181.0, "volume": 50000000},
        {"ticker": "AAPL", "date": "2024-01-03", "open": 181.0, "high": 183.0,
         "low": 180.0, "close": 182.5, "adj_close": 182.5, "volume": 45000000},
    ]
    tmp_db.insert_prices(rows)

    result = tmp_db.get_prices("AAPL", "2024-01-01", "2024-01-05")
    assert len(result) == 2
    assert result[0]["close"] == 181.0
    assert result[1]["close"] == 182.5


def test_insert_and_get_universe(tmp_db):
    tmp_db.insert_universe([
        {"ticker": "AAPL", "company_name": "Apple", "sector": "Tech",
         "industry": "CE", "market_cap": 3e12},
    ])

    tickers = tmp_db.get_all_tickers()
    assert "AAPL" in tickers


def test_trade_lifecycle(tmp_db):
    trade_id = tmp_db.insert_trade({
        "ticker": "AAPL",
        "direction": "long",
        "signal_type": "insider",
        "signal_score": 0.8,
        "entry_date": "2024-01-02",
        "entry_price": 180.0,
        "shares": 10,
        "stop_loss_price": 171.0,
        "target_exit_date": "2024-01-09",
        "status": "open",
    })

    assert trade_id is not None

    open_trades = tmp_db.get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0]["ticker"] == "AAPL"

    tmp_db.close_trade(trade_id, "2024-01-09", 185.0, "time_exit")

    open_trades = tmp_db.get_open_trades()
    assert len(open_trades) == 0


def test_equity_snapshot(tmp_db):
    tmp_db.insert_equity_snapshot({
        "date": "2024-01-02",
        "cash": 8000,
        "positions_value": 2000,
        "total_equity": 10000,
        "daily_pnl": 0,
        "drawdown_pct": 0,
        "open_positions": 1,
    })

    curve = tmp_db.get_equity_curve()
    assert len(curve) == 1
    assert curve[0]["total_equity"] == 10000

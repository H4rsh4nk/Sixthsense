"""Tests for backtesting engine."""

from datetime import date

from src.backtest.engine import BacktestEngine
from src.signals.price_action import PriceActionSignal


def test_backtest_runs_without_crash(tmp_config, seeded_db):
    engine = BacktestEngine(tmp_config, seeded_db)
    signal = PriceActionSignal(tmp_config, seeded_db)

    result = engine.run(
        signals=[signal],
        start_date=date(2024, 3, 1),
        end_date=date(2024, 6, 30),
        hold_days=5,
        stop_loss_pct=0.05,
    )

    assert result is not None
    assert result.initial_capital == 10000
    assert len(result.equity_curve) > 0


def test_backtest_no_tickers(tmp_config, tmp_db):
    """Backtest with empty universe should return empty result."""
    engine = BacktestEngine(tmp_config, tmp_db)
    signal = PriceActionSignal(tmp_config, tmp_db)

    result = engine.run(
        signals=[signal],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
    )

    assert result.num_trades == 0


def test_backtest_result_metrics(tmp_config, seeded_db):
    """Verify result metrics are computed."""
    engine = BacktestEngine(tmp_config, seeded_db)
    signal = PriceActionSignal(tmp_config, seeded_db)

    result = engine.run(
        signals=[signal],
        start_date=date(2024, 2, 1),
        end_date=date(2024, 8, 30),
        hold_days=5,
        stop_loss_pct=0.05,
    )

    # These should be computed regardless of whether trades happened
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert result.max_drawdown_pct >= 0
    assert result.parameters["hold_days"] == 5
    assert result.parameters["stop_loss_pct"] == 0.05

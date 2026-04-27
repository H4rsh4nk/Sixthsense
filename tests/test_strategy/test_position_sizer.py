"""Tests for position sizing logic."""

from src.strategy.position_sizer import PositionSizer


def test_basic_sizing(tmp_config):
    sizer = PositionSizer(tmp_config)

    result = sizer.calculate(
        account_equity=10000,
        available_cash=10000,
        entry_price=100.0,
        stop_loss_pct=0.05,
    )

    assert result is not None
    # Risk per trade = 10000 * 0.02 = $200
    # Risk per share = 100 * 0.05 = $5
    # Shares by risk = 200 / 5 = 40
    # Shares by max position = 10000 * 0.30 / 100 = 30 (limiting factor)
    assert result.shares == 30
    assert result.stop_loss_price == 95.0
    assert result.position_value == 3000.0
    assert result.risk_amount == 150.0


def test_max_position_cap(tmp_config):
    sizer = PositionSizer(tmp_config)

    # With a $2 stock, risk sizing would give huge position
    # Max position = 30% of $10K = $3000
    result = sizer.calculate(
        account_equity=10000,
        available_cash=10000,
        entry_price=2.0,
        stop_loss_pct=0.05,
    )

    assert result is not None
    # Max by value = 3000 / 2 = 1500 shares
    # Max by risk = 200 / 0.10 = 2000 shares
    assert result.shares == 1500
    assert result.position_value == 3000.0


def test_cash_limited(tmp_config):
    sizer = PositionSizer(tmp_config)

    result = sizer.calculate(
        account_equity=10000,
        available_cash=500,  # Very low cash
        entry_price=100.0,
        stop_loss_pct=0.05,
    )

    assert result is not None
    assert result.shares == 5  # 500 / 100
    assert result.position_value == 500.0


def test_insufficient_cash(tmp_config):
    sizer = PositionSizer(tmp_config)

    result = sizer.calculate(
        account_equity=10000,
        available_cash=50,  # Not enough for even 1 share
        entry_price=100.0,
        stop_loss_pct=0.05,
    )

    assert result is None


def test_zero_stop_loss(tmp_config):
    sizer = PositionSizer(tmp_config)

    result = sizer.calculate(
        account_equity=10000,
        available_cash=10000,
        entry_price=100.0,
        stop_loss_pct=0.0,
    )

    assert result is None


def test_high_priced_stock(tmp_config):
    """With a $3000 stock, verify we can still size correctly."""
    sizer = PositionSizer(tmp_config)

    result = sizer.calculate(
        account_equity=10000,
        available_cash=10000,
        entry_price=3000.0,
        stop_loss_pct=0.05,
    )

    assert result is not None
    # Risk per share = 3000 * 0.05 = $150
    # Shares by risk = 200 / 150 = 1
    # Shares by max position = 3000 / 3000 = 1
    assert result.shares == 1

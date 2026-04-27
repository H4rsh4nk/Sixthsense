"""Tests for risk manager circuit breaker and limits."""

from src.strategy.risk_manager import RiskManager


def test_circuit_breaker_triggers(tmp_config, tmp_db):
    rm = RiskManager(tmp_config, tmp_db)

    assert not rm.circuit_breaker_active

    # Equity drops 10%: $10000 → $9000
    rm.update_equity(10000)
    assert not rm.circuit_breaker_active

    rm.update_equity(9500)
    assert not rm.circuit_breaker_active

    rm.update_equity(9000)  # Exactly 10% drawdown
    assert rm.circuit_breaker_active


def test_circuit_breaker_reset(tmp_config, tmp_db):
    rm = RiskManager(tmp_config, tmp_db)

    rm.update_equity(10000)
    rm.update_equity(9000)
    assert rm.circuit_breaker_active

    rm.reset_circuit_breaker()
    assert not rm.circuit_breaker_active


def test_daily_loss_limit(tmp_config, tmp_db):
    rm = RiskManager(tmp_config, tmp_db)

    rm.start_new_day(10000)
    assert not rm.check_daily_loss(9800)  # 2% loss, limit is 3%
    assert rm.check_daily_loss(9690)  # 3.1% loss, over limit


def test_position_limits(tmp_config, seeded_db):
    rm = RiskManager(tmp_config, seeded_db)

    # No open trades — should be allowed
    allowed, reason = rm.can_open_position("AAPL", "Technology")
    assert allowed

    # Circuit breaker blocks everything
    rm._circuit_breaker_active = True
    allowed, reason = rm.can_open_position("AAPL", "Technology")
    assert not allowed
    assert "Circuit breaker" in reason


def test_drawdown_calculation(tmp_config, tmp_db):
    rm = RiskManager(tmp_config, tmp_db)

    rm.update_equity(12000)  # New peak
    assert rm.get_current_drawdown(12000) == 0.0
    assert abs(rm.get_current_drawdown(10800) - 0.10) < 0.001


def test_status_report(tmp_config, tmp_db):
    rm = RiskManager(tmp_config, tmp_db)
    rm.update_equity(10000)

    status = rm.get_status(9500)
    assert status["current_equity"] == 9500
    assert status["peak_equity"] == 10000
    assert abs(status["drawdown_pct"] - 0.05) < 0.001
    assert not status["circuit_breaker_active"]

"""Tests for exit manager."""

from datetime import date

from src.strategy.exit_manager import ExitManager


def test_target_exit_date(tmp_config, tmp_db):
    em = ExitManager(tmp_config, tmp_db)

    # Monday → following Monday (5 trading days)
    entry = date(2024, 1, 8)  # Monday
    target = em.compute_target_exit_date(entry)
    assert target == date(2024, 1, 15)  # Following Monday


def test_target_exit_skips_weekends(tmp_config, tmp_db):
    em = ExitManager(tmp_config, tmp_db)

    # Friday → following Friday (5 trading days, skips 2 weekends)
    entry = date(2024, 1, 5)  # Friday
    target = em.compute_target_exit_date(entry)
    assert target == date(2024, 1, 12)  # Following Friday


def test_stop_loss_long(tmp_config, tmp_db):
    em = ExitManager(tmp_config, tmp_db)

    stop = em.compute_stop_loss(100.0, "long")
    assert stop == 95.0  # 5% below entry


def test_stop_loss_short(tmp_config, tmp_db):
    em = ExitManager(tmp_config, tmp_db)

    stop = em.compute_stop_loss(100.0, "short")
    assert stop == 105.0  # 5% above entry

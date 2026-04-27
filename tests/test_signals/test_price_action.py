"""Tests for price action signal generator."""

from datetime import date

from src.signals.price_action import PriceActionSignal


def test_generate_with_data(tmp_config, seeded_db):
    signal = PriceActionSignal(tmp_config, seeded_db)

    # Should not crash, may or may not produce a signal
    result = signal.generate("AAPL", date(2024, 6, 15))
    # Result could be None if no strong signal, that's OK
    if result:
        assert result.ticker == "AAPL"
        assert result.signal_type == "price_action"
        assert -1.0 <= result.strength <= 1.0
        assert result.direction in ("long", "short")


def test_generate_no_data(tmp_config, seeded_db):
    signal = PriceActionSignal(tmp_config, seeded_db)

    # Ticker not in DB
    result = signal.generate("ZZZZ", date(2024, 6, 15))
    assert result is None


def test_generate_bulk(tmp_config, seeded_db):
    signal = PriceActionSignal(tmp_config, seeded_db)

    results = signal.generate_bulk(["AAPL", "MSFT", "GOOGL"], date(2024, 6, 15))
    assert isinstance(results, list)
    for r in results:
        assert r.signal_type == "price_action"

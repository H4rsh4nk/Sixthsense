"""Lightweight validation for persisted market rows (schema + sanity checks)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_ohlcv_row(row: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason_if_bad) for one daily OHLCV dict."""
    try:
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = int(row["volume"])
    except (KeyError, TypeError, ValueError):
        return False, "parse_error"

    if min(o, h, l, c) <= 0 or v < 0:
        return False, "non_positive_ohlc_or_negative_volume"

    if h + 1e-9 < max(o, c) or l - 1e-9 > min(o, c):
        return False, "high_low_vs_oc"

    if h < l:
        return False, "high_below_low"

    return True, ""


def filter_anomalous_daily_returns(
    rows: list[dict[str, Any]],
    max_day_return_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    """Drop streak-breaking daily moves per ticker chronologically."""

    if not rows or max_day_return_frac <= 0:
        return rows, 0

    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["ticker"]].append(r)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for ticker, ticker_rows in grouped.items():
        ticker_rows_sorted = sorted(ticker_rows, key=lambda x: x["date"])
        prev_accepted_close: float | None = None

        for r in ticker_rows_sorted:
            ok, _ = validate_ohlcv_row(r)
            if not ok:
                dropped += 1
                logger.debug("Drop invalid OHLCV row %s %s", ticker, r.get("date"))
                continue

            close = float(r["close"])

            if prev_accepted_close is not None and prev_accepted_close > 0:
                day_ret = abs(close / prev_accepted_close - 1.0)
                if day_ret > max_day_return_frac:
                    dropped += 1
                    logger.warning(
                        "Drop anomalous return for %s on %s: %.2f%%",
                        ticker,
                        r["date"],
                        day_ret * 100.0,
                    )
                    continue

            prev_accepted_close = close
            kept.append(r)

    return kept, dropped

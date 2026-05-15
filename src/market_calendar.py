"""NYSE exchange-calendar helpers (holidays, early closes).

Thin wrapper around `exchange_calendars` (XNYS). Used by the live scheduler
to skip non-trading days and shorten the in-market window on early-close days.

See `docs/decisions/0007-exchange-calendar-awareness.md` for rationale.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from functools import lru_cache

logger = logging.getLogger(__name__)

_DEFAULT_CALENDAR = "XNYS"
_REGULAR_CLOSE_LOCAL = time(16, 0)  # 16:00 ET, NYSE regular session close


@lru_cache(maxsize=1)
def _calendar(name: str = _DEFAULT_CALENDAR):
    """Return a cached `exchange_calendars` calendar object.

    The library lazily compiles its holiday list; caching avoids repeated work
    across scheduler ticks. We tolerate the import failing so the scheduler can
    still run (degraded: treat every weekday as a trading day) — see ADR-0007.
    """

    try:
        import exchange_calendars as ec  # type: ignore
    except ImportError:
        logger.warning(
            "exchange_calendars not installed; calendar awareness disabled "
            "(assuming weekdays are trading days). Install with `pip install exchange_calendars`."
        )
        return None

    return ec.get_calendar(name)


def _to_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


def is_trading_day(d: date | datetime) -> bool:
    """True if NYSE is open at all on the given calendar date.

    Falls back to "Mon-Fri" if the library is unavailable.
    """

    target = _to_date(d)
    cal = _calendar()
    if cal is None:
        return target.weekday() < 5

    try:
        import pandas as pd

        ts = pd.Timestamp(target)
        return bool(cal.is_session(ts))
    except Exception as e:  # noqa: BLE001 - log and degrade gracefully
        logger.debug("calendar lookup failed for %s: %s", target, e)
        return target.weekday() < 5


def session_close_time(d: date | datetime) -> time | None:
    """Local-time (US/Eastern) close for the given session, or None if closed.

    Returns the *early* close time (e.g. 13:00) on early-close days, otherwise
    the regular 16:00 close. Returns None on non-trading days.
    """

    target = _to_date(d)
    if not is_trading_day(target):
        return None

    cal = _calendar()
    if cal is None:
        return _REGULAR_CLOSE_LOCAL

    try:
        import pandas as pd

        ts = pd.Timestamp(target)
        close_utc = cal.session_close(ts)
        # exchange_calendars returns a tz-aware UTC pandas.Timestamp.
        # Convert to US/Eastern (where the rest of the system lives).
        if close_utc.tzinfo is None:
            close_utc = close_utc.tz_localize("UTC")
        close_local = close_utc.tz_convert("US/Eastern")
        return close_local.time().replace(microsecond=0)
    except Exception as e:  # noqa: BLE001
        logger.debug("session_close lookup failed for %s: %s", target, e)
        return _REGULAR_CLOSE_LOCAL


def is_early_close(d: date | datetime) -> bool:
    """True if today's NYSE session closes before 16:00 ET."""

    close = session_close_time(d)
    if close is None:
        return False
    return close < _REGULAR_CLOSE_LOCAL


def summary(d: date | datetime) -> str:
    """Human-readable one-line summary for startup logging."""

    target = _to_date(d)
    if not is_trading_day(target):
        return f"{target}: NYSE closed (holiday or weekend)"
    close = session_close_time(target) or _REGULAR_CLOSE_LOCAL
    tag = " (early close)" if close < _REGULAR_CLOSE_LOCAL else ""
    return f"{target}: NYSE open, close at {close.strftime('%H:%M')} ET{tag}"

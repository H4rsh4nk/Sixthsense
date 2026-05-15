"""End-to-end smoke test — verify each piece of Sixthsense works with real data.

Runs against:
- A **temporary SQLite DB** so your production `data/swing_trader.db` is not touched.
- **Real external APIs where freely available** (Yahoo Finance, Federal Register,
  SEC EDGAR). Auto-skips checks whose dependencies are unreachable (Ollama not
  running, Alpaca keys missing, etc.) with a clear reason.

Exit codes:
- 0 — every required check passed (skips are allowed).
- 1 — one or more required checks failed.

Run::

    python -m scripts.smoketest
    python -m scripts.smoketest --skip news  # disable a slow check
    python -m scripts.smoketest --include-live-broker  # opt-in Alpaca reads

See ``TESTING.md`` for the full playbook.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure repo root is importable when this script is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import AppConfig, load_config  # noqa: E402
from src.database import Database  # noqa: E402


logger = logging.getLogger("smoketest")


# ---- result reporting ------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: str            # PASS | FAIL | SKIP
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class SmoketestReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)
        symbol = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(r.status, "[??]  ")
        print(f"{symbol} {r.name}  ({r.duration_ms:.0f} ms)")
        if r.detail:
            for line in r.detail.splitlines():
                print(f"        {line}")

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.status == "PASS")
        n_fail = sum(1 for r in self.results if r.status == "FAIL")
        n_skip = sum(1 for r in self.results if r.status == "SKIP")
        return f"{n_pass} passed, {n_fail} failed, {n_skip} skipped"


@contextmanager
def _timed(report: SmoketestReport, name: str, *, optional: bool = False):
    """Context manager: capture a check's outcome as a `CheckResult`."""
    start = datetime.utcnow()

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.detail = ""
    ctx.skip_reason: str | None = None

    try:
        yield ctx
    except _SkipCheck as e:
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000.0
        report.add(CheckResult(name=name, status="SKIP", detail=str(e), duration_ms=elapsed))
        return
    except Exception as e:  # noqa: BLE001 — capture *any* failure with traceback
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000.0
        tb = traceback.format_exc(limit=3).strip().splitlines()
        report.add(CheckResult(
            name=name,
            status="SKIP" if optional else "FAIL",
            detail=f"{type(e).__name__}: {e}\n" + "\n".join(tb[-3:]),
            duration_ms=elapsed,
        ))
        return

    elapsed = (datetime.utcnow() - start).total_seconds() * 1000.0
    report.add(CheckResult(name=name, status="PASS", detail=ctx.detail, duration_ms=elapsed))


class _SkipCheck(Exception):
    """Raise inside a check body to skip it cleanly."""


# ---- fixture: temp config + DB ---------------------------------------------


def _make_temp_config(tmpdir: Path) -> AppConfig:
    """Build an AppConfig with a temp DB path; keep secrets from env / real config."""
    # Reuse the user's real config so credentials / settings carry through, but
    # rewrite the DB path so nothing here can touch the production DB.
    cfg = load_config()
    cfg.data.db_path = str(tmpdir / "smoketest.db")
    return cfg


# ---- checks ----------------------------------------------------------------


def check_calendar(report: SmoketestReport) -> None:
    """ADR-0007: NYSE calendar awareness."""
    with _timed(report, "Calendar awareness (exchange_calendars / XNYS)") as ctx:
        try:
            from src import market_calendar
        except ImportError as e:
            raise _SkipCheck(f"market_calendar module unavailable: {e}") from e

        christmas_2026 = date(2026, 12, 25)
        weekday = date(2026, 5, 13)  # Wednesday
        if market_calendar.is_trading_day(christmas_2026):
            raise AssertionError("Expected NYSE closed on 2026-12-25 (Christmas)")
        if not market_calendar.is_trading_day(weekday):
            raise AssertionError("Expected NYSE open on 2026-05-13 (Wednesday)")
        ctx.detail = f"summary: {market_calendar.summary(date.today())}"


def check_database_schema(report: SmoketestReport, db: Database) -> None:
    """Verify schema migrations applied and expected tables exist."""
    with _timed(report, "Database schema + migrations") as ctx:
        with db.connect() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        required = {
            "prices", "signals", "trades", "decision_logs", "equity_snapshots",
            "universe", "insider_filings", "news_articles", "political_events",
            "signal_source_stats", "phase_state", "regression_alerts", "tool_call_log",
        }
        missing = required - tables
        if missing:
            raise AssertionError(f"Missing tables: {missing}")
        ctx.detail = f"{len(tables)} tables present, all required present"


def check_universe_seed(report: SmoketestReport, db: Database) -> None:
    """Seed a tiny custom universe so other checks have tickers to work with."""
    with _timed(report, "Universe seed (custom 4-ticker)") as ctx:
        db.insert_universe([
            {"ticker": "AAPL", "company_name": "Apple", "sector": "Technology",
             "industry": "Hardware", "market_cap": 3e12},
            {"ticker": "MSFT", "company_name": "Microsoft", "sector": "Technology",
             "industry": "Software", "market_cap": 3e12},
            {"ticker": "SPY", "company_name": "SPDR S&P 500", "sector": "Index",
             "industry": "ETF", "market_cap": 0},
            {"ticker": "JPM", "company_name": "JPMorgan", "sector": "Financials",
             "industry": "Bank", "market_cap": 5e11},
        ])
        n = len(db.get_all_tickers())
        if n < 4:
            raise AssertionError(f"Expected 4 tickers seeded, got {n}")
        ctx.detail = f"{n} tickers seeded"


def check_price_ingest(report: SmoketestReport, cfg: AppConfig, db: Database) -> None:
    """Real Yahoo download via `download_price_history` — small batch."""
    with _timed(report, "Price ingest via yfinance (AAPL + SPY, ~30 days)") as ctx:
        try:
            from src.backtest.data_loader import download_price_history
        except ImportError as e:
            raise _SkipCheck(f"data_loader unavailable: {e}") from e

        # Temporarily reduce history to keep the smoketest snappy.
        original_years = cfg.data.price_history_years
        cfg.data.price_history_years = 1
        try:
            download_price_history(db, ["AAPL", "SPY"], years=1, batch_size=2, config=cfg)
        except Exception as e:  # network failures shouldn't fail the suite
            raise _SkipCheck(f"yfinance unreachable: {e}") from e
        finally:
            cfg.data.price_history_years = original_years

        rows = db.get_prices("AAPL", (date.today() - timedelta(days=400)).isoformat(),
                             date.today().isoformat())
        if len(rows) < 10:
            raise AssertionError(f"Only {len(rows)} AAPL rows after download")
        ctx.detail = f"{len(rows)} AAPL rows ingested; OHLCV validation engaged"


def check_position_sizer(report: SmoketestReport, cfg: AppConfig) -> None:
    """ADR-0003: fixed-risk and fractional-Kelly modes."""
    with _timed(report, "PositionSizer (fixed_risk + fractional_kelly + fallback)") as ctx:
        from src.strategy.position_sizer import PositionSizer

        sizer = PositionSizer(cfg)
        s = sizer.calculate(account_equity=100000, available_cash=50000,
                            entry_price=100, stop_loss_pct=0.05)
        if s is None or s.sizing_mode_used != "fixed_risk":
            raise AssertionError(f"Fixed-risk sizing failed: {s}")

        # Force Kelly mode
        cfg.trading.sizing_mode = "fractional_kelly"
        sizer_k = PositionSizer(cfg)
        ki = {"news": {"n": 40, "wins": 24, "losses": 16,
                       "avg_win_pct": 0.04, "avg_loss_pct": 0.02}}
        sk = sizer_k.calculate(account_equity=100000, available_cash=50000,
                               entry_price=100, stop_loss_pct=0.05,
                               signal_sources=["news"], kelly_inputs=ki)
        if sk is None or sk.sizing_mode_used != "fractional_kelly":
            raise AssertionError(f"Kelly sizing failed: {sk}")

        # Fallback path
        sf = sizer_k.calculate(account_equity=100000, available_cash=50000,
                               entry_price=100, stop_loss_pct=0.05,
                               signal_sources=["news"],
                               kelly_inputs={"news": {"n": 5, "wins": 3, "losses": 2,
                                                      "avg_win_pct": 0.04, "avg_loss_pct": 0.02}})
        if sf is None or sf.sizing_mode_used != "fixed_risk" or not sf.kelly_fallback_reason:
            raise AssertionError(f"Kelly fallback did not engage: {sf}")

        cfg.trading.sizing_mode = "fixed_risk"  # restore for downstream checks
        ctx.detail = (
            f"fixed_risk shares={s.shares}; "
            f"kelly shares={sk.shares} target={sk.kelly_target_pct:.4f}; "
            f"fallback reason='{sf.kelly_fallback_reason[:50]}...'"
        )


def check_risk_manager(report: SmoketestReport, cfg: AppConfig, db: Database) -> None:
    """RiskManager: drawdown → circuit breaker; daily-loss limit."""
    with _timed(report, "RiskManager (circuit breaker + daily loss)") as ctx:
        from src.strategy.risk_manager import RiskManager

        rm = RiskManager(cfg, db)
        starting = cfg.trading.capital
        rm.start_new_day(starting)
        rm.update_equity(starting * 0.95)  # mild drawdown
        if rm.circuit_breaker_active:
            raise AssertionError("Circuit breaker tripped at 5% drawdown (should not)")
        rm.update_equity(starting * (1 - cfg.trading.max_drawdown_pct - 0.01))
        if not rm.circuit_breaker_active:
            raise AssertionError("Circuit breaker did NOT trip past max_drawdown_pct")
        ctx.detail = f"max_drawdown_pct={cfg.trading.max_drawdown_pct:.2f} -> breaker tripped as expected"


def check_regression_detector(report: SmoketestReport, cfg: AppConfig, db: Database) -> None:
    """ADR-0006: synthetic trades + equity → regression detector fires alerts."""
    with _timed(report, "Performance regression detector (synthetic data)") as ctx:
        from src.monitoring import regression

        with db.connect() as conn:
            # Seed 40 closed trades for 'news': oldest 10 win, most recent 30 lose.
            # Layout makes both alert conditions fire: recent 30-trade win rate = 0%
            # (below floor) and trails the 40-trade baseline of 25% by 25 pts.
            base_day = date(2026, 1, 1)
            for i in range(40):
                is_win = i < 10  # earliest inserted are winners
                exit_day = (base_day + timedelta(days=i)).isoformat()
                conn.execute(
                    """INSERT INTO trades
                       (ticker, direction, signal_type, signal_score, entry_reason,
                        entry_date, entry_price, shares, stop_loss_price,
                        target_exit_date, exit_date, exit_price, exit_reason,
                        pnl, pnl_pct, hold_days, status)
                       VALUES (?, 'long', 'news', 0.5, 'smoketest',
                               ?, 100, 10, 95, ?, ?, ?, 'time_exit',
                               ?, ?, 4, 'closed')""",
                    (
                        "TST", exit_day, exit_day, exit_day,
                        110 if is_win else 95,
                        100.0 if is_win else -50.0,
                        0.10 if is_win else -0.05,
                    ),
                )

        cfg.monitoring.regression_min_trades = 20
        cfg.monitoring.regression_win_rate_floor = 0.5
        cfg.monitoring.regression_win_rate_delta = 0.05
        cfg.monitoring.regression_min_equity_days = 1000  # disable Sharpe path
        findings = regression.evaluate(db, cfg)
        if not findings or not any(f.alert_type == "win_rate" for f in findings):
            raise AssertionError(f"Expected win_rate regression alert, got: {findings}")
        ctx.detail = (
            f"{len(findings)} findings; example: {findings[0].alert_type} "
            f"severity={findings[0].severity} "
            f"metric={findings[0].metric_value:.3f}"
        )


def check_calendar_phase_state(report: SmoketestReport, db: Database) -> None:
    """ADR-0008: persisted phase_state round-trips."""
    with _timed(report, "Persisted phase_state round-trip") as ctx:
        today = date.today().isoformat()
        db.set_phase_state("pre_market", today, datetime.utcnow().isoformat())
        loaded = db.load_phase_state()
        if "pre_market" not in loaded:
            raise AssertionError("phase_state row not persisted")
        if loaded["pre_market"]["last_run_date"] != today:
            raise AssertionError(
                f"phase_state date mismatch: {loaded['pre_market']['last_run_date']} != {today}"
            )
        db.clear_phase_state("pre_market")
        loaded2 = db.load_phase_state()
        if "pre_market" in loaded2:
            raise AssertionError("clear_phase_state did not erase the row")
        ctx.detail = "set / load / clear all behave"


def check_sticky_failover(report: SmoketestReport, cfg: AppConfig) -> None:
    """ADR-0004: BrokerCascade circuit breaker trips and recovers (mock-driven)."""
    with _timed(report, "BrokerCascade sticky failover (mocked legs)") as ctx:
        from src.execution.broker import AccountInfo, BrokerCascade

        # Build a cascade with synthetic legs so we never touch a real API.
        cascade = BrokerCascade.__new__(BrokerCascade)
        cascade.config = cfg
        cascade._failure_threshold = 2
        from datetime import timedelta as _td
        cascade._cooldown = _td(milliseconds=50)
        cascade._state = BrokerCascade.STATE_CLOSED
        cascade._opened_at = None
        cascade._consecutive_failures = 0
        cascade._recovery_cb = None

        class _FailingBroker:
            def __init__(self):
                self.calls = 0

            def get_account(self):
                self.calls += 1
                raise RuntimeError("primary down (simulated)")

        class _OKBroker:
            def get_account(self):
                return AccountInfo(equity=100.0, cash=50.0,
                                   buying_power=200.0, portfolio_value=150.0, status="ACTIVE")

        cascade.primary = _FailingBroker()
        cascade.fallback = _OKBroker()

        # First two calls trip the breaker; third should skip primary.
        cascade.get_account()
        cascade.get_account()
        if cascade.circuit_state != BrokerCascade.STATE_OPEN:
            raise AssertionError(
                f"Expected breaker OPEN after 2 failures, got {cascade.circuit_state}"
            )
        primary_calls_before = cascade.primary.calls
        cascade.get_account()
        if cascade.primary.calls != primary_calls_before:
            raise AssertionError("Open breaker should skip the primary leg")

        # After cooldown elapses, primary should be probed once (half_open).
        import time as _t
        _t.sleep(0.07)
        cascade.primary = _OKBroker()  # primary recovers
        # Need to satisfy a downstream isinstance check the broker uses internally
        cascade.primary.calls = 0  # type: ignore[attr-defined]

        # Use a callback to verify recovery hook fires.
        fired = {"v": False}

        def _cb():
            fired["v"] = True

        cascade.set_recovery_callback(_cb)
        cascade.get_account()  # probes primary, succeeds → breaker closes
        if cascade.circuit_state != BrokerCascade.STATE_CLOSED:
            raise AssertionError(
                f"Expected breaker CLOSED after recovery, got {cascade.circuit_state}"
            )
        if not fired["v"]:
            raise AssertionError("Recovery callback did not fire")
        ctx.detail = "OPEN -> HALF_OPEN -> CLOSED with recovery callback"


def check_tool_registry(
    report: SmoketestReport, cfg: AppConfig, db: Database, args: argparse.Namespace
) -> None:
    """Phase A: every tool resolves; non-network tools succeed."""
    with _timed(report, "Tool registry: build_default_registry + read tools") as ctx:
        from src.agent.tool_registry import (
            ToolContext,
            build_default_registry,
            execute_tool,
        )
        from src.signals.macro import MacroSignal
        from src.signals.price_action import PriceActionSignal
        from src.strategy.scorer import SignalScorer

        scorer = SignalScorer(cfg, db)
        tools = build_default_registry(include_write=False)

        # Skip news/insider by default — they hit external APIs and we cover them
        # with their own dedicated checks. Same for portfolio_state (needs broker).
        skip = {"get_portfolio_state", "get_news_sentiment", "get_insider_filings"}
        skip |= set(args.skip or [])

        signal_generators = {
            "price_action": PriceActionSignal(cfg, db),
            "macro": MacroSignal(cfg, db),
        }
        toolctx = ToolContext(
            config=cfg, db=db, scorer=scorer,
            signal_generators=signal_generators,
        )

        summary_parts: list[str] = []
        for name in sorted(tools):
            if name in skip:
                summary_parts.append(f"{name}=skip")
                continue
            kwargs: dict[str, Any] = {}
            if name in {"get_price_history", "compute_indicators"}:
                kwargs["ticker"] = "AAPL"
            res = execute_tool(tools, toolctx, name, **kwargs)
            tag = "ok" if res.ok else f"err({res.error[:30] if res.error else '?'}...)"
            summary_parts.append(f"{name}={tag}")
            if not res.ok and name in {"get_signal_source_stats", "query_regression_alerts",
                                      "get_political_events", "get_rules_ranking",
                                      "get_macro_regime", "get_price_history",
                                      "compute_indicators"}:
                # These tools should not fail given seeded data; treat as failure.
                raise AssertionError(f"Tool '{name}' failed unexpectedly: {res.error}")
        ctx.detail = " | ".join(summary_parts)


def check_broker_readonly(report: SmoketestReport, cfg: AppConfig) -> None:
    """Optional: read-only Alpaca round-trip (`get_account` + `get_latest_price('SPY')`)."""
    with _timed(report, "Broker read-only (Alpaca paper account + SPY price)",
                optional=True) as ctx:
        if not cfg.secrets.alpaca_api_key or not cfg.secrets.alpaca_secret_key:
            raise _SkipCheck("Alpaca credentials not configured (set in config/secrets.yaml)")
        from src.execution.broker import create_execution_broker

        broker = create_execution_broker(cfg)
        account = broker.get_account()
        price = broker.get_latest_price("SPY")
        if price is None or price <= 0:
            raise AssertionError(f"Got invalid SPY price from broker: {price}")
        ctx.detail = f"equity=${account.equity:,.2f} | SPY=${price:.2f}"


def check_news_signal_real(report: SmoketestReport, cfg: AppConfig, db: Database) -> None:
    """Optional: hit Google News + Ollama for an AAPL headline sentiment."""
    with _timed(report, "NewsSignal real fetch + LLM scoring (AAPL)",
                optional=True) as ctx:
        from src.signals.news import NewsSignal

        # If Ollama isn't running, treat as skip.
        api_base = cfg.agent.api_base or ""
        if "ollama" in (cfg.agent.provider or "").lower() or "11434" in api_base:
            try:
                import requests as _r
                _r.get(api_base.rstrip("/") + "/api/tags", timeout=1.5)
            except Exception as e:
                raise _SkipCheck(f"Ollama not reachable at {api_base}: {e}") from e

        ns = NewsSignal(cfg, db)
        try:
            result = ns.fetch_and_score("AAPL")
        except Exception as e:
            raise _SkipCheck(f"NewsSignal fetch failed: {e}") from e
        if result is None:
            ctx.detail = "AAPL fetch ran; no in-window scored signal (acceptable)"
        else:
            ctx.detail = (
                f"strength={result.strength:+.2f} dir={result.direction} "
                f"conf={result.confidence:.2f}"
            )


def check_political_signal_real(report: SmoketestReport, cfg: AppConfig, db: Database) -> None:
    """Optional: Federal Register fetch — small window."""
    with _timed(report, "PoliticalSignal real fetch (Federal Register, 7d)",
                optional=True) as ctx:
        from src.signals.political import PoliticalSignal

        ps = PoliticalSignal(cfg, db)
        try:
            n = ps.fetch_and_store_events(days_back=7)
        except Exception as e:
            raise _SkipCheck(f"Federal Register API failed: {e}") from e
        ctx.detail = f"{n} events fetched and stored"


# ---- runner ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sixthsense smoke test")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip a tool by name (repeatable).")
    parser.add_argument("--include-live-broker", action="store_true",
                        help="Run the optional Alpaca read-only check.")
    parser.add_argument("--include-news", action="store_true",
                        help="Run the optional NewsSignal real-fetch check.")
    parser.add_argument("--include-political", action="store_true",
                        help="Run the optional Federal Register fetch check.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress library logging (cleaner output).")
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Sixthsense smoke test -- every check uses a temp DB; production data is untouched.\n")

    report = SmoketestReport()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cfg = _make_temp_config(tmpdir)
        db = Database(cfg)

        # Required checks (no network needed)
        check_calendar(report)
        check_database_schema(report, db)
        check_universe_seed(report, db)
        check_position_sizer(report, cfg)
        check_risk_manager(report, cfg, db)
        check_calendar_phase_state(report, db)
        check_regression_detector(report, cfg, db)
        check_sticky_failover(report, cfg)

        # Network-dependent (free APIs) — auto-skip on failure
        check_price_ingest(report, cfg, db)

        # Tool registry (uses seeded DB)
        check_tool_registry(report, cfg, db, args)

        # Opt-in checks
        if args.include_live_broker:
            check_broker_readonly(report, cfg)
        if args.include_news:
            check_news_signal_real(report, cfg, db)
        if args.include_political:
            check_political_signal_real(report, cfg, db)

    print("\n" + report.summary())
    return 1 if any(r.status == "FAIL" for r in report.results) else 0


if __name__ == "__main__":
    sys.exit(main())

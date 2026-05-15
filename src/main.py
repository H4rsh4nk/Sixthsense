"""Main entry point — scheduler-driven trading system."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

from src.config import ROOT_DIR, load_config
from src import market_calendar

# Setup logging
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "swing_trader.log"),
    ],
)
logger = logging.getLogger("swing_trader")

def persist_decision_trace(db, scorer, stage: str, when: date):
    """Persist latest scorer decision trace to DB."""
    trace = getattr(scorer, "last_decision_trace", []) or []
    if not trace:
        return

    decision_time = datetime.now().isoformat()
    rows = []
    for d in trace:
        rows.append({
            "decision_time": decision_time,
            "decision_date": when.isoformat(),
            "stage": stage,
            "mode": d.get("mode", "unknown"),
            "ticker": d.get("ticker", ""),
            "direction": d.get("direction", ""),
            "score": d.get("score"),
            "selected": 1 if d.get("selected") else 0,
            "signal_sources": ",".join(d.get("signal_sources", [])),
            "reasoning": d.get("reasoning", ""),
            "rejection_reason": d.get("rejection_reason", ""),
            "signal_details": json.dumps(d.get("signal_details", [])),
            "agent_trace": json.dumps(d.get("agent_trace", {})),
        })
    db.insert_decision_logs(rows)


def run_backtest(args):
    """Run backtesting mode."""
    config = load_config()

    from src.backtest.analytics import (
        print_backtest_summary,
        run_parameter_sweep,
        run_walk_forward,
        save_results,
    )
    from src.backtest.data_loader import run_data_pipeline
    from src.backtest.engine import BacktestEngine
    from src.database import Database
    from src.signals.insider import InsiderSignal
    from src.signals.news import NewsSignal
    from src.signals.political import PoliticalSignal
    from src.signals.macro import MacroSignal
    from src.signals.price_action import PriceActionSignal

    # Initialize database and load data
    logger.info("Initializing database and loading data...")
    db = Database(config)

    if args.download_data:
        logger.info("Downloading historical data (this may take a while)...")
        run_data_pipeline(config)

    # Initialize signals
    signals = []
    if config.signals.insider.enabled:
        signals.append(InsiderSignal(config, db))
    if config.signals.news.enabled:
        signals.append(NewsSignal(config, db))
    if config.signals.political.enabled:
        signals.append(PoliticalSignal(config, db))
    if config.signals.price_action.enabled:
        signals.append(PriceActionSignal(config, db))
    if config.signals.macro.enabled:
        signals.append(MacroSignal(config, db))

    if not signals:
        logger.error("No signals enabled. Enable at least one in config/settings.yaml")
        return

    if args.sweep:
        # Parameter sweep
        results = run_parameter_sweep(config, db, signals)
        if results:
            save_results(results[0])  # Save best result
    elif args.walk_forward:
        # Walk-forward validation
        run_walk_forward(config, db, signals)
    else:
        # Single backtest with default params
        engine = BacktestEngine(config, db)
        start = date.fromisoformat(config.backtest.start_date)
        end = date.fromisoformat(config.backtest.end_date)

        result = engine.run(signals=signals, start_date=start, end_date=end)
        print_backtest_summary(result)
        save_results(result)


def run_live(args):
    """Run live/paper trading mode."""
    config = load_config()

    from apscheduler.schedulers.blocking import BlockingScheduler
    import pytz

    from src.database import Database
    from src.execution.broker import create_execution_broker
    from src.execution.order_manager import OrderManager
    from src.monitoring.alerts import AlertManager
    from src.monitoring import regression as regression_monitor
    from src.signals.insider import InsiderSignal
    from src.signals.news import NewsSignal
    from src.signals.political import PoliticalSignal
    from src.signals.macro import MacroSignal
    from src.signals.price_action import PriceActionSignal
    from src.strategy.exit_manager import ExitManager
    from src.strategy.position_sizer import PositionSizer
    from src.strategy.risk_manager import RiskManager
    from src.strategy.scorer import SignalScorer

    db = Database(config)
    broker = create_execution_broker(config)
    risk_manager = RiskManager(config, db)
    exit_manager = ExitManager(config, db)
    position_sizer = PositionSizer(config)
    scorer = SignalScorer(config, db)
    order_manager = OrderManager(
        config, db, broker, risk_manager, exit_manager, position_sizer
    )
    alerts = AlertManager(config)

    # Wire reconcile-on-recovery for sticky failover (see ADR-0004). The
    # callback runs when the broker circuit breaker closes after recovering
    # from a primary outage. AlpacaBroker (single-leg) has no breaker.
    if hasattr(broker, "set_recovery_callback"):
        broker.set_recovery_callback(order_manager.reconcile)

    # Initialize signal generators
    signal_generators = []
    if config.signals.insider.enabled:
        signal_generators.append(InsiderSignal(config, db))
    if config.signals.news.enabled:
        signal_generators.append(NewsSignal(config, db))
    if config.signals.political.enabled:
        signal_generators.append(PoliticalSignal(config, db))
    if config.signals.price_action.enabled:
        signal_generators.append(PriceActionSignal(config, db))
    if config.signals.macro.enabled:
        signal_generators.append(MacroSignal(config, db))

    tz = pytz.timezone(config.scheduler.timezone)

    def pre_market_scan():
        """Pre-market: scan for signals and prepare trade candidates."""
        logger.info("=== PRE-MARKET SCAN ===")
        if risk_manager.circuit_breaker_active:
            logger.warning("Circuit breaker active — skipping scan")
            return

        today = date.today()
        tickers = db.get_all_tickers()

        all_signals = []
        for gen in signal_generators:
            try:
                results = gen.generate_bulk(tickers, today)
                all_signals.extend(results)
                logger.info(f"  {gen.signal_type}: {len(results)} signals")
            except Exception as e:
                logger.error(f"  {gen.signal_type} failed: {e}")

        # Persist signals to local database
        signal_rows = []
        for s in all_signals:
            signal_rows.append({
                "signal_date": s.signal_date.isoformat() if hasattr(s.signal_date, "isoformat") else s.signal_date,
                "ticker": s.ticker,
                "signal_type": s.signal_type,
                "strength": s.strength,
                "direction": s.direction,
                "metadata": json.dumps(s.metadata) if s.metadata else "{}"
            })
        db.insert_signals(signal_rows)

        candidates = scorer.score(all_signals, today)
        persist_decision_trace(db, scorer, stage="pre_market", when=today)
        logger.info(f"  Scored candidates: {len(candidates)}")

        for c in candidates[:10]:
            logger.info(
                f"    {c.ticker}: score={c.combined_score:.2f} "
                f"direction={c.direction} signals={c.signal_sources}"
            )

    def market_open_entry():
        """Market open: execute entries for top candidates."""
        logger.info("=== MARKET OPEN ENTRY ===")
        if risk_manager.circuit_breaker_active:
            return

        account = broker.get_account()
        risk_manager.start_new_day(account.equity)
        risk_manager.update_equity(account.equity)

        today = date.today()
        tickers = db.get_all_tickers()

        # Re-generate and score (fresh data)
        all_signals = []
        for gen in signal_generators:
            try:
                all_signals.extend(gen.generate_bulk(tickers, today))
            except Exception:
                pass

        candidates = scorer.score(all_signals, today)
        persist_decision_trace(db, scorer, stage="market_open", when=today)

        for candidate in candidates:
            trade_id = order_manager.enter_trade(candidate)
            if trade_id:
                logger.info(f"  Entered trade #{trade_id}: {candidate.ticker}")

    def intraday_check():
        """Intraday: monitor positions and check stops."""
        logger.info("--- Intraday Check ---")
        account = broker.get_account()
        risk_manager.update_equity(account.equity)

        if risk_manager.circuit_breaker_active:
            logger.critical("CIRCUIT BREAKER — closing all positions")
            order_manager.close_all("circuit_breaker")
            return

        if risk_manager.check_daily_loss(account.equity):
            logger.warning("Daily loss limit hit — no new entries today")

        # Check exits
        open_trades = db.get_open_trades()
        tickers = [t["ticker"] for t in open_trades]
        prices = broker.get_latest_prices(tickers)
        closed = order_manager.process_exits(prices)

        if closed:
            logger.info(f"  Closed {len(closed)} positions")

    def heartbeat():
        """Periodic heartbeat to confirm backend health."""
        try:
            account = broker.get_account()
            status = risk_manager.get_status(account.equity)
            sg = db.count_signals_recent(days=7)
            sg_part = "".join(f" {k}:{v}" for k, v in sorted(sg.items())) if sg else " none"
            logger.info(
                "HEARTBEAT | mode=%s | equity=$%0.2f | open_positions=%s/%s | drawdown=%0.2f%% | circuit_breaker=%s"
                "| signal_rows_7d=%s",
                "PAPER" if config.broker.paper else "LIVE",
                account.equity,
                status["open_positions"],
                status["max_positions"],
                status["drawdown_pct"] * 100,
                status["circuit_breaker_active"],
                sg_part,
            )
        except Exception as e:
            logger.error(f"HEARTBEAT FAILED: {e}")

    def post_market_review():
        """Post-market: daily P&L review and equity snapshot."""
        logger.info("=== POST-MARKET REVIEW ===")
        account = broker.get_account()

        open_trades = db.get_open_trades()
        positions = broker.get_positions()
        positions_value = sum(p["market_value"] for p in positions)

        drawdown = risk_manager.get_current_drawdown(account.equity)

        db.insert_equity_snapshot({
            "date": date.today().isoformat(),
            "cash": account.cash,
            "positions_value": positions_value,
            "total_equity": account.equity,
            "daily_pnl": account.equity - risk_manager._daily_starting_equity,
            "drawdown_pct": drawdown,
            "open_positions": len(open_trades),
        })

        status = risk_manager.get_status(account.equity)
        logger.info(f"  Equity: ${account.equity:,.2f} | Drawdown: {drawdown:.2%} | "
                     f"Open: {len(open_trades)} positions")

        # Performance regression detection — ADR-0006. Runs after the equity
        # snapshot so today's row is included in the rolling window.
        if config.monitoring.regression_check_enabled:
            try:
                regression_monitor.run_check(
                    db, config, alerts, when=datetime.now(tz)
                )
            except Exception as e:  # noqa: BLE001 - monitoring is best-effort
                logger.error("Regression check failed: %s", e)

    def post_market_refresh():
        """Post-market: refresh external datasets."""
        logger.info("=== POST-MARKET DATA REFRESH ===")
        for gen in signal_generators:
            if hasattr(gen, "fetch_and_store_events"):
                try:
                    gen.fetch_and_store_events()
                except Exception as e:
                    logger.error(f"Data refresh failed for {gen.signal_type}: {e}")
            if hasattr(gen, "fetch_and_score"):
                tickers = db.get_all_tickers()[:50]  # Limit to top 50 to avoid rate limits
                for ticker in tickers:
                    try:
                        gen.fetch_and_score(ticker)
                    except Exception:
                        pass

    def parse_hhmm(value: str) -> time:
        hh, mm = map(int, value.split(":"))
        return time(hour=hh, minute=mm)

    sched = config.scheduler
    pre_market_t = parse_hhmm(sched.pre_market)
    market_open_t = parse_hhmm(sched.market_open_entry)
    market_close_t = parse_hhmm(sched.market_close_exit)
    post_market_t = parse_hhmm(sched.post_market)
    overnight_t = parse_hhmm(sched.overnight_scan)

    # Buffer (minutes) the configured market_close_exit lives ahead of the
    # regular 16:00 ET close. Re-applied on early-close days so we still flat
    # positions before the bell. See ADR-0007.
    close_buffer_minutes = max(
        0, (16 * 60) - (market_close_t.hour * 60 + market_close_t.minute)
    )

    def effective_close_time(today: date) -> time:
        """Compute today's effective market-close exit time.

        Honors NYSE early-close sessions when calendar awareness is enabled.
        Falls back to the static config value otherwise.
        """
        if not config.broker.calendar_aware:
            return market_close_t
        cal_close = market_calendar.session_close_time(today)
        if cal_close is None or cal_close >= time(16, 0):
            return market_close_t
        # Early close: apply the same pre-close buffer as the regular config.
        cal_dt = datetime.combine(today, cal_close) - timedelta(minutes=close_buffer_minutes)
        return cal_dt.time()

    # ---- persisted phase state (idempotency; see ADR-0008) ----------------

    _STATE_KEYS = (
        "pre_market",
        "market_open_entry",
        "post_market_review",
        "post_market_refresh",
        "intraday_check",
    )

    def _load_persisted_phase_state() -> dict:
        rows = db.load_phase_state()
        out: dict = {
            "pre_market_date": None,
            "market_open_entry_date": None,
            "post_market_review_date": None,
            "post_market_refresh_date": None,
            "last_intraday_check": None,
        }
        for key in _STATE_KEYS:
            row = rows.get(key)
            if not row:
                continue
            if key == "intraday_check":
                detail = row.get("detail") or row.get("last_run_at")
                if detail:
                    try:
                        out["last_intraday_check"] = datetime.fromisoformat(detail)
                    except ValueError:
                        out["last_intraday_check"] = None
            else:
                out[f"{key}_date"] = row.get("last_run_date")
        return out

    phase_state = _load_persisted_phase_state()
    logger.info(
        "Loaded persisted phase_state: pre_market=%s market_open_entry=%s "
        "post_market_review=%s post_market_refresh=%s last_intraday_check=%s",
        phase_state["pre_market_date"],
        phase_state["market_open_entry_date"],
        phase_state["post_market_review_date"],
        phase_state["post_market_refresh_date"],
        phase_state["last_intraday_check"],
    )

    def _mark_phase(key: str, today_iso: str, *, detail: str | None = None) -> None:
        """Persist phase row after a successful run. See ADR-0008."""
        db.set_phase_state(
            key,
            today_iso,
            datetime.utcnow().isoformat(),
            detail,
        )

    def get_market_phase(now_local: datetime, close_t: time) -> str:
        current = now_local.time()
        if pre_market_t <= current < market_open_t:
            return "pre_market"
        if market_open_t <= current < close_t:
            return "in_market"
        return "post_market"

    def phase_orchestrator():
        """Single 24/7 orchestrator that routes work by market phase."""
        now_local = datetime.now(tz)
        today_date = now_local.date()
        today = today_date.isoformat()
        is_trading = (
            market_calendar.is_trading_day(today_date)
            if config.broker.calendar_aware
            else (now_local.weekday() < 5)
        )
        close_t = effective_close_time(today_date)
        phase = get_market_phase(now_local, close_t)
        logger.info(
            "=== PHASE ORCHESTRATOR === phase=%s time=%s close=%s trading_day=%s",
            phase,
            now_local.strftime("%H:%M:%S"),
            close_t.strftime("%H:%M"),
            is_trading,
        )

        if not is_trading:
            logger.info(
                "Non-trading day — skipping trading jobs, allowing post-market refresh only"
            )
            if (
                now_local.time() >= overnight_t
                and phase_state["post_market_refresh_date"] != today
            ):
                post_market_refresh()
                phase_state["post_market_refresh_date"] = today
                _mark_phase("post_market_refresh", today)
            return

        if phase == "pre_market":
            if phase_state["pre_market_date"] != today:
                pre_market_scan()
                phase_state["pre_market_date"] = today
                _mark_phase("pre_market", today)
            return

        if phase == "in_market":
            if phase_state["market_open_entry_date"] != today:
                market_open_entry()
                phase_state["market_open_entry_date"] = today
                _mark_phase("market_open_entry", today)

            last_intraday = phase_state["last_intraday_check"]
            interval_minutes = sched.intraday_check_interval_minutes
            should_run_intraday = (
                last_intraday is None
                or (now_local - last_intraday).total_seconds() >= interval_minutes * 60
            )
            if should_run_intraday:
                intraday_check()
                phase_state["last_intraday_check"] = now_local
                _mark_phase("intraday_check", today, detail=now_local.isoformat())
            return

        # post_market phase
        if now_local.time() >= post_market_t and phase_state["post_market_review_date"] != today:
            post_market_review()
            phase_state["post_market_review_date"] = today
            _mark_phase("post_market_review", today)

        if now_local.time() >= overnight_t and phase_state["post_market_refresh_date"] != today:
            post_market_refresh()
            phase_state["post_market_refresh_date"] = today
            _mark_phase("post_market_refresh", today)

    # Setup scheduler
    scheduler = BlockingScheduler(timezone=tz)
    scheduler.add_job(phase_orchestrator, "interval", minutes=1, id="phase_orchestrator")
    scheduler.add_job(heartbeat, "interval", minutes=sched.heartbeat_interval_minutes, id="heartbeat")

    mode = "PAPER" if config.broker.paper else "LIVE"
    decision_mode = f"AI Agent ({config.agent.model})" if config.agent.enabled else "Rules"
    logger.info(f"Starting swing-trader in {mode} mode...")
    logger.info(f"Decision mode: {decision_mode}")
    logger.info(
        f"24/7 phases: pre_market={sched.pre_market}->{sched.market_open_entry}, "
        f"in_market={sched.market_open_entry}->{sched.market_close_exit}, "
        f"post_market=all other times"
    )
    logger.info(f"Heartbeat interval: every {sched.heartbeat_interval_minutes} minutes")
    if config.broker.calendar_aware:
        logger.info("Calendar awareness: %s", market_calendar.summary(datetime.now(tz).date()))
    else:
        logger.info("Calendar awareness: disabled (weekday-only check)")

    # --force-phase support: clear persisted phase rows so the next tick re-runs them.
    forced_phases = list(getattr(args, "force_phase", None) or [])
    for raw in forced_phases:
        key = raw.strip()
        if key not in _STATE_KEYS:
            logger.warning(
                "--force-phase: unknown phase '%s' (expected one of %s); ignoring",
                key, ", ".join(_STATE_KEYS),
            )
            continue
        db.clear_phase_state(key)
        if key == "intraday_check":
            phase_state["last_intraday_check"] = None
        else:
            phase_state[f"{key}_date"] = None
        logger.info("Cleared persisted phase '%s' — it will run on the next tick", key)

    try:
        if getattr(args, "now", False):
            logger.info("Executing immediate phase orchestration cycle (--now flag passed)")
            phase_orchestrator()
            
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler.shutdown()


def main():
    parser = argparse.ArgumentParser(description="News-Based Swing Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    bt = subparsers.add_parser("backtest", help="Run backtesting")
    bt.add_argument("--download-data", action="store_true",
                    help="Download historical data before backtesting")
    bt.add_argument("--sweep", action="store_true",
                    help="Run parameter sweep")
    bt.add_argument("--walk-forward", action="store_true",
                    help="Run walk-forward validation")

    # Live/paper trading command
    live = subparsers.add_parser("trade", help="Run live/paper trading")
    live.add_argument("--now", action="store_true", help="Trigger an immediate scan and entry cycle on startup")
    live.add_argument(
        "--force-phase",
        action="append",
        choices=[
            "pre_market",
            "market_open_entry",
            "intraday_check",
            "post_market_review",
            "post_market_refresh",
        ],
        help="Clear the persisted phase row so it re-runs on the next tick. "
             "Repeat to force multiple phases. See docs/decisions/0008-idempotent-phase-state.md.",
    )

    # Data download command
    data = subparsers.add_parser("download", help="Download historical data only")

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "trade":
        run_live(args)
    elif args.command == "download":
        config = load_config()
        from src.backtest.data_loader import run_data_pipeline
        run_data_pipeline(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Main entry point — scheduler-driven trading system."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from src.config import ROOT_DIR, load_config

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
    from apscheduler.triggers.cron import CronTrigger
    import pytz

    from src.database import Database
    from src.execution.broker import AlpacaBroker
    from src.execution.order_manager import OrderManager
    from src.signals.insider import InsiderSignal
    from src.signals.news import NewsSignal
    from src.signals.political import PoliticalSignal
    from src.signals.price_action import PriceActionSignal
    from src.strategy.exit_manager import ExitManager
    from src.strategy.position_sizer import PositionSizer
    from src.strategy.risk_manager import RiskManager
    from src.strategy.scorer import SignalScorer

    db = Database(config)
    broker = AlpacaBroker(config)
    risk_manager = RiskManager(config, db)
    exit_manager = ExitManager(config, db)
    position_sizer = PositionSizer(config)
    scorer = SignalScorer(config, db)
    order_manager = OrderManager(
        config, db, broker, risk_manager, exit_manager, position_sizer
    )

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
        import json
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

    def overnight_data_refresh():
        """Overnight: refresh data from external sources."""
        logger.info("=== OVERNIGHT DATA REFRESH ===")
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

    # Setup scheduler
    scheduler = BlockingScheduler(timezone=tz)

    # Parse schedule times
    sched = config.scheduler
    pre_h, pre_m = map(int, sched.pre_market.split(":"))
    entry_h, entry_m = map(int, sched.market_open_entry.split(":"))
    close_h, close_m = map(int, sched.market_close_exit.split(":"))
    post_h, post_m = map(int, sched.post_market.split(":"))
    night_h, night_m = map(int, sched.overnight_scan.split(":"))

    scheduler.add_job(pre_market_scan, CronTrigger(
        hour=pre_h, minute=pre_m, day_of_week="mon-fri"
    ), id="pre_market")

    scheduler.add_job(market_open_entry, CronTrigger(
        hour=entry_h, minute=entry_m, day_of_week="mon-fri"
    ), id="market_open")

    scheduler.add_job(intraday_check, "interval",
        minutes=sched.intraday_check_interval_minutes,
        id="intraday"
    )

    scheduler.add_job(post_market_review, CronTrigger(
        hour=post_h, minute=post_m, day_of_week="mon-fri"
    ), id="post_market")

    scheduler.add_job(overnight_data_refresh, CronTrigger(
        hour=night_h, minute=night_m, day_of_week="mon-fri"
    ), id="overnight")

    mode = "PAPER" if config.broker.paper else "LIVE"
    decision_mode = f"AI Agent ({config.agent.model})" if config.agent.enabled else "Rules"
    logger.info(f"Starting swing-trader in {mode} mode...")
    logger.info(f"Decision mode: {decision_mode}")
    logger.info(f"Schedule: pre={sched.pre_market}, entry={sched.market_open_entry}, "
                f"close={sched.market_close_exit}, post={sched.post_market}")

    try:
        if getattr(args, "now", False):
            logger.info("Executing immediate manual scan (--now flag passed)")
            pre_market_scan()
            market_open_entry()
            
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

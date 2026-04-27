"""Performance analytics and visualization for backtest results."""

from __future__ import annotations

import json
import logging
from datetime import date
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.config import AppConfig, ROOT_DIR
from src.database import Database
from src.signals.base import Signal

logger = logging.getLogger(__name__)


def print_backtest_summary(result: BacktestResult, label: str = "") -> None:
    """Print a formatted summary of backtest results."""
    title = f"Backtest Results: {label}" if label else "Backtest Results"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Period:          {result.parameters.get('start_date', 'N/A')} → "
          f"{result.parameters.get('end_date', 'N/A')}")
    print(f"  Signals:         {', '.join(result.parameters.get('signal_types', []))}")
    print(f"  Hold Days:       {result.parameters.get('hold_days', 'N/A')}")
    print(f"  Stop Loss:       {result.parameters.get('stop_loss_pct', 0):.1%}")
    print(f"{'─'*60}")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Equity:    ${result.final_equity:,.2f}")
    print(f"  Total Return:    {result.total_return_pct:+.2%}")
    print(f"{'─'*60}")
    print(f"  Total Trades:    {result.num_trades}")
    print(f"  Win Rate:        {result.win_rate:.1%}")
    print(f"  Avg Win:         {result.avg_win_pct:+.2%}")
    print(f"  Avg Loss:        {result.avg_loss_pct:+.2%}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Avg Hold Days:   {result.avg_hold_days:.1f}")
    print(f"{'─'*60}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2%}")
    print(f"{'='*60}\n")


def plot_equity_curve(result: BacktestResult, output_path: Path | None = None) -> None:
    """Plot equity curve and drawdown chart."""
    if not result.equity_curve:
        logger.warning("No equity curve data to plot.")
        return

    df = pd.DataFrame(result.equity_curve)
    df["date"] = pd.to_datetime(df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1.plot(df["date"], df["total_equity"], linewidth=1.5, color="#2196F3", label="Portfolio")
    ax1.axhline(
        y=result.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital"
    )
    ax1.set_title(
        f"Equity Curve — Return: {result.total_return_pct:+.2%} | "
        f"Sharpe: {result.sharpe_ratio:.2f} | MaxDD: {result.max_drawdown_pct:.2%}",
        fontsize=12,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Drawdown
    ax2.fill_between(
        df["date"], 0, -df["drawdown_pct"] * 100, color="#F44336", alpha=0.4
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved equity curve to {output_path}")
    plt.close()


def plot_trade_distribution(result: BacktestResult, output_path: Path | None = None) -> None:
    """Plot trade P&L distribution."""
    if not result.trades:
        return

    pnl_pcts = [t["pnl_pct"] * 100 for t in result.trades]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # P&L histogram
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnl_pcts]
    ax1.hist(pnl_pcts, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    ax1.set_title("Trade P&L Distribution")
    ax1.set_xlabel("P&L (%)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    # P&L by signal type
    signal_pnl: dict[str, list[float]] = {}
    for t in result.trades:
        for st in t["signal_type"].split(","):
            signal_pnl.setdefault(st, []).append(t["pnl_pct"] * 100)

    labels = list(signal_pnl.keys())
    means = [sum(v) / len(v) for v in signal_pnl.values()]
    bar_colors = ["#4CAF50" if m > 0 else "#F44336" for m in means]
    ax2.bar(labels, means, color=bar_colors, alpha=0.7, edgecolor="white")
    ax2.set_title("Avg P&L by Signal Type")
    ax2.set_ylabel("Avg P&L (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_parameter_sweep(
    config: AppConfig,
    db: Database,
    signals: list[Signal],
    hold_days_list: list[int] | None = None,
    stop_loss_list: list[float] | None = None,
) -> list[BacktestResult]:
    """Run backtests across parameter combinations and return ranked results."""
    hold_days_list = hold_days_list or config.exit_rules.sweep_hold_days
    stop_loss_list = stop_loss_list or config.exit_rules.sweep_stop_loss_pct

    engine = BacktestEngine(config, db)
    start = date.fromisoformat(config.backtest.start_date)
    end = date.fromisoformat(config.backtest.end_date)

    results = []
    combos = list(product(hold_days_list, stop_loss_list))
    total = len(combos)

    logger.info(f"Running parameter sweep: {total} combinations...")

    for i, (hold, stop) in enumerate(combos):
        logger.info(f"  [{i+1}/{total}] hold={hold}d, stop={stop:.1%}")
        try:
            result = engine.run(
                signals=signals,
                start_date=start,
                end_date=end,
                hold_days=hold,
                stop_loss_pct=stop,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Rank by Sharpe ratio
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    # Print summary table
    print(f"\n{'='*80}")
    print("  Parameter Sweep Results (ranked by Sharpe)")
    print(f"{'='*80}")
    print(f"  {'Hold':>6} {'Stop':>7} {'Return':>9} {'Sharpe':>8} {'Sortino':>9} "
          f"{'MaxDD':>8} {'WinRate':>9} {'Trades':>8} {'PF':>6}")
    print(f"  {'─'*6} {'─'*7} {'─'*9} {'─'*8} {'─'*9} {'─'*8} {'─'*9} {'─'*8} {'─'*6}")

    for r in results:
        hold = r.parameters.get("hold_days", "?")
        stop = r.parameters.get("stop_loss_pct", 0)
        print(
            f"  {hold:>5}d {stop:>6.1%} {r.total_return_pct:>+8.2%} "
            f"{r.sharpe_ratio:>8.2f} {r.sortino_ratio:>9.2f} "
            f"{r.max_drawdown_pct:>7.2%} {r.win_rate:>8.1%} "
            f"{r.num_trades:>8} {r.profit_factor:>5.2f}"
        )
    print(f"{'='*80}\n")

    return results


def run_walk_forward(
    config: AppConfig,
    db: Database,
    signals: list[Signal],
) -> dict:
    """Run walk-forward validation: train on N years, test on M years."""
    train_years = config.backtest.walk_forward_train_years
    test_years = config.backtest.walk_forward_test_years

    start = date.fromisoformat(config.backtest.start_date)
    end = date.fromisoformat(config.backtest.end_date)

    total_years = (end - start).days / 365
    if total_years < train_years + test_years:
        logger.error("Not enough data for walk-forward validation")
        return {}

    # Split periods
    train_end = date(start.year + train_years, start.month, start.day)
    test_start = train_end
    test_end = date(test_start.year + test_years, test_start.month, test_start.day)
    test_end = min(test_end, end)

    logger.info(f"Walk-forward: Train {start}→{train_end}, Test {test_start}→{test_end}")

    # Train: find best parameters
    engine = BacktestEngine(config, db)
    best_sharpe = -999
    best_params = {}

    for hold in config.exit_rules.sweep_hold_days:
        for stop in config.exit_rules.sweep_stop_loss_pct:
            result = engine.run(
                signals=signals,
                start_date=start,
                end_date=train_end,
                hold_days=hold,
                stop_loss_pct=stop,
            )
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = {"hold_days": hold, "stop_loss_pct": stop}

    logger.info(f"Best train params: {best_params} (Sharpe={best_sharpe:.2f})")

    # Test: apply best params to out-of-sample period
    test_result = engine.run(
        signals=signals,
        start_date=test_start,
        end_date=test_end,
        hold_days=best_params["hold_days"],
        stop_loss_pct=best_params["stop_loss_pct"],
    )

    print_backtest_summary(test_result, "Walk-Forward Test (Out-of-Sample)")

    return {
        "train_best_params": best_params,
        "train_sharpe": best_sharpe,
        "test_result": test_result,
    }


def save_results(result: BacktestResult, output_dir: Path | None = None) -> None:
    """Save backtest results to JSON and charts."""
    output_dir = output_dir or ROOT_DIR / "data" / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename from parameters
    params = result.parameters
    label = (
        f"{'-'.join(params.get('signal_types', ['unknown']))}"
        f"_hold{params.get('hold_days', 0)}"
        f"_stop{int(params.get('stop_loss_pct', 0) * 100)}"
    )

    # Save trades and equity curve as JSON
    data = {
        "parameters": result.parameters,
        "summary": {
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "num_trades": result.num_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "avg_hold_days": result.avg_hold_days,
        },
        "trades": result.trades,
    }

    json_path = output_dir / f"{label}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Save charts
    plot_equity_curve(result, output_dir / f"{label}_equity.png")
    plot_trade_distribution(result, output_dir / f"{label}_distribution.png")

    logger.info(f"Results saved to {output_dir / label}*")

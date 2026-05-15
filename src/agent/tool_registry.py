"""Tool registry — the surface the decision agent calls into.

This module wraps existing components (`InsiderSignal`, `NewsSignal`,
`PoliticalSignal`, `PriceActionSignal`, `MacroSignal`, `RiskManager`,
`PositionSizer`, `SignalScorer`, `OrderManager`, `Database`) as a uniform
set of named tools with JSON-schema arg signatures.

Phase A (this module) is **behaviour-neutral**: nothing in the live trading
loop calls these tools yet. They exist so:

1. The forthcoming agent-driver (ADR-0010 Phase B) can call them directly.
2. The smoketest (`scripts/smoketest.py`) can exercise each piece of the
   system through a single uniform contract.

Each tool function:
- Takes a `ToolContext` (dependencies) plus keyword args.
- Returns a JSON-serialisable dict (so traces persist cleanly).
- Raises `ToolError` on user-facing failures (missing data, invalid args).

`execute_tool(...)` is the safe entry-point — it catches errors, times the
call, and produces a `ToolResult` envelope suitable for logging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from src.config import AppConfig
from src.database import Database

logger = logging.getLogger(__name__)


# ---- contract types ---------------------------------------------------------


class ToolError(Exception):
    """Raised by a tool to signal a user-visible failure (bad args, no data).

    The agent should treat these as informative messages, not crashes.
    """


@dataclass
class ToolContext:
    """Dependencies passed into every tool. Built once per agent invocation.

    Anything optional (broker, agent-scorer, alerts) is allowed to be ``None``
    so the registry can be exercised in tests / smoke runs without a live
    broker or LLM provider.
    """

    config: AppConfig
    db: Database
    broker: Any = None                # AlpacaBroker | BrokerCascade | None
    risk_manager: Any = None          # RiskManager | None
    position_sizer: Any = None        # PositionSizer | None
    order_manager: Any = None         # OrderManager | None
    scorer: Any = None                # SignalScorer | None
    signal_generators: dict[str, Any] = field(default_factory=dict)
    # ``signal_generators`` maps signal_type → instance, e.g. {"news": NewsSignal(...)}


@dataclass
class ToolResult:
    """Envelope returned by `execute_tool` regardless of success."""

    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class Tool:
    """One tool registered with the agent."""

    name: str
    description: str
    parameters: dict[str, Any]                # JSON-schema (OpenAI tool-use format)
    fn: Callable[..., dict[str, Any]]
    is_side_effecting: bool = False           # True for `submit_paper_trade`

    def openai_schema(self) -> dict[str, Any]:
        """Schema in the format LiteLLM / OpenAI tool-use expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---- helpers ----------------------------------------------------------------


def _require(ctx: ToolContext, attr: str, tool_name: str) -> Any:
    val = getattr(ctx, attr, None)
    if val is None:
        raise ToolError(
            f"Tool '{tool_name}' requires `ToolContext.{attr}` — not provided "
            "(typical when running offline or without broker credentials)."
        )
    return val


def _parse_date(value: str | None, default: date | None = None) -> date:
    if not value:
        return default or date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ToolError(f"Invalid ISO date '{value}': {e}") from e


# ---- read tools -------------------------------------------------------------


def get_portfolio_state(ctx: ToolContext) -> dict[str, Any]:
    """Snapshot of equity, cash, open positions, and risk-engine status."""
    broker = _require(ctx, "broker", "get_portfolio_state")
    account = broker.get_account()
    positions = broker.get_positions() or []
    risk = ctx.risk_manager
    status = risk.get_status(account.equity) if risk is not None else None
    return {
        "equity": account.equity,
        "cash": account.cash,
        "buying_power": account.buying_power,
        "portfolio_value": account.portfolio_value,
        "open_positions": positions,
        "n_positions": len(positions),
        "risk_status": status,
    }


def get_price_history(
    ctx: ToolContext, ticker: str, days: int = 30
) -> dict[str, Any]:
    """Recent OHLCV rows from SQLite (no live fetch).

    Use `compute_indicators` for derived metrics; this tool is the raw data.
    """
    if not ticker:
        raise ToolError("`ticker` is required.")
    days = max(1, min(int(days), 365))
    end = date.today()
    start = end - timedelta(days=days * 2)  # widen to absorb weekends/holidays
    rows = ctx.db.get_prices(ticker, start.isoformat(), end.isoformat())
    if not rows:
        raise ToolError(
            f"No price rows for {ticker} between {start} and {end} "
            "(run `swing-trader download` first)."
        )
    return {
        "ticker": ticker,
        "start": rows[0]["date"],
        "end": rows[-1]["date"],
        "rows": rows[-days:],  # at most `days` most recent rows
    }


def compute_indicators(ctx: ToolContext, ticker: str) -> dict[str, Any]:
    """RSI / MACD / volume-spike via `PriceActionSignal`."""
    pa = ctx.signal_generators.get("price_action")
    if pa is None:
        raise ToolError(
            "`price_action` signal generator not registered on ToolContext. "
            "Wire it via ToolContext.signal_generators['price_action']."
        )
    df = pa._get_price_df(ticker, date.today())  # noqa: SLF001 (private helper reuse)
    if df.empty:
        raise ToolError(f"Not enough price history for {ticker} to compute indicators.")
    rsi = pa._analyze_rsi(df)  # noqa: SLF001
    macd = pa._analyze_macd(df)  # noqa: SLF001
    volume = pa._analyze_volume(df)  # noqa: SLF001
    return {
        "ticker": ticker,
        "rsi": rsi,
        "macd": macd,
        "volume": volume,
        "as_of": df.index[-1].date().isoformat() if hasattr(df.index[-1], "date") else None,
    }


def get_news_sentiment(
    ctx: ToolContext, ticker: str, days_back: int = 3
) -> dict[str, Any]:
    """Run `NewsSignal.fetch_and_score(ticker)` — hits RSS + LLM."""
    news = ctx.signal_generators.get("news")
    if news is None:
        raise ToolError("`news` signal generator not registered.")
    # The signal exposes `fetch_and_score(ticker)` which persists articles and
    # returns the SignalResult (or None) — wrap that envelope for the LLM.
    try:
        result = news.fetch_and_score(ticker)
    except AttributeError as e:
        raise ToolError(f"news.fetch_and_score(...) missing: {e}") from e
    if result is None:
        return {"ticker": ticker, "signal": None, "reason": "no relevant articles in window"}
    return {
        "ticker": ticker,
        "signal": {
            "strength": result.strength,
            "direction": result.direction,
            "confidence": result.confidence,
            "metadata": result.metadata,
        },
        "days_back": days_back,
    }


def get_insider_filings(
    ctx: ToolContext, ticker: str, days: int = 30
) -> dict[str, Any]:
    """Insider Form-4 cluster summary for a ticker, from `InsiderSignal`."""
    insider = ctx.signal_generators.get("insider")
    if insider is None:
        raise ToolError("`insider` signal generator not registered.")
    today = date.today()
    sigres = insider.generate(ticker, today)
    return {
        "ticker": ticker,
        "as_of": today.isoformat(),
        "signal": None if sigres is None else {
            "strength": sigres.strength,
            "direction": sigres.direction,
            "confidence": sigres.confidence,
            "metadata": sigres.metadata,
        },
    }


def get_political_events(
    ctx: ToolContext, sector: str | None = None, days: int = 7
) -> dict[str, Any]:
    """Recent political / regulatory events from the cached `political_events` table."""
    end = date.today()
    start = end - timedelta(days=max(1, int(days)))
    with ctx.db.connect() as conn:
        cursor = conn.execute(
            """SELECT event_date, event_type, title, affected_sectors, impact_score
               FROM political_events
               WHERE event_date >= ? AND event_date <= ?
               ORDER BY event_date DESC, id DESC
               LIMIT 50""",
            (start.isoformat(), end.isoformat()),
        )
        rows = [dict(r) for r in cursor.fetchall()]
    if sector:
        rows = [r for r in rows if sector.lower() in (r.get("affected_sectors") or "").lower()]
    return {"sector": sector, "start": start.isoformat(), "end": end.isoformat(), "events": rows}


def get_macro_regime(ctx: ToolContext) -> dict[str, Any]:
    """Current macro regime tilt from `MacroSignal`."""
    macro = ctx.signal_generators.get("macro")
    if macro is None:
        raise ToolError("`macro` signal generator not registered or not enabled.")
    today = date.today()
    snap = macro._regime_snapshot(today)  # noqa: SLF001
    if snap is None:
        return {"as_of": today.isoformat(), "regime": None, "reason": "insufficient benchmark history"}
    return {"as_of": today.isoformat(), "regime": snap}


def query_regression_alerts(ctx: ToolContext, days: int = 7) -> dict[str, Any]:
    """Recent rows from `regression_alerts` (ADR-0006)."""
    cutoff = (date.today() - timedelta(days=max(1, int(days)))).isoformat()
    with ctx.db.connect() as conn:
        cursor = conn.execute(
            """SELECT check_date, alert_type, signal_type, severity, metric_value, threshold, detail
               FROM regression_alerts
               WHERE check_date >= ?
               ORDER BY check_date DESC, id DESC
               LIMIT 100""",
            (cutoff,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
    return {"since": cutoff, "alerts": rows}


def get_signal_source_stats(ctx: ToolContext) -> dict[str, Any]:
    """Closed-trade win/loss counts per signal source plus Beta-prior multipliers."""
    with ctx.db.connect() as conn:
        rows = conn.execute(
            "SELECT signal_type, wins, losses, updated_at FROM signal_source_stats"
        ).fetchall()
    stats = [dict(r) for r in rows]
    multipliers = ctx.db.get_signal_accuracy_multipliers()
    return {"stats": stats, "accuracy_multipliers": multipliers}


def get_rules_ranking(
    ctx: ToolContext, as_of_date: str | None = None
) -> dict[str, Any]:
    """Run the rules-based scorer end-to-end and return its candidates.

    Lets the agent treat the existing rules pipeline as *one opinion among many*.
    """
    scorer = _require(ctx, "scorer", "get_rules_ranking")
    target = _parse_date(as_of_date)

    # Pull recent signals from the DB and let the scorer rank them.
    cutoff = (target - timedelta(days=2)).isoformat()
    with ctx.db.connect() as conn:
        rows = conn.execute(
            """SELECT signal_date, ticker, signal_type, strength, direction, metadata
               FROM signals
               WHERE signal_date >= ? AND signal_date <= ?
               ORDER BY signal_date DESC""",
            (cutoff, target.isoformat()),
        ).fetchall()

    if not rows:
        return {"as_of": target.isoformat(), "candidates": [], "n_signals": 0}

    # Reconstruct SignalResult-like objects the scorer expects.
    from src.signals.base import SignalResult
    import json
    sig_results = []
    for r in rows:
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except (TypeError, ValueError):
            meta = {}
        sig_results.append(SignalResult(
            ticker=r["ticker"],
            signal_date=date.fromisoformat(r["signal_date"]),
            signal_type=r["signal_type"],
            strength=float(r["strength"]),
            direction=r["direction"],
            confidence=float(meta.get("confidence", 1.0)),
            metadata=meta,
        ))

    candidates = scorer._score_with_rules(sig_results, target)  # noqa: SLF001
    return {
        "as_of": target.isoformat(),
        "n_signals": len(sig_results),
        "candidates": [
            {
                "ticker": c.ticker,
                "direction": c.direction,
                "score": c.combined_score,
                "signal_sources": c.signal_sources,
                "sector": c.sector,
            }
            for c in candidates
        ],
    }


# ---- write tool (gated) -----------------------------------------------------


def submit_paper_trade(
    ctx: ToolContext,
    ticker: str,
    direction: str,
    reasoning: str = "",
    size_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> dict[str, Any]:
    """Propose one trade. Always runs RiskManager / PositionSizer checks.

    Honours `OrderManager.dry_run` — when True, this tool returns the *would-be*
    trade and writes a `decision_logs` row but does not call the broker.

    Args:
        ticker: ticker symbol.
        direction: "long" or "short".
        reasoning: free-text explanation for the audit trail.
        size_pct: optional hint to the sizer (clamped to caps regardless).
        stop_loss_pct: optional override; defaults to ``exit_rules.stop_loss_pct``.
    """
    if direction not in ("long", "short"):
        raise ToolError(f"`direction` must be 'long' or 'short', got '{direction}'.")
    om = _require(ctx, "order_manager", "submit_paper_trade")

    from src.strategy.scorer import TradeCandidate
    candidate = TradeCandidate(
        ticker=ticker,
        combined_score=1.0,  # agent-supplied; rules-style score not meaningful here
        direction=direction,
        signal_sources=["agent"],
        num_signals=0,
        signals=[],
        sector="",
        metadata={
            "reasoning": reasoning or "agent-proposed trade",
            "decision_mode": "agent",
            "size_pct_hint": size_pct,
            "stop_loss_pct_hint": stop_loss_pct,
        },
    )

    trade_id = om.enter_trade(candidate)
    return {
        "ticker": ticker,
        "direction": direction,
        "trade_id": trade_id,
        "submitted": trade_id is not None,
        "dry_run": getattr(om, "dry_run", False),
    }


# ---- registry construction --------------------------------------------------


def build_default_registry(include_write: bool = True) -> dict[str, Tool]:
    """Return the standard tool registry.

    `include_write=False` is used by the smoketest to guarantee no broker
    calls happen during read-only verification.
    """

    tools: list[Tool] = [
        Tool(
            name="get_portfolio_state",
            description="Snapshot of account equity, cash, open positions, and risk-engine status.",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=get_portfolio_state,
        ),
        Tool(
            name="get_price_history",
            description="Recent daily OHLCV bars for a ticker from local SQLite (no live fetch).",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."},
                    "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30},
                },
                "required": ["ticker"],
            },
            fn=get_price_history,
        ),
        Tool(
            name="compute_indicators",
            description="Compute RSI, MACD, and volume-spike indicators for a ticker.",
            parameters={
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
            fn=compute_indicators,
        ),
        Tool(
            name="get_news_sentiment",
            description=(
                "Fetch recent news headlines for a ticker and score sentiment via LLM. "
                "May take seconds; results are typically cached."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days_back": {"type": "integer", "minimum": 1, "maximum": 14, "default": 3},
                },
                "required": ["ticker"],
            },
            fn=get_news_sentiment,
        ),
        Tool(
            name="get_insider_filings",
            description="Insider Form-4 cluster summary for a ticker over the lookback window.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 90, "default": 30},
                },
                "required": ["ticker"],
            },
            fn=get_insider_filings,
        ),
        Tool(
            name="get_political_events",
            description="Recent political / regulatory events; optionally filtered by sector.",
            parameters={
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Optional GICS sector filter."},
                    "days": {"type": "integer", "minimum": 1, "maximum": 30, "default": 7},
                },
                "required": [],
            },
            fn=get_political_events,
        ),
        Tool(
            name="get_macro_regime",
            description="Current benchmark-momentum macro regime tilt (risk-on / risk-off).",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=get_macro_regime,
        ),
        Tool(
            name="query_regression_alerts",
            description="Recent rows from the regression_alerts table (win-rate / Sharpe / drift).",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "minimum": 1, "maximum": 30, "default": 7},
                },
                "required": [],
            },
            fn=query_regression_alerts,
        ),
        Tool(
            name="get_signal_source_stats",
            description="Per-source win/loss counts plus Beta-prior accuracy multipliers.",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=get_signal_source_stats,
        ),
        Tool(
            name="get_rules_ranking",
            description=(
                "Run the legacy rules-based scorer on recent signals and return its candidates. "
                "Use as one opinion among many; defer to it on uncertain calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "as_of_date": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD). Defaults to today.",
                    }
                },
                "required": [],
            },
            fn=get_rules_ranking,
        ),
    ]

    if include_write:
        tools.append(Tool(
            name="submit_paper_trade",
            description=(
                "Propose a paper trade. Always runs RiskManager and PositionSizer. "
                "If OrderManager is in dry-run mode, logs the intent instead of placing the order."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "direction": {"type": "string", "enum": ["long", "short"]},
                    "reasoning": {"type": "string"},
                    "size_pct": {
                        "type": "number",
                        "description": "Optional sizing hint as a fraction of equity (0-1).",
                    },
                    "stop_loss_pct": {"type": "number"},
                },
                "required": ["ticker", "direction"],
            },
            fn=submit_paper_trade,
            is_side_effecting=True,
        ))

    return {t.name: t for t in tools}


# ---- safe execution entry-point ---------------------------------------------


def execute_tool(
    registry: dict[str, Tool], ctx: ToolContext, name: str, **kwargs: Any
) -> ToolResult:
    """Run one tool with timing and uniform error wrapping."""
    tool = registry.get(name)
    if tool is None:
        return ToolResult(tool=name, ok=False, error=f"Unknown tool '{name}'.")

    started = time.perf_counter()
    try:
        data = tool.fn(ctx, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ToolResult(tool=name, ok=True, data=data, latency_ms=latency_ms)
    except ToolError as e:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ToolResult(tool=name, ok=False, error=str(e), latency_ms=latency_ms)
    except Exception as e:  # noqa: BLE001 — surface unexpected errors as ToolResult
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.exception("Tool '%s' raised unexpectedly", name)
        return ToolResult(tool=name, ok=False, error=f"{type(e).__name__}: {e}", latency_ms=latency_ms)

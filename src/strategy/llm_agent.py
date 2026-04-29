"""LLM-powered trading agent — model-agnostic via LiteLLM.

Supports any LLM provider: Gemini, OpenAI, Anthropic, Ollama, Together, Groq, etc.
Uses LiteLLM's unified API for tool-calling across all providers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any

import litellm

from src.config import AppConfig
from src.database import Database
from src.signals.base import SignalResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert swing trader AI assistant. Your job is to analyze 
trading signals and decide which stocks to trade.

## Your Role
- You receive raw signals from 4 sources: insider filings (SEC Form 4), news sentiment 
  (FinBERT NLP), political events (Federal Register), and price action (RSI/MACD/volume).
- You have tools to query additional data: prices, positions, risk status, etc.
- You must decide which signals are worth trading and rank them by conviction.

## Risk Rules (MUST follow)
- Max {max_positions} concurrent positions
- Max {max_sector} positions in the same sector
- Max {risk_per_trade}% risk per trade
- Stop-loss at {stop_loss}%
- Circuit breaker at {max_drawdown}% total drawdown
- Daily loss limit: {daily_loss}%

## Decision Guidelines
- Be selective. Skip weak or ambiguous signals.
- Multiple confirming signals on the same ticker = stronger conviction.
- Insider buying clusters are historically the strongest alpha signal.
- News sentiment is most reliable when multiple articles agree.
- Price action alone is weak — use it as confirmation only.
- Consider sector exposure: don't pile into one sector.
- Consider the current portfolio: avoid correlated positions.

## Output Format
After analyzing the signals and using your tools, respond with a JSON array of 
trade decisions. Each decision must have:
```json
[
  {{
    "ticker": "AAPL",
    "direction": "long",
    "score": 0.85,
    "reasoning": "3 insiders bought $2M+ in last 5 days, FinBERT sentiment 0.82 across 4 articles, RSI at 32 (oversold). Strong multi-signal convergence.",
    "signal_sources": ["insider", "news", "price_action"]
  }}
]
```

If no signals are worth trading, return an empty array: []

IMPORTANT:
- `score` must be between 0.0 and 1.0 (your conviction level)
- `direction` must be "long" or "short"
- `reasoning` should be 1-2 sentences explaining your logic
- Only include tickers you genuinely recommend trading
- Return ONLY the JSON array, no other text
"""


# ── Tool definitions (OpenAI function-calling format, used by all providers) ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get recent OHLCV price data for a ticker. Use this to check price trends, support/resistance levels, and recent volatility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"},
                    "days": {"type": "integer", "description": "Number of days of history (default 30, max 90)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_positions",
            "description": "Get all currently open positions with entry price, P&L, and days held. Use this to check portfolio exposure before recommending new trades.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_status",
            "description": "Get current risk metrics: drawdown, daily P&L, circuit breaker status, and whether new trades are allowed. Always check this before recommending trades.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_exposure",
            "description": "Get a breakdown of current portfolio exposure by sector. Use this to avoid concentrated sector bets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insider_details",
            "description": "Get detailed insider filing information for a ticker: who bought/sold, amounts, dates, and whether trades are part of a planned 10b5-1 program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "days_back": {"type": "integer", "description": "Number of days to look back (default 30)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_headlines",
            "description": "Get recent news headlines and their FinBERT sentiment scores for a ticker. Use this to validate news signal strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Get current technical indicators (RSI, MACD, volume ratio) for a ticker. Use this to confirm or deny price action signals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
]


class TradingAgent:
    """LLM-powered trading decision maker — model-agnostic via LiteLLM.

    Supports any provider: Gemini, OpenAI, Anthropic, Ollama, Together, Groq, etc.
    Configure via settings.yaml:
        model: "gemini/gemini-2.0-flash"   → Google Gemini
        model: "gpt-4o"                    → OpenAI
        model: "claude-sonnet-4-20250514"            → Anthropic
        model: "ollama/llama3"             → Local Ollama
        model: "together_ai/meta-llama/Llama-3-70b"  → Together AI
    """

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self._model = config.agent.model
        self._temperature = config.agent.temperature
        self._max_tool_calls = config.agent.max_tool_calls
        self.last_trace: dict[str, Any] = {}

        # Set the API key for the configured provider
        self._setup_provider(config)

    def _setup_provider(self, config: AppConfig) -> None:
        """Configure LiteLLM with the appropriate API key and base URL."""
        api_key = config.secrets.llm_api_key
        provider = config.agent.provider.lower()

        # Set the appropriate env var for LiteLLM based on provider
        provider_env_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "together": "TOGETHER_API_KEY",
            "together_ai": "TOGETHERAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        env_var = provider_env_map.get(provider)
        if env_var and api_key and not os.environ.get(env_var):
            os.environ[env_var] = api_key

        # Set custom API base if provided (for Ollama, vLLM, etc.)
        if config.agent.api_base:
            self._api_base = config.agent.api_base
        else:
            self._api_base = None

        # Suppress LiteLLM debug noise
        litellm.suppress_debug_info = True

        logger.info(f"Agent initialized: provider={provider}, model={self._model}")

    def decide(
        self, signals: list[SignalResult], as_of_date: date
    ) -> list[dict[str, Any]]:
        """Run the LLM agent to analyze signals and produce trade decisions.

        Returns a list of dicts with keys: ticker, direction, score, reasoning, signal_sources
        """
        if not signals:
            logger.info("Agent: No signals to analyze")
            return []

        # Build the system prompt with current risk parameters
        system = SYSTEM_PROMPT.format(
            max_positions=self.config.trading.max_concurrent_positions,
            max_sector=self.config.trading.max_sector_positions,
            risk_per_trade=self.config.trading.risk_per_trade_pct * 100,
            stop_loss=self.config.exit_rules.stop_loss_pct * 100,
            max_drawdown=self.config.trading.max_drawdown_pct * 100,
            daily_loss=self.config.trading.daily_loss_limit_pct * 100,
        )

        # Format signals into a readable summary
        signal_summary = self._format_signals(signals, as_of_date)

        user_message = (
            f"Date: {as_of_date.isoformat()}\n\n"
            f"## Raw Signals Detected Today\n\n{signal_summary}\n\n"
            f"Analyze these signals. Use your tools to gather additional context as needed, "
            f"then provide your ranked trade recommendations as a JSON array."
        )

        try:
            decisions = self._run_agent_loop(system, user_message)
            logger.info(f"Agent produced {len(decisions)} trade decision(s)")
            return decisions
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            raise

    def _run_agent_loop(self, system: str, user_message: str) -> list[dict[str, Any]]:
        """Run the agent loop: call model → handle tool calls → repeat until done."""
        self.last_trace = {
            "model": self._model,
            "tool_calls": [],
            "final_response": "",
            "forced_final_answer": False,
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        tool_call_count = 0

        while tool_call_count < self._max_tool_calls:
            # Call the model via LiteLLM (unified API)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "tools": TOOLS,
                "tool_choice": "auto",
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.completion(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message

            # No tool calls → model is done, extract text
            if not assistant_msg.tool_calls:
                text = assistant_msg.content or "[]"
                self.last_trace["final_response"] = text
                return self._parse_decisions(text)

            # Process tool calls
            messages.append(assistant_msg.model_dump())

            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                logger.info(f"Agent tool call: {fn_name}({fn_args})")
                result = self._execute_tool(fn_name, fn_args)
                self.last_trace["tool_calls"].append({
                    "name": fn_name,
                    "args": fn_args,
                    "result_preview": result[:1000],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                tool_call_count += 1

        # Hit tool call limit — force a final answer without tools
        logger.warning(f"Agent hit tool call limit ({self._max_tool_calls}), forcing final answer")
        self.last_trace["forced_final_answer"] = True
        messages.append({
            "role": "user",
            "content": "You've used all available tool calls. Provide your final trade recommendations now as a JSON array.",
        })

        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = litellm.completion(**kwargs)
        text = response.choices[0].message.content or "[]"
        self.last_trace["final_response"] = text
        return self._parse_decisions(text)

    def _format_signals(self, signals: list[SignalResult], as_of_date: date) -> str:
        """Format raw signals into a readable summary for the LLM."""
        by_ticker: dict[str, list[SignalResult]] = {}
        for sig in signals:
            by_ticker.setdefault(sig.ticker, []).append(sig)

        lines = []
        for ticker, sigs in sorted(by_ticker.items()):
            lines.append(f"### {ticker}")
            for sig in sigs:
                meta_str = ""
                if sig.metadata:
                    meta_items = [f"{k}={v}" for k, v in sig.metadata.items()]
                    meta_str = f" | Details: {', '.join(meta_items)}"
                lines.append(
                    f"- **{sig.signal_type}**: strength={sig.strength:.2f}, "
                    f"direction={sig.direction}, confidence={sig.confidence:.2f}"
                    f"{meta_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the result as a JSON string."""
        try:
            handler = {
                "get_price_history": self._tool_get_price_history,
                "get_current_positions": self._tool_get_current_positions,
                "get_risk_status": self._tool_get_risk_status,
                "get_sector_exposure": self._tool_get_sector_exposure,
                "get_insider_details": self._tool_get_insider_details,
                "get_news_headlines": self._tool_get_news_headlines,
                "get_technical_indicators": self._tool_get_technical_indicators,
            }.get(tool_name)

            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            return handler(**args)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    # ── Tool implementations ───────────────────────────────────────────────

    def _tool_get_price_history(self, ticker: str, days: int = 30) -> str:
        """Fetch recent OHLCV price data from the database."""
        days = min(days, 90)
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days)).isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT date, open, high, low, close, volume "
                "FROM prices WHERE ticker = ? AND date BETWEEN ? AND ? "
                "ORDER BY date DESC LIMIT 20",
                (ticker, start_date, end_date),
            )
            rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return json.dumps({"ticker": ticker, "data": [], "message": "No price data available"})

        return json.dumps({"ticker": ticker, "days_requested": days, "data": rows})

    def _tool_get_current_positions(self) -> str:
        """Get all open positions from the database."""
        trades = self.db.get_open_trades()
        positions = []
        for t in trades:
            positions.append({
                "ticker": t["ticker"],
                "direction": t["direction"],
                "entry_date": t["entry_date"],
                "entry_price": t["entry_price"],
                "shares": t["shares"],
                "stop_loss_price": t["stop_loss_price"],
                "target_exit_date": t["target_exit_date"],
                "signal_type": t["signal_type"],
            })
        return json.dumps({"open_positions": positions, "count": len(positions)})

    def _tool_get_risk_status(self) -> str:
        """Get current risk metrics."""
        trades = self.db.get_open_trades()
        num_open = len(trades)
        max_positions = self.config.trading.max_concurrent_positions
        can_open = num_open < max_positions

        return json.dumps({
            "open_positions": num_open,
            "max_positions": max_positions,
            "can_open_new": can_open,
            "slots_available": max_positions - num_open,
            "max_drawdown_pct": self.config.trading.max_drawdown_pct * 100,
            "daily_loss_limit_pct": self.config.trading.daily_loss_limit_pct * 100,
            "stop_loss_pct": self.config.exit_rules.stop_loss_pct * 100,
            "risk_per_trade_pct": self.config.trading.risk_per_trade_pct * 100,
        })

    def _tool_get_sector_exposure(self) -> str:
        """Get sector breakdown of open positions."""
        trades = self.db.get_open_trades()
        sector_counts: dict[str, int] = {}

        for t in trades:
            with self.db.connect() as conn:
                cursor = conn.execute(
                    "SELECT sector FROM universe WHERE ticker = ?", (t["ticker"],)
                )
                row = cursor.fetchone()
                sector = row["sector"] if row else "Unknown"
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return json.dumps({
            "sectors": sector_counts,
            "max_per_sector": self.config.trading.max_sector_positions,
        })

    def _tool_get_insider_details(self, ticker: str, days_back: int = 30) -> str:
        """Get insider filing details for a ticker."""
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days_back)).isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT filing_date, insider_name, insider_title, transaction_type, "
                "shares, price_per_share, total_value, is_10b5_1 "
                "FROM insider_filings WHERE ticker = ? AND filing_date BETWEEN ? AND ? "
                "ORDER BY filing_date DESC LIMIT 10",
                (ticker, start_date, end_date),
            )
            rows = [dict(r) for r in cursor.fetchall()]

        return json.dumps({"ticker": ticker, "filings": rows, "count": len(rows)})

    def _tool_get_news_headlines(self, ticker: str) -> str:
        """Get recent news headlines and sentiment for a ticker."""
        start_date = (date.today() - timedelta(days=7)).isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT published_date, headline, source, sentiment_score, sentiment_label "
                "FROM news_articles WHERE ticker = ? AND published_date >= ? "
                "ORDER BY published_date DESC LIMIT 10",
                (ticker, start_date),
            )
            rows = [dict(r) for r in cursor.fetchall()]

        return json.dumps({"ticker": ticker, "articles": rows, "count": len(rows)})

    def _tool_get_technical_indicators(self, ticker: str) -> str:
        """Get current technical indicators from recent price data."""
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=60)).isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT date, close, volume FROM prices "
                "WHERE ticker = ? AND date BETWEEN ? AND ? "
                "ORDER BY date ASC",
                (ticker, start_date, end_date),
            )
            rows = [dict(r) for r in cursor.fetchall()]

        if len(rows) < 14:
            return json.dumps({"ticker": ticker, "error": "Insufficient data for indicators"})

        closes = [r["close"] for r in rows]
        rsi = self._calculate_rsi(closes)

        macd_line, signal_line, histogram = self._calculate_macd(closes)

        volumes = [r["volume"] for r in rows]
        avg_vol_20 = sum(volumes[-20:]) / min(len(volumes), 20) if volumes else 0
        current_vol = volumes[-1] if volumes else 0
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0

        return json.dumps({
            "ticker": ticker,
            "rsi_14": round(rsi, 2),
            "macd": {
                "macd_line": round(macd_line, 4),
                "signal_line": round(signal_line, 4),
                "histogram": round(histogram, 4),
                "crossover": "bullish" if histogram > 0 and macd_line > signal_line else "bearish",
            },
            "volume": {
                "current": current_vol,
                "avg_20d": round(avg_vol_20),
                "ratio": round(vol_ratio, 2),
                "is_spike": vol_ratio >= self.config.signals.price_action.volume_spike_multiplier,
            },
            "latest_close": closes[-1],
        })

    @staticmethod
    def _calculate_rsi(closes: list[float], period: int = 14) -> float:
        """Calculate RSI from a list of closing prices."""
        if len(closes) < period + 1:
            return 50.0

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(
        closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        def ema(data: list[float], period: int) -> list[float]:
            k = 2 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append(data[i] * k + result[-1] * (1 - k))
            return result

        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line_values = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_values = ema(macd_line_values, signal)

        return macd_line_values[-1], signal_values[-1], macd_line_values[-1] - signal_values[-1]

    # ── Response parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_decisions(text: str) -> list[dict[str, Any]]:
        """Parse the LLM's JSON response into trade decisions."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            decisions = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent response as JSON: {e}\nRaw: {text[:500]}")
            return []

        if isinstance(decisions, dict):
            # Open-source models often wrap arrays in a parent object
            for _, v in decisions.items():
                if isinstance(v, list):
                    decisions = v
                    break
            else:
                # If it's just a single decision object, wrap it
                if all(k in decisions for k in ("ticker", "direction", "score")):
                    decisions = [decisions]

        if not isinstance(decisions, list):
            logger.error(f"Agent response is not a list: {type(decisions)}")
            return []

        valid = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
                
            # Map common LLM hallucinations
            if "symbol" in d and "ticker" not in d:
                d["ticker"] = d["symbol"]
            if "confidence" in d and "score" not in d:
                d["score"] = d["confidence"]
            elif "signalStrength" in d and "score" not in d:
                d["score"] = abs(float(d.get("signalStrength", 0.0)))
                
            if not all(k in d for k in ("ticker", "direction", "score")):
                logger.warning(f"Skipping incomplete decision: {d}")
                continue
            if d["direction"] not in ("long", "short"):
                logger.warning(f"Invalid direction '{d['direction']}' for {d['ticker']}")
                continue
            score = float(d["score"])
            score = max(0.0, min(1.0, score))

            valid.append({
                "ticker": d["ticker"].upper(),
                "direction": d["direction"],
                "score": score,
                "reasoning": d.get("reasoning", "No reasoning provided by Agent."),
                "signal_sources": d.get("signal_sources", []),
            })

        valid.sort(key=lambda x: x["score"], reverse=True)
        return valid

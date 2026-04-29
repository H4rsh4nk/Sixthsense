"""Signal scorer and ranker — combines multiple signals into trade candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)


@dataclass
class TradeCandidate:
    """A scored trade candidate ready for position sizing."""
    ticker: str
    combined_score: float
    direction: str  # long | short
    signal_sources: list[str]
    num_signals: int
    signals: list[SignalResult]
    sector: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalScorer:
    """Combines signals from multiple generators into ranked trade candidates.

    Supports two modes:
    - Rules mode (default): weighted average of signal strengths
    - Agent mode: LLM-powered decision making with tool-calling
    """

    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db
        self.agent_mode = config.agent.enabled
        self.weight_map = {
            "insider": config.signals.insider.weight,
            "news": config.signals.news.weight,
            "political": config.signals.political.weight,
            "price_action": config.signals.price_action.weight,
        }
        self.last_decision_trace: list[dict[str, Any]] = []
        # Initialize the LLM agent if enabled
        self._agent = None
        if self.agent_mode:
            try:
                from src.strategy.llm_agent import TradingAgent
                self._agent = TradingAgent(config, db)
                logger.info(
                    f"Agent mode enabled: provider={config.agent.provider}, "
                    f"model={config.agent.model}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM agent: {e}")
                if not config.agent.fallback_to_rules:
                    raise
                logger.warning("Falling back to rule-based scoring")
                self.agent_mode = False

    @staticmethod
    def _signal_details(signals: list[SignalResult]) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for s in signals:
            details.append({
                "signal_type": s.signal_type,
                "strength": s.strength,
                "direction": s.direction,
                "confidence": s.confidence,
                "metadata": s.metadata or {},
            })
        return details

    def score(
        self, signals: list[SignalResult], as_of_date: date
    ) -> list[TradeCandidate]:
        """Score and rank signal results into trade candidates."""
        if self.agent_mode and self._agent is not None:
            return self._score_with_agent(signals, as_of_date)
        return self._score_with_rules(signals, as_of_date)

    def _score_with_agent(
        self, signals: list[SignalResult], as_of_date: date
    ) -> list[TradeCandidate]:
        """Use the LLM agent to score and rank signals."""
        try:
            decisions = self._agent.decide(signals, as_of_date)

            # Build a lookup for signals by ticker
            by_ticker: dict[str, list[SignalResult]] = {}
            for sig in signals:
                by_ticker.setdefault(sig.ticker, []).append(sig)

            candidates = []
            selected_tickers: set[str] = set()
            for d in decisions:
                ticker = d["ticker"]
                selected_tickers.add(ticker)
                ticker_signals = by_ticker.get(ticker, [])
                sector = self._get_sector(ticker)

                candidates.append(TradeCandidate(
                    ticker=ticker,
                    combined_score=d["score"],
                    direction=d["direction"],
                    signal_sources=d.get("signal_sources", [s.signal_type for s in ticker_signals]),
                    num_signals=len(ticker_signals),
                    signals=ticker_signals,
                    sector=sector,
                    metadata={
                        "decision_mode": "agent",
                        "reasoning": d.get("reasoning", ""),
                        "model": self.config.agent.model,
                        "individual_scores": {
                            s.signal_type: {"strength": s.strength, "confidence": s.confidence}
                            for s in ticker_signals
                        },
                    },
                ))

            trace: list[dict[str, Any]] = []
            for c in candidates:
                trace.append({
                    "mode": "agent",
                    "ticker": c.ticker,
                    "direction": c.direction,
                    "score": c.combined_score,
                    "selected": True,
                    "signal_sources": c.signal_sources,
                    "reasoning": c.metadata.get("reasoning", ""),
                    "rejection_reason": "",
                    "signal_details": self._signal_details(c.signals),
                    "agent_trace": getattr(self._agent, "last_trace", {}),
                })
            for ticker, ticker_signals in by_ticker.items():
                if ticker in selected_tickers:
                    continue
                trace.append({
                    "mode": "agent",
                    "ticker": ticker,
                    "direction": "",
                    "score": None,
                    "selected": False,
                    "signal_sources": [s.signal_type for s in ticker_signals],
                    "reasoning": "",
                    "rejection_reason": "not_selected_by_agent",
                    "signal_details": self._signal_details(ticker_signals),
                    "agent_trace": getattr(self._agent, "last_trace", {}),
                })
            self.last_decision_trace = trace

            logger.info(f"Agent scored {len(candidates)} candidates from {len(signals)} signals")
            return candidates

        except Exception as e:
            logger.error(f"Agent scoring failed: {e}", exc_info=True)
            if self.config.agent.fallback_to_rules:
                logger.warning("Falling back to rule-based scoring")
                return self._score_with_rules(signals, as_of_date)
            self.last_decision_trace = []
            return []

    def _score_with_rules(
        self, signals: list[SignalResult], as_of_date: date
    ) -> list[TradeCandidate]:
        """Original rule-based scoring: weighted average of signal strengths."""
        # Group by ticker
        by_ticker: dict[str, list[SignalResult]] = {}
        for sig in signals:
            by_ticker.setdefault(sig.ticker, []).append(sig)

        candidates = []
        trace: list[dict[str, Any]] = []
        for ticker, sigs in by_ticker.items():
            # Weighted combination
            weighted_sum = 0.0
            total_weight = 0.0
            direction_votes = {"long": 0, "short": 0}

            for sig in sigs:
                w = self.weight_map.get(sig.signal_type, 0.25)
                weighted_sum += sig.strength * w
                total_weight += w
                direction_votes[sig.direction] += w

            if total_weight == 0:
                trace.append({
                    "mode": "rules",
                    "ticker": ticker,
                    "direction": "",
                    "score": None,
                    "selected": False,
                    "signal_sources": [s.signal_type for s in sigs],
                    "reasoning": "",
                    "rejection_reason": "zero_total_weight",
                    "signal_details": self._signal_details(sigs),
                    "agent_trace": {},
                })
                continue

            combined = weighted_sum / total_weight
            direction = "long" if direction_votes["long"] >= direction_votes["short"] else "short"
            signal_sources = [s.signal_type for s in sigs]

            # Apply minimum thresholds
            if abs(combined) < self.config.scoring.min_combined_score:
                trace.append({
                    "mode": "rules",
                    "ticker": ticker,
                    "direction": direction,
                    "score": abs(combined),
                    "selected": False,
                    "signal_sources": signal_sources,
                    "reasoning": "",
                    "rejection_reason": "below_min_combined_score",
                    "signal_details": self._signal_details(sigs),
                    "agent_trace": {},
                })
                continue
            if len(sigs) < self.config.scoring.min_signals_agreeing:
                trace.append({
                    "mode": "rules",
                    "ticker": ticker,
                    "direction": direction,
                    "score": abs(combined),
                    "selected": False,
                    "signal_sources": signal_sources,
                    "reasoning": "",
                    "rejection_reason": "below_min_signals_agreeing",
                    "signal_details": self._signal_details(sigs),
                    "agent_trace": {},
                })
                continue

            # Get sector info
            sector = self._get_sector(ticker)

            candidates.append(TradeCandidate(
                ticker=ticker,
                combined_score=abs(combined),
                direction=direction,
                signal_sources=signal_sources,
                num_signals=len(sigs),
                signals=sigs,
                sector=sector,
                metadata={
                    "decision_mode": "rules",
                    "individual_scores": {
                        s.signal_type: {"strength": s.strength, "confidence": s.confidence}
                        for s in sigs
                    },
                },
            ))
            trace.append({
                "mode": "rules",
                "ticker": ticker,
                "direction": direction,
                "score": abs(combined),
                "selected": True,
                "signal_sources": signal_sources,
                "reasoning": "",
                "rejection_reason": "",
                "signal_details": self._signal_details(sigs),
                "agent_trace": {},
            })

        # Rank by combined score (highest first)
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        self.last_decision_trace = trace
        return candidates

    def _get_sector(self, ticker: str) -> str:
        """Look up ticker's sector from universe table."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT sector FROM universe WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            return row["sector"] if row else ""


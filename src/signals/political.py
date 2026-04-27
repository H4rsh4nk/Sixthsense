"""Political event signal generator.

Detects trade-relevant political events (tariffs, executive orders, Fed decisions)
and maps them to affected sectors/tickers.
Data sources: Federal Register API, Congress.gov API.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

import requests

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)

# Sector → keyword mapping for political event impact
SECTOR_KEYWORDS = {
    "Technology": ["tech", "semiconductor", "chip", "ai", "artificial intelligence", "data",
                   "cyber", "software", "silicon", "huawei", "tiktok", "ban"],
    "Energy": ["oil", "gas", "energy", "pipeline", "drill", "opec", "lng", "solar",
               "wind", "renewable", "fossil", "carbon", "emission", "climate"],
    "Health Care": ["pharma", "drug", "health", "medicare", "medicaid", "fda",
                    "vaccine", "biotech", "hospital", "insurance", "aca"],
    "Financials": ["bank", "financial", "fed", "interest rate", "monetary", "dodd-frank",
                   "regulation", "lending", "credit", "reserve", "treasury"],
    "Consumer Discretionary": ["tariff", "trade", "import", "export", "retail", "consumer",
                                "auto", "housing", "luxury"],
    "Industrials": ["infrastructure", "defense", "military", "contract", "manufacturing",
                    "supply chain", "steel", "aluminum", "construction"],
    "Materials": ["steel", "aluminum", "tariff", "mining", "commodity", "rare earth",
                  "lumber", "chemical"],
    "Communication Services": ["media", "broadcast", "telecom", "5g", "social media",
                                "content", "streaming", "antitrust"],
    "Consumer Staples": ["food", "agriculture", "farm", "grocery", "subsidy", "snap",
                          "nutrition"],
    "Utilities": ["utility", "power", "grid", "electric", "water", "nuclear",
                  "regulation"],
    "Real Estate": ["housing", "real estate", "mortgage", "rent", "zoning",
                    "property", "construction"],
}

# Impact direction heuristics
BEARISH_KEYWORDS = ["tariff", "ban", "sanction", "restrict", "penalty", "tax", "fine",
                    "lawsuit", "antitrust", "regulation", "investigation"]
BULLISH_KEYWORDS = ["subsidy", "stimulus", "infrastructure", "deregulat", "tax cut",
                    "incentive", "deal", "agreement", "approve", "grant", "fund"]

FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1/documents.json"


class PoliticalSignal(Signal):
    """Detect politically-driven trading opportunities."""

    signal_type = "political"

    def __init__(self, config: AppConfig, db: Database):
        self.config = config.signals.political
        self.app_config = config
        self.db = db
        self.session = requests.Session()

    def _fetch_federal_register(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch executive orders and presidential documents from Federal Register."""
        params = {
            "conditions[publication_date][gte]": start_date,
            "conditions[publication_date][lte]": end_date,
            "conditions[type][]": ["PRESDOCU", "RULE"],
            "per_page": 50,
            "order": "newest",
        }

        events = []
        try:
            resp = self.session.get(FEDERAL_REGISTER_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for doc in data.get("results", []):
                events.append({
                    "event_date": doc.get("publication_date", ""),
                    "event_type": "executive_order" if doc.get("type") == "Presidential Document"
                                 else "regulation",
                    "title": doc.get("title", ""),
                    "description": doc.get("abstract", ""),
                    "source_url": doc.get("html_url", ""),
                })
        except Exception as e:
            logger.warning(f"Federal Register API failed: {e}")

        return events

    def _classify_event(self, title: str, description: str) -> dict:
        """Classify a political event by affected sectors and expected impact."""
        text = (title + " " + (description or "")).lower()

        affected_sectors = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                affected_sectors.append(sector)

        # Determine impact direction
        bearish_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        bullish_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in text)

        if bullish_hits > bearish_hits:
            impact = 0.5 + min(bullish_hits * 0.15, 0.5)
        elif bearish_hits > bullish_hits:
            impact = -(0.5 + min(bearish_hits * 0.15, 0.5))
        else:
            impact = 0.0

        return {
            "affected_sectors": affected_sectors,
            "impact_score": impact,
        }

    def _get_sector_tickers(self, sectors: list[str]) -> list[str]:
        """Get tickers belonging to the given sectors from universe."""
        if not sectors:
            return []

        placeholders = ",".join("?" for _ in sectors)
        with self.db.connect() as conn:
            cursor = conn.execute(
                f"SELECT ticker FROM universe WHERE sector IN ({placeholders})",
                sectors,
            )
            return [row["ticker"] for row in cursor.fetchall()]

    def _get_events_from_db(self, as_of_date: date, lookback_days: int = 7) -> list[dict]:
        """Get political events from database."""
        start = (as_of_date - timedelta(days=lookback_days)).isoformat()
        end = as_of_date.isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                """SELECT * FROM political_events
                   WHERE event_date >= ? AND event_date <= ?
                   ORDER BY event_date DESC""",
                (start, end),
            )
            return [dict(row) for row in cursor.fetchall()]

    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Generate political signal for a specific ticker."""
        # Get ticker's sector
        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT sector FROM universe WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            sector = row["sector"]

        # Get recent political events
        events = self._get_events_from_db(as_of_date)
        if not events:
            return None

        # Find events affecting this ticker's sector
        relevant_events = []
        for event in events:
            affected = (event.get("affected_sectors") or "").split(",")
            if sector in affected or not affected[0]:
                relevant_events.append(event)

        if not relevant_events:
            return None

        # Aggregate impact
        total_impact = sum(e.get("impact_score", 0) for e in relevant_events)
        avg_impact = total_impact / len(relevant_events)

        if abs(avg_impact) < 0.2:
            return None

        direction = "long" if avg_impact > 0 else "short"

        return SignalResult(
            ticker=ticker,
            signal_date=as_of_date,
            signal_type=self.signal_type,
            strength=max(min(avg_impact, 1.0), -1.0),
            direction=direction,
            confidence=min(len(relevant_events) / 5.0, 1.0),
            metadata={
                "num_events": len(relevant_events),
                "avg_impact": avg_impact,
                "sector": sector,
                "events": [e["title"] for e in relevant_events[:3]],
            },
        )

    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        """Generate political signals for all tickers."""
        results = []
        for ticker in tickers:
            signal = self.generate(ticker, as_of_date)
            if signal:
                results.append(signal)
        return results

    def fetch_and_store_events(self, days_back: int = 30) -> int:
        """Fetch recent political events and store in database."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        events = self._fetch_federal_register(start_date, end_date)

        rows = []
        for event in events:
            classification = self._classify_event(event["title"], event.get("description", ""))

            rows.append({
                "event_date": event["event_date"],
                "event_type": event["event_type"],
                "title": event["title"],
                "description": event.get("description", ""),
                "affected_sectors": ",".join(classification["affected_sectors"]),
                "affected_tickers": "",
                "impact_score": classification["impact_score"],
                "source_url": event.get("source_url", ""),
            })

        if rows:
            with self.db.connect() as conn:
                conn.executemany(
                    """INSERT OR IGNORE INTO political_events
                       (event_date, event_type, title, description,
                        affected_sectors, affected_tickers, impact_score, source_url)
                       VALUES (:event_date, :event_type, :title, :description,
                               :affected_sectors, :affected_tickers, :impact_score, :source_url)""",
                    rows,
                )

        logger.info(f"Stored {len(rows)} political events")
        return len(rows)

    def backfill(self, start_date: date, end_date: date) -> None:
        """Backfill political events from Federal Register."""
        logger.info(f"Backfilling political events from {start_date} to {end_date}...")

        # Federal Register API supports date ranges natively
        current = start_date
        while current < end_date:
            batch_end = min(current + timedelta(days=90), end_date)
            events = self._fetch_federal_register(
                current.isoformat(), batch_end.isoformat()
            )

            rows = []
            for event in events:
                classification = self._classify_event(
                    event["title"], event.get("description", "")
                )
                rows.append({
                    "event_date": event["event_date"],
                    "event_type": event["event_type"],
                    "title": event["title"],
                    "description": event.get("description", ""),
                    "affected_sectors": ",".join(classification["affected_sectors"]),
                    "affected_tickers": "",
                    "impact_score": classification["impact_score"],
                    "source_url": event.get("source_url", ""),
                })

            if rows:
                with self.db.connect() as conn:
                    conn.executemany(
                        """INSERT OR IGNORE INTO political_events
                           (event_date, event_type, title, description,
                            affected_sectors, affected_tickers, impact_score, source_url)
                           VALUES (:event_date, :event_type, :title, :description,
                                   :affected_sectors, :affected_tickers, :impact_score,
                                   :source_url)""",
                        rows,
                    )

            logger.info(f"  {current} to {batch_end}: {len(rows)} events")
            current = batch_end + timedelta(days=1)

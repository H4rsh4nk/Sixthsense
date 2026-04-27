"""SEC Form 4 insider filing signal generator.

Detects clusters of insider buying as bullish signals.
Data source: SEC EDGAR FULL-TEXT search API (free, public).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from xml.etree import ElementTree as ET

import requests

from src.config import AppConfig, InsiderSignalConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)

SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_FILINGS = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FILING_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"


class InsiderSignal(Signal):
    """Detect insider buying clusters from SEC Form 4 filings."""

    signal_type = "insider"

    def __init__(self, config: AppConfig, db: Database):
        self.config = config.signals.insider
        self.app_config = config
        self.db = db
        self.user_agent = config.secrets.sec_edgar_user_agent or "SwingTrader bot@example.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def _fetch_recent_filings_for_ticker(self, ticker: str, days_back: int = 30) -> list[dict]:
        """Fetch recent Form 4 filings for a ticker from SEC EDGAR."""
        # Use the full-text search endpoint
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "forms": "4",
        }

        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            resp = self.session.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"EDGAR search failed for {ticker}: {e}")
            return []

    def _parse_form4_filing(self, filing_url: str) -> list[dict]:
        """Parse a Form 4 XML filing to extract transaction details."""
        try:
            time.sleep(0.1)  # SEC rate limit: 10 req/sec
            resp = self.session.get(filing_url, timeout=10)
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            ns = {"": "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"}

            transactions = []

            # Try to find non-derivative transactions
            for txn in root.iter("nonDerivativeTransaction"):
                try:
                    security_title = txn.findtext(".//securityTitle/value", "")
                    txn_date = txn.findtext(".//transactionDate/value", "")
                    txn_code = txn.findtext(".//transactionCoding/transactionCode", "")
                    shares = txn.findtext(".//transactionAmounts/transactionShares/value", "0")
                    price = txn.findtext(
                        ".//transactionAmounts/transactionPricePerShare/value", "0"
                    )
                    acq_disp = txn.findtext(
                        ".//transactionAmounts/transactionAcquiredDisposedCode/value", ""
                    )

                    shares_f = float(shares) if shares else 0
                    price_f = float(price) if price else 0

                    transactions.append({
                        "transaction_date": txn_date,
                        "transaction_type": txn_code,
                        "shares": shares_f,
                        "price_per_share": price_f,
                        "total_value": shares_f * price_f,
                        "acquired_disposed": acq_disp,
                        "security_title": security_title,
                    })
                except (ValueError, TypeError):
                    continue

            return transactions
        except Exception as e:
            logger.warning(f"Failed to parse Form 4: {filing_url}: {e}")
            return []

    def _get_insider_buys_from_db(
        self, ticker: str, as_of_date: date, lookback_days: int
    ) -> list[dict]:
        """Get insider buy filings from local database."""
        start = (as_of_date - timedelta(days=lookback_days)).isoformat()
        end = as_of_date.isoformat()
        filings = self.db.get_insider_filings(ticker, start, end)

        # Filter to purchases only, exclude 10b5-1 planned trades
        buys = [
            f for f in filings
            if f["transaction_type"] == "P"
            and not f["is_10b5_1"]
            and (f["total_value"] or 0) >= self.config.min_transaction_value
        ]
        return buys

    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Generate insider signal for a single ticker.

        Looks for clusters of insider buying within the configured window.
        """
        buys = self._get_insider_buys_from_db(
            ticker, as_of_date, self.config.cluster_window_days
        )

        if not buys:
            return None

        # Cluster scoring:
        # - More unique insiders buying = stronger signal
        # - Higher total value = stronger signal
        unique_insiders = len(set(b["insider_name"] for b in buys))
        total_value = sum(b["total_value"] or 0 for b in buys)
        total_shares = sum(b["shares"] for b in buys)

        # Normalize: 1 insider buying $100K = 0.3, 3+ insiders buying $500K+ = 1.0
        insider_score = min(unique_insiders / 3.0, 1.0)
        value_score = min(total_value / 500_000, 1.0)
        strength = (insider_score * 0.6 + value_score * 0.4)

        return SignalResult(
            ticker=ticker,
            signal_date=as_of_date,
            signal_type=self.signal_type,
            strength=strength,
            direction="long",
            confidence=min(unique_insiders / 5.0, 1.0),
            metadata={
                "unique_insiders": unique_insiders,
                "total_value": total_value,
                "total_shares": total_shares,
                "num_transactions": len(buys),
                "insiders": [b["insider_name"] for b in buys],
            },
        )

    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        """Generate insider signals for all tickers."""
        results = []
        for ticker in tickers:
            signal = self.generate(ticker, as_of_date)
            if signal:
                results.append(signal)
        return results

    def backfill(self, start_date: date, end_date: date) -> None:
        """Download historical insider filings from SEC EDGAR.

        Note: This is rate-limited and can take a while for the full S&P 500.
        In practice, you may want to use a bulk data source or third-party API.
        """
        tickers = self.db.get_all_tickers()
        logger.info(f"Backfilling insider filings for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(tickers)}")

            try:
                self._fetch_and_store_filings(ticker, start_date, end_date)
                time.sleep(0.15)  # Respect SEC rate limits
            except Exception as e:
                logger.warning(f"Failed to backfill {ticker}: {e}")
                continue

    def _fetch_and_store_filings(self, ticker: str, start_date: date, end_date: date):
        """Fetch Form 4 filings for a ticker and store in database."""
        # Use SEC EDGAR full-text search
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": start_date.isoformat(),
            "enddt": end_date.isoformat(),
            "forms": "4",
        }

        try:
            resp = self.session.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            rows = []
            for hit in hits[:20]:  # Limit to most recent 20
                source = hit.get("_source", {})
                filing_date = source.get("file_date", "")
                display_names = source.get("display_names", [])
                insider_name = display_names[0] if display_names else "Unknown"

                rows.append({
                    "filing_date": filing_date,
                    "ticker": ticker,
                    "insider_name": insider_name,
                    "insider_title": "",
                    "transaction_type": "P",  # Will be refined by XML parsing
                    "shares": 0,
                    "price_per_share": 0,
                    "total_value": 0,
                    "shares_owned_after": 0,
                    "is_10b5_1": 0,
                    "source_url": source.get("file_url", ""),
                })

            if rows:
                self.db.insert_insider_filings(rows)

        except Exception as e:
            logger.warning(f"EDGAR fetch failed for {ticker}: {e}")

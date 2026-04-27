"""News sentiment signal generator using FinBERT NLP.

Parses news from free RSS feeds and scores sentiment per ticker.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import feedparser
import requests

from src.config import AppConfig
from src.database import Database
from src.signals.base import Signal, SignalResult

logger = logging.getLogger(__name__)

# FinBERT model for financial sentiment (loaded lazily)
_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Lazy-load the FinBERT sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading FinBERT sentiment model (first time only)...")
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU
        )
    return _sentiment_pipeline


# Free news RSS sources
FINVIZ_RSS = "https://finviz.com/quote.ashx?t={ticker}&ty=c&p=d&b=1"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"


class NewsSignal(Signal):
    """Generate trading signals from news sentiment analysis."""

    signal_type = "news"

    def __init__(self, config: AppConfig, db: Database):
        self.config = config.signals.news
        self.app_config = config
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SwingTrader/1.0)"
        })

    def _fetch_news_google(self, ticker: str, days_back: int = 3) -> list[dict]:
        """Fetch news articles from Google News RSS."""
        url = GOOGLE_NEWS_RSS.format(ticker=ticker)
        articles = []

        try:
            feed = feedparser.parse(url)
            cutoff = datetime.now() - timedelta(days=days_back)

            for entry in feed.entries[:20]:
                try:
                    published = datetime(*entry.published_parsed[:6])
                    if published < cutoff:
                        continue

                    articles.append({
                        "headline": entry.title,
                        "published_date": published.strftime("%Y-%m-%d"),
                        "source": entry.get("source", {}).get("title", "Google News"),
                        "url": entry.link,
                    })
                except (AttributeError, TypeError):
                    continue
        except Exception as e:
            logger.warning(f"Google News RSS failed for {ticker}: {e}")

        return articles

    def _fetch_news_finviz(self, ticker: str) -> list[dict]:
        """Scrape news headlines from FinViz."""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        articles = []

        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return articles

            # Simple regex to extract news table rows
            # FinViz news table has pattern: date | headline | source
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            news_table = soup.find("table", {"id": "news-table"})

            if not news_table:
                return articles

            current_date = datetime.now().strftime("%Y-%m-%d")
            for row in news_table.find_all("tr")[:15]:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                date_cell = cells[0].text.strip()
                headline_cell = cells[1]
                link = headline_cell.find("a")

                if not link:
                    continue

                # Parse date (FinViz uses "Mon-DD-YY HH:MM" or just "HH:MM" for today)
                if len(date_cell) > 8:
                    try:
                        parsed_date = datetime.strptime(
                            date_cell.split()[0], "%b-%d-%y"
                        ).strftime("%Y-%m-%d")
                        current_date = parsed_date
                    except ValueError:
                        pass

                articles.append({
                    "headline": link.text.strip(),
                    "published_date": current_date,
                    "source": "FinViz",
                    "url": link.get("href", ""),
                })

        except Exception as e:
            logger.warning(f"FinViz scrape failed for {ticker}: {e}")

        return articles

    def _score_sentiment(self, headlines: list[str]) -> list[dict[str, Any]]:
        """Score sentiment of headlines using FinBERT."""
        if not headlines:
            return []

        pipeline = get_sentiment_pipeline()

        # FinBERT has a max token length; truncate long headlines
        truncated = [h[:512] for h in headlines]

        try:
            results = pipeline(truncated)
            scored = []
            for headline, result in zip(headlines, results):
                label = result["label"]  # positive, negative, neutral
                score = result["score"]  # confidence

                # Map to [-1, 1] scale
                if label == "positive":
                    sentiment = score
                elif label == "negative":
                    sentiment = -score
                else:
                    sentiment = 0.0

                scored.append({
                    "headline": headline,
                    "sentiment_score": sentiment,
                    "sentiment_label": label,
                    "confidence": score,
                })
            return scored
        except Exception as e:
            logger.error(f"FinBERT scoring failed: {e}")
            return []

    def _get_articles_from_db(self, ticker: str, as_of_date: date, days_back: int = 3) -> list[dict]:
        """Get cached articles from database."""
        start = (as_of_date - timedelta(days=days_back)).isoformat()
        end = as_of_date.isoformat()

        with self.db.connect() as conn:
            cursor = conn.execute(
                """SELECT * FROM news_articles
                   WHERE ticker = ? AND published_date >= ? AND published_date <= ?
                   ORDER BY published_date DESC""",
                (ticker, start, end),
            )
            return [dict(row) for row in cursor.fetchall()]

    def generate(self, ticker: str, as_of_date: date) -> SignalResult | None:
        """Generate news sentiment signal for a ticker."""
        articles = self._get_articles_from_db(ticker, as_of_date)

        if len(articles) < self.config.min_articles:
            return None

        # Filter to articles that have been scored
        scored = [a for a in articles if a.get("sentiment_score") is not None]
        if not scored:
            return None

        # Volume-weighted average sentiment
        total_weight = len(scored)
        avg_sentiment = sum(a["sentiment_score"] for a in scored) / total_weight

        # Only generate signal if sentiment exceeds threshold
        if abs(avg_sentiment) < self.config.sentiment_threshold:
            return None

        direction = "long" if avg_sentiment > 0 else "short"
        # Confidence increases with more articles agreeing
        agreement = sum(
            1 for a in scored
            if (a["sentiment_score"] > 0) == (avg_sentiment > 0)
        ) / len(scored)

        return SignalResult(
            ticker=ticker,
            signal_date=as_of_date,
            signal_type=self.signal_type,
            strength=avg_sentiment,
            direction=direction,
            confidence=agreement,
            metadata={
                "num_articles": len(scored),
                "avg_sentiment": avg_sentiment,
                "agreement_pct": agreement,
                "top_headlines": [a["headline"] for a in scored[:5]],
            },
        )

    def generate_bulk(self, tickers: list[str], as_of_date: date) -> list[SignalResult]:
        """Generate news signals for all tickers."""
        results = []
        for ticker in tickers:
            signal = self.generate(ticker, as_of_date)
            if signal:
                results.append(signal)
        return results

    def fetch_and_score(self, ticker: str) -> list[dict]:
        """Fetch latest news and score sentiment. Stores results in DB."""
        # Fetch from multiple sources
        articles = self._fetch_news_google(ticker)
        articles.extend(self._fetch_news_finviz(ticker))

        if not articles:
            return []

        # Score sentiment
        headlines = [a["headline"] for a in articles]
        sentiments = self._score_sentiment(headlines)

        # Merge and store
        rows = []
        for article, sentiment in zip(articles, sentiments):
            rows.append({
                "published_date": article["published_date"],
                "ticker": ticker,
                "headline": article["headline"],
                "source": article["source"],
                "url": article["url"],
                "sentiment_score": sentiment["sentiment_score"],
                "sentiment_label": sentiment["sentiment_label"],
            })

        if rows:
            with self.db.connect() as conn:
                conn.executemany(
                    """INSERT OR IGNORE INTO news_articles
                       (published_date, ticker, headline, source, url,
                        sentiment_score, sentiment_label)
                       VALUES (:published_date, :ticker, :headline, :source, :url,
                               :sentiment_score, :sentiment_label)""",
                    rows,
                )

        return rows

    def backfill(self, start_date: date, end_date: date) -> None:
        """Backfill news data. Limited by free RSS feed history (usually ~1 week)."""
        tickers = self.db.get_all_tickers()
        logger.info(f"Backfilling news for {len(tickers)} tickers (limited to recent data)...")

        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(tickers)}")
            try:
                self.fetch_and_score(ticker)
            except Exception as e:
                logger.warning(f"News backfill failed for {ticker}: {e}")

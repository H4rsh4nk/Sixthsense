"""Historical data fetching and storage for backtesting."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from src.config import AppConfig
from src.database import Database

logger = logging.getLogger(__name__)


def fetch_sp500_tickers() -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia."""
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        df = df.rename(columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        })
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "company_name", "sector", "industry"]]
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed ({e}), using top 50 tickers as fallback")
        return _fallback_tickers()


def _fallback_tickers() -> pd.DataFrame:
    """Fallback: top 50 S&P 500 stocks by market cap."""
    tickers = [
        ("AAPL", "Apple", "Information Technology", "Technology Hardware"),
        ("MSFT", "Microsoft", "Information Technology", "Systems Software"),
        ("AMZN", "Amazon", "Consumer Discretionary", "Internet Retail"),
        ("NVDA", "NVIDIA", "Information Technology", "Semiconductors"),
        ("GOOGL", "Alphabet A", "Communication Services", "Interactive Media"),
        ("META", "Meta Platforms", "Communication Services", "Interactive Media"),
        ("TSLA", "Tesla", "Consumer Discretionary", "Auto Manufacturers"),
        ("BRK-B", "Berkshire Hathaway", "Financials", "Multi-Sector Holdings"),
        ("UNH", "UnitedHealth", "Health Care", "Managed Health Care"),
        ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals"),
        ("JPM", "JPMorgan Chase", "Financials", "Diversified Banks"),
        ("V", "Visa", "Financials", "Transaction Processing"),
        ("XOM", "Exxon Mobil", "Energy", "Integrated Oil & Gas"),
        ("PG", "Procter & Gamble", "Consumer Staples", "Household Products"),
        ("MA", "Mastercard", "Financials", "Transaction Processing"),
        ("HD", "Home Depot", "Consumer Discretionary", "Home Improvement"),
        ("CVX", "Chevron", "Energy", "Integrated Oil & Gas"),
        ("MRK", "Merck", "Health Care", "Pharmaceuticals"),
        ("ABBV", "AbbVie", "Health Care", "Biotechnology"),
        ("LLY", "Eli Lilly", "Health Care", "Pharmaceuticals"),
        ("PEP", "PepsiCo", "Consumer Staples", "Soft Drinks"),
        ("KO", "Coca-Cola", "Consumer Staples", "Soft Drinks"),
        ("COST", "Costco", "Consumer Staples", "Hypermarkets"),
        ("AVGO", "Broadcom", "Information Technology", "Semiconductors"),
        ("WMT", "Walmart", "Consumer Staples", "Hypermarkets"),
        ("MCD", "McDonald's", "Consumer Discretionary", "Restaurants"),
        ("CSCO", "Cisco", "Information Technology", "Communications Equipment"),
        ("TMO", "Thermo Fisher", "Health Care", "Life Sciences Tools"),
        ("CRM", "Salesforce", "Information Technology", "Application Software"),
        ("ABT", "Abbott Labs", "Health Care", "Health Care Equipment"),
        ("ACN", "Accenture", "Information Technology", "IT Consulting"),
        ("NFLX", "Netflix", "Communication Services", "Movies & Entertainment"),
        ("AMD", "AMD", "Information Technology", "Semiconductors"),
        ("LIN", "Linde", "Materials", "Industrial Gases"),
        ("DHR", "Danaher", "Health Care", "Health Care Equipment"),
        ("ORCL", "Oracle", "Information Technology", "Application Software"),
        ("TXN", "Texas Instruments", "Information Technology", "Semiconductors"),
        ("PM", "Philip Morris", "Consumer Staples", "Tobacco"),
        ("NEE", "NextEra Energy", "Utilities", "Electric Utilities"),
        ("UNP", "Union Pacific", "Industrials", "Railroads"),
        ("BA", "Boeing", "Industrials", "Aerospace & Defense"),
        ("RTX", "RTX Corp", "Industrials", "Aerospace & Defense"),
        ("AMGN", "Amgen", "Health Care", "Biotechnology"),
        ("INTC", "Intel", "Information Technology", "Semiconductors"),
        ("CAT", "Caterpillar", "Industrials", "Construction Machinery"),
        ("GS", "Goldman Sachs", "Financials", "Investment Banking"),
        ("DIS", "Walt Disney", "Communication Services", "Movies & Entertainment"),
        ("SPGI", "S&P Global", "Financials", "Financial Exchanges"),
        ("DE", "Deere & Co", "Industrials", "Farm Machinery"),
        ("LOW", "Lowe's", "Consumer Discretionary", "Home Improvement"),
    ]
    return pd.DataFrame(tickers, columns=["ticker", "company_name", "sector", "industry"])


def load_universe(db: Database, config: AppConfig) -> list[str]:
    """Load the trading universe into the database and return ticker list."""
    if config.data.universe == "custom" and config.data.custom_tickers:
        tickers = config.data.custom_tickers
        rows = [{"ticker": t, "company_name": "", "sector": "", "industry": "", "market_cap": 0}
                for t in tickers]
        db.insert_universe(rows)
        return tickers

    logger.info("Fetching S&P 500 constituents from Wikipedia...")
    df = fetch_sp500_tickers()
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "sector": row["sector"],
            "industry": row["industry"],
            "market_cap": 0,
        })
    db.insert_universe(rows)
    logger.info(f"Loaded {len(rows)} tickers into universe")
    return [r["ticker"] for r in rows]


def download_price_history(
    db: Database,
    tickers: list[str],
    years: int = 3,
    batch_size: int = 50,
) -> None:
    """Download daily OHLCV data for all tickers and store in SQLite."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)
        logger.info(f"Downloading prices: batch {i // batch_size + 1} ({len(batch)} tickers)...")

        try:
            data = yf.download(
                batch_str,
                start=start_str,
                end=end_str,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"Failed to download batch: {e}")
            continue

        if data.empty:
            continue

        rows = []
        for ticker in batch:
            try:
                if len(batch) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker]

                if ticker_data.empty:
                    continue

                ticker_data = ticker_data.dropna(subset=["Close"])
                for date_idx, row in ticker_data.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")
                    rows.append({
                        "ticker": ticker,
                        "date": date_str,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "adj_close": float(row.get("Adj Close", row["Close"])),
                        "volume": int(row["Volume"]),
                    })
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue

        if rows:
            db.insert_prices(rows)
            logger.info(f"  Stored {len(rows)} price rows for {len(batch)} tickers")


def run_data_pipeline(config: AppConfig) -> Database:
    """Run the full data loading pipeline."""
    db = Database(config)

    # Step 1: Load universe
    tickers = load_universe(db, config)

    # Step 2: Download price history
    download_price_history(db, tickers, years=config.data.price_history_years)

    logger.info("Data pipeline complete.")
    return db

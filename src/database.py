"""SQLite database manager for price data, signals, and trade logs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import AppConfig, ROOT_DIR


SCHEMA = """
-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- SEC Form 4 insider filings
CREATE TABLE IF NOT EXISTS insider_filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    insider_name TEXT NOT NULL,
    insider_title TEXT,
    transaction_type TEXT NOT NULL,  -- P=purchase, S=sale, A=award
    shares REAL NOT NULL,
    price_per_share REAL,
    total_value REAL,
    shares_owned_after REAL,
    is_10b5_1 INTEGER DEFAULT 0,
    source_url TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker_date ON insider_filings(ticker, filing_date);

-- News articles and sentiment
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    published_date TEXT NOT NULL,
    ticker TEXT,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    sentiment_score REAL,  -- -1.0 to 1.0
    sentiment_label TEXT,  -- positive, negative, neutral
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_articles(ticker, published_date);

-- Political events
CREATE TABLE IF NOT EXISTS political_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    affected_sectors TEXT,  -- comma-separated
    affected_tickers TEXT,  -- comma-separated
    impact_score REAL,  -- -1.0 to 1.0
    source_url TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_political_date ON political_events(event_date);

-- Generated signals
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- insider, news, political, price_action
    strength REAL NOT NULL,  -- -1.0 to 1.0
    direction TEXT NOT NULL,  -- long, short
    metadata TEXT,  -- JSON blob with signal-specific details
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(signal_date, ticker);

-- Trade log
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,  -- long, short
    signal_type TEXT NOT NULL,
    signal_score REAL NOT NULL,
    entry_reason TEXT,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    shares REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    target_exit_date TEXT NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,  -- time_exit, stop_loss, trailing_stop, manual, circuit_breaker
    pnl REAL,
    pnl_pct REAL,
    hold_days INTEGER,
    status TEXT NOT NULL DEFAULT 'open',  -- open, closed
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

-- Decision trace log (selected + rejected candidates)
CREATE TABLE IF NOT EXISTS decision_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_time TEXT NOT NULL,
    decision_date TEXT NOT NULL,
    stage TEXT NOT NULL,  -- pre_market | market_open | manual
    mode TEXT NOT NULL,   -- agent | rules
    ticker TEXT NOT NULL,
    direction TEXT,
    score REAL,
    selected INTEGER NOT NULL,  -- 1=yes, 0=no
    signal_sources TEXT,  -- comma-separated
    reasoning TEXT,
    rejection_reason TEXT,
    signal_details TEXT,  -- JSON details for each signal
    agent_trace TEXT,  -- JSON trace (tool calls, raw model output)
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_decision_logs_date ON decision_logs(decision_date);
CREATE INDEX IF NOT EXISTS idx_decision_logs_stage ON decision_logs(stage);

-- Daily equity snapshots
CREATE TABLE IF NOT EXISTS equity_snapshots (
    date TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_equity REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    open_positions INTEGER NOT NULL
);

-- S&P 500 universe
CREATE TABLE IF NOT EXISTS universe (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


class Database:
    def __init__(self, config: AppConfig):
        self.db_path = ROOT_DIR / config.data.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Apply lightweight non-destructive schema migrations."""
        cursor = conn.execute("PRAGMA table_info(trades)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "entry_reason" not in columns:
            conn.execute("ALTER TABLE trades ADD COLUMN entry_reason TEXT")
        cursor = conn.execute("PRAGMA table_info(decision_logs)")
        decision_cols = {row["name"] for row in cursor.fetchall()}
        if decision_cols:
            if "signal_details" not in decision_cols:
                conn.execute("ALTER TABLE decision_logs ADD COLUMN signal_details TEXT")
            if "agent_trace" not in decision_cols:
                conn.execute("ALTER TABLE decision_logs ADD COLUMN agent_trace TEXT")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_prices(self, rows: list[dict]):
        """Bulk insert OHLCV price rows."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO prices
                   (ticker, date, open, high, low, close, adj_close, volume)
                   VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)""",
                rows,
            )

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        """Get price data for a ticker in a date range."""
        with self.connect() as conn:
            cursor = conn.execute(
                """SELECT * FROM prices
                   WHERE ticker = ? AND date >= ? AND date <= ?
                   ORDER BY date""",
                (ticker, start_date, end_date),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_tickers(self) -> list[str]:
        """Get all tickers in the universe."""
        with self.connect() as conn:
            cursor = conn.execute("SELECT ticker FROM universe ORDER BY ticker")
            return [row["ticker"] for row in cursor.fetchall()]

    def insert_universe(self, rows: list[dict]):
        """Bulk insert universe tickers."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO universe
                   (ticker, company_name, sector, industry, market_cap)
                   VALUES (:ticker, :company_name, :sector, :industry, :market_cap)""",
                rows,
            )

    def insert_insider_filings(self, rows: list[dict]):
        """Bulk insert insider filings."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO insider_filings
                   (filing_date, ticker, insider_name, insider_title,
                    transaction_type, shares, price_per_share, total_value,
                    shares_owned_after, is_10b5_1, source_url)
                   VALUES (:filing_date, :ticker, :insider_name, :insider_title,
                           :transaction_type, :shares, :price_per_share, :total_value,
                           :shares_owned_after, :is_10b5_1, :source_url)""",
                rows,
            )

    def get_insider_filings(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        """Get insider filings for a ticker in a date range."""
        with self.connect() as conn:
            cursor = conn.execute(
                """SELECT * FROM insider_filings
                   WHERE ticker = ? AND filing_date >= ? AND filing_date <= ?
                   ORDER BY filing_date""",
                (ticker, start_date, end_date),
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_signals(self, rows: list[dict]):
        """Bulk insert generated signals."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO signals
                   (signal_date, ticker, signal_type, strength, direction, metadata)
                   VALUES (:signal_date, :ticker, :signal_type, :strength, :direction, :metadata)""",
                rows,
            )

    def insert_trade(self, trade: dict) -> int:
        """Insert a trade record and return its ID."""
        trade = {**trade}
        trade.setdefault("entry_reason", "")
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (ticker, direction, signal_type, signal_score, entry_reason, entry_date,
                    entry_price, shares, stop_loss_price, target_exit_date, status)
                   VALUES (:ticker, :direction, :signal_type, :signal_score, :entry_reason, :entry_date,
                           :entry_price, :shares, :stop_loss_price, :target_exit_date, :status)""",
                trade,
            )
            return cursor.lastrowid

    def close_trade(self, trade_id: int, exit_date: str, exit_price: float, exit_reason: str):
        """Close an open trade with exit details."""
        with self.connect() as conn:
            # Get trade entry data
            cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            trade = dict(cursor.fetchone())

            pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            if trade["direction"] == "short":
                pnl = -pnl
            pnl_pct = pnl / (trade["entry_price"] * trade["shares"])

            conn.execute(
                """UPDATE trades
                   SET exit_date = ?, exit_price = ?, exit_reason = ?,
                       pnl = ?, pnl_pct = ?,
                       hold_days = julianday(?) - julianday(entry_date),
                       status = 'closed'
                   WHERE id = ?""",
                (exit_date, exit_price, exit_reason, pnl, pnl_pct, exit_date, trade_id),
            )

    def get_open_trades(self) -> list[dict]:
        """Get all currently open trades."""
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_date"
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_decision_logs(self, rows: list[dict]):
        """Bulk insert decision trace rows."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO decision_logs
                   (decision_time, decision_date, stage, mode, ticker, direction, score,
                    selected, signal_sources, reasoning, rejection_reason, signal_details, agent_trace)
                   VALUES (:decision_time, :decision_date, :stage, :mode, :ticker, :direction, :score,
                           :selected, :signal_sources, :reasoning, :rejection_reason, :signal_details, :agent_trace)""",
                rows,
            )

    def get_recent_decision_logs(self, limit: int = 300) -> list[dict]:
        """Get recent decision traces."""
        with self.connect() as conn:
            cursor = conn.execute(
                """SELECT * FROM decision_logs
                   ORDER BY decision_time DESC, id DESC
                   LIMIT ?""",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_equity_snapshot(self, snapshot: dict):
        """Insert a daily equity snapshot."""
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO equity_snapshots
                   (date, cash, positions_value, total_equity, daily_pnl,
                    drawdown_pct, open_positions)
                   VALUES (:date, :cash, :positions_value, :total_equity,
                           :daily_pnl, :drawdown_pct, :open_positions)""",
                snapshot,
            )

    def get_equity_curve(self) -> list[dict]:
        """Get the full equity curve."""
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM equity_snapshots ORDER BY date"
            )
            return [dict(row) for row in cursor.fetchall()]

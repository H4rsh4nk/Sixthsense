"""Trade logging — journal every trade with full context."""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path

from src.config import ROOT_DIR
from src.database import Database

logger = logging.getLogger(__name__)


class TradeLogger:
    """Exports trade history to CSV and provides summary statistics."""

    def __init__(self, db: Database):
        self.db = db
        self.export_dir = ROOT_DIR / "data" / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_trades_csv(self, output_path: Path | None = None) -> Path:
        """Export all closed trades to CSV."""
        output_path = output_path or self.export_dir / "trades.csv"

        with self.db.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM trades WHERE status = 'closed' ORDER BY entry_date"
            )
            trades = [dict(row) for row in cursor.fetchall()]

        if not trades:
            logger.info("No closed trades to export.")
            return output_path

        fieldnames = trades[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"Exported {len(trades)} trades to {output_path}")
        return output_path

    def export_equity_csv(self, output_path: Path | None = None) -> Path:
        """Export equity curve to CSV."""
        output_path = output_path or self.export_dir / "equity_curve.csv"
        curve = self.db.get_equity_curve()

        if not curve:
            return output_path

        fieldnames = curve[0].keys()
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(curve)

        logger.info(f"Exported equity curve to {output_path}")
        return output_path

    def get_daily_summary(self, for_date: date | None = None) -> dict:
        """Get trade summary for a specific day."""
        date_str = (for_date or date.today()).isoformat()

        with self.db.connect() as conn:
            # Today's entries
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE entry_date = ?", (date_str,)
            )
            entries = cursor.fetchone()["cnt"]

            # Today's exits
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl), 0) as total_pnl "
                "FROM trades WHERE exit_date = ? AND status = 'closed'",
                (date_str,),
            )
            row = cursor.fetchone()
            exits = row["cnt"]
            daily_pnl = row["total_pnl"]

            # Open positions
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'"
            )
            open_count = cursor.fetchone()["cnt"]

            # Equity snapshot
            cursor = conn.execute(
                "SELECT * FROM equity_snapshots WHERE date = ?", (date_str,)
            )
            equity = cursor.fetchone()

        return {
            "date": date_str,
            "entries_today": entries,
            "exits_today": exits,
            "daily_pnl": daily_pnl,
            "open_positions": open_count,
            "equity": dict(equity) if equity else None,
        }

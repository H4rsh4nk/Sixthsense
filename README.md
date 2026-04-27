# Sixthsense

> Detect the invisible forces that move markets — before they move.

Sixthsense is a multi-signal swing trading engine that fuses **insider activity**, **news sentiment**, **political events**, and **technical indicators** into a single scoring pipeline — then executes trades automatically via Alpaca.

---

## Architecture

```
                          ┌─────────────────────────────────┐
                          │        External Data Sources     │
                          │                                  │
                          │  SEC EDGAR ─── Form 4 filings   │
                          │  Google News / FinViz ─── NLP    │
                          │  Federal Register ─── Policy     │
                          │  Yahoo Finance ─── OHLCV         │
                          └──────────┬──────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             ┌────────────┐  ┌────────────┐  ┌────────────────┐
             │  Insider    │  │   News     │  │  Political     │
             │  Signal     │  │  Signal    │  │  Signal        │
             │             │  │  (FinBERT) │  │                │
             └─────┬──────┘  └─────┬──────┘  └───────┬────────┘
                   │               │                  │
                   │    ┌──────────┘                  │
                   │    │    ┌────────────┐           │
                   │    │    │ Price      │           │
                   │    │    │ Action     │           │
                   │    │    │ Signal     │           │
                   │    │    └─────┬──────┘           │
                   │    │          │                  │
                   ▼    ▼          ▼                  ▼
              ┌──────────────────────────────────────────┐
              │           Signal Scorer                   │
              │   weighted aggregation + ranking          │
              │   (insider 35% / news 25% /               │
              │    political 20% / price 20%)             │
              └────────────────┬─────────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────────┐
              │          Strategy Layer                   │
              │                                          │
              │  ┌──────────────┐  ┌──────────────────┐  │
              │  │ Position     │  │  Risk Manager    │  │
              │  │ Sizer        │  │  - drawdown      │  │
              │  │ - % risk     │  │  - circuit       │  │
              │  │ - hard caps  │  │    breaker       │  │
              │  └──────────────┘  │  - sector limits │  │
              │                    └──────────────────┘  │
              │  ┌──────────────────────────────────────┐│
              │  │ Exit Manager                         ││
              │  │ stop-loss · time-exit · trailing     ││
              │  └──────────────────────────────────────┘│
              └────────────────┬─────────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────────┐
              │         Order Manager                     │
              │   entry lifecycle · exit lifecycle        │
              │   stop order tracking · reconciliation   │
              └────────────────┬─────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            ┌──────────────┐     ┌──────────────┐
            │  Alpaca API  │     │   SQLite DB  │
            │  paper/live  │     │   (WAL mode) │
            └──────────────┘     └──────────────┘
                                        │
                               ┌────────┴────────┐
                               ▼                  ▼
                     ┌──────────────┐    ┌──────────────┐
                     │  Streamlit   │    │  Telegram    │
                     │  Dashboard   │    │  Alerts      │
                     └──────────────┘    └──────────────┘
```

---

## Signal Pipeline

| Signal | Source | Method | Weight |
|--------|--------|--------|--------|
| **Insider** | SEC EDGAR Form 4 | Cluster detection — counts unique insiders buying + total dollar volume. 3+ insiders buying $500k+ = max strength | 35% |
| **News** | Google News RSS + FinViz | **FinBERT** (financial-domain BERT) sentiment analysis on headlines. Requires ≥2 articles agreeing | 25% |
| **Political** | Federal Register API | Classifies executive orders, tariffs, sanctions, trade deals by GICS sector impact using keyword dictionaries | 20% |
| **Price Action** | SQLite OHLCV | RSI (14), MACD crossovers, volume spikes (2x 20-day avg) | 20% |

All signals output a normalized `SignalResult` with strength (-1.0 to +1.0), direction, and confidence. The scorer aggregates them into ranked `TradeCandidate` objects.

---

## Risk Controls

```
┌─────────────────────────────────────────────────────┐
│                    RISK LAYERS                       │
│                                                     │
│  1. Position Sizing                                 │
│     └─ Fixed 2% risk per trade                      │
│     └─ Max 30% of equity in single position         │
│                                                     │
│  2. Portfolio Limits                                 │
│     └─ Max 3 concurrent positions                   │
│     └─ Max 2 positions per sector                   │
│     └─ No duplicate tickers                         │
│                                                     │
│  3. Loss Limits                                     │
│     └─ 5% stop-loss per trade                       │
│     └─ 3% daily loss limit                          │
│     └─ 10% max drawdown → circuit breaker           │
│           └─ Auto-liquidates all positions           │
│           └─ Blocks all new entries                  │
│                                                     │
│  4. Exit Rules                                      │
│     └─ 5-day time exit (trading days only)          │
│     └─ Stop-loss at daily low                       │
│     └─ Optional trailing stop (3% from peak)        │
└─────────────────────────────────────────────────────┘
```

---

## Daily Schedule (US/Eastern)

| Time | Job | Description |
|------|-----|-------------|
| `23:00` | Overnight Refresh | Fetch political events, pre-score news for top 50 tickers |
| `08:00` | Pre-Market Scan | Generate all signals, rank candidates (read-only) |
| `09:35` | Market Open Entry | Re-score and execute entries (5 min after open) |
| Every 30m | Intraday Check | Poll equity, enforce risk limits, process exits |
| `15:55` | Exit Window | Close positions before market close |
| `17:00` | Post-Market Review | Record equity snapshot |

---

## Backtesting

Event-driven simulation that replays each trading day:

1. **Check exits** — stop-loss at daily low, time-based exits
2. **Generate signals** — all four signal types with strict `as_of_date` (no look-ahead)
3. **Score and enter** — position sizing with slippage model
4. **Mark-to-market** — daily equity, drawdown tracking

### Optimization

- **Parameter sweep**: grid search over hold days × stop-loss percentages, ranked by Sharpe ratio
- **Walk-forward validation**: 2-year train → 1-year test windows, prevents overfitting

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.11 |
| NLP | FinBERT via HuggingFace Transformers + PyTorch (CPU) |
| Broker | Alpaca Markets (paper + live) |
| Data | Yahoo Finance, SEC EDGAR, Federal Register, Google News RSS, FinViz |
| Database | SQLite (WAL mode, 7 tables) |
| Scheduling | APScheduler |
| Dashboard | Streamlit + Plotly |
| Alerts | Telegram Bot API |
| Indicators | `ta` library (RSI, MACD, Bollinger, volume) |
| Config | Pydantic v2 + YAML |
| Testing | pytest (29 tests) + ruff + mypy |
| Infra | Docker Compose (trader + dashboard services) |

---

## Quick Start

```bash
# Clone
git clone git@github.com:H4rsh4nk/Sixthsense.git
cd Sixthsense

# Configure
cp config/secrets.yaml.template config/secrets.yaml
# Edit secrets.yaml with your Alpaca API keys

# Download data
python -m src.main download

# Run backtest
python -m src.main backtest

# Start live paper trading
python -m src.main trade

# Launch dashboard
streamlit run src/monitoring/dashboard.py
```

### Docker

```bash
docker compose up -d        # starts trader + dashboard
docker compose logs -f      # follow logs
# Dashboard at http://localhost:8501
```

---

## Project Structure

```
sixthsense/
├── config/
│   ├── settings.yaml           # All tunable parameters
│   └── secrets.yaml.template   # API keys template
├── src/
│   ├── signals/
│   │   ├── base.py             # Signal ABC + SignalResult
│   │   ├── insider.py          # SEC Form 4 cluster detection
│   │   ├── news.py             # FinBERT sentiment analysis
│   │   ├── political.py        # Federal Register event classifier
│   │   └── price_action.py     # RSI / MACD / volume spikes
│   ├── strategy/
│   │   ├── scorer.py           # Weighted signal aggregation
│   │   ├── position_sizer.py   # Fixed-% risk sizing
│   │   ├── risk_manager.py     # Drawdown + circuit breaker
│   │   └── exit_manager.py     # Stop-loss / time / trailing
│   ├── execution/
│   │   ├── broker.py           # Alpaca API wrapper
│   │   └── order_manager.py    # Full order lifecycle
│   ├── backtest/
│   │   ├── engine.py           # Event-driven backtester
│   │   ├── data_loader.py      # S&P 500 + Yahoo Finance
│   │   └── analytics.py        # Charts, sweep, walk-forward
│   ├── monitoring/
│   │   ├── dashboard.py        # Streamlit web UI
│   │   ├── alerts.py           # Telegram notifications
│   │   └── trade_log.py        # CSV export + summaries
│   ├── config.py               # Pydantic config system
│   ├── database.py             # SQLite persistence layer
│   └── main.py                 # CLI entry point + scheduler
├── tests/                      # 29 pytest tests
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── requirements.txt
```

---

## License

Private — not for distribution.

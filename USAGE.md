# Swing Trader — Usage Guide

A news-based swing trading system that detects short-term opportunities from insider filings, news sentiment, political events, and price action.

---

## Prerequisites

- Python 3.10+
- Free Alpaca account (for paper/live trading)

---

## 1. Installation

```bash
cd swing-trader
pip install -r requirements.txt
```

Or with Docker:

```bash
docker-compose build
```

---

## 2. Configuration

### API Keys

```bash
cp config/secrets.yaml.template config/secrets.yaml
```

Edit `config/secrets.yaml` with your keys:

```yaml
alpaca:
  api_key: "PKXXXXXXXXXXXXXXXX"
  secret_key: "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

telegram:
  bot_token: "optional — for alerts"

sec_edgar:
  user_agent: "YourName your.email@example.com"
```

Get Alpaca keys free at [alpaca.markets](https://alpaca.markets).

### Settings

All tunable parameters live in `config/settings.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `trading.capital` | 10000 | Starting capital ($) |
| `trading.risk_per_trade_pct` | 0.02 | Max risk per trade (2%) |
| `trading.max_drawdown_pct` | 0.10 | Circuit breaker threshold (10%) |
| `trading.max_concurrent_positions` | 3 | Max open positions |
| `exit_rules.default_hold_days` | 5 | Days to hold before exiting |
| `exit_rules.stop_loss_pct` | 0.05 | Stop-loss distance (5%) |
| `broker.paper` | true | `true` = paper, `false` = live |
| `signals.insider.enabled` | true | Toggle each signal on/off |

---

## 3. Download Historical Data

Before backtesting, download S&P 500 price history (3 years):

```bash
python3 -m src.main download
```

This fetches ~500 tickers from Yahoo Finance and stores them in `data/swing_trader.db`. Takes roughly 10–15 minutes on first run.

---

## 4. Backtesting

### Single Backtest

Run with default parameters from `settings.yaml`:

```bash
python3 -m src.main backtest
```

Output:
- Summary table printed to terminal (return, Sharpe, drawdown, win rate)
- Equity curve chart saved to `data/backtest_results/`
- Trade distribution chart saved alongside
- Full trade log in JSON

### Parameter Sweep

Find the optimal hold period and stop-loss:

```bash
python3 -m src.main backtest --sweep
```

Tests all combinations of:
- Hold days: 3, 5, 10, 15
- Stop-loss: 3%, 5%, 7%

Results are ranked by Sharpe ratio in a summary table.

### Walk-Forward Validation

Train on 2 years, test on 1 year (out-of-sample):

```bash
python3 -m src.main backtest --walk-forward
```

This is the most realistic test — it finds the best parameters on historical data, then validates them on unseen data. If out-of-sample Sharpe drops significantly, the strategy is overfit.

### Download + Backtest in One Step

```bash
python3 -m src.main backtest --download-data
```

---

## 5. Paper Trading

Start the automated paper trading system:

```bash
python3 -m src.main trade
```

This runs on a schedule (all times US/Eastern):

| Time | Action |
|------|--------|
| 8:00 AM | Scan all signals, rank candidates |
| 9:35 AM | Execute entry orders (5 min after open) |
| Every 30 min | Monitor positions, check stop-losses |
| 3:55 PM | Execute time-based exits |
| 5:00 PM | Daily P&L review, equity snapshot |
| 11:00 PM | Refresh data from SEC, news, political sources |

Press `Ctrl+C` to stop.

### What Happens Automatically

1. System scans 500 stocks for signals every morning
2. Top candidates are scored and ranked
3. Positions are entered with calculated size and stop-loss
4. Stop-losses and time exits are monitored throughout the day
5. Circuit breaker closes everything if drawdown hits 10%

### Switch to Live Trading

After 30–60 days of successful paper trading, change one line in `config/settings.yaml`:

```yaml
broker:
  paper: false  # was: true
```

Everything else stays identical.

---

## 6. Dashboard

Launch the Streamlit monitoring dashboard:

```bash
streamlit run src/monitoring/dashboard.py
```

Open `http://localhost:8501` in your browser.

Pages:
- **Overview** — equity, daily P&L, recent trades
- **Open Positions** — current holdings with entry/stop/target
- **Trade History** — all closed trades with P&L breakdown by signal
- **Equity Curve** — interactive chart with drawdown
- **Signal Pipeline** — recent signals detected across all types
- **Backtest Results** — load and compare saved backtest runs

With Docker:

```bash
docker-compose up dashboard
```

---

## 7. Telegram Alerts

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Add to `config/secrets.yaml`:

```yaml
telegram:
  bot_token: "123456:ABC-DEF..."
```

4. Enable in `config/settings.yaml`:

```yaml
alerts:
  telegram_enabled: true
  telegram_chat_id: "your_chat_id"
```

You'll receive alerts for: new signals, trade entries/exits, daily P&L summary, circuit breaker events, and system errors.

---

## 8. Docker Deployment

Run both the trader and dashboard:

```bash
docker-compose up -d
```

- Trader runs as a background service
- Dashboard available at `http://localhost:8501`
- Data persists in `./data/` volume
- Logs persist in `./logs/`

Check logs:

```bash
docker logs -f swing-trader
```

---

## 9. Running Tests

```bash
python3 -m pytest tests/ -v
```

29 tests covering config, database, signals, position sizing, risk management, exit logic, and the backtest engine.

---

## 10. Project Structure

```
swing-trader/
├── config/
│   ├── settings.yaml          # All tunable parameters
│   └── secrets.yaml           # API keys (gitignored)
├── data/
│   ├── swing_trader.db        # SQLite database (auto-created)
│   └── backtest_results/      # Charts and JSON from backtests
├── src/
│   ├── signals/               # 4 signal generators
│   │   ├── insider.py         # SEC Form 4 insider buying clusters
│   │   ├── news.py            # FinBERT NLP sentiment on news
│   │   ├── political.py       # Federal Register / Congress events
│   │   └── price_action.py    # RSI, MACD, volume spikes
│   ├── strategy/              # Trade decision engine
│   │   ├── scorer.py          # Combine and rank signals
│   │   ├── position_sizer.py  # Risk-based position sizing
│   │   ├── risk_manager.py    # Drawdown tracking, circuit breaker
│   │   └── exit_manager.py    # Time-based + stop-loss exits
│   ├── execution/             # Broker integration
│   │   ├── broker.py          # Alpaca API wrapper
│   │   └── order_manager.py   # Order lifecycle management
│   ├── backtest/              # Historical testing
│   │   ├── engine.py          # Event-driven backtest loop
│   │   ├── analytics.py       # Sharpe, drawdown, charts
│   │   └── data_loader.py     # S&P 500 price data via yfinance
│   ├── monitoring/            # Observability
│   │   ├── dashboard.py       # Streamlit web dashboard
│   │   ├── alerts.py          # Telegram notifications
│   │   └── trade_log.py       # CSV export, daily summaries
│   └── main.py                # CLI entry point + scheduler
├── tests/                     # 29 unit/integration tests
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

---

## 11. Signal Types Explained

### Insider (SEC Form 4)
Detects clusters of insider buying. When multiple executives buy stock with their own money (not planned 10b5-1 trades), it's historically a bullish signal. Academic research shows ~5–8% annual alpha.

### News Sentiment (FinBERT NLP)
Scores news headlines using FinBERT, a financial-domain language model. Aggregates sentiment per ticker. Trades when sentiment is strongly positive or negative with multiple confirming articles.

### Political Events
Monitors the Federal Register for executive orders, tariffs, regulations, and trade deals. Maps events to affected sectors using keyword matching and estimates bullish/bearish impact.

### Price Action (Technical)
Combines RSI (oversold/overbought), MACD crossovers, and volume spikes. Acts as a confirmation signal — strongest when it agrees with a fundamental signal.

---

## 12. Risk Management Rules

| Rule | Value | What It Does |
|------|-------|--------------|
| Risk per trade | 2% | Max $200 loss on any single trade (on $10K) |
| Stop-loss | 5% | Auto-exit if price drops 5% from entry |
| Max drawdown | 10% | Circuit breaker — closes all positions, halts trading |
| Daily loss limit | 3% | No new entries if daily loss exceeds 3% |
| Max positions | 3 | Never more than 3 concurrent trades |
| Max per sector | 2 | Prevents correlated blowups |
| Max single position | 30% | No single stock exceeds 30% of account |

---

## Quick Reference

```bash
# Download data
python3 -m src.main download

# Backtest (single run)
python3 -m src.main backtest

# Backtest (parameter sweep)
python3 -m src.main backtest --sweep

# Backtest (walk-forward validation)
python3 -m src.main backtest --walk-forward

# Paper trade (automated)
python3 -m src.main trade

# Dashboard
streamlit run src/monitoring/dashboard.py

# Tests
python3 -m pytest tests/ -v
```

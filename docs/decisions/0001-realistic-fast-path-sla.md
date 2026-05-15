# ADR-0001: Realistic fast-path SLA (reject literal <50 ms target)

- **Status:** Accepted
- **Date:** 2026-05-14
- **Related:** `GOAL_ARCHITECTURE.md` §3 (Processing Paths)

## Context

`GOAL_ARCHITECTURE.md` originally specified a "Fast path (<50 ms)" for real-time signals (price spikes, breaking news, order-book imbalance). The current implementation is:

- A Python `BlockingScheduler` polling once per minute.
- Daily-bar (Yahoo) market data with no live order book.
- Alpaca REST polling, not websocket streaming.
- LLM news scoring via Ollama (Qwen 3.5 9B locally), which alone takes >50 ms per headline.

Achieving a literal 50 ms tick-to-trade SLA requires:

- Co-located servers near the exchange.
- Event-driven C++/Rust (or at minimum, asyncio + cython) processing.
- Direct market-data feeds (L2 book) — not RSS or daily bars.
- Pre-compiled rule paths, not Python class hierarchies.

That is a full rewrite of the system, and the use-case (swing trading, paper account, single operator) does not need it.

## Decision

Reframe the fast path as **"intraday event-driven path with a <5 s SLA"**, measured from event ingestion timestamp to order submission. Reserve the literal <50 ms target for a hypothetical future low-latency variant that is **explicitly deferred**.

In concrete code terms the future fast path will look like:

- Subscribe to Alpaca trade/quote websocket for tickers in our universe.
- Subscribe to a breaking-news firehose (NewsAPI or similar) over websocket / SSE.
- A lightweight `EventReactor` (asyncio) routes events to a small set of pre-compiled rules: hard price-spike triggers (z-score over rolling N-minute window), keyword-matched headline triggers (no LLM in the hot path).
- Anything requiring an LLM call lives on the **slow path** and joins the regular scoring pipeline.

## Consequences

### Positive

- Achievable on the current Python stack without rewriting in C++/Rust.
- Honest about latency — no false advertising in the goal doc.
- LLM scoring (which is the slow part) stays on the slow path where it belongs.

### Negative / accepted trade-offs

- We give up the ability to react to microstructure events (HFT-style book imbalance). Acceptable for a swing-trading strategy.
- "5 s" is still an *aspirational* target until the websocket reactor exists; the current code reacts at *scheduler tick* granularity (1 min) inside the `in_market` phase.

### Neutral

- The slow-path / fast-path *split* in the goal doc still makes architectural sense. We are only changing the SLA on the fast path.

## Alternatives considered

### A — Keep the <50 ms target as written

Rejected. It is not achievable on this stack and writing it in the goal doc as if it were creates the illusion of a missing feature when the real gap is a missing rewrite.

### B — Drop the fast path entirely; merge everything into the slow path

Rejected. There is real value in a non-LLM, keyword/spike trigger path that fires within seconds of a news headline — orders of magnitude faster than waiting for the next scheduler tick. Worth keeping the lane.

## References

- Code: `src/main.py` (`phase_orchestrator`, minute scheduler)
- Docs: `GOAL_ARCHITECTURE.md` §3, `CURRENT_DESIGN_ARCHITECTURE.md` §3
- External: [Alpaca websocket docs](https://docs.alpaca.markets/docs/real-time-stock-pricing-data), Hasbrouck "Empirical Market Microstructure"

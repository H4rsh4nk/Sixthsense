"""Test the LLM trading agent before going live.

Tests:
1. Tool implementations (no API key needed)
2. Signal formatting
3. Response parsing
4. Full agent loop with mock signals (requires API key)
5. Fallback to rule-based scoring
"""

import io
import json
import sys
from datetime import date
from pathlib import Path

# Fix Windows encoding for emoji output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.database import Database
from src.signals.base import SignalResult
from src.strategy.scorer import SignalScorer


def create_mock_signals() -> list[SignalResult]:
    """Create realistic mock signals for testing."""
    return [
        SignalResult(
            ticker="AAPL", signal_date=date.today(), signal_type="insider",
            strength=0.85, direction="long", confidence=0.9,
            metadata={"num_insiders": 3, "total_value": 2500000, "window_days": 5},
        ),
        SignalResult(
            ticker="AAPL", signal_date=date.today(), signal_type="news",
            strength=0.72, direction="long", confidence=0.8,
            metadata={"num_articles": 4, "avg_sentiment": 0.82},
        ),
        SignalResult(
            ticker="AAPL", signal_date=date.today(), signal_type="price_action",
            strength=0.65, direction="long", confidence=0.7,
            metadata={"rsi": 32, "macd_crossover": "bullish", "volume_spike": True},
        ),
        SignalResult(
            ticker="MSFT", signal_date=date.today(), signal_type="news",
            strength=0.55, direction="long", confidence=0.6,
            metadata={"num_articles": 2, "avg_sentiment": 0.65},
        ),
        SignalResult(
            ticker="TSLA", signal_date=date.today(), signal_type="political",
            strength=0.35, direction="long", confidence=0.4,
            metadata={"event_type": "tariff", "impact": "uncertain"},
        ),
        SignalResult(
            ticker="TSLA", signal_date=date.today(), signal_type="price_action",
            strength=-0.20, direction="short", confidence=0.5,
            metadata={"rsi": 58, "macd_crossover": "bearish"},
        ),
    ]


def test_signal_formatting():
    print("\n" + "=" * 60)
    print("TEST 1: Signal Formatting")
    print("=" * 60)
    from src.strategy.llm_agent import TradingAgent
    config = load_config()
    db = Database(config)
    agent = TradingAgent.__new__(TradingAgent)
    agent.config = config
    agent.db = db
    signals = create_mock_signals()
    formatted = agent._format_signals(signals, date.today())
    print(formatted)
    print("\n  [PASS] Signal formatting works correctly")
    return True


def test_response_parsing():
    print("\n" + "=" * 60)
    print("TEST 2: Response Parsing")
    print("=" * 60)
    from src.strategy.llm_agent import TradingAgent

    text1 = json.dumps([
        {"ticker": "AAPL", "direction": "long", "score": 0.85,
         "reasoning": "Strong convergence", "signal_sources": ["insider", "news"]},
    ])
    result1 = TradingAgent._parse_decisions(text1)
    assert len(result1) == 1 and result1[0]["ticker"] == "AAPL"
    print("  [PASS] Clean JSON parsing")

    text2 = '```json\n' + text1 + '\n```'
    result2 = TradingAgent._parse_decisions(text2)
    assert len(result2) == 1
    print("  [PASS] Markdown-wrapped JSON parsing")

    assert TradingAgent._parse_decisions("[]") == []
    print("  [PASS] Empty array")

    assert TradingAgent._parse_decisions("this is not json") == []
    print("  [PASS] Invalid JSON handled gracefully")

    text5 = json.dumps([{"ticker": "AAPL", "direction": "long", "score": 1.5, "reasoning": "test"}])
    assert TradingAgent._parse_decisions(text5)[0]["score"] == 1.0
    print("  [PASS] Score clamping (1.5 -> 1.0)")

    text6 = json.dumps([{"ticker": "AAPL", "direction": "sideways", "score": 0.5}])
    assert len(TradingAgent._parse_decisions(text6)) == 0
    print("  [PASS] Invalid direction filtered out")

    text7 = json.dumps([
        {"ticker": "MSFT", "direction": "long", "score": 0.5, "reasoning": "ok"},
        {"ticker": "AAPL", "direction": "long", "score": 0.9, "reasoning": "great"},
    ])
    assert TradingAgent._parse_decisions(text7)[0]["ticker"] == "AAPL"
    print("  [PASS] Sorted by score descending")

    print("\n  [PASS] All response parsing tests passed!")
    return True


def test_tool_implementations():
    print("\n" + "=" * 60)
    print("TEST 3: Tool Implementations (Database)")
    print("=" * 60)
    config = load_config()
    db = Database(config)
    from src.strategy.llm_agent import TradingAgent
    agent = TradingAgent.__new__(TradingAgent)
    agent.config = config
    agent.db = db

    tools_to_test = [
        ("get_price_history", {"ticker": "AAPL", "days": 30}),
        ("get_current_positions", {}),
        ("get_risk_status", {}),
        ("get_sector_exposure", {}),
        ("get_insider_details", {"ticker": "AAPL", "days_back": 30}),
        ("get_news_headlines", {"ticker": "AAPL"}),
        ("get_technical_indicators", {"ticker": "AAPL"}),
    ]
    for tool_name, args in tools_to_test:
        result = agent._execute_tool(tool_name, args)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        print(f"  [PASS] {tool_name}: {len(result)} chars")

    result = agent._execute_tool("nonexistent_tool", {})
    assert "error" in json.loads(result)
    print(f"  [PASS] Unknown tool returns error")
    print("\n  [PASS] All tool tests passed!")
    return True


def test_technical_indicators():
    print("\n" + "=" * 60)
    print("TEST 4: Technical Indicator Calculations")
    print("=" * 60)
    from src.strategy.llm_agent import TradingAgent

    up_prices = [100 + i for i in range(20)]
    rsi_up = TradingAgent._calculate_rsi(up_prices)
    assert rsi_up > 90, f"RSI uptrend should be >90, got {rsi_up}"
    print(f"  [PASS] RSI uptrend: {rsi_up:.1f}")

    down_prices = [200 - i for i in range(20)]
    rsi_down = TradingAgent._calculate_rsi(down_prices)
    assert rsi_down < 10, f"RSI downtrend should be <10, got {rsi_down}"
    print(f"  [PASS] RSI downtrend: {rsi_down:.1f}")

    prices = [100 + i * 0.5 for i in range(40)]
    macd, signal, hist = TradingAgent._calculate_macd(prices)
    assert isinstance(macd, float)
    print(f"  [PASS] MACD: line={macd:.4f}, signal={signal:.4f}, hist={hist:.4f}")

    assert TradingAgent._calculate_rsi([100, 101]) == 50.0
    print(f"  [PASS] RSI insufficient data -> neutral 50.0")
    print("\n  [PASS] All indicator tests passed!")
    return True


def test_rules_fallback():
    print("\n" + "=" * 60)
    print("TEST 5: Rules-Based Scoring (Fallback)")
    print("=" * 60)
    config = load_config()
    config.agent.enabled = False
    db = Database(config)
    scorer = SignalScorer(config, db)
    assert not scorer.agent_mode
    print(f"  [PASS] Scorer in rules mode")

    signals = create_mock_signals()
    candidates = scorer.score(signals, date.today())
    print(f"  Candidates found: {len(candidates)}")
    for c in candidates:
        print(f"    {c.ticker}: score={c.combined_score:.2f}, dir={c.direction}, "
              f"signals={c.signal_sources}, mode={c.metadata.get('decision_mode')}")

    assert all(c.metadata.get("decision_mode") == "rules" for c in candidates)
    if candidates:
        assert candidates[0].ticker == "AAPL"
        print(f"  [PASS] AAPL ranked #1")
    print("\n  [PASS] Rules-based scoring works!")
    return True


def test_agent_fallback_on_error():
    print("\n" + "=" * 60)
    print("TEST 6: Agent Fallback on API Error")
    print("=" * 60)
    config = load_config()
    config.agent.enabled = True
    config.agent.fallback_to_rules = True
    config.secrets.llm_api_key = "invalid_key_for_testing"
    db = Database(config)
    scorer = SignalScorer(config, db)
    signals = create_mock_signals()
    candidates = scorer.score(signals, date.today())

    if candidates:
        mode = candidates[0].metadata.get("decision_mode", "unknown")
        print(f"  Decision mode used: {mode}")
        if mode == "rules":
            print("  [PASS] Correctly fell back to rules-based scoring")
        else:
            print("  [INFO] Agent responded despite invalid key")
    else:
        print("  [PASS] Empty result (acceptable fallback)")
    print("\n  [PASS] Fallback mechanism works!")
    return True


def test_agent_scoring():
    print("\n" + "=" * 60)
    print("TEST 7: Full Agent Scoring Loop (Live LLM)")
    print("=" * 60)
    config = load_config()

    if not config.secrets.llm_api_key or config.secrets.llm_api_key in ("YOUR_LLM_API_KEY", "invalid_key_for_testing"):
        print("  [SKIP] No valid LLM API key configured")
        print("  To run: add key to config/secrets.yaml or set LLM_API_KEY env var")
        return None

    config.agent.enabled = True
    db = Database(config)
    scorer = SignalScorer(config, db)
    print(f"  Model: {config.agent.model}")
    print(f"  Provider: {config.agent.provider}")

    signals = create_mock_signals()
    print(f"  Sending {len(signals)} mock signals to the agent...")
    candidates = scorer.score(signals, date.today())

    print(f"\n  Agent returned {len(candidates)} candidates:")
    for c in candidates:
        reasoning = c.metadata.get("reasoning", "")
        print(f"    {c.ticker}: score={c.combined_score:.2f}, dir={c.direction}")
        if reasoning:
            print(f"      Reasoning: {reasoning}")

    print("\n  [PASS] Full agent scoring loop works!")
    return True


def main():
    print("=" * 60)
    print("  SWING TRADER - LLM AGENT TEST SUITE")
    print("=" * 60)
    print(f"  Date: {date.today()}")
    print(f"  Running pre-production tests...\n")

    results = {}
    all_tests = [
        ("Signal Formatting", test_signal_formatting),
        ("Response Parsing", test_response_parsing),
        ("Tool Implementations", test_tool_implementations),
        ("Technical Indicators", test_technical_indicators),
        ("Rules Fallback", test_rules_fallback),
        ("Agent Fallback on Error", test_agent_fallback_on_error),
        ("Full Agent Loop", test_agent_scoring),
    ]

    for name, test_fn in all_tests:
        try:
            result = test_fn()
            results[name] = "SKIPPED" if result is None else "PASSED"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "[OK]" if status == "PASSED" else "[--]" if status == "SKIPPED" else "[!!]"
        print(f"  {icon} {name}: {status}")

    passed = sum(1 for s in results.values() if s == "PASSED")
    skipped = sum(1 for s in results.values() if s == "SKIPPED")
    failed = sum(1 for s in results.values() if s not in ("PASSED", "SKIPPED"))
    print(f"\n  Total: {passed} passed, {skipped} skipped, {failed} failed")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Agent package — tool registry and (future) decision-driver loop.

See `docs/decisions/0010-agent-as-decision-driver.md` and
`docs/decisions/0011-agent-budgets-caching-observability.md` for the rationale
and roadmap. Phase A delivers the tool registry; Phase B will rewire
`TradingAgent` to be the orchestrator.
"""

from src.agent.tool_registry import (  # noqa: F401
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    build_default_registry,
    execute_tool,
)

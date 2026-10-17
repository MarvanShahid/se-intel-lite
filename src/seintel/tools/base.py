from __future__ import annotations

from typing import Callable

from ..agent.state import ToolResult

TOOL_SPECS: dict[str, dict] = {}


def tool_spec(name: str, description: str, args: dict):
    """Register a tool description (shown in the agent prompt)."""
    def deco(fn: Callable):
        TOOL_SPECS[name] = {"name": name, "description": description, "args": args}
        fn.tool_name = name
        return fn
    return deco


def err(msg: str) -> ToolResult:
    return ToolResult(ok=False, text=f"ERROR: {msg}")

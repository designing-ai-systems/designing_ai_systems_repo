"""
Map Tool Service definitions to model-provider tool JSON (e.g. OpenAI Chat tools).

Registry names may contain dots; many providers expect function names matching
^[a-zA-Z0-9_-]+$. We derive a stable LLM-safe name and record the mapping so
callers can route tool_calls back to platform.tools.execute(tool_name=...).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from services.tools.models import ToolDefinition

_LLM_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def platform_tool_name_to_llm_function_name(platform_name: str) -> str:
    """Turn a Tool Service tool name into a provider-safe function identifier."""
    s = platform_name.replace(".", "_")
    s = _LLM_UNSAFE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tool"


def tool_definitions_to_model_tools(
    definitions: List[ToolDefinition],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build OpenAI-style tool specs and an LLM-name -> platform-name map.

    Raises:
        ValueError: if two definitions collapse to the same LLM function name.
    """
    llm_to_platform: Dict[str, str] = {}
    model_tools: List[Dict[str, Any]] = []
    for t in definitions:
        llm_name = platform_tool_name_to_llm_function_name(t.name)
        existing = llm_to_platform.get(llm_name)
        if existing is not None and existing != t.name:
            raise ValueError(
                f"LLM function name collision {llm_name!r}: tools {existing!r} and {t.name!r}"
            )
        llm_to_platform[llm_name] = t.name
        params = t.parameters if t.parameters is not None else {"type": "object", "properties": {}}
        model_tools.append(
            {
                "type": "function",
                "function": {
                    "name": llm_name,
                    "description": t.description or f"Platform tool {t.name}",
                    "parameters": params,
                },
            }
        )
    return model_tools, llm_to_platform

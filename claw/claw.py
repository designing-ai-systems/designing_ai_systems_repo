"""
Claw: a personal AI assistant built on the GenAI platform (Chapter 9).

Implements the *final* assistant from Listing 9.16. Each step in the body
of :func:`claw_assistant` corresponds directly to a chapter listing:

  - Listing 9.4   long-term memory tool (``claw.memory.save_fact``)
  - Listing 9.7   agentic RAG tool (``claw.knowledge.search``)
  - Listing 9.9   action tools (calendar, tickets, MCP web search)
  - Listing 9.11  three-layer guardrails (input / tool args / output)
  - Listing 9.13  tool-argument validation inside the agent loop
  - Listing 9.14  versioned system prompt
  - Listing 9.15  context engineering with an explicit token budget
  - Listing 9.16  the complete workflow function
  - Listing 9.18  per-turn quality scoring via the Observability Service

The README in this folder explains every deviation from the chapter
listings (places where the chapter's SDK shapes are aspirational and
don't yet exist on the platform).

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from genai_platform import GenAIPlatform, workflow
from services.tools.models import RateLimits, ToolBehavior

CLAW_NAMESPACE = "claw.*"  # fnmatch pattern used by platform.tools.discover
CLAW_KNOWLEDGE_INDEX = os.environ.get("CLAW_KNOWLEDGE_INDEX", "company-knowledge")
CLAW_MODEL = os.environ.get("CLAW_MODEL", "gpt-4o")
MAX_AGENT_ITERATIONS = 5
TOKEN_BUDGET = 120_000
RESPONSE_HEADROOM_TOKENS = 8_000

platform = GenAIPlatform(gateway_url=os.environ.get("GENAI_GATEWAY_URL", "localhost:50051"))

# Listing 9.14: the system prompt is registered with the Model Service so it
# can be versioned and rolled back without redeploying the workflow.
CLAW_SYSTEM_PROMPT = """You are Claw, a personal AI assistant for the team at {organization_name}.

## Your capabilities
You can:
- Answer questions using the company knowledge base
- Schedule meetings and manage calendar events
- Create and manage tickets in the project tracker
- Search the web for current information
- Remember user preferences and project context across sessions

## How to behave
- Be concise and direct. Avoid filler phrases.
- When answering from the knowledge base, cite your sources.
- When uncertain, say so. Never fabricate information.
- When you remember something about the user, use it naturally without
  announcing that you remembered it.
- Ask clarifying questions rather than guessing when the request is
  ambiguous.

## Tool usage guidelines
- Search the knowledge base before answering factual questions about the
  company.
- Only create calendar events or tickets when the user explicitly asks.
  Never proactively create them based on implications.
- When a tool call fails, explain the issue and suggest alternatives.
- Save important facts about users to memory when they share preferences
  or project details.

## Safety boundaries
- Never provide medical, legal, or financial advice.
- Never share one user's personal information with another user.
- When your response includes information from the knowledge base,
  clearly attribute it to the source document.
- If you cannot help with a request, explain why and suggest who to
  contact instead.
"""


def register_claw_assets() -> None:
    """Idempotent setup: prompt + tool registry. Safe to call on every boot."""
    platform.models.register_prompt(
        name="claw-assistant",
        content=CLAW_SYSTEM_PROMPT,
        author="sarah@company.com",
        tags=["production", "claw-assistant"],
    )
    # --- Listing 9.4: long-term memory tool. ---
    platform.tools.register(
        name="claw.memory.save_fact",
        description=(
            "Save an important fact about the user for future reference. Use "
            "this when the user shares preferences, project details, or "
            "personal information they'd expect you to remember."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Short identifier for this fact, e.g. "
                        "'preferred_meeting_time', 'current_project'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The fact itself, e.g. 'morning'.",
                },
            },
            "required": ["key", "value"],
        },
        capabilities=["memory"],
        tags=["claw", "internal"],
        endpoint="internal://claw.memory.save_fact",
    )
    # --- Listing 9.7: agentic RAG tool. ---
    platform.tools.register(
        name="claw.knowledge.search",
        description=(
            "Search the company knowledge base. Use when the user asks about "
            "policies, processes, or organizational information. Rewrite the "
            "user's question into a search-optimized query that includes "
            "relevant context from the conversation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "index": {
                    "type": "string",
                    "enum": [
                        "company-knowledge",
                        "engineering-docs",
                        "hr-policies",
                    ],
                    "description": "Which knowledge index to search.",
                },
            },
            "required": ["query"],
        },
        capabilities=["knowledge", "retrieval"],
        tags=["claw", "internal"],
        endpoint="internal://claw.knowledge.search",
    )
    # --- Listing 9.9: action tools. ---
    platform.tools.register(
        name="claw.calendar.create_event",
        description=(
            "Create a calendar event when the user wants to schedule a meeting or block time."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "format": "date-time"},
                "end_time": {"type": "string", "format": "date-time"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email addresses of attendees.",
                },
            },
            "required": ["title", "start_time", "end_time"],
        },
        endpoint=os.environ.get("CLAW_CALENDAR_URL", "https://calendar-api.internal/v1/events"),
        credential_ref="calendar-service-account",
        behavior=ToolBehavior(is_read_only=False, is_idempotent=False),
        rate_limits=RateLimits(requests_per_session=5),
        capabilities=["scheduling"],
        tags=["claw", "external"],
    )
    platform.tools.register(
        name="claw.tickets.create",
        description="Create a ticket in the project tracker.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
            },
            "required": ["title"],
        },
        endpoint=os.environ.get("CLAW_TICKETS_URL", "https://tickets-api.internal/v1/tickets"),
        credential_ref="tickets-api-key",
        behavior=ToolBehavior(is_read_only=False, is_idempotent=False),
        capabilities=["task-management"],
        tags=["claw", "external"],
    )
    # MCP web search (Listing 9.9). Only registered when the operator has
    # actually pointed Claw at an MCP server, since registration calls the
    # remote server during import.
    mcp_url = os.environ.get("CLAW_MCP_WEB_URL")
    if mcp_url:
        platform.tools.register_mcp_server(server_url=mcp_url, namespace="claw.web")


# Listing 9.11 / 9.13: a small in-app stand-in for the chapter's
# `claw-action-limits` policy. The platform's Guardrails Service doesn't
# expose a runtime policy-registration RPC today; the README documents this.
_action_counters: Dict[str, int] = {}


def _check_tool_arguments(
    session_id: str, tool_name: str, arguments: Dict[str, Any]
) -> Tuple[bool, str]:
    """Listing 9.13: validate a proposed tool call before execution."""
    if tool_name == "claw.calendar.create_event":
        raw_start = arguments.get("start_time", "")
        try:
            start = datetime.fromisoformat(str(raw_start).replace("Z", "+00:00"))
        except ValueError:
            return False, "start_time must be a valid ISO-8601 timestamp"
        if start <= datetime.now(timezone.utc):
            return False, "start_time must be in the future"
        attendees = arguments.get("attendees") or []
        if len(attendees) > 20:
            return False, "attendees must contain 20 or fewer entries"

    if tool_name == "claw.tickets.create":
        key = f"{session_id}:claw.tickets.create"
        if _action_counters.get(key, 0) >= 10:
            return False, "maximum 10 ticket creations per session"
        _action_counters[key] = _action_counters.get(key, 0) + 1

    return True, ""


def _estimate_tokens(text: Any) -> int:
    """Cheap 4-chars-per-token estimate; good enough for budgeting."""
    return max(1, len(str(text or "")) // 4)


def _format_docs_within_budget(docs, max_tokens: int) -> str:
    parts: List[str] = []
    used = 0
    for doc in docs:
        chunk = f"[Source: {doc.document_id}]\n{doc.text}"
        chunk_tokens = _estimate_tokens(chunk)
        if used + chunk_tokens > max_tokens:
            break
        parts.append(chunk)
        used += chunk_tokens
    return "\n\n".join(parts)


def _msg_to_dict(msg) -> Dict[str, Any]:
    """Flatten a Session Service Message dataclass into a chat-API dict."""
    out: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        out["tool_call_id"] = msg.tool_call_id
    return out


def _trim_history_to_budget(history, max_tokens: int) -> List[Dict[str, Any]]:
    """Listing 9.15: keep the most recent messages that fit the budget."""
    out: List[Dict[str, Any]] = []
    used = 0
    for msg in reversed(history):
        msg_tokens = _estimate_tokens(msg.content) + 4
        if used + msg_tokens > max_tokens:
            break
        out.append(_msg_to_dict(msg))
        used += msg_tokens
    out.reverse()
    return out


def assemble_context(
    user_id: str,
    session_id: str,
    message: str,
    *,
    token_budget: int = TOKEN_BUDGET,
) -> Tuple[List[Dict[str, Any]], str]:
    """Listing 9.15: assemble the full messages list within a token budget."""
    used = 0

    system_prompt = platform.models.get_prompt("claw-assistant")
    used += _estimate_tokens(system_prompt["content"])

    used += _estimate_tokens(message)

    user_memory = platform.sessions.get_memory(user_id=user_id)
    memory_text = "\n".join(f"- {key}: {value}" for key, value in user_memory.items())
    used += _estimate_tokens(memory_text)

    remaining_for_docs = max(0, token_budget - used - RESPONSE_HEADROOM_TOKENS)
    docs_budget = int(remaining_for_docs * 0.4)
    relevant_docs = platform.data.search(
        index_name=CLAW_KNOWLEDGE_INDEX,
        query=message,
        top_k=5,
        score_threshold=0.7,
    )
    doc_text = _format_docs_within_budget(relevant_docs, docs_budget)
    used += _estimate_tokens(doc_text)

    history_budget = max(0, token_budget - used - RESPONSE_HEADROOM_TOKENS)
    raw_history, _ = platform.sessions.get_messages(session_id=session_id, limit=50)
    trimmed_history = _trim_history_to_budget(raw_history, history_budget)

    messages = [
        {"role": "system", "content": system_prompt["content"]},
        {
            "role": "system",
            "content": f"User memory:\n{memory_text}\n\nRetrieved documents:\n{doc_text}",
        },
        *trimmed_history,
        {"role": "user", "content": message},
    ]
    return messages, doc_text


def _execute_tool(session_id: str, user_id: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Dispatch a tool call. Internal tools run locally; MCP tools go through
    ``platform.tools.execute``; the calendar/ticket tools return a deterministic
    mock so the demo is self-contained (production registers real HTTP
    endpoints; the README documents the swap).
    """
    if tool_name == "claw.memory.save_fact":
        platform.sessions.save_memory(
            user_id=user_id,
            key=arguments["key"],
            value=arguments["value"],
        )
        return {"status": "saved", "key": arguments["key"]}

    if tool_name == "claw.knowledge.search":
        results = platform.data.search(
            index_name=arguments.get("index") or CLAW_KNOWLEDGE_INDEX,
            query=arguments["query"],
            top_k=5,
            score_threshold=0.6,
        )
        return [{"text": r.text, "source": r.document_id, "score": r.score} for r in results]

    if tool_name.startswith("claw.web."):
        execution = platform.tools.execute(
            tool_name=tool_name, arguments=arguments, session_id=session_id
        )
        return execution.result or {"error": execution.error}

    if tool_name == "claw.calendar.create_event":
        return {
            "event_id": f"evt-{uuid.uuid4().hex[:8]}",
            "status": "scheduled",
            **arguments,
        }
    if tool_name == "claw.tickets.create":
        return {
            "ticket_id": f"tkt-{uuid.uuid4().hex[:8]}",
            "status": "open",
            **arguments,
        }

    return {"error": f"unknown tool {tool_name!r}"}


@workflow(
    name="claw-assistant",
    api_path="/claw/chat",
    response_mode="sync",
    min_replicas=2,
    max_replicas=10,
    target_cpu_percent=70,
    cpu="1000m",
    memory="2Gi",
    timeout_seconds=30,
    max_retries=3,
)
def claw_assistant(message: str, user_id: str, session_id: str = "") -> dict:
    """Listing 9.16: the complete Claw workflow."""
    # --- Step 1: input validation (Listing 9.11). ---
    input_check = platform.guardrails.validate_input(
        content=message, checks=["prompt_injection", "pii_detection"]
    )
    if not input_check["allowed"]:
        return {
            "response": (
                "I can help with scheduling, tasks, and finding information. "
                "I'm not able to process that request."
            ),
            "denial_reason": input_check["denial_reason"],
            "triggered_checks": input_check["triggered_checks"],
        }

    # --- Step 2: load session. ---
    session = platform.sessions.get_or_create(user_id=user_id, session_id=session_id or None)

    # --- Step 3: assemble context (Listing 9.15). ---
    messages, doc_text = assemble_context(
        user_id=user_id, session_id=session.session_id, message=message
    )

    # --- Step 4: agent loop (Listing 9.10), wrapped in a trace span. ---
    model_tools, llm_to_platform = platform.tools.build_model_tools(namespace=CLAW_NAMESPACE)
    final_response: str = ""
    used_knowledge = bool(doc_text)
    with platform.observability.trace_operation(
        "claw.agent_loop",
        workflow_id="claw-assistant",
        user_id=user_id,
        session_id=session.session_id,
    ) as trace_ctx:
        for _ in range(MAX_AGENT_ITERATIONS):
            response = platform.models.chat(model=CLAW_MODEL, messages=messages, tools=model_tools)
            if not response.tool_calls:
                final_response = response.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
            )
            for tool_call in response.tool_calls:
                llm_name = tool_call["function"]["name"]
                platform_name = llm_to_platform.get(llm_name, llm_name)
                try:
                    raw_args = tool_call["function"].get("arguments") or "{}"
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                # --- Step 4a: tool argument validation (Listing 9.13). ---
                allowed, reason = _check_tool_arguments(session.session_id, platform_name, args)
                if not allowed:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Action blocked: {reason}",
                        }
                    )
                    continue
                # --- Step 4b: execute the tool. ---
                try:
                    result = _execute_tool(session.session_id, user_id, platform_name, args)
                except Exception as exc:  # noqa: BLE001 — surface to the model as a tool result
                    result = {"error": str(exc)}
                if platform_name == "claw.knowledge.search":
                    used_knowledge = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, default=str),
                    }
                )
        else:
            final_response = response.content or ""

        # --- Step 7: per-turn quality scoring (Listing 9.18, simplified). ---
        platform.observability.record_score(
            trace_id=trace_ctx.trace_id,
            name="retrieval-usage",
            value=1.0 if used_knowledge else 0.0,
            source="AUTOMATED",
        )

    # --- Step 5: filter output (Listing 9.11). ---
    filtered = platform.guardrails.filter_output(content=final_response, filters=["pii_redaction"])

    # --- Step 6: store the exchange. ---
    platform.sessions.add_messages(
        session_id=session.session_id,
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": filtered["content"]},
        ],
    )

    # Make this turn's trace and score visible immediately. The SDK's buffered
    # observability client otherwise flushes on a background timer.
    platform.observability.flush()

    return {
        "response": filtered["content"],
        "session_id": session.session_id,
        "modified": filtered["modified"],
        "applied_filters": filtered["applied_filters"],
        "trace_id": trace_ctx.trace_id,
    }

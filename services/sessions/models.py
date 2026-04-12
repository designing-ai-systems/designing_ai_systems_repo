"""
Session Service domain models.

Python dataclasses representing sessions, messages, and memory entries.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 4.1: Message dataclass
  - Listing 4.2: Session dataclass
  - Listing 4.19: MemoryEntry dataclass
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Literal, Optional


# Part of Listing 4.7 (proto) and Listing 4.1 (domain representation)
@dataclass
class Function:
    name: str
    arguments: str  # JSON-encoded string


@dataclass
class ToolCall:
    id: str
    type: Literal["function"]
    function: Function


# Listing 4.1
@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Listing 4.2
@dataclass
class Session:
    session_id: str
    user_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


# Listing 4.19
@dataclass
class MemoryEntry:
    key: str
    value: Any
    user_id: str
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

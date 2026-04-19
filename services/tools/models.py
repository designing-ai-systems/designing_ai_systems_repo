"""
Tool Service domain models.

Python dataclasses for tool definitions, execution results,
and operational metadata.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 6.2: ToolDefinition (identity + schema)
  - Listing 6.3: ToolBehavior, RateLimits, CostMetadata
  - Listing 6.15: ToolTask (async execution)
  - Listing 6.17: ExecutionLimits
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Listing 6.3
@dataclass
class ToolBehavior:
    is_read_only: bool = False
    is_idempotent: bool = False
    requires_confirmation: bool = False
    typical_latency_ms: int = 0
    side_effects: List[str] = field(default_factory=list)


# Listing 6.3
@dataclass
class RateLimits:
    requests_per_minute: int = 60
    requests_per_session: int = 0
    daily_limit: int = 0


# Listing 6.3
@dataclass
class CostMetadata:
    estimated_cost_usd: float = 0.0
    billing_category: str = ""


# Listing 6.17
@dataclass
class ExecutionLimits:
    timeout_seconds: int = 30
    memory_limit_mb: int = 256
    cpu_limit_millicores: int = 500
    max_response_size_kb: int = 1024
    max_retries: int = 3


# Listing 6.2
@dataclass
class ToolDefinition:
    name: str = ""
    version: str = "1.0.0"
    owner: str = ""
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None
    returns_schema: Optional[Dict[str, Any]] = None
    behavior: Optional[ToolBehavior] = None
    rate_limits: Optional[RateLimits] = None
    cost: Optional[CostMetadata] = None
    required_permissions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    endpoint: str = ""
    credential_ref: str = ""
    execution_limits: Optional[ExecutionLimits] = None


@dataclass
class ToolExecutionResult:
    tool_name: str = ""
    success: bool = True
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int = 0


# Listing 6.15
@dataclass
class ToolTask:
    id: str = ""
    status: str = "pending"
    tool_name: str = ""
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[ToolExecutionResult] = None

"""
Model Service domain models.

Python dataclasses for chat requests/responses, model capabilities,
provider configuration, and operational configs.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 3.1: Basic chat request/response structure (conceptual)
  - Listing 3.2: Model discovery response with capabilities
  - Listing 3.3: Prompt registration request/response
  - Listing 3.4: Custom model registration request/response
  - Listing 3.11: RetryConfig dataclass
  - Listing 3.12: FallbackConfig dataclass
  - Listing 3.13: RoutingConfig dataclass
  - Listing 3.14: RateLimitConfig dataclass
  - Listing 3.15: CacheConfig and ResponseCache interface
  - Listing 3.16: RequestMetrics dataclass
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --- Core chat types (Listings 3.1-3.4) ---

@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ChatConfig:
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    content: Optional[str] = None
    model: str = ""
    provider: str = ""
    usage: Optional[TokenUsage] = None
    finish_reason: str = "stop"
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatChunk:
    token: str = ""
    model: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


# --- Model discovery (Listing 3.2) ---

@dataclass
class ModelCapability:
    context_window: int = 0
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False


@dataclass
class ModelInfo:
    name: str = ""
    provider: str = ""
    capabilities: ModelCapability = field(default_factory=ModelCapability)


# --- Tool definitions ---

@dataclass
class FunctionDefinition:
    name: str = ""
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    type: str = "function"
    function: Optional[FunctionDefinition] = None


@dataclass
class ResponseFormat:
    type: str = "text"


# --- Operational configs (Listings 3.11-3.16) ---

# Listing 3.11
@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    exponential_backoff: bool = True
    max_delay: float = 60.0
    retry_on: Optional[List[str]] = None


# Listing 3.12
@dataclass
class FallbackConfig:
    enabled: bool = True
    providers: Optional[List[str]] = None
    retry_config: Optional[RetryConfig] = None
    fail_on: Optional[List[str]] = None


# Listing 3.13
@dataclass
class RoutingConfig:
    strategy: str = "priority"
    models: List[str] = field(default_factory=list)
    weights: Optional[List[float]] = None


# Listing 3.14
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10


# Listing 3.15
@dataclass
class CacheConfig:
    enabled: bool = False
    ttl_seconds: int = 3600
    max_entries: int = 1000


# Listing 3.16
@dataclass
class RequestMetrics:
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    cached: bool = False
    retries: int = 0
    tokens_used: Optional[TokenUsage] = None

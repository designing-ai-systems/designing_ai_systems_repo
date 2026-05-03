"""
GenAIPlatform - Main SDK entry point

Provides access to all platform services through a unified interface.
"""

import os
from typing import Optional

from .clients.data import DataClient
from .clients.experiments import ExperimentsClient
from .clients.guardrails import GuardrailsClient
from .clients.models import ModelClient
from .clients.observability import ObservabilityClient
from .clients.sessions import SessionClient
from .clients.tools import ToolClient
from .clients.workflow import WorkflowClient


class GenAIPlatform:
    """
    Main platform SDK entry point.

    Provides lazy-initialized access to all platform services.
    Services are initialized on first access to avoid unnecessary
    network connections.

    Example:
        platform = GenAIPlatform()
        session = platform.sessions.get_or_create(user_id="user-123")
        response = platform.models.chat(model="gpt-4o", query="Hello")
    """

    def __init__(self, gateway_url: Optional[str] = None):
        """
        Initialize the platform SDK.

        Args:
            gateway_url: Optional explicit gateway URL. If not provided,
                       checks GENAI_GATEWAY_URL environment variable.
                       If neither exists, defaults to "localhost:50051"
        """
        if gateway_url:
            self.gateway_url = gateway_url
        else:
            self.gateway_url = os.getenv("GENAI_GATEWAY_URL", "localhost:50051")

        # Lazy initialization - clients created on first access
        self._sessions = None
        self._models = None
        self._data = None
        self._guardrails = None
        self._tools = None
        self._observability = None
        self._experiments = None
        self._workflows = None

    @property
    def sessions(self) -> SessionClient:
        """Access the Session Service client."""
        if self._sessions is None:
            self._sessions = SessionClient(self)
        return self._sessions

    @property
    def models(self) -> ModelClient:
        """Access the Model Service client."""
        if self._models is None:
            self._models = ModelClient(self)
        return self._models

    @property
    def data(self) -> DataClient:
        """Access the Data Service client."""
        if self._data is None:
            self._data = DataClient(self)
        return self._data

    @property
    def guardrails(self) -> GuardrailsClient:
        """Access the Guardrails Service client."""
        if self._guardrails is None:
            self._guardrails = GuardrailsClient(self)
        return self._guardrails

    @property
    def tools(self) -> ToolClient:
        """Access the Tool Service client."""
        if self._tools is None:
            self._tools = ToolClient(self)
        return self._tools

    @property
    def observability(self) -> ObservabilityClient:
        """Access the Observability Service client (Chapter 7)."""
        if self._observability is None:
            self._observability = ObservabilityClient(self)
        return self._observability

    @property
    def experiments(self) -> ExperimentsClient:
        """Access the Experimentation Service client (Chapter 7)."""
        if self._experiments is None:
            self._experiments = ExperimentsClient(self)
        return self._experiments

    @property
    def workflows(self) -> WorkflowClient:
        """Access the Workflow Service client (Listing 8.10–8.11)."""
        if self._workflows is None:
            self._workflows = WorkflowClient(self)
        return self._workflows

"""
Guardrails Service client.

Enforces safety policies and compliance checks.
"""

from typing import List, Optional

from .base import BaseClient


class ValidationResult:
    """Result from guardrails validation."""

    def __init__(self, allowed: bool, reason: Optional[str] = None):
        self.allowed = allowed
        self.reason = reason


class GuardrailsClient(BaseClient):
    """
    Client for Guardrails Service.

    Validates inputs and outputs against safety policies.
    """

    def __init__(self, platform):
        super().__init__(platform, "guardrails")

    def validate_input(self, content: str, policies: List[str]) -> ValidationResult:
        """
        Validate input content against safety policies.

        Args:
            content: Content to validate
            policies: List of policy names to apply

        Returns:
            ValidationResult indicating if content is allowed
        """
        # TODO: Implement in Chapter 6
        return ValidationResult(allowed=True)

    def validate_output(self, content: str, policies: List[str]) -> ValidationResult:
        """
        Validate output content against safety policies.

        Args:
            content: Content to validate
            policies: List of policy names to apply

        Returns:
            ValidationResult indicating if content is allowed
        """
        # TODO: Implement in Chapter 6
        return ValidationResult(allowed=True)

    def policy(self, name: str):
        """
        Decorator for registering custom safety policies.

        Args:
            name: Policy identifier

        Example:
            @platform.guardrails.policy(name="no_medical_advice")
            def check_medical_advice(query):
                # Policy implementation
                return True
        """

        # TODO: Implement in Chapter 6
        def decorator(func):
            return func

        return decorator

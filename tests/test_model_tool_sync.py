"""Tests for Tool Service -> model tool JSON sync."""

from unittest.mock import MagicMock

import pytest

from genai_platform.clients.tools import ToolClient
from proto import tools_pb2
from services.tools.models import ToolDefinition
from services.tools.model_sync import (
    platform_tool_name_to_llm_function_name,
    tool_definitions_to_model_tools,
)


class TestPlatformToolNameToLlmFunctionName:
    def test_replaces_dots(self) -> None:
        assert (
            platform_tool_name_to_llm_function_name("healthcare.scheduling.check_availability")
            == "healthcare_scheduling_check_availability"
        )

    def test_strips_unsafe_chars(self) -> None:
        assert platform_tool_name_to_llm_function_name("foo@bar#baz") == "foo_bar_baz"


class TestToolDefinitionsToModelTools:
    def test_builds_openai_style_tools_and_map(self) -> None:
        defs = [
            ToolDefinition(
                name="healthcare.scheduling.check_availability",
                version="1.0.0",
                description="Check slots.",
                parameters={
                    "type": "object",
                    "required": ["provider_id"],
                    "properties": {"provider_id": {"type": "string"}},
                },
            )
        ]
        tools, m = tool_definitions_to_model_tools(defs)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        fn = tools[0]["function"]
        assert fn["name"] == "healthcare_scheduling_check_availability"
        assert fn["description"] == "Check slots."
        assert fn["parameters"]["required"] == ["provider_id"]
        assert m["healthcare_scheduling_check_availability"] == "healthcare.scheduling.check_availability"

    def test_default_parameters_when_missing(self) -> None:
        defs = [ToolDefinition(name="a.b", description="d")]
        tools, _ = tool_definitions_to_model_tools(defs)
        assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_collision_raises(self) -> None:
        defs = [
            ToolDefinition(name="a.b", description="x"),
            ToolDefinition(name="a_b", description="y"),
        ]
        with pytest.raises(ValueError, match="collision"):
            tool_definitions_to_model_tools(defs)


class TestToolClientBuildModelTools:
    def test_filters_by_names(self) -> None:
        platform = MagicMock()
        platform.gateway_url = "localhost:50051"
        client = ToolClient(platform)
        client._stub = MagicMock()
        client._stub.DiscoverTools.return_value = tools_pb2.DiscoverToolsResponse(
            tools=[
                tools_pb2.ToolDefinition(
                    name="one.two",
                    version="1.0.0",
                    description="A",
                    parameters_json='{"type": "object", "properties": {}}',
                ),
                tools_pb2.ToolDefinition(
                    name="other.tool",
                    version="1.0.0",
                    description="B",
                    parameters_json="",
                ),
            ]
        )

        tools, m = client.build_model_tools(names=["other.tool"])

        assert len(tools) == 1
        assert m["other_tool"] == "other.tool"
        assert tools[0]["function"]["name"] == "other_tool"

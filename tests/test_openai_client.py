import pytest

from agent.models import AgentMessage, AgentToolCall
from agent.providers.openai_client import OpenAIClient


class MockMessage:
    content = "The service has a CPU bottleneck."


class MockChoice:
    message = MockMessage()


class MockResponse:
    choices = [MockChoice()]


class MockCompletions:

    async def create(self, **kwargs):
        return MockResponse()


class MockChat:
    completions = MockCompletions()


class MockClient:
    chat = MockChat()


@pytest.mark.asyncio
async def test_openai_client():
    client = OpenAIClient.__new__(OpenAIClient)

    client.client = MockClient()
    client.model = "test-model"

    response = await client.generate(
        [
            AgentMessage(
                role="user",
                content="Investigate the incident.",
            )
        ]
    )

    assert response.answer == (
        "The service has a CPU bottleneck."
    )

    assert response.tool_calls == []

@pytest.mark.asyncio
async def test_openai_client_parses_tool_calls():
    class MockToolCall:
        id = "call_123"

        class function:
            name = "get_service_health"
            arguments = (
                '{"service_name": "image-ranking-service"}'
            )

    class ToolCallMessage:
        content = ""
        tool_calls = [MockToolCall()]

    class ToolCallChoice:
        message = ToolCallMessage()

    class ToolCallResponse:
        choices = [ToolCallChoice()]

    class ToolCallCompletions:

        async def create(self, **kwargs):
            assert kwargs["model"] == "test-model"
            assert kwargs["tools"] == [
                {
                    "type": "function",
                    "function": {
                        "name": "get_service_health",
                    },
                }
            ]

            return ToolCallResponse()

    class ToolCallChat:
        completions = ToolCallCompletions()

    class ToolCallClient:
        chat = ToolCallChat()

    client = OpenAIClient.__new__(OpenAIClient)

    client.client = ToolCallClient()
    client.model = "test-model"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_service_health",
            },
        }
    ]

    response = await client.generate(
        [
            AgentMessage(
                role="user",
                content="Investigate the incident.",
            )
        ],
        tools=tools,
    )

    assert response.answer == ""

    assert response.tool_calls == [
        AgentToolCall(
            tool_name="get_service_health",
            arguments={
                "service_name": "image-ranking-service",
            },
            tool_call_id="call_123",
        )
    ]
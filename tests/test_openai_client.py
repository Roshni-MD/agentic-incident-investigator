import pytest

from agent.models import AgentMessage
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
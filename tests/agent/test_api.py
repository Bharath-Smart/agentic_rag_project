"""
Tests for API orchestration and message construction.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain.messages import AIMessage

from agent import api
from agent.models import ChatRequest


def history_messages(count: int = 10) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(count)
    ]


def test_build_agent_messages_preserves_roles_and_current_message_once():
    context = history_messages()

    messages = api.build_agent_messages(
        "current question",
        context,
    )

    assert messages[:-1] == context
    assert messages[-1] == {"role": "user", "content": "current question"}
    assert sum(message["content"] == "current question" for message in messages) == 1


@pytest.mark.asyncio
async def test_get_conversation_context_restores_chronological_order():
    newest_first = [
        {"role": "assistant", "content": "third"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "first"},
    ]

    get_messages = AsyncMock(return_value=newest_first)
    with patch.object(
        api,
        "get_session_messages",
        new=get_messages,
    ):
        context = await api.get_conversation_context("session-123", max_messages=10)

    get_messages.assert_awaited_once_with("session-123", limit=10)
    assert [message["content"] for message in context] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_execute_agent_passes_role_specific_history():
    agent_result = {"messages": [AIMessage(content="answer", tool_calls=[])]}
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = agent_result
    context = history_messages()

    with patch.object(api, "rag_agent", mock_agent), patch.object(
        api,
        "get_conversation_context",
        new=AsyncMock(return_value=context),
    ):
        response, tools = await api.execute_agent(
            "current question",
            "session-123",
            save_conversation=False,
        )

    assert response == "answer"
    assert tools == []
    messages = mock_agent.ainvoke.await_args.args[0]["messages"]
    assert messages[:-1] == context
    assert messages[-1] == {"role": "user", "content": "current question"}


@pytest.mark.asyncio
async def test_chat_preserves_original_search_type_behavior():
    with patch.object(
        api,
        "get_or_create_session",
        new=AsyncMock(return_value="session-123"),
    ), patch.object(
        api,
        "execute_agent",
        new=AsyncMock(return_value=("answer", [])),
    ) as execute_agent:
        response = await api.chat(
            ChatRequest(message="question", search_type="vector")
        )

    assert response.message == "answer"
    execute_agent.assert_awaited_once_with(
        message="question",
        session_id="session-123",
        user_id=None,
    )


@pytest.mark.asyncio
async def test_chat_stream_passes_role_specific_history():
    captured = {}

    class FakeAgent:
        async def astream(self, payload, **kwargs):
            captured["payload"] = payload
            yield {"type": "updates", "data": {}}

    with patch.object(
        api,
        "get_or_create_session",
        new=AsyncMock(return_value="session-123"),
    ), patch.object(
        api,
        "get_conversation_context",
        new=AsyncMock(return_value=history_messages()),
    ), patch.object(
        api,
        "add_message",
        new=AsyncMock(),
    ), patch.object(api, "rag_agent", FakeAgent()):
        response = await api.chat_stream(
            ChatRequest(message="current question", search_type="vector")
        )
        body = "".join([chunk async for chunk in response.body_iterator])

    messages = captured["payload"]["messages"]
    assert messages[:-1] == history_messages()
    assert messages[-1] == {"role": "user", "content": "current question"}
    assert '"type": "end"' in body

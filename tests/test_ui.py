"""
Tests for the Streamlit streaming client behavior.
"""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def load_ui_functions():
    source_path = Path(__file__).parents[1] / "ui" / "app.py"
    tree = ast.parse(source_path.read_text())
    function_names = {"stream_chat", "run_async"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name in function_names
    ]
    namespace = {"asyncio": asyncio, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source_path, "exec"), namespace)
    return namespace


class FakePlaceholder:
    def __init__(self):
        self.writes = []
        self.errors = []

    def write(self, value):
        self.writes.append(value)

    def error(self, value):
        self.errors.append(value)


class FakeStreamlit:
    def __init__(self):
        self.session_state = SimpleNamespace(session_id=None, messages=[])
        self.placeholder = FakePlaceholder()

    def empty(self):
        return self.placeholder


class FakeResponse:
    def __init__(self, events, status=200):
        self.status = status
        self.content = [f"data: {json.dumps(event)}\n".encode() for event in events]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None


class FakeClientSession:
    def __init__(self, response):
        self.response = response
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def post(self, url, json):
        self.request = {"url": url, "json": json}
        return self.response


class AsyncContent:
    def __init__(self, lines):
        self.lines = lines

    def __aiter__(self):
        self.iterator = iter(self.lines)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


def make_client(response):
    response.content = AsyncContent(response.content)
    client = FakeClientSession(response)
    return SimpleNamespace(ClientSession=lambda: client), client


@pytest.mark.asyncio
async def test_stream_chat_renders_normal_completion_without_final_duplicate_write():
    st = FakeStreamlit()
    response = FakeResponse([
        {"type": "session", "session_id": "session-123"},
        {"type": "text", "content": "Answer"},
        {"type": "tools", "tools": [{"tool_name": "vector_search", "args": {"limit": 5}}]},
        {"type": "end"},
    ])
    aiohttp, client = make_client(response)
    namespace = load_ui_functions()
    namespace.update({"st": st, "aiohttp": aiohttp, "USER_ID": "test-user"})

    await namespace["stream_chat"]("question", "http://api")

    assert st.session_state.session_id == "session-123"
    assert st.placeholder.errors == []
    assert st.placeholder.writes == [
        "Answer",
        "Answer\n [Tool: vector_search] \n Args: {'limit': 5}",
    ]
    assert st.session_state.messages == [{
        "role": "assistant",
        "content": "Answer\n [Tool: vector_search] \n Args: {'limit': 5}",
    }]
    assert client.request["json"]["message"] == "question"


@pytest.mark.asyncio
async def test_stream_chat_displays_sse_error_without_success_message():
    st = FakeStreamlit()
    response = FakeResponse([
        {"type": "session", "session_id": "session-123"},
        {"type": "text", "content": "Partial response"},
        {"type": "error", "content": "The streaming response could not be completed."},
    ])
    aiohttp, _ = make_client(response)
    namespace = load_ui_functions()
    namespace.update({"st": st, "aiohttp": aiohttp, "USER_ID": "test-user"})

    await namespace["stream_chat"]("question", "http://api")

    assert st.placeholder.errors == ["The streaming response could not be completed."]
    assert st.session_state.messages == [{
        "role": "assistant",
        "content": "Error: The streaming response could not be completed.",
    }]


@pytest.mark.asyncio
async def test_stream_chat_handles_incomplete_stream_without_success_message():
    st = FakeStreamlit()
    response = FakeResponse([
        {"type": "session", "session_id": "session-123"},
        {"type": "text", "content": "Partial response"},
    ])
    aiohttp, _ = make_client(response)
    namespace = load_ui_functions()
    namespace.update({"st": st, "aiohttp": aiohttp, "USER_ID": "test-user"})

    await namespace["stream_chat"]("question", "http://api")

    assert st.placeholder.errors == [
        "The streaming response ended before completion."
    ]
    assert st.session_state.messages == [{
        "role": "assistant",
        "content": "Error: The streaming response ended before completion.",
    }]


def test_run_async_uses_asyncio_run():
    namespace = load_ui_functions()

    async def noop():
        return None

    coroutine = noop()
    namespace["stream_chat"] = Mock(return_value=coroutine)
    with patch.object(namespace["asyncio"], "run") as run:
        namespace["run_async"]("question", "http://api")

    run.assert_called_once_with(coroutine)
    coroutine.close()

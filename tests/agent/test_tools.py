"""
Tests for agent-facing tools.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from agent.tools import get_document_tool


DOCUMENT_ID = "ead204d0-04cf-4ba1-b7ae-c186c4881857"
DOCUMENT_CONTENT = "0123456789" * 60


def make_document(document_id: str = DOCUMENT_ID) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "id": document_id,
        "title": "sr_2024_cb_v",
        "source": "sr_2024_cb_v.pdf",
        "content": DOCUMENT_CONTENT,
        "metadata": {},
        "created_at": timestamp,
        "updated_at": timestamp,
    }


@pytest.mark.asyncio
async def test_get_document_tool_accepts_uuid():
    with patch(
        "agent.tools.get_document",
        new=AsyncMock(return_value=make_document()),
    ) as get_document, patch(
        "agent.tools.find_document_ids_by_title",
        new=AsyncMock(),
    ) as find_by_title:
        result = await get_document_tool.ainvoke({"document_id": DOCUMENT_ID})

    get_document.assert_awaited_once_with(DOCUMENT_ID)
    find_by_title.assert_not_awaited()
    assert result.id == DOCUMENT_ID
    assert len(result.content) == 500
    assert result.content == DOCUMENT_CONTENT[:500]
    assert "chunks" not in result.model_dump()


@pytest.mark.asyncio
async def test_get_document_tool_resolves_exact_title():
    with patch(
        "agent.tools.find_document_ids_by_title",
        new=AsyncMock(return_value=[DOCUMENT_ID]),
    ) as find_by_title, patch(
        "agent.tools.get_document",
        new=AsyncMock(return_value=make_document()),
    ) as get_document:
        result = await get_document_tool.ainvoke({"document_id": "sr_2024_cb_v"})

    find_by_title.assert_awaited_once_with("sr_2024_cb_v")
    get_document.assert_awaited_once_with(DOCUMENT_ID)
    assert result.id == DOCUMENT_ID
    assert len(result.content) <= 500
    assert result.content == DOCUMENT_CONTENT[:500]
    assert "chunks" not in result.model_dump()


@pytest.mark.asyncio
async def test_get_document_tool_returns_none_for_unknown_title():
    with patch(
        "agent.tools.find_document_ids_by_title",
        new=AsyncMock(return_value=[]),
    ) as find_by_title, patch(
        "agent.tools.get_document",
        new=AsyncMock(),
    ) as get_document:
        result = await get_document_tool.ainvoke({"document_id": "unknown-title"})

    find_by_title.assert_awaited_once_with("unknown-title")
    get_document.assert_not_awaited()
    assert result is None


@pytest.mark.asyncio
async def test_get_document_tool_returns_none_for_duplicate_title():
    with patch(
        "agent.tools.find_document_ids_by_title",
        new=AsyncMock(return_value=[DOCUMENT_ID, "another-document-id"]),
    ) as find_by_title, patch(
        "agent.tools.get_document",
        new=AsyncMock(),
    ) as get_document:
        result = await get_document_tool.ainvoke({"document_id": "duplicate-title"})

    find_by_title.assert_awaited_once_with("duplicate-title")
    get_document.assert_not_awaited()
    assert result is None

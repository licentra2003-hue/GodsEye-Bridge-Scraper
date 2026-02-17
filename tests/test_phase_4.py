"""Tests for Phase 4: API Endpoints (The Bridge)

Validates:
  - ``POST /api/v1/optimize/start`` — credit check, snapshot creation,
    background task dispatch, and HTTP 202 response.
  - ``GET /api/v1/optimize/status/{snapshot_id}`` — returns correct snapshot
    status from Supabase.

All external services (Supabase, scraper, Gemini) are fully mocked.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://testserver"

VALID_PAYLOAD: Dict[str, Any] = {
    "product_id": "prod-123",
    "user_id": "user-456",
    "batch_id": "batch-789",
    "perplexity_queries": ["best shampoo", "best conditioner"],
    "google_queries": ["best soap"],
    "client_product_json": {"name": "TestProduct"},
    "debug": False,
}


def _mock_verify_credits(result: bool = True):
    """Return an AsyncMock for ``verify_and_deduct_credits``."""

    async def _verify(*args, **kwargs):
        return result

    return _verify


def _mock_create_snapshot(snapshot_id: str = "snap-789"):
    """Return an AsyncMock for ``create_analysis_snapshot``."""

    async def _create(*args, **kwargs):
        return snapshot_id

    return _create


def _mock_get_snapshot(data: Dict[str, Any] | None = None):
    """Return an AsyncMock for ``get_snapshot_status``."""

    async def _get(*args, **kwargs):
        return data or {}

    return _get


def _noop_background(*args, **kwargs):
    """A no-op replacement for ``run_optimization_background``."""
    pass


# Patch targets
_P_VERIFY = "app.main.verify_and_deduct_credits"
_P_CREATE = "app.main.create_analysis_snapshot"
_P_BG = "app.main.run_optimization_background"
_P_STATUS = "app.main.get_snapshot_status"


# ---------------------------------------------------------------------------
# POST /api/v1/optimize/start
# ---------------------------------------------------------------------------

class TestOptimizeStartEndpoint:
    """Tests for the start endpoint."""

    @pytest.mark.asyncio
    async def test_returns_202_with_snapshot_id(self):
        """Happy path: valid payload returns 202 with snapshot_id."""
        with (
            patch(_P_VERIFY, new=_mock_verify_credits(True)),
            patch(_P_CREATE, new=_mock_create_snapshot("snap-789")),
            patch(_P_BG, new=_noop_background),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=VALID_PAYLOAD)

        assert resp.status_code == 202
        body = resp.json()
        assert body["snapshot_id"] == "snap-789"
        assert body["status"] == "running"
        assert body["total_queries"] == 3

    @pytest.mark.asyncio
    async def test_insufficient_credits_returns_402(self):
        """Credits check fails → 402."""
        with (
            patch(_P_VERIFY, new=_mock_verify_credits(False)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=VALID_PAYLOAD)

        assert resp.status_code == 402
        assert "Insufficient credits" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_empty_queries_returns_400(self):
        """No queries at all → 400."""
        payload = {**VALID_PAYLOAD, "perplexity_queries": [], "google_queries": []}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL
        ) as client:
            resp = await client.post("/api/v1/optimize/start", json=payload)

        assert resp.status_code == 400
        assert "At least one query" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_snapshot_creation_failure_returns_500(self):
        """If snapshot creation blows up, return 500."""

        async def _fail(*args, **kwargs):
            raise Exception("DB write failed")

        with (
            patch(_P_VERIFY, new=_mock_verify_credits(True)),
            patch(_P_CREATE, new=_fail),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=VALID_PAYLOAD)

        assert resp.status_code == 500
        assert "Snapshot creation failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_credit_check_exception_returns_500(self):
        """Unexpected exception during credit check → 500."""

        async def _fail(*args, **kwargs):
            raise RuntimeError("Supabase timeout")

        with patch(_P_VERIFY, new=_fail):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=VALID_PAYLOAD)

        assert resp.status_code == 500
        assert "Credit check failed" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_background_task_is_dispatched(self):
        """Verify that ``run_optimization_background`` is actually added to BackgroundTasks."""
        dispatched = {"called": False}

        async def _track_bg(*args, **kwargs):
            dispatched["called"] = True

        with (
            patch(_P_VERIFY, new=_mock_verify_credits(True)),
            patch(_P_CREATE, new=_mock_create_snapshot("snap-bg")),
            patch(_P_BG, new=_track_bg),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=VALID_PAYLOAD)

        assert resp.status_code == 202
        # BackgroundTasks runs after the response — it should have been invoked
        assert dispatched["called"] is True

    @pytest.mark.asyncio
    async def test_response_contains_correct_total(self):
        """Total queries should be sum of perplexity + google queries."""
        payload = {
            **VALID_PAYLOAD,
            "perplexity_queries": ["q1"],
            "google_queries": ["g1", "g2", "g3"],
        }

        with (
            patch(_P_VERIFY, new=_mock_verify_credits(True)),
            patch(_P_CREATE, new=_mock_create_snapshot("snap-count")),
            patch(_P_BG, new=_noop_background),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json=payload)

        assert resp.status_code == 202
        assert resp.json()["total_queries"] == 4


# ---------------------------------------------------------------------------
# GET /api/v1/optimize/status/{snapshot_id}
# ---------------------------------------------------------------------------

class TestOptimizeStatusEndpoint:
    """Tests for the status polling endpoint."""

    @pytest.mark.asyncio
    async def test_returns_running_status(self):
        """Snapshot exists and is running."""
        snapshot_data = {
            "id": "snap-789",
            "status": "running",
            "total_no_of_query": 3,
            "no_of_query": 1,
        }
        with patch(_P_STATUS, new=_mock_get_snapshot(snapshot_data)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/snap-789")

        assert resp.status_code == 200
        body = resp.json()
        assert body["snapshot_id"] == "snap-789"
        assert body["status"] == "running"
        assert body["total_queries"] == 3
        assert body["completed_queries"] == 1

    @pytest.mark.asyncio
    async def test_returns_completed_status(self):
        """Snapshot has finished."""
        snapshot_data = {
            "id": "snap-789",
            "status": "completed",
            "total_no_of_query": 3,
            "no_of_query": 3,
        }
        with patch(_P_STATUS, new=_mock_get_snapshot(snapshot_data)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/snap-789")

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert resp.json()["completed_queries"] == 3

    @pytest.mark.asyncio
    async def test_returns_failed_status(self):
        """Snapshot failed."""
        snapshot_data = {
            "id": "snap-789",
            "status": "failed",
            "total_no_of_query": 3,
            "no_of_query": 0,
        }
        with patch(_P_STATUS, new=_mock_get_snapshot(snapshot_data)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/snap-789")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["completed_queries"] == 0

    @pytest.mark.asyncio
    async def test_returns_partial_status(self):
        """Some tasks failed → status is 'partial'."""
        snapshot_data = {
            "id": "snap-789",
            "status": "partial",
            "total_no_of_query": 3,
            "no_of_query": 2,
        }
        with patch(_P_STATUS, new=_mock_get_snapshot(snapshot_data)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/snap-789")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "partial"
        assert body["completed_queries"] == 2
        assert body["total_queries"] == 3

    @pytest.mark.asyncio
    async def test_snapshot_not_found_returns_404(self):
        """Unknown snapshot_id → 404."""
        with patch(_P_STATUS, new=_mock_get_snapshot(None)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/nonexistent")

        assert resp.status_code == 404
        assert "Snapshot not found" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_status_supabase_error_returns_500(self):
        """If Supabase throws, return 500."""

        async def _fail(*args, **kwargs):
            raise RuntimeError("DB connection lost")

        with patch(_P_STATUS, new=_fail):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/snap-789")

        assert resp.status_code == 500
        assert "Failed to fetch status" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Health endpoint still works (regression check)
# ---------------------------------------------------------------------------

class TestHealthEndpointRegression:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL
        ) as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

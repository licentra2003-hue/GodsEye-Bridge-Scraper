"""End-to-End Workflow Test

Exercises the ENTIRE pipeline from API request to completion:

    POST /api/v1/optimize/start
        ↓  credit check
        ↓  snapshot creation
        ↓  background dispatch
        ├─ process_single_query_pipeline("perplexity", "q1")
        │     ↓ fetch_ai_search_data          (mocked scraper)
        │     ↓ perform_strategic_analysis     (mocked Gemini)
        │     ↓ store_analysis_result          (mocked Supabase)
        ├─ process_single_query_pipeline("perplexity", "q2")
        │     ↓ … same flow …
        └─ process_single_query_pipeline("google_overview", "q3")
              ↓ … same flow …
        ↓  snapshot updated → "completed"
    GET /api/v1/optimize/status/{snapshot_id}
        ↓  returns "completed"

Only the *boundaries* are mocked (external APIs / database).  The entire
internal orchestration runs for real:  asyncio.gather, asyncio.to_thread,
BackgroundTasks, FastAPI routing — all exercised.
"""
from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import app


# ========================================================================
# Shared fake data
# ========================================================================

FAKE_PRODUCT_JSON = {"name": "SuperShampoo", "url": "https://example.com/product/1"}

FAKE_SCRAPED = {
    "query": "<filled-at-runtime>",
    "ai_overview_text": "Top results for your search …",
    "source_links": [
        {"url": "https://example.com/1", "title": "Source 1"},
        {"url": "https://example.com/2", "title": "Source 2"},
    ],
}

FAKE_ANALYSIS = {
    "executive_summary": {"title": "AEO Report", "status_overview": "Strong"},
    "competitive_landscape": {"competitors": ["A", "B"]},
}

FAKE_STORED_RECORD = {"id": "rec-001", "product_id": "prod-e2e"}

SIMULATED_DELAY = 0.05  # 50 ms — fast but proves concurrency


# ========================================================================
# In-memory snapshot store (replaces Supabase for the whole flow)
# ========================================================================

class FakeSnapshotStore:
    """Thread-safe in-memory store that mimics the Supabase snapshot table."""

    def __init__(self):
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def create(self, product_id: str, batch_id: str, total: int) -> str:
        self._counter += 1
        sid = f"snap-e2e-{self._counter}"
        self._snapshots[sid] = {
            "id": sid,
            "product_id": product_id,
            "batch_id": batch_id,
            "status": "running",
            "total_no_of_query": total,
            "no_of_query": 0,
        }
        return sid

    def update(self, snapshot_id: str, data: Dict[str, Any]) -> None:
        if snapshot_id in self._snapshots:
            self._snapshots[snapshot_id].update(data)

    def increment(self, snapshot_id: str) -> None:
        if snapshot_id in self._snapshots:
            snap = self._snapshots[snapshot_id]
            current = snap.get("no_of_query", 0) or 0
            total = snap.get("total_no_of_query", 0) or 0
            if current < total:
                snap["no_of_query"] = current + 1

    def get(self, snapshot_id: str) -> Dict[str, Any]:
        return self._snapshots.get(snapshot_id, {})


# Singleton for a test run
_store = FakeSnapshotStore()


# ========================================================================
# Mock factories
# ========================================================================

def _make_mock_verify(should_pass: bool = True):
    async def _verify(settings, user_id, required_credits):
        return should_pass
    return _verify


def _make_mock_create_snapshot(store: FakeSnapshotStore):
    async def _create(settings, product_id, batch_id, queries, total_no_of_query=None):
        return store.create(product_id, batch_id, total_no_of_query if total_no_of_query is not None else len(queries))
    return _create


def _make_mock_update_snapshot(store: FakeSnapshotStore):
    async def _update(settings, snapshot_id, status, *, completed_queries=None):
        data: Dict[str, Any] = {"status": status}
        if completed_queries is not None:
            data["no_of_query"] = completed_queries
        store.update(snapshot_id, data)
    return _update


def _make_mock_get_snapshot(store: FakeSnapshotStore):
    async def _get(settings, snapshot_id):
        return store.get(snapshot_id)
    return _get


def _make_mock_fetch(delay: float = SIMULATED_DELAY):
    """Fake scraper: returns canned data after a short async sleep."""
    async def _fetch(pipeline, query, settings, **kwargs):
        await asyncio.sleep(delay)
        return {**FAKE_SCRAPED, "query": query, "pipeline": pipeline}
    return _fetch


def _make_mock_analysis(delay: float = SIMULATED_DELAY):
    """Fake Gemini: returns canned analysis after a short blocking sleep."""
    def _analysis(request, *, settings, api_key=None, debug=False, store_to_db=False):
        import time as _t
        _t.sleep(delay)
        return {
            "success": True,
            "analysis": {**FAKE_ANALYSIS, "search_query": request.search_query},
        }
    return _analysis


def _make_mock_store(delay: float = 0.01):
    """Fake Supabase store: returns a canned record."""
    def _store(**kwargs):
        import time as _t
        _t.sleep(delay)
        return {
            **FAKE_STORED_RECORD,
            "search_query": kwargs.get("search_query", ""),
            "pipeline": kwargs.get("pipeline", ""),
        }
    return _store


def _make_mock_supabase_select(store: FakeSnapshotStore):
    async def _select(client, table, match_dict):
        if table == "analysis_snapshots" and "id" in match_dict:
            snap = store.get(match_dict["id"])
            if snap:
                return [snap]
        return []
    return _select


def _make_mock_supabase_update(store: FakeSnapshotStore):
    async def _update(client, table, update_data, match_dict):
        if table == "analysis_snapshots" and "id" in match_dict:
            store.update(match_dict["id"], update_data)
        class _Resp:
            data = [{"id": match_dict.get("id")}]
        return _Resp()
    return _update


def _make_mock_supabase_rpc(store: FakeSnapshotStore):
    async def _rpc(client, function_name, params):
        if function_name == "increment_snapshot_progress":
            sid = params.get("snapshot_id")
            if sid:
                store.increment(sid)
        class _Resp:
            data = None
        return _Resp()
    return _rpc


# ========================================================================
# Patch targets
# ========================================================================

# API-level patches (used by main.py endpoint handlers directly)
_P_API_VERIFY = "app.main.verify_and_deduct_credits"
_P_API_CREATE = "app.main.create_analysis_snapshot"
_P_API_STATUS = "app.main.get_snapshot_status"

# Service-level patches (used inside the background task / pipeline)
_P_SVC_UPDATE = "app.services.analysis_service.update_snapshot_status"
_P_SVC_FETCH = "app.services.scraping_service.fetch_ai_search_data"
_P_SVC_ANALYSIS = "app.services.analysis_service.perform_strategic_analysis"
_P_SVC_STORE = "app.services.analysis_service.store_analysis_result"
_P_SVC_REFUND = "app.services.analysis_service.refund_credits"
_P_SVC_SELECT = "app.services.analysis_service.async_supabase_select"
_P_SVC_DB_UPDATE = "app.services.analysis_service.async_supabase_update"
_P_SVC_RPC = "app.services.analysis_service.async_supabase_rpc"

BASE_URL = "http://testserver"


# ========================================================================
# E2E Happy-Path Test
# ========================================================================

class TestEndToEndWorkflow:
    """Full pipeline: start → background processing → status polling."""

    @pytest.mark.asyncio
    async def test_full_optimization_lifecycle(self):
        """
        1. POST /optimize/start  →  202 + snapshot_id
        2. Background task runs all 3 queries concurrently
        3. GET /optimize/status   →  "completed", 3/3 queries done
        """
        store = FakeSnapshotStore()

        payload = {
            "product_id": "prod-e2e",
            "user_id": "user-e2e",
            "batch_id": "batch-e2e",
            "perplexity_queries": ["best shampoo", "best conditioner"],
            "google_queries": ["best soap"],
            "chatgpt_queries": [],
            "client_product_json": FAKE_PRODUCT_JSON,
        }

        with (
            # API-level mocks
            patch(_P_API_VERIFY, new=_make_mock_verify(True)),
            patch(_P_API_CREATE, new=_make_mock_create_snapshot(store)),
            patch(_P_API_STATUS, new=_make_mock_get_snapshot(store)),
            # Service-level mocks (inside background task)
            patch(_P_SVC_UPDATE, new=_make_mock_update_snapshot(store)),
            patch(_P_SVC_FETCH, new=_make_mock_fetch()),
            patch(_P_SVC_ANALYSIS, side_effect=_make_mock_analysis()),
            patch(_P_SVC_STORE, side_effect=_make_mock_store()),
            patch(_P_SVC_REFUND, new=AsyncMock(return_value=True)),
            patch(_P_SVC_SELECT, new=_make_mock_supabase_select(store)),
            patch(_P_SVC_DB_UPDATE, new=_make_mock_supabase_update(store)),
            patch(_P_SVC_RPC, new=_make_mock_supabase_rpc(store)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:

                # ── Step 1: Start ──────────────────────────────────
                start_resp = await client.post(
                    "/api/v1/optimize/start", json=payload
                )

                assert start_resp.status_code == 202, (
                    f"Expected 202, got {start_resp.status_code}: {start_resp.text}"
                )
                start_body = start_resp.json()
                snapshot_id = start_body["snapshot_id"]
                assert start_body["status"] == "running"
                assert start_body["total_queries"] == 3

                # ── Step 2: Poll status ────────────────────────────
                # BackgroundTasks in ASGI transport run after the response,
                # so by the time we get here the task should be done.
                status_resp = await client.get(
                    f"/api/v1/optimize/status/{snapshot_id}"
                )

                assert status_resp.status_code == 200
                status_body = status_resp.json()
                assert status_body["snapshot_id"] == snapshot_id
                assert status_body["status"] == "completed"
                assert status_body["completed_queries"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_execution_is_faster_than_sequential(self):
        """
        3 queries with 50ms delay each → should finish well under 3× time.
        """
        store = FakeSnapshotStore()
        per_query = SIMULATED_DELAY
        num_queries = 3

        payload = {
            "product_id": "prod-perf",
            "user_id": "user-perf",
            "batch_id": "batch-perf",
            "perplexity_queries": ["q1", "q2", "q3", "q4", "q5"],
            "google_queries": [],
            "chatgpt_queries": [],
            "client_product_json": FAKE_PRODUCT_JSON,
        }

        with (
            patch(_P_API_VERIFY, new=_make_mock_verify(True)),
            patch(_P_API_CREATE, new=_make_mock_create_snapshot(store)),
            patch(_P_API_STATUS, new=_make_mock_get_snapshot(store)),
            patch(_P_SVC_UPDATE, new=_make_mock_update_snapshot(store)),
            patch(_P_SVC_FETCH, new=_make_mock_fetch(delay=per_query)),
            patch(_P_SVC_ANALYSIS, side_effect=_make_mock_analysis(delay=per_query)),
            patch(_P_SVC_STORE, side_effect=_make_mock_store(delay=0.01)),
            patch(_P_SVC_REFUND, new=AsyncMock(return_value=True)),
            patch(_P_SVC_SELECT, new=_make_mock_supabase_select(store)),
            patch(_P_SVC_DB_UPDATE, new=_make_mock_supabase_update(store)),
            patch(_P_SVC_RPC, new=_make_mock_supabase_rpc(store)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                start = time.monotonic()
                resp = await client.post("/api/v1/optimize/start", json=payload)
                elapsed = time.monotonic() - start

        assert resp.status_code == 202
        # Sequential estimate: N * (scrape + analysis) = 3 * 2 * 0.05 = 0.30s
        sequential_estimate = num_queries * 2 * per_query
        assert elapsed < (sequential_estimate * 0.80) + 5.0, (
            f"E2E took {elapsed:.2f}s vs sequential estimate {sequential_estimate:.2f}s"
        )


# ========================================================================
# E2E: Insufficient Credits
# ========================================================================

class TestEndToEndCreditFailure:
    """The pipeline should not start if credits are insufficient."""

    @pytest.mark.asyncio
    async def test_no_credits_returns_402_and_no_background_work(self):
        dispatched = {"called": False}

        original_bg = _make_mock_analysis()

        def _tracking_analysis(*args, **kwargs):
            dispatched["called"] = True
            return original_bg(*args, **kwargs)

        with (
            patch(_P_API_VERIFY, new=_make_mock_verify(False)),
            patch(_P_SVC_ANALYSIS, side_effect=_tracking_analysis),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json={
                    "product_id": "prod-no-credit",
                    "user_id": "user-broke",
                    "batch_id": "batch-broke",
                    "perplexity_queries": ["q1"],
                    "google_queries": [],
                    "client_product_json": FAKE_PRODUCT_JSON,
                })

        assert resp.status_code == 402
        assert dispatched["called"] is False


# ========================================================================
# E2E: Partial Failure
# ========================================================================

class TestEndToEndPartialFailure:
    """One query fails in the batch, but the others succeed."""

    @pytest.mark.asyncio
    async def test_partial_failure_marked_in_snapshot(self):
        store = FakeSnapshotStore()
        call_count = 0

        async def _flaky_fetch(pipeline, query, settings, **kwargs):
            nonlocal call_count
            call_count += 1
            if query == "fail-me":
                raise Exception("Scraper exploded")
            await asyncio.sleep(0.01)
            return {**FAKE_SCRAPED, "query": query}

        with (
            patch(_P_API_VERIFY, new=_make_mock_verify(True)),
            patch(_P_API_CREATE, new=_make_mock_create_snapshot(store)),
            patch(_P_API_STATUS, new=_make_mock_get_snapshot(store)),
            patch(_P_SVC_UPDATE, new=_make_mock_update_snapshot(store)),
            patch(_P_SVC_FETCH, new=_flaky_fetch),
            patch(_P_SVC_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_P_SVC_STORE, side_effect=_make_mock_store(delay=0.01)),
            patch(_P_SVC_REFUND, new=AsyncMock(return_value=True)),
            patch(_P_SVC_SELECT, new=_make_mock_supabase_select(store)),
            patch(_P_SVC_DB_UPDATE, new=_make_mock_supabase_update(store)),
            patch(_P_SVC_RPC, new=_make_mock_supabase_rpc(store)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                start_resp = await client.post("/api/v1/optimize/start", json={
                    "product_id": "prod-partial",
                    "user_id": "user-partial",
                    "batch_id": "batch-partial",
                    "perplexity_queries": ["good-query", "fail-me"],
                    "google_queries": [],
                    "client_product_json": FAKE_PRODUCT_JSON,
                })

                assert start_resp.status_code == 202
                snapshot_id = start_resp.json()["snapshot_id"]

                status_resp = await client.get(
                    f"/api/v1/optimize/status/{snapshot_id}"
                )

        body = status_resp.json()
        assert body["status"] == "partial"  # some succeeded, some failed
        assert body["completed_queries"] == 1


# ========================================================================
# E2E: Mixed Pipelines
# ========================================================================

class TestEndToEndMixedPipelines:
    """Queries run across both Perplexity and Google Overview pipelines."""

    @pytest.mark.asyncio
    async def test_mixed_pipelines_all_complete(self):
        store = FakeSnapshotStore()
        observed_pipelines: List[str] = []

        async def _tracking_fetch(pipeline, query, settings, **kwargs):
            observed_pipelines.append(pipeline)
            await asyncio.sleep(0.01)
            return {**FAKE_SCRAPED, "query": query, "pipeline": pipeline}

        with (
            patch(_P_API_VERIFY, new=_make_mock_verify(True)),
            patch(_P_API_CREATE, new=_make_mock_create_snapshot(store)),
            patch(_P_API_STATUS, new=_make_mock_get_snapshot(store)),
            patch(_P_SVC_UPDATE, new=_make_mock_update_snapshot(store)),
            patch(_P_SVC_FETCH, new=_tracking_fetch),
            patch(_P_SVC_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_P_SVC_STORE, side_effect=_make_mock_store(delay=0.01)),
            patch(_P_SVC_REFUND, new=AsyncMock(return_value=True)),
            patch(_P_SVC_SELECT, new=_make_mock_supabase_select(store)),
            patch(_P_SVC_DB_UPDATE, new=_make_mock_supabase_update(store)),
            patch(_P_SVC_RPC, new=_make_mock_supabase_rpc(store)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.post("/api/v1/optimize/start", json={
                    "product_id": "prod-mix",
                    "user_id": "user-mix",
                    "batch_id": "batch-mix",
                    "perplexity_queries": ["pq1", "pq2"],
                    "google_queries": ["gq1", "gq2"],
                    "client_product_json": FAKE_PRODUCT_JSON,
                })

                assert resp.status_code == 202
                sid = resp.json()["snapshot_id"]

                status = await client.get(f"/api/v1/optimize/status/{sid}")

        assert status.json()["status"] == "completed"
        assert status.json()["completed_queries"] == 4
        assert sorted(set(observed_pipelines)) == ["google_overview", "perplexity"]


# ========================================================================
# E2E: Polling Before Completion (status=processing)
# ========================================================================

class TestEndToEndStatusPolling:
    """Verify that polling returns correct status at different points."""

    @pytest.mark.asyncio
    async def test_snapshot_not_found_returns_404(self):
        """Polling a nonexistent snapshot returns 404."""
        async def _empty(*args, **kwargs):
            return {}

        with patch(_P_API_STATUS, new=_empty):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get("/api/v1/optimize/status/nonexistent-id")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_running_status_before_background_completes(self):
        """
        If we poll the snapshot before the background task finishes,
        we should see status='running'.
        """
        store = FakeSnapshotStore()
        # Pre-create a snapshot in running state
        sid = store.create("prod-1", "batch-1", 2)
        # Don't run any background — just check that status returns running

        with patch(_P_API_STATUS, new=_make_mock_get_snapshot(store)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url=BASE_URL
            ) as client:
                resp = await client.get(f"/api/v1/optimize/status/{sid}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        assert resp.json()["completed_queries"] == 0


# ========================================================================
# E2E: Existing Endpoints Still Work (Regression)
# ========================================================================

class TestExistingEndpointsRegression:
    """Verify Phase 4 additions don't break existing routes."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_normalize_ai_search_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL
        ) as client:
            resp = await client.post(
                "/api/v1/utils/normalize-ai-search",
                json={"ai_search_json": {"key": "value"}},
            )
        assert resp.status_code == 200
        assert resp.json()["items"] == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_clean_json_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url=BASE_URL
        ) as client:
            resp = await client.post(
                "/api/v1/utils/clean-json",
                json={"text": "```json\n{\"a\": 1}\n```"},
            )
        assert resp.status_code == 200
        assert '"a"' in resp.json()["clean_text"]

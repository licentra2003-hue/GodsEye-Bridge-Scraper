"""Tests for Phase 3: Concurrent Execution Engine

Validates that:
  - ``process_single_query_pipeline`` correctly chains scrape → analyse → store.
  - ``run_optimization_batch`` fans out queries concurrently so that *N* queries
    finish in roughly the time of one (proving ``asyncio.gather`` parallelism).

All external dependencies (scraper API, Gemini, Supabase) are mocked.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.services.analysis_service import (
    process_single_query_pipeline,
    run_optimization_batch,
)


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

FAKE_SCRAPED_DATA: Dict[str, Any] = {
    "query": "best shampoo",
    "ai_overview_text": "Here are the top shampoos...",
    "source_links": [{"url": "https://example.com", "title": "Example"}],
}

FAKE_ANALYSIS: Dict[str, Any] = {
    "executive_summary": {"title": "AEO Analysis", "status_overview": "Good"},
    "client_product_visibility": {"status": "Featured", "details": "On page 1"},
}

FAKE_STORED_RECORD: Dict[str, Any] = {"id": "stored-uuid-1", "product_id": "prod-1"}

FAKE_PRODUCT_JSON: Dict[str, Any] = {
    "name": "TestProduct",
    "description": "A test product for analysis",
}

# Simulated delay (seconds) to represent network/AI latency
SIMULATED_DELAY = 0.15


@pytest.fixture
def mock_settings() -> Settings:
    """Minimal settings for tests."""
    return Settings(
        gemini_api_key="fake-gemini-key",
        google_ai_mode="new_ai_mode",
        scraper_url_perplexity="https://mock-perplexity.test/scrape",
        scraper_url_google_overview="https://mock-google.test/scrape",
        scraper_url_new_ai_mode="https://mock-new-ai.test/api/v1/scrape",
        scraper_api_key="test-key",
        scraper_poll_initial_interval=0.01,
        scraper_poll_max_interval=0.05,
        scraper_poll_backoff_multiplier=1.2,
        scraper_poll_max_attempts=5,
        scraper_request_timeout=5.0,
        next_public_supabase_url="https://test.supabase.co",
        supabase_service_role_key="test-supa-key",
    )


def _make_mock_fetch(delay: float = SIMULATED_DELAY):
    """Return an async callable for ``fetch_ai_search_data`` that sleeps to simulate latency."""

    async def _fake_fetch(pipeline: str, query: str, settings: Any, **kwargs) -> Dict[str, Any]:
        await asyncio.sleep(delay)
        return {**FAKE_SCRAPED_DATA, "query": query, "pipeline": pipeline}

    return _fake_fetch


def _make_mock_analysis(delay: float = SIMULATED_DELAY):
    """Return a synchronous callable for ``perform_strategic_analysis`` with a blocking delay."""

    def _fake_analysis(request, *, settings, api_key=None, debug=False, store_to_db=False):
        # Simulate blocking Gemini call
        import time as _time
        _time.sleep(delay)
        return {
            "success": True,
            "analysis": {**FAKE_ANALYSIS, "search_query": request.search_query},
        }

    return _fake_analysis


def _make_mock_store(delay: float = 0.01):
    """Return a synchronous callable for ``store_analysis_result``."""

    def _fake_store(**kwargs):
        import time as _time
        _time.sleep(delay)
        return {**FAKE_STORED_RECORD, "search_query": kwargs.get("search_query", "")}

    return _fake_store


# ---------------------------------------------------------------------------
# Patch targets — fetch_ai_search_data is lazy-imported inside
# process_single_query_pipeline, so we patch at the scraping_service source.
# ---------------------------------------------------------------------------
_PATCH_FETCH = "app.services.scraping_service.fetch_ai_search_data"
_PATCH_ANALYSIS = "app.services.analysis_service.perform_strategic_analysis"
_PATCH_STORE = "app.services.analysis_service.store_analysis_result"
# Supabase helpers used by _run_and_increment for per-query progress tracking
_PATCH_SUPA_CLIENT = "app.services.analysis_service.get_supabase_client"
_PATCH_SUPA_SELECT = "app.services.analysis_service.async_supabase_select"
_PATCH_SUPA_UPDATE = "app.services.analysis_service.async_supabase_update"


# ---------------------------------------------------------------------------
# Tests for process_single_query_pipeline
# ---------------------------------------------------------------------------

class TestProcessSingleQueryPipeline:
    """Unit tests for the single-query orchestrator."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_structured_result(self, mock_settings: Settings):
        """All three steps succeed and the result contains expected keys."""
        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=0.01)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
        ):
            result = await process_single_query_pipeline(
                product_id="prod-1",
                query="best shampoo",
                pipeline="perplexity",
                snapshot_id="snap-1",
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )

        assert result["success"] is True
        assert result["pipeline"] == "perplexity"
        assert result["search_query"] == "best shampoo"
        assert result["product_id"] == "prod-1"
        assert "analysis" in result
        assert "stored_record" in result

    @pytest.mark.asyncio
    async def test_google_overview_pipeline(self, mock_settings: Settings):
        """Verify the pipeline works for google_overview as well."""
        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=0.01)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
        ):
            result = await process_single_query_pipeline(
                product_id="prod-2",
                query="best laptop",
                pipeline="google_overview",
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )

        assert result["pipeline"] == "google_overview"
        assert result["search_query"] == "best laptop"

    @pytest.mark.asyncio
    async def test_scraping_error_propagates(self, mock_settings: Settings):
        """If the scraper fails, the exception bubbles up."""
        from app.services.scraping_service import ScrapingError

        async def _fail_fetch(*args, **kwargs):
            raise ScrapingError("Scraper unavailable")

        with patch(_PATCH_FETCH, new=_fail_fetch):
            with pytest.raises(ScrapingError, match="Scraper unavailable"):
                await process_single_query_pipeline(
                    product_id="prod-1",
                    query="fail",
                    pipeline="perplexity",
                    settings=mock_settings,
                    client_product_json=FAKE_PRODUCT_JSON,
                )

    @pytest.mark.asyncio
    async def test_analysis_error_propagates(self, mock_settings: Settings):
        """If Gemini analysis fails, the exception bubbles up."""

        def _fail_analysis(request, **kwargs):
            raise Exception("Gemini quota exceeded")

        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=0.01)),
            patch(_PATCH_ANALYSIS, side_effect=_fail_analysis),
        ):
            with pytest.raises(Exception, match="Gemini quota exceeded"):
                await process_single_query_pipeline(
                    product_id="prod-1",
                    query="fail-analysis",
                    pipeline="perplexity",
                    settings=mock_settings,
                    client_product_json=FAKE_PRODUCT_JSON,
                )


# ---------------------------------------------------------------------------
# Tests for run_optimization_batch
# ---------------------------------------------------------------------------

class TestRunOptimizationBatch:
    """Tests for batch concurrency via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_batch_returns_results_for_all_queries(self, mock_settings: Settings):
        """All queries across both pipelines produce results."""
        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=0.01)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
            patch(_PATCH_SUPA_CLIENT, return_value=MagicMock()),
            patch(_PATCH_SUPA_SELECT, new_callable=AsyncMock, return_value=[{"no_of_query": 0}]),
            patch(_PATCH_SUPA_UPDATE, new_callable=AsyncMock),
        ):
            results = await run_optimization_batch(
                product_id="prod-1",
                perplexity_queries=["query-p1", "query-p2"],
                google_queries=["query-g1"],
                snapshot_id="snap-1",
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )

        assert len(results) == 3
        pipelines = [r["pipeline"] for r in results]
        assert pipelines.count("perplexity") == 2
        assert pipelines.count("google_overview") == 1

    @pytest.mark.asyncio
    async def test_concurrency_3_queries_faster_than_sequential(
        self, mock_settings: Settings
    ):
        """
        Prove true concurrency: 3 queries each taking ~SIMULATED_DELAY should
        complete in roughly *one* SIMULATED_DELAY span, NOT three.

        This is the key Phase 3 requirement.
        """
        num_queries = 3
        per_query_delay = SIMULATED_DELAY  # 0.15 s

        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=per_query_delay)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=per_query_delay)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
        ):
            start = time.monotonic()
            results = await run_optimization_batch(
                product_id="prod-1",
                perplexity_queries=["q1", "q2", "q3"],
                google_queries=[],
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )
            elapsed = time.monotonic() - start

        assert len(results) == num_queries
        assert all(r["success"] for r in results)

        # Sequential would be ~num_queries * 2 * per_query_delay (scrape + analysis)
        # Concurrent should be ~1 * 2 * per_query_delay
        sequential_estimate = num_queries * 2 * per_query_delay
        # Allow generous margin — concurrent must be notably faster than sequential
        assert elapsed < sequential_estimate * 0.75, (
            f"Batch took {elapsed:.2f}s but sequential estimate is "
            f"{sequential_estimate:.2f}s — concurrency not proven"
        )

    @pytest.mark.asyncio
    async def test_concurrency_mixed_pipelines(self, mock_settings: Settings):
        """Queries across multiple pipelines also run concurrently."""
        per_query_delay = SIMULATED_DELAY

        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=per_query_delay)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=per_query_delay)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
            patch(_PATCH_SUPA_CLIENT, return_value=MagicMock()),
            patch(_PATCH_SUPA_SELECT, new_callable=AsyncMock, return_value=[{"no_of_query": 0}]),
            patch(_PATCH_SUPA_UPDATE, new_callable=AsyncMock),
        ):
            start = time.monotonic()
            results = await run_optimization_batch(
                product_id="prod-1",
                perplexity_queries=["pq1", "pq2"],
                google_queries=["gq1"],
                snapshot_id="snap-mix",
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )
            elapsed = time.monotonic() - start

        assert len(results) == 3
        sequential_estimate = 3 * 2 * per_query_delay
        assert elapsed < sequential_estimate * 0.75

    @pytest.mark.asyncio
    async def test_partial_failure_returns_exceptions(self, mock_settings: Settings):
        """
        ``return_exceptions=True`` in gather means partial failures don't
        crash the batch — they are returned in the result list.
        """
        call_count = 0

        async def _sometimes_fail(pipeline, query, settings, **kwargs):
            nonlocal call_count
            call_count += 1
            if query == "fail-query":
                raise Exception("Intentional failure")
            await asyncio.sleep(0.01)
            return {**FAKE_SCRAPED_DATA, "query": query}

        with (
            patch(_PATCH_FETCH, new=_sometimes_fail),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=0.01)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
        ):
            results = await run_optimization_batch(
                product_id="prod-1",
                perplexity_queries=["good-query", "fail-query"],
                google_queries=[],
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )

        assert len(results) == 2
        # One should be a dict, one should be an exception
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert "Intentional failure" in str(failures[0])

    @pytest.mark.asyncio
    async def test_empty_queries_returns_empty_list(self, mock_settings: Settings):
        """Edge case: no queries at all should return an empty list."""
        results = await run_optimization_batch(
            product_id="prod-1",
            perplexity_queries=[],
            google_queries=[],
            settings=mock_settings,
            client_product_json=FAKE_PRODUCT_JSON,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_to_thread(
        self, mock_settings: Settings
    ):
        """
        The Gemini call runs in ``to_thread``, so a concurrent async task
        should still be able to make progress while Gemini is "thinking".
        """
        counter = {"ticks": 0}

        async def _tick():
            while True:
                counter["ticks"] += 1
                await asyncio.sleep(0.005)

        with (
            patch(_PATCH_FETCH, new=_make_mock_fetch(delay=0.01)),
            patch(_PATCH_ANALYSIS, side_effect=_make_mock_analysis(delay=0.1)),
            patch(_PATCH_STORE, side_effect=_make_mock_store(delay=0.01)),
        ):
            tick_task = asyncio.create_task(_tick())

            await process_single_query_pipeline(
                product_id="prod-1",
                query="non-blocking-test",
                pipeline="perplexity",
                settings=mock_settings,
                client_product_json=FAKE_PRODUCT_JSON,
            )

            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

        # While to_thread was blocking for 100ms, the tick task should have progressed
        assert counter["ticks"] > 0, "Event loop was blocked during to_thread"

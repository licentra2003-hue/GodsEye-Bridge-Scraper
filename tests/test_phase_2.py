"""Tests for Phase 2: Async Scraping Orchestrator — 3 Scrapers + Mode Switching

Validates the three scraper patterns:
  - **Perplexity** (direct): POST {query, location, keep_open: false} → result
  - **Google Overview** (direct, mode=google_overview): POST {query, location, max_retries: 3} → result
  - **New AI Mode** (polling, mode=new_ai_mode): POST {query, location} → {job_id} → poll

And the mode switching: ``GOOGLE_AI_MODE`` controls which handler google
queries use, while perplexity is always separate.

All external HTTP calls are mocked with ``respx``.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from app.config import Settings
from app.services.scraping_service import (
    ScrapingError,
    ScrapingTimeoutError,
    fetch_ai_search_data,
)


# ---------------------------------------------------------------------------
# Mock URLs
# ---------------------------------------------------------------------------

MOCK_PERPLEXITY_URL = "https://mock-perplexity.test/scrape"
MOCK_GOOGLE_OVERVIEW_URL = "https://mock-google.test/scrape"
MOCK_NEW_AI_MODE_URL = "https://mock-new-ai.test/api/v1/scrape"

# Poll URL for New AI Mode (origin + /api/job-result/{job_id})
MOCK_NEW_AI_POLL_URL = "https://mock-new-ai.test/api/job-result/job-abc-123"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_new_ai_mode() -> Settings:
    """Settings with google_ai_mode=new_ai_mode (default)."""
    return Settings(
        google_ai_mode="new_ai_mode",
        scraper_url_perplexity=MOCK_PERPLEXITY_URL,
        scraper_url_google_overview=MOCK_GOOGLE_OVERVIEW_URL,
        scraper_url_new_ai_mode=MOCK_NEW_AI_MODE_URL,
        scraper_api_key="test-key-123",
        scraper_poll_initial_interval=0.05,   # 50 ms — fast for tests
        scraper_poll_max_interval=0.2,
        scraper_poll_backoff_multiplier=1.0,
        scraper_poll_max_attempts=20,
        scraper_request_timeout=10.0,
    )


@pytest.fixture
def settings_google_overview() -> Settings:
    """Settings with google_ai_mode=google_overview."""
    return Settings(
        google_ai_mode="google_overview",
        scraper_url_perplexity=MOCK_PERPLEXITY_URL,
        scraper_url_google_overview=MOCK_GOOGLE_OVERVIEW_URL,
        scraper_url_new_ai_mode=MOCK_NEW_AI_MODE_URL,
        scraper_api_key="test-key-123",
        scraper_poll_initial_interval=0.05,
        scraper_poll_max_interval=0.2,
        scraper_poll_backoff_multiplier=1.0,
        scraper_poll_max_attempts=20,
        scraper_request_timeout=10.0,
    )


# ---------------------------------------------------------------------------
# Perplexity — Direct response (keep_open: false)
# ---------------------------------------------------------------------------

class TestPerplexityDirect:
    """Perplexity scraper always returns results directly."""

    @pytest.mark.asyncio
    async def test_perplexity_returns_direct_result(
        self, settings_new_ai_mode: Settings
    ):
        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "ai_overview_text": "Top organic shampoos include...",
                        "query": "best organic shampoo",
                    },
                )
            )

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="perplexity",
                    query="best organic shampoo",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

            assert "ai_overview_text" in result
            assert result["ai_overview_text"] == "Top organic shampoos include..."

    @pytest.mark.asyncio
    async def test_perplexity_sends_correct_payload(
        self, settings_new_ai_mode: Settings
    ):
        captured_body = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"ai_overview_text": "result here"})

        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(side_effect=_capture)

            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="perplexity",
                    query="test query",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

        assert captured_body["query"] == "test query"
        assert captured_body["location"] == "India"
        assert captured_body["keep_open"] is False
        # Perplexity should NOT have max_retries
        assert "max_retries" not in captured_body

    @pytest.mark.asyncio
    async def test_perplexity_empty_content_raises(
        self, settings_new_ai_mode: Settings
    ):
        """Empty ai_overview_text should raise ScrapingError."""
        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(
                return_value=httpx.Response(
                    200, json={"ai_overview_text": ""}
                )
            )

            with pytest.raises(ScrapingError, match="empty or insufficient"):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="perplexity",
                        query="test",
                        settings=settings_new_ai_mode,
                        http_client=client,
                    )

    @pytest.mark.asyncio
    async def test_perplexity_unaffected_by_google_mode(
        self, settings_google_overview: Settings
    ):
        """Perplexity always uses its own URL regardless of google_ai_mode."""
        called_urls: list[str] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            called_urls.append(str(request.url))
            return httpx.Response(200, json={"ai_overview_text": "perplexity result"})

        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(side_effect=_capture)

            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="perplexity",
                    query="test",
                    settings=settings_google_overview,
                    http_client=client,
                )

        assert MOCK_PERPLEXITY_URL in called_urls[0]


# ---------------------------------------------------------------------------
# Google Overview — Direct response (max_retries: 3)
# ---------------------------------------------------------------------------

class TestGoogleOverviewDirect:
    """Google Overview when google_ai_mode=google_overview."""

    @pytest.mark.asyncio
    async def test_google_overview_returns_direct_result(
        self, settings_google_overview: Settings
    ):
        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "success": True,
                        "ai_overview_text": "Top shampoos include...",
                        "source_links": [{"url": "https://example.com"}],
                    },
                )
            )

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="best shampoo",
                    settings=settings_google_overview,
                    http_client=client,
                )

            assert result["success"] is True
            assert result["ai_overview_text"] == "Top shampoos include..."

    @pytest.mark.asyncio
    async def test_google_overview_sends_correct_payload(
        self, settings_google_overview: Settings
    ):
        captured_body = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={
                "success": True, "ai_overview_text": "ok", "source_links": [],
            })

        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(side_effect=_capture)

            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="test query",
                    settings=settings_google_overview,
                    http_client=client,
                )

        assert captured_body["query"] == "test query"
        assert captured_body["location"] == "India"
        assert captured_body["max_retries"] == 3
        # Google Overview should NOT have keep_open
        assert "keep_open" not in captured_body

    @pytest.mark.asyncio
    async def test_google_overview_failure_response_raises(
        self, settings_google_overview: Settings
    ):
        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(
                return_value=httpx.Response(
                    200, json={"success": False, "error_message": "Block detected"}
                )
            )

            with pytest.raises(ScrapingError, match="Block detected"):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="google_overview",
                        query="test",
                        settings=settings_google_overview,
                        http_client=client,
                    )

    @pytest.mark.asyncio
    async def test_google_overview_http_error_propagates(
        self, settings_google_overview: Settings
    ):
        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(
                return_value=httpx.Response(500, json={"error": "Internal"})
            )

            with pytest.raises(httpx.HTTPStatusError):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="google_overview",
                        query="error-test",
                        settings=settings_google_overview,
                        http_client=client,
                    )


# ---------------------------------------------------------------------------
# New AI Mode — Job polling (when google_ai_mode=new_ai_mode)
# ---------------------------------------------------------------------------

class TestNewAIModePolling:
    """New AI Mode scraper uses submit → poll → result pattern."""

    @pytest.mark.asyncio
    async def test_new_ai_completes_after_polling(
        self, settings_new_ai_mode: Settings
    ):
        poll_count = 0

        def _poll_side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            poll_count += 1
            if poll_count < 3:
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "result": {
                        "ai_overview_text": "Here are the top results...",
                        "source_links": [{"url": "https://example.com"}],
                    },
                },
            )

        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
            )
            respx.get(MOCK_NEW_AI_POLL_URL).mock(side_effect=_poll_side_effect)

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="google_overview",  # google pipeline, but mode=new_ai_mode
                    query="best shampoo",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

            assert result["ai_overview_text"] == "Here are the top results..."
            assert poll_count == 3

    @pytest.mark.asyncio
    async def test_new_ai_sends_correct_payload(
        self, settings_new_ai_mode: Settings
    ):
        captured_body = {}

        def _capture(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "job-abc-123"})

        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(side_effect=_capture)
            respx.get(MOCK_NEW_AI_POLL_URL).mock(
                return_value=httpx.Response(
                    200, json={"status": "completed", "result": {"q": "test"}},
                )
            )

            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="test query",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

        assert captured_body["query"] == "test query"
        assert captured_body["location"] == "India"
        # New AI Mode should NOT have keep_open or max_retries
        assert "keep_open" not in captured_body
        assert "max_retries" not in captured_body

    @pytest.mark.asyncio
    async def test_new_ai_job_failure(self, settings_new_ai_mode: Settings):
        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
            )
            respx.get(MOCK_NEW_AI_POLL_URL).mock(
                return_value=httpx.Response(
                    200,
                    json={"status": "failed", "error_message": "Rate limited"},
                )
            )

            with pytest.raises(ScrapingError, match="Rate limited"):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="google_overview",
                        query="test",
                        settings=settings_new_ai_mode,
                        http_client=client,
                    )

    @pytest.mark.asyncio
    async def test_polling_timeout(self, settings_new_ai_mode: Settings):
        settings_new_ai_mode.scraper_poll_max_attempts = 3

        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
            )
            respx.get(MOCK_NEW_AI_POLL_URL).mock(
                return_value=httpx.Response(200, json={"status": "processing"})
            )

            with pytest.raises(ScrapingTimeoutError, match="timed out"):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="google_overview",
                        query="timeout-test",
                        settings=settings_new_ai_mode,
                        http_client=client,
                    )

    @pytest.mark.asyncio
    async def test_submit_missing_job_id_raises(self, settings_new_ai_mode: Settings):
        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"status": "queued"})
            )

            with pytest.raises(ScrapingError, match="did not return a job_id"):
                async with httpx.AsyncClient() as client:
                    await fetch_ai_search_data(
                        pipeline="google_overview",
                        query="no-id-test",
                        settings=settings_new_ai_mode,
                        http_client=client,
                    )


# ---------------------------------------------------------------------------
# Mode switching: google_overview vs new_ai_mode
# ---------------------------------------------------------------------------

class TestModeSwitching:
    """Verify the GOOGLE_AI_MODE switch routes correctly."""

    @pytest.mark.asyncio
    async def test_google_mode_google_overview_uses_direct_url(
        self, settings_google_overview: Settings
    ):
        """When mode=google_overview, google pipeline hits google_overview URL."""
        called_url = None

        def _capture(request: httpx.Request) -> httpx.Response:
            nonlocal called_url
            called_url = str(request.url)
            return httpx.Response(200, json={
                "success": True, "ai_overview_text": "direct result", "source_links": [],
            })

        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(side_effect=_capture)

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="test",
                    settings=settings_google_overview,
                    http_client=client,
                )

        assert called_url and MOCK_GOOGLE_OVERVIEW_URL in called_url
        assert result["ai_overview_text"] == "direct result"

    @pytest.mark.asyncio
    async def test_google_mode_new_ai_uses_polling_url(
        self, settings_new_ai_mode: Settings
    ):
        """When mode=new_ai_mode, google pipeline hits new_ai_mode URL + polls."""
        called_urls: list[str] = []

        def _capture_submit(request: httpx.Request) -> httpx.Response:
            called_urls.append(str(request.url))
            return httpx.Response(200, json={"job_id": "job-abc-123"})

        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(side_effect=_capture_submit)
            respx.get(MOCK_NEW_AI_POLL_URL).mock(
                return_value=httpx.Response(
                    200, json={"status": "completed", "result": {"ai_overview_text": "polled result"}},
                )
            )

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="test",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

        assert any(MOCK_NEW_AI_MODE_URL in url for url in called_urls)
        assert result["ai_overview_text"] == "polled result"

    def test_pipeline_registry_reflects_mode_google_overview(
        self, settings_google_overview: Settings
    ):
        """Registry maps google_overview → google overview URL."""
        pipelines = settings_google_overview.scraper_pipelines
        assert pipelines["google_overview"] == MOCK_GOOGLE_OVERVIEW_URL
        assert pipelines["perplexity"] == MOCK_PERPLEXITY_URL

    def test_pipeline_registry_reflects_mode_new_ai(
        self, settings_new_ai_mode: Settings
    ):
        """Registry maps google_overview → new_ai_mode URL when mode=new_ai_mode."""
        pipelines = settings_new_ai_mode.scraper_pipelines
        assert pipelines["google_overview"] == MOCK_NEW_AI_MODE_URL
        assert pipelines["perplexity"] == MOCK_PERPLEXITY_URL


# ---------------------------------------------------------------------------
# Each pipeline hits its own URL
# ---------------------------------------------------------------------------

class TestPipelineRouting:
    """Verify pipelines route to different URLs."""

    @pytest.mark.asyncio
    async def test_perplexity_and_google_hit_different_urls(
        self, settings_google_overview: Settings
    ):
        perplexity_called = False
        google_called = False

        def _perplexity(request: httpx.Request) -> httpx.Response:
            nonlocal perplexity_called
            perplexity_called = True
            return httpx.Response(200, json={"ai_overview_text": "perp result"})

        def _google(request: httpx.Request) -> httpx.Response:
            nonlocal google_called
            google_called = True
            return httpx.Response(200, json={
                "success": True, "ai_overview_text": "google result", "source_links": [],
            })

        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(side_effect=_perplexity)
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(side_effect=_google)

            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="perplexity", query="p",
                    settings=settings_google_overview, http_client=client,
                )
                await fetch_ai_search_data(
                    pipeline="google_overview", query="g",
                    settings=settings_google_overview, http_client=client,
                )

        assert perplexity_called
        assert google_called

    @pytest.mark.asyncio
    async def test_unsupported_pipeline_raises(self, settings_new_ai_mode: Settings):
        with pytest.raises(ValueError, match="Unknown scraper pipeline"):
            async with httpx.AsyncClient() as client:
                await fetch_ai_search_data(
                    pipeline="unknown_engine",
                    query="test",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )


# ---------------------------------------------------------------------------
# Polling behaviour
# ---------------------------------------------------------------------------

class TestPollingBehaviour:
    """Test polling loop non-blocking behaviour."""

    @pytest.mark.asyncio
    async def test_polling_loop_does_not_block_event_loop(
        self, settings_new_ai_mode: Settings
    ):
        poll_count = 0
        REQUIRED_POLLS = 5

        def _poll(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            poll_count += 1
            if poll_count < REQUIRED_POLLS:
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(
                200, json={"status": "completed", "result": {"done": True}},
            )

        counter = {"ticks": 0}

        async def _tick():
            while True:
                counter["ticks"] += 1
                await asyncio.sleep(0.01)

        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
            )
            respx.get(MOCK_NEW_AI_POLL_URL).mock(side_effect=_poll)

            tick_task = asyncio.create_task(_tick())

            async with httpx.AsyncClient() as client:
                result = await fetch_ai_search_data(
                    pipeline="google_overview",
                    query="test",
                    settings=settings_new_ai_mode,
                    http_client=client,
                )

            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

        assert result["done"] is True
        assert poll_count == REQUIRED_POLLS
        assert counter["ticks"] > 0, "Event loop was blocked during polling"


# ---------------------------------------------------------------------------
# Auto-managed client path
# ---------------------------------------------------------------------------

class TestAutoManagedClient:

    @pytest.mark.asyncio
    async def test_works_without_injected_client_perplexity(
        self, settings_new_ai_mode: Settings
    ):
        with respx.mock:
            respx.post(MOCK_PERPLEXITY_URL).mock(
                return_value=httpx.Response(
                    200, json={"ai_overview_text": "auto-client-perp"},
                )
            )

            result = await fetch_ai_search_data(
                pipeline="perplexity",
                query="auto-client",
                settings=settings_new_ai_mode,
            )

        assert result["ai_overview_text"] == "auto-client-perp"

    @pytest.mark.asyncio
    async def test_works_without_injected_client_google_direct(
        self, settings_google_overview: Settings
    ):
        with respx.mock:
            respx.post(MOCK_GOOGLE_OVERVIEW_URL).mock(
                return_value=httpx.Response(
                    200, json={"success": True, "ai_overview_text": "auto-google", "source_links": []},
                )
            )

            result = await fetch_ai_search_data(
                pipeline="google_overview",
                query="auto-client",
                settings=settings_google_overview,
            )

        assert result["ai_overview_text"] == "auto-google"

    @pytest.mark.asyncio
    async def test_works_without_injected_client_new_ai(
        self, settings_new_ai_mode: Settings
    ):
        with respx.mock:
            respx.post(MOCK_NEW_AI_MODE_URL).mock(
                return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
            )
            respx.get(MOCK_NEW_AI_POLL_URL).mock(
                return_value=httpx.Response(
                    200, json={"status": "completed", "result": {"ai_overview_text": "auto-ai"}},
                )
            )

            result = await fetch_ai_search_data(
                pipeline="google_overview",
                query="auto-client",
                settings=settings_new_ai_mode,
            )

        assert result["ai_overview_text"] == "auto-ai"

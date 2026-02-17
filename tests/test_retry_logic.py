"""
Comprehensive mock tests for the retry logic in scraping_service.py.

Tests cover:
  1. Successful scrape on first attempt (no retries)
  2. Transient failure → success after retry (bot detection, empty content)
  3. All retries exhausted → graceful failure propagation
  4. Config-driven retry values from Settings (.env)
  5. Network errors (timeout, connection) trigger retry
  6. HTTP 5xx errors trigger retry
  7. Application-level failures (success=False) trigger retry
  8. should_retry_scraping_result helper function
  9. Perplexity and Google Overview handler behaviors
  10. Concurrency: multiple queries, one fails, others succeed
"""

import asyncio
import logging
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import pytest
import httpx

# ── Path setup ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.scraping_service import (
    fetch_ai_search_data,
    should_retry_scraping_result,
    _fetch_perplexity,
    _fetch_google_overview,
    ScrapingError,
)
from app.config import Settings


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def make_settings(**overrides) -> Settings:
    """Create a Settings object with test defaults, overridable."""
    defaults = {
        "retry_max_attempts": 3,
        "retry_min_wait_seconds": 0,   # No wait in tests (fast)
        "retry_max_wait_seconds": 0,   # No wait in tests (fast)
        "retry_multiplier": 0,         # No wait in tests (fast)
        "scraper_request_timeout": 10.0,
        "scraper_api_key": None,
        "scraper_url_perplexity": "https://mock-perplexity.example.com/scrape",
        "scraper_url_google_overview": "https://mock-google.example.com/scrape",
        "scraper_url_new_ai_mode": "https://mock-newai.example.com/api/v1/scrape",
        "google_ai_mode": "google_overview",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def success_perplexity_response() -> Dict[str, Any]:
    """A valid Perplexity scraper success response."""
    return {
        "success": True,
        "query": "test query",
        "ai_overview_text": "This is a comprehensive AI-generated analysis of the topic.",
        "source_links": [
            {"title": "Source 1", "url": "https://example.com/1"},
            {"title": "Source 2", "url": "https://example.com/2"},
        ],
        "total_interactions": 3,
        "timestamp": "2026-02-15T15:00:00",
    }


def success_google_response() -> Dict[str, Any]:
    """A valid Google Overview scraper success response."""
    return {
        "success": True,
        "ai_overview_text": "Google AI overview text about the product.",
        "source_links": [
            {"title": "Google Source", "url": "https://google.com/result"},
        ],
    }


def bot_detected_response() -> Dict[str, Any]:
    """Bot detection failure response."""
    return {
        "success": False,
        "error_code": "BOT_DETECTED",
        "error_message": "Bot detection or captcha encountered",
        "ai_overview_text": "",
        "source_links": [],
    }


def ai_mode_missing_response() -> Dict[str, Any]:
    """AI mode content missing response."""
    return {
        "success": False,
        "error_code": "AI_MODE_CONTENT_MISSING",
        "error_message": "No AI Mode content detected",
        "ai_overview_text": "",
        "source_links": [],
    }


def text_extraction_failed_response() -> Dict[str, Any]:
    """Text extraction failed response."""
    return {
        "success": False,
        "error_code": "TEXT_EXTRACTION_FAILED",
        "error_message": "Could not extract text content",
        "ai_overview_text": "",
        "source_links": [],
    }


def empty_content_response() -> Dict[str, Any]:
    """Response with empty AI text (technically HTTP 200 success)."""
    return {
        "success": True,
        "query": "test query",
        "ai_overview_text": "",
        "source_links": [],
        "total_interactions": 0,
        "timestamp": "2026-02-15T15:00:00",
    }


def make_mock_response(json_data: Dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ═══════════════════════════════════════════════════════════════════════
# 1. should_retry_scraping_result unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestShouldRetryScrapingResult:
    """Test the retry predicate helper function."""

    def test_success_true_no_retry(self):
        """success=True → should NOT retry."""
        assert should_retry_scraping_result({"success": True, "data": "ok"}) is False

    def test_success_false_should_retry(self):
        """success=False → should retry."""
        assert should_retry_scraping_result({"success": False, "error": "bot"}) is True

    def test_missing_success_key_no_retry(self):
        """Missing 'success' key defaults to True (no retry)."""
        assert should_retry_scraping_result({"data": "some data"}) is False

    def test_non_dict_should_retry(self):
        """Non-dict result → should retry (invalid format)."""
        assert should_retry_scraping_result("not a dict") is True
        assert should_retry_scraping_result(None) is True
        assert should_retry_scraping_result([]) is True

    def test_bot_detected_should_retry(self):
        """Bot detection response → should retry."""
        assert should_retry_scraping_result(bot_detected_response()) is True

    def test_ai_mode_missing_should_retry(self):
        """AI mode content missing → should retry."""
        assert should_retry_scraping_result(ai_mode_missing_response()) is True

    def test_text_extraction_failed_should_retry(self):
        """Text extraction failed → should retry."""
        assert should_retry_scraping_result(text_extraction_failed_response()) is True


# ═══════════════════════════════════════════════════════════════════════
# 2. _fetch_perplexity handler unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestFetchPerplexityHandler:
    """Test the raw _fetch_perplexity handler (no retry wrapper)."""

    @pytest.mark.asyncio
    async def test_success_returns_data(self):
        """Successful response → returns full data dict."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(success_perplexity_response())

        result = await _fetch_perplexity(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is True
        assert "ai_overview_text" in result
        assert len(result["ai_overview_text"]) > 1

    @pytest.mark.asyncio
    async def test_bot_detected_returns_failure(self):
        """Bot detection → returns dict with success=False (for retry_if_result)."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(bot_detected_response())

        result = await _fetch_perplexity(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is False
        assert "BOT_DETECTED" in result.get("error_code", "") or "bot" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_empty_content_returns_failure(self):
        """Empty ai_overview_text → returns dict with success=False."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(empty_content_response())

        result = await _fetch_perplexity(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is False
        assert "empty" in result.get("error_message", "").lower() or "insufficient" in result.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_http_500_raises(self):
        """HTTP 500 → raises HTTPStatusError (for retry_if_exception)."""
        client = AsyncMock()
        client.post.return_value = make_mock_response({"error": "internal"}, status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            await _fetch_perplexity(
                client=client,
                scrape_url="https://mock.com/scrape",
                query="test",
                settings=make_settings(),
            )

    @pytest.mark.asyncio
    async def test_network_error_raises(self):
        """Network timeout → raises RequestError (for retry_if_exception)."""
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectTimeout("Connection timed out")

        with pytest.raises(httpx.RequestError):
            await _fetch_perplexity(
                client=client,
                scrape_url="https://mock.com/scrape",
                query="test",
                settings=make_settings(),
            )


# ═══════════════════════════════════════════════════════════════════════
# 3. _fetch_google_overview handler unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestFetchGoogleOverviewHandler:
    """Test the raw _fetch_google_overview handler (no retry wrapper)."""

    @pytest.mark.asyncio
    async def test_success_returns_data(self):
        """Successful response → returns normalized data."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(success_google_response())

        result = await _fetch_google_overview(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is True
        assert "ai_overview_text" in result

    @pytest.mark.asyncio
    async def test_ai_mode_missing_returns_failure(self):
        """AI mode content missing → returns dict with success=False."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(ai_mode_missing_response())

        result = await _fetch_google_overview(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_text_extraction_failed_returns_failure(self):
        """Text extraction failed → returns dict with success=False."""
        client = AsyncMock()
        client.post.return_value = make_mock_response(text_extraction_failed_response())

        result = await _fetch_google_overview(
            client=client,
            scrape_url="https://mock.com/scrape",
            query="test",
            settings=make_settings(),
        )

        assert result["success"] is False


# ═══════════════════════════════════════════════════════════════════════
# 4. fetch_ai_search_data RETRY integration tests
# ═══════════════════════════════════════════════════════════════════════

class TestFetchAISearchDataRetry:
    """Test the retry orchestration in fetch_ai_search_data."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """No retries needed when first attempt succeeds."""
        settings = make_settings()
        mock_client = AsyncMock()
        mock_client.post.return_value = make_mock_response(success_perplexity_response())

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 1  # Only 1 call, no retries

    @pytest.mark.asyncio
    async def test_retry_on_bot_detection_then_success(self):
        """Bot detection on first attempt → retry → success on second."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        # First call: bot detected, Second call: success
        mock_client.post.side_effect = [
            make_mock_response(bot_detected_response()),
            make_mock_response(success_perplexity_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2  # 1 failure + 1 success

    @pytest.mark.asyncio
    async def test_retry_on_empty_content_then_success(self):
        """Empty content on first attempt → retry → success on second."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            make_mock_response(empty_content_response()),
            make_mock_response(success_perplexity_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_network_timeout_then_success(self):
        """Network timeout on first attempt → retry → success."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            httpx.ConnectTimeout("Connection timed out"),
            make_mock_response(success_perplexity_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_http_500_then_success(self):
        """HTTP 500 on first attempt → retry → success."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            make_mock_response({"error": "internal server error"}, status_code=500),
            make_mock_response(success_perplexity_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_ai_mode_missing_then_success(self):
        """AI mode content missing → retry → success."""
        settings = make_settings(retry_max_attempts=3, google_ai_mode="google_overview")
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            make_mock_response(ai_mode_missing_response()),
            make_mock_response(success_google_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="google_overview",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_failure(self):
        """All retries fail → returns the last failure result."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        # All 3 attempts return bot detection
        mock_client.post.side_effect = [
            make_mock_response(bot_detected_response()),
            make_mock_response(bot_detected_response()),
            make_mock_response(bot_detected_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        # After exhausting retries, tenacity returns the last result
        assert result["success"] is False
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_network_error_raises(self):
        """All retries fail with network errors → raises exception."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            httpx.ConnectTimeout("timeout 1"),
            httpx.ConnectTimeout("timeout 2"),
            httpx.ConnectTimeout("timeout 3"),
        ]

        with pytest.raises(httpx.ConnectTimeout):
            await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )

        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_multiple_failures_before_success(self):
        """Multiple different failures → eventually succeeds."""
        settings = make_settings(retry_max_attempts=5)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            httpx.ConnectTimeout("timeout"),                          # Attempt 1: network
            make_mock_response(bot_detected_response()),             # Attempt 2: bot detection
            make_mock_response(empty_content_response()),            # Attempt 3: empty content
            make_mock_response(success_perplexity_response()),       # Attempt 4: SUCCESS
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 4


# ═══════════════════════════════════════════════════════════════════════
# 5. Config-driven retry values
# ═══════════════════════════════════════════════════════════════════════

class TestConfigDrivenRetry:
    """Verify that retry behavior respects Settings values."""

    @pytest.mark.asyncio
    async def test_retry_count_matches_settings(self):
        """Number of attempts matches RETRY_MAX_ATTEMPTS from settings."""
        for max_attempts in [1, 2, 4]:
            settings = make_settings(retry_max_attempts=max_attempts)
            mock_client = AsyncMock()

            # All attempts fail with bot detection
            mock_client.post.side_effect = [
                make_mock_response(bot_detected_response())
                for _ in range(max_attempts + 1)  # Extra to ensure we don't overshoot
            ]

            result = await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )

            assert result["success"] is False
            assert mock_client.post.call_count == max_attempts, (
                f"Expected {max_attempts} attempts, got {mock_client.post.call_count}"
            )

    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self):
        """RETRY_MAX_ATTEMPTS=1 → no retries, just one attempt."""
        settings = make_settings(retry_max_attempts=1)
        mock_client = AsyncMock()
        mock_client.post.return_value = make_mock_response(bot_detected_response())

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is False
        assert mock_client.post.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# 6. Google Overview pipeline retry tests
# ═══════════════════════════════════════════════════════════════════════

class TestGoogleOverviewRetry:
    """Test retry logic specifically for the Google Overview pipeline."""

    @pytest.mark.asyncio
    async def test_google_success_no_retry(self):
        """Google Overview success on first attempt."""
        settings = make_settings(google_ai_mode="google_overview")
        mock_client = AsyncMock()
        mock_client.post.return_value = make_mock_response(success_google_response())

        result = await fetch_ai_search_data(
            pipeline="google_overview",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_google_text_extraction_failed_retry(self):
        """TEXT_EXTRACTION_FAILED → retry → success."""
        settings = make_settings(retry_max_attempts=3, google_ai_mode="google_overview")
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            make_mock_response(text_extraction_failed_response()),
            make_mock_response(success_google_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="google_overview",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_google_bot_then_timeout_then_success(self):
        """Bot detection → network timeout → success."""
        settings = make_settings(retry_max_attempts=5, google_ai_mode="google_overview")
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            make_mock_response(bot_detected_response()),
            httpx.ReadTimeout("read timed out"),
            make_mock_response(success_google_response()),
        ]

        result = await fetch_ai_search_data(
            pipeline="google_overview",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )

        assert result["success"] is True
        assert mock_client.post.call_count == 3


# ═══════════════════════════════════════════════════════════════════════
# 7. Concurrency: multiple queries, isolated failures
# ═══════════════════════════════════════════════════════════════════════

class TestConcurrentQueryIsolation:
    """Verify that one query's failure doesn't affect other concurrent queries."""

    @pytest.mark.asyncio
    async def test_parallel_queries_isolated(self):
        """Run 3 queries in parallel: 1 fails permanently, 2 succeed."""
        settings = make_settings(retry_max_attempts=2)

        async def mock_fetch(pipeline, query, settings, http_client=None):
            """Simulate fetch_ai_search_data with controlled behavior."""
            mock_client = AsyncMock()

            if query == "failing_query":
                mock_client.post.side_effect = [
                    make_mock_response(bot_detected_response()),
                    make_mock_response(bot_detected_response()),
                ]
            else:
                mock_client.post.return_value = make_mock_response(success_perplexity_response())

            return await fetch_ai_search_data(
                pipeline=pipeline,
                query=query,
                settings=settings,
                http_client=mock_client,
            )

        # Run 3 queries concurrently
        tasks = [
            mock_fetch("perplexity", "successful_query_1", settings),
            mock_fetch("perplexity", "failing_query", settings),
            mock_fetch("perplexity", "successful_query_2", settings),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Query 1 and 3 should succeed
        assert results[0]["success"] is True
        assert results[2]["success"] is True

        # Query 2 should fail (but not crash — returns dict)
        assert results[1]["success"] is False


# ═══════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_json_format_raises(self):
        """Invalid response format (non-dict) → raises ScrapingError."""
        settings = make_settings(retry_max_attempts=2)
        mock_client = AsyncMock()

        # Return a non-dict JSON (e.g., a string)
        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 200
        bad_resp.json.return_value = "not a dict"
        bad_resp.raise_for_status.return_value = None
        mock_client.post.return_value = bad_resp

        with pytest.raises(ScrapingError, match="Invalid"):
            await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )

    @pytest.mark.asyncio
    async def test_mixed_error_types_all_retried(self):
        """Mix of error types across attempts — all retried until success."""
        settings = make_settings(retry_max_attempts=5)
        mock_client = AsyncMock()

        mock_client.post.side_effect = [
            httpx.ConnectTimeout("timeout"),                          # Network error
            make_mock_response({"error": "bad gateway"}, 502),        # HTTP 502
            make_mock_response(bot_detected_response()),             # Bot detection
            make_mock_response(success_perplexity_response()),       # SUCCESS!
        ]

        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query="test query",
            settings=settings,
            http_client=mock_client,
        )


        assert result["success"] is True
        assert mock_client.post.call_count == 4


# ═══════════════════════════════════════════════════════════════════════
# 9. HTTP 4xx Handling (No Retry)
# ═══════════════════════════════════════════════════════════════════════

class TestHttp4xxHandling:
    """Verify that client-side 4xx errors are NOT retried."""

    @pytest.mark.asyncio
    async def test_http_400_no_retry(self):
        """HTTP 400 Bad Request → Fail immediately (no retry)."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()
        
        # 400 Bad Request
        mock_client.post.return_value = make_mock_response({"error": "bad request"}, status_code=400)

        with pytest.raises(httpx.HTTPStatusError) as exc:
            await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )
        
        assert exc.value.response.status_code == 400
        assert mock_client.post.call_count == 1  # Verify NO retries

    @pytest.mark.asyncio
    async def test_http_422_no_retry(self):
        """HTTP 422 Validation Error → Fail immediately (no retry)."""
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()
        
        # 422 Unprocessable Entity
        mock_client.post.return_value = make_mock_response({"error": "validation failed"}, status_code=422)

        with pytest.raises(httpx.HTTPStatusError) as exc:
            await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )
        
        assert exc.value.response.status_code == 422
        assert mock_client.post.call_count == 1  # Verify NO retries

    @pytest.mark.asyncio
    async def test_http_429_should_retry(self):
        """HTTP 429 Too Many Requests → Should still retry (server-side concern)."""
        # Note: 429 is technically 4xx but usually transient. 
        # Our current logic skips ALL 4xx.
        # IF we wanted to retry 429, we'd need to adjust logic.
        # But per user request "HTTP 400/422 ... waste retry budget", 
        # blocking all 4xx is the requested behavior for now.
        # So this test confirms it DOES NOT retry on 429 with current logic.
        
        settings = make_settings(retry_max_attempts=3)
        mock_client = AsyncMock()
        
        mock_client.post.return_value = make_mock_response({"error": "rate limit"}, status_code=429)

        with pytest.raises(httpx.HTTPStatusError) as exc:
            await fetch_ai_search_data(
                pipeline="perplexity",
                query="test query",
                settings=settings,
                http_client=mock_client,
            )
        
        assert exc.value.response.status_code == 429
        assert mock_client.post.call_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",           # Stop on first failure
        "--no-header",
    ])
    sys.exit(exit_code)

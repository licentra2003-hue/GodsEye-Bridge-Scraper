"""Async scraping orchestrator — 3 scraper endpoints, mode switching.

Three scraper patterns:

  **Perplexity** (direct):
      POST ``{query, location, keep_open: false}`` → ``{ai_overview_text, ...}``

  **Google Overview** (direct) — active when ``GOOGLE_AI_MODE=google_overview``:
      POST ``{query, location, max_retries: 3}`` → ``{success, ai_overview_text, source_links}``

  **New AI Mode** (job polling) — active when ``GOOGLE_AI_MODE=new_ai_mode`` (default):
      POST ``{query, location}`` → ``{job_id}``
      GET  ``<origin>/api/job-result/{job_id}`` → poll every 3s (max 100 attempts)

The ``google_overview`` pipeline dynamically uses either the Google Overview
or New AI Mode scraper based on ``settings.google_ai_mode``.  Perplexity is
always separate.

To add a new pipeline:
1. Set ``SCRAPER_URL_<NAME>`` in ``.env``
2. Add the corresponding field in ``Settings``
3. Register it in ``Settings.scraper_pipelines``
4. If it has a unique flow, add a handler in ``_PIPELINE_HANDLERS``
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import httpx
import tenacity
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
)

from app.config import Settings

logger = logging.getLogger(__name__)


def should_retry_scraping_result(result: Any) -> bool:
    """
    Retry if the scraper result explicitly indicates failure.
    
    This covers:
    - BOT_DETECTED / Captcha
    - AI_MODE_CONTENT_MISSING
    - TEXT_EXTRACTION_FAILED
    - Generic errors
    
    If 'success' is explicitly False, we retry.
    """
    if not isinstance(result, dict):
        return True  # Invalid/Empty response -> Retry
    
    # If success is missing, assume True (legacy) or False? 
    # Usually scrapers return success=True/False. Safe to default to True (don't retry unknown)
    # UNLESS success is explicitly False.
    return result.get("success", True) is False


class ScrapingError(Exception):
    """Raised when the scraper API returns an unrecoverable error."""


class ScrapingTimeoutError(ScrapingError):
    """Raised when polling exceeds the maximum number of attempts."""


def should_retry_exception(exc: BaseException) -> bool:
    """
    Retry on network errors, ScrapingError, or 5xx HTTP errors.
    
    Do NOT retry on 4xx client errors (e.g. 400, 422), as these 
    indicate invalid requests that will likely fail again.
    """
    if isinstance(exc, httpx.RequestError):
        # Connection timeouts, DNS failures, etc.
        return True
    if isinstance(exc, ScrapingError):
        # Application-level scraping failures
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # Don't retry client errors (400-499)
        if 400 <= exc.response.status_code < 500:
            return False
        # Do retry server errors (500-599)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Perplexity — Direct response (keep_open: false)
# ═══════════════════════════════════════════════════════════════════════

async def _fetch_perplexity(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    settings: Settings,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perplexity scraper returns results directly in the response.

    POST ``{query, location, keep_open}`` → ``{ai_overview_text, ...}``
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Add browser-like User-Agent to avoid blocking
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    logger.info("[Scraper:perplexity] POST %s  query=%r", scrape_url, query)

    response = await client.post(
        scrape_url,
        json={
            "query": query,
            "location": "India",
        },
        headers=headers,
    )
    response.raise_for_status()

    data = response.json()

    # Validate response format first (before calling .get())
    if not data or not isinstance(data, dict):
        raise ScrapingError("Invalid Perplexity scraper response format")

    logger.info("[Scraper:perplexity] Direct response received  success=%s", data.get("success"))

    # Check for application-level failure (bot detection, empty content, etc.)
    # Return as-is so should_retry_scraping_result() can trigger a retry.
    if data.get("success") is False:
        error_msg = data.get("error_message", "Unknown scraper error")
        logger.warning("[Scraper:perplexity] Server returned failure: %s", error_msg)
        return data  # retry_if_result will inspect this

    ai_text = data.get("ai_overview_text", "")
    if not ai_text or len(ai_text) <= 1:
        logger.warning("[Scraper:perplexity] Empty or insufficient content")
        return {**data, "success": False, "error_message": "Empty or insufficient content"}

    return data


# ═══════════════════════════════════════════════════════════════════════
# Google Overview — Direct response (max_retries: 3)
# ═══════════════════════════════════════════════════════════════════════

async def _fetch_google_overview(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    settings: Settings,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Google Overview scraper returns results directly in the response.

    POST ``{query, location, max_retries}`` → ``{success, ai_overview_text, source_links}``
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("[Scraper:google_overview] POST %s  query=%r", scrape_url, query)

    response = await client.post(
        scrape_url,
        json={
            "query": query,
            "location": "India",
            "max_retries": 3,
        },
        headers=headers,
    )
    response.raise_for_status()

    data = response.json()

    # Validate response format first (before calling .get())
    if not data or not isinstance(data, dict):
        raise ScrapingError("Invalid Google Overview scraper response format")

    logger.info("[Scraper:google_overview] Direct response received  success=%s", data.get("success"))

    # Check for application-level failure (bot detection, DOM changes, etc.)
    # Return as-is so should_retry_scraping_result() can trigger a retry.
    if data.get("success") is False:
        error_msg = data.get("error_message", "Scraper failed to get AI Overview")
        logger.warning("[Scraper:google_overview] Server returned failure: %s", error_msg)
        return data  # retry_if_result will inspect this

    ai_text = data.get("ai_overview_text", "")
    if not ai_text or len(ai_text) <= 1:
        logger.warning("[Scraper:google_overview] Empty or insufficient AI overview content")
        return {**data, "success": False, "error_message": "Empty or insufficient content"}

    return {
        "success": True,
        "ai_overview_text": data["ai_overview_text"],
        "source_links": data.get("source_links", []),
    }


# ═══════════════════════════════════════════════════════════════════════
# New AI Mode — Job-based polling
# ═══════════════════════════════════════════════════════════════════════

async def _submit_new_ai_job(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Submit a scraping job and return the ``job_id``.

    POST ``{query, location}`` → ``{job_id}``
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.info("[Scraper:new_ai_mode] Submitting job  url=%s  query=%r", scrape_url, query)

    response = await client.post(
        scrape_url,
        json={"query": query, "location": "India"},
        headers=headers,
    )
    response.raise_for_status()

    data = response.json()
    job_id = data.get("job_id")
    if not job_id:
        raise ScrapingError(f"New AI Mode scraper did not return a job_id: {data}")

    logger.info("[Scraper:new_ai_mode] Job submitted  job_id=%s", job_id)
    return job_id


async def _poll_job_result(
    client: httpx.AsyncClient,
    scrape_url: str,
    job_id: str,
    *,
    api_key: Optional[str] = None,
    poll_interval: float = 3.0,
    max_attempts: int = 100,
) -> Dict[str, Any]:
    """
    Poll ``<origin>/api/job-result/{job_id}`` until the job completes.

    The poll URL is derived by replacing the scrape path with
    ``/api/job-result/{job_id}`` on the same origin, matching the
    frontend logic::

        config.scraperEndpoint.replace('/api/v1/scrape', '/api/job-result')

    Polls every ``poll_interval`` seconds (default 3s, matching frontend).
    Max ``max_attempts`` attempts (default 100 = 5 minutes).

    Returns the result payload on success.
    """
    # Derive poll URL matching frontend logic:
    # config.scraperEndpoint.replace('/api/v1/scrape', '/api/job-result')
    if "/api/v1/scrape" in scrape_url:
        base_url = scrape_url.replace("/api/v1/scrape", "/api/job-result")
    else:
        # Fallback if URL structure changes: assume root /api/job-result
        parsed = urlparse(scrape_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        base_url = f"{origin}/api/job-result"
    
    poll_url = f"{base_url}/{job_id}"

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    attempt = 0

    while True:
        attempt += 1
        if attempt > max_attempts:
            raise ScrapingTimeoutError(
                f"Polling timed out after {max_attempts} attempts for job {job_id}"
            )

        logger.debug(
            "[Scraper:new_ai_mode] poll attempt=%d/%d  interval=%.1fs  job_id=%s",
            attempt, max_attempts, poll_interval, job_id,
        )

        response = await client.get(poll_url, headers=headers)
        response.raise_for_status()

        data: Dict[str, Any] = response.json()
        status = (data.get("status", "") or "").lower()

        # Check terminal states (matching frontend logic)
        is_completed = status == "completed"
        is_failed = status == "failed"
        has_data = (
            isinstance(data.get("data"), dict) and
            data["data"].get("success") is not None
        )

        if is_completed or is_failed or has_data:
            if is_failed or (has_data and data["data"].get("success") is False):
                error_msg = (
                    data.get("error_message") or
                    (data.get("data", {}).get("error_message") if isinstance(data.get("data"), dict) else None) or
                    "Job failed to complete"
                )
                raise ScrapingError(f"New AI Mode job {job_id} failed: {error_msg}")

            logger.info("[Scraper:new_ai_mode] Job completed  job_id=%s", job_id)
            return data.get("result", data.get("data", data))

        # Still processing — wait then retry (fixed interval, matching frontend)
        await asyncio.sleep(poll_interval)


async def _fetch_new_ai_mode(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    settings: Settings,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit → poll flow for New AI Mode scraper."""
    job_id = await _submit_new_ai_job(client, scrape_url, query, api_key)
    return await _poll_job_result(
        client,
        scrape_url,
        job_id,
        api_key=api_key,
        poll_interval=settings.scraper_poll_initial_interval,
        max_attempts=settings.scraper_poll_max_attempts,
    )


# ═══════════════════════════════════════════════════════════════════════
# ChatGPT — Job-based polling
# ═══════════════════════════════════════════════════════════════════════

async def _submit_chatgpt_job(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    api_key: Optional[str] = None,
) -> str:
    """Submit a ChatGPT scraping job and return the job_id."""
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # The scrape_url might already be /api/scrape, depending on env Var. Ensure it is correct.
    submit_url = scrape_url
    if not submit_url.endswith("/api/scrape"):
        submit_url = submit_url.rstrip("/") + "/api/scrape"

    logger.info("[Scraper:chatgpt] Submitting job  url=%s  query=%r", submit_url, query)

    response = await client.post(
        submit_url,
        json={"query": query},
        headers=headers,
    )
    if response.status_code != 201:
        raise ScrapingError(f"Failed to submit ChatGPT job: {response.status_code} - {response.text}")
        
    data = response.json()
    job_id = data.get("job_id")
    if not job_id:
        raise ScrapingError(f"ChatGPT scraper did not return a job_id: {data}")

    logger.info("[Scraper:chatgpt] Job submitted  job_id=%s", job_id)
    return job_id


async def _poll_chatgpt_result(
    client: httpx.AsyncClient,
    scrape_url: str,
    job_id: str,
    *,
    api_key: Optional[str] = None,
    poll_interval: float = 3.0,
    max_attempts: int = 100,
) -> Dict[str, Any]:
    """Poll the ChatGPT result until completed."""
    # Base URL parsing
    if "/api/scrape" in scrape_url:
        base_url = scrape_url.replace("/api/scrape", "/api/result")
    else:
        parsed = urlparse(scrape_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        base_url = f"{origin}/api/result"
        
    poll_url = f"{base_url}/{job_id}"

    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    attempt = 0
    while True:
        attempt += 1
        if attempt > max_attempts:
            raise ScrapingTimeoutError(
                f"Polling timed out after {max_attempts} attempts for ChatGPT job {job_id}"
            )

        logger.debug(
            "[Scraper:chatgpt] poll attempt=%d/%d  interval=%.1fs  job_id=%s",
            attempt, max_attempts, poll_interval, job_id,
        )

        response = await client.get(poll_url, headers=headers)
        response.raise_for_status()

        data: Dict[str, Any] = response.json()
        status = (data.get("status", "") or "").lower()

        is_completed = status == "completed"
        is_failed = status == "failed"
        has_data = (
            isinstance(data.get("data"), dict) and
            data["data"].get("success") is not None
        )
        
        if is_completed or is_failed or has_data:
            if is_failed or (has_data and data["data"].get("success") is False):
                error_msg = (
                    data.get("error_message") or
                    (data.get("data", {}).get("error_message") if isinstance(data.get("data"), dict) else None) or
                    "Job failed to complete"
                )
                raise ScrapingError(f"ChatGPT job {job_id} failed: {error_msg}")

            logger.info("[Scraper:chatgpt] Job completed  job_id=%s", job_id)

            # --- Extract the actual content ---
            # Priority: data["result"] → data["data"] → data itself (only if it
            # carries a known content key, NOT just a job-metadata envelope).
            content = data.get("result", data.get("data"))
            if content is not None:
                return content

            # Last resort: use the top-level dict ONLY if it contains actual
            # scrape content (answer_text, ai_overview_text, response, etc.).
            # If it only has job-metadata keys we treat it as empty and let
            # the content-validation in _fetch_chatgpt trigger a retry.
            CONTENT_KEYS = {"answer_text", "ai_overview_text", "response", "text", "content"}
            if any(k in data for k in CONTENT_KEYS):
                return data

            # No usable content — return a failure-shaped dict so retry fires
            logger.warning(
                "[Scraper:chatgpt] Job %s completed but response contains no content keys. "
                "Raw keys: %s",
                job_id, list(data.keys()),
            )
            return {"success": False, "error_message": "ChatGPT job completed but returned no content"}

        # Still processing — wait then retry
        await asyncio.sleep(poll_interval)


async def _fetch_chatgpt(
    client: httpx.AsyncClient,
    scrape_url: str,
    query: str,
    settings: Settings,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit → poll flow for ChatGPT scraper."""
    job_id = await _submit_chatgpt_job(client, scrape_url, query, api_key)
    result = await _poll_chatgpt_result(
        client,
        scrape_url,
        job_id,
        api_key=api_key,
        poll_interval=settings.scraper_poll_initial_interval,
        max_attempts=settings.scraper_poll_max_attempts,
    )

    # --- Content validation (mirrors Perplexity and Google patterns) ---
    # If the result already signals failure, pass it through for retry.
    if result.get("success") is False:
        error_msg = result.get("error_message", "Unknown ChatGPT scraper error")
        logger.warning("[Scraper:chatgpt] Server returned failure: %s", error_msg)
        return result  # retry_if_result will inspect this

    # Try to find the main text content under several possible field names.
    # The ChatGPT scraper may use different keys depending on its version.
    CONTENT_FIELD_CANDIDATES = ["answer_text", "ai_overview_text", "response", "text", "content"]
    ai_text = ""
    for field in CONTENT_FIELD_CANDIDATES:
        candidate = result.get(field, "")
        if isinstance(candidate, str) and len(candidate) > 1:
            ai_text = candidate
            break

    if not ai_text:
        logger.warning(
            "[Scraper:chatgpt] Empty or missing content. Available keys: %s",
            list(result.keys()),
        )
        return {**result, "success": False, "error_message": "Empty or insufficient ChatGPT response content"}

    return result



# ═══════════════════════════════════════════════════════════════════════
# Pipeline → handler mapping
# ═══════════════════════════════════════════════════════════════════════

# Static handler map for all 3 scraper types.
_SCRAPER_HANDLERS: Dict[str, Callable] = {
    "perplexity": _fetch_perplexity,
    "google_overview": _fetch_google_overview,
    "new_ai_mode": _fetch_new_ai_mode,
    "chatgpt": _fetch_chatgpt,
}


def _resolve_handler(pipeline: str, settings: Settings) -> Callable:
    """
    Resolve the correct scraper handler for a pipeline.

    For ``"perplexity"`` → always ``_fetch_perplexity``
    For ``"google_overview"`` → depends on ``settings.google_ai_mode``:
      - ``"google_overview"`` → ``_fetch_google_overview``
      - ``"new_ai_mode"`` → ``_fetch_new_ai_mode``
    """
    if pipeline == "perplexity":
        return _SCRAPER_HANDLERS["perplexity"]

    if pipeline == "google_overview":
        mode = settings.google_ai_mode
        handler = _SCRAPER_HANDLERS.get(mode)
        if handler is None:
            raise ScrapingError(
                f"No handler for google_ai_mode='{mode}'. "
                f"Valid modes: google_overview, new_ai_mode"
            )
        logger.info(
            "[Scraper] google_overview pipeline using mode=%s", mode
        )
        return handler

    if pipeline == "chatgpt":
        if not settings.enable_chatgpt:
            raise ScrapingError("ChatGPT pipeline is disabled in settings.")
        return _SCRAPER_HANDLERS["chatgpt"]

    raise ScrapingError(
        f"No handler registered for pipeline: '{pipeline}'. "
        f"Supported: perplexity, google_overview"
    )


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

async def fetch_ai_search_data(
    pipeline: str,
    query: str,
    settings: Settings,
    *,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """
    End-to-end async function: scrape AI search data for a given pipeline.

    Retry policy is driven by ``.env`` settings:
        - ``RETRY_MAX_ATTEMPTS``  → max number of attempts (default 5)
        - ``RETRY_MIN_WAIT_SECONDS`` → min backoff wait (default 1)
        - ``RETRY_MAX_WAIT_SECONDS`` → max backoff wait (default 15)
        - ``RETRY_MULTIPLIER`` → exponential multiplier (default 1)

    Retries are triggered on:
        - Network / HTTP errors (httpx.RequestError, httpx.HTTPStatusError)
        - Application-level failures (success=False in response JSON),
          covering bot detection, DOM changes, content missing, etc.

    Args:
        pipeline: Pipeline id (``"perplexity"`` or ``"google_overview"``).
        query: The search query to send to the scraper.
        settings: Application settings containing scraper configuration.
        http_client: Optional pre-built ``httpx.AsyncClient`` (for tests).

    Returns:
        The scraped AI search data as a dictionary.

    Raises:
        ScrapingError: on submission or job failure (after retries exhausted).
        ScrapingTimeoutError: if polling exceeds max attempts.
        ValueError: if the pipeline is not registered.
    """
    scrape_url = settings.get_scraper_url(pipeline)
    api_key = settings.scraper_api_key
    handler = _resolve_handler(pipeline, settings)

    # Build settings-driven retry policy
    retrier = AsyncRetrying(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            multiplier=settings.retry_multiplier,
            min=settings.retry_min_wait_seconds,
            max=settings.retry_max_wait_seconds,
        ),
        retry=(
            retry_if_exception(should_retry_exception)
            | retry_if_result(should_retry_scraping_result)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    logger.info(
        "[Scraper] fetch_ai_search_data  pipeline=%s  query=%r  max_attempts=%d",
        pipeline, query, settings.retry_max_attempts,
    )

    async def _execute_with_retry(client: httpx.AsyncClient) -> Dict[str, Any]:
        """Execute handler with retry, gracefully handling exhaustion."""
        async def _call() -> Dict[str, Any]:
            return await handler(
                client=client,
                scrape_url=scrape_url,
                query=query,
                settings=settings,
                api_key=api_key,
            )
        try:
            return await retrier(_call)
        except tenacity.RetryError as e:
            # All retries exhausted. If the last attempt returned a result
            # (rather than raising), extract and return it so callers get
            # a clean {success: False, ...} dict instead of a crash.
            last = e.last_attempt
            if last and not last.failed:
                logger.warning(
                    "[Scraper] All %d retries exhausted for pipeline=%s query=%r — returning last failure result",
                    settings.retry_max_attempts, pipeline, query,
                )
                return last.result()
            # Last attempt was an exception — re-raise it
            raise last.result()

    if http_client is not None:
        return await _execute_with_retry(http_client)

    async with httpx.AsyncClient(timeout=settings.scraper_request_timeout) as client:
        return await _execute_with_retry(client)

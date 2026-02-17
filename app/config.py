"""Application configuration — Pydantic Settings.

Scraper Architecture
--------------------
Three scraper endpoints:

  **Perplexity** (always direct):
      POST ``{query, location, keep_open}`` → immediate result

  **Google Overview** (direct, when ``GOOGLE_AI_MODE=google_overview``):
      POST ``{query, location, max_retries}`` → immediate result

  **New AI Mode** (job polling, when ``GOOGLE_AI_MODE=new_ai_mode`` — default):
      POST ``{query, location}`` → ``{job_id}``
      GET  ``<origin>/api/job-result/{job_id}`` → poll until completed

The ``GOOGLE_AI_MODE`` environment variable controls which scraper
is used for "google" queries.  Default is ``new_ai_mode``.
Perplexity is always separate and unaffected by this switch.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Literal

from pydantic import BaseModel, Field


GoogleAIMode = Literal["google_overview", "new_ai_mode"]


class Settings(BaseModel):
    app_name: str = Field(default_factory=lambda: os.getenv("APP_NAME", "strategic-analysis-backend"))
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_debug: bool = Field(default_factory=lambda: os.getenv("APP_DEBUG", "false").lower() == "true")
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model_name: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
    gemini_temperature: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TEMPERATURE", "0.7")))
    gemini_top_p: float = Field(default_factory=lambda: float(os.getenv("GEMINI_TOP_P", "0.95")))
    gemini_top_k: int = Field(default_factory=lambda: int(os.getenv("GEMINI_TOP_K", "40")))
    gemini_max_output_tokens: int = Field(default_factory=lambda: int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192")))

    retry_max_attempts: int = Field(default_factory=lambda: int(os.getenv("RETRY_MAX_ATTEMPTS", "5")))
    retry_min_wait_seconds: int = Field(default_factory=lambda: int(os.getenv("RETRY_MIN_WAIT_SECONDS", "1")))
    retry_max_wait_seconds: int = Field(default_factory=lambda: int(os.getenv("RETRY_MAX_WAIT_SECONDS", "15")))
    retry_multiplier: int = Field(default_factory=lambda: int(os.getenv("RETRY_MULTIPLIER", "1")))

    next_public_supabase_url: str | None = Field(default_factory=lambda: os.getenv("NEXT_PUBLIC_SUPABASE_URL"))
    supabase_service_role_key: str | None = Field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

    # ── Google AI Mode Switch ─────────────────────────────────────────
    # Controls which scraper is used for google queries:
    #   "google_overview" → direct call to Google Overview scraper
    #   "new_ai_mode"     → job polling via New AI Mode scraper (default)
    google_ai_mode: GoogleAIMode = Field(
        default_factory=lambda: os.getenv("GOOGLE_AI_MODE", "new_ai_mode")  # type: ignore[return-value]
    )

    # ── Scraper URLs (all 3 endpoints) ────────────────────────────────
    scraper_url_perplexity: str = Field(
        default_factory=lambda: os.getenv(
            "SCRAPER_URL_PERPLEXITY",
            "https://perplexity-scraper-new-production.up.railway.app/scrape",
        )
    )
    scraper_url_google_overview: str = Field(
        default_factory=lambda: os.getenv(
            "SCRAPER_URL_GOOGLE_OVERVIEW",
            "https://google-ai-overview-scraper-production.up.railway.app/scrape",
        )
    )
    scraper_url_new_ai_mode: str = Field(
        default_factory=lambda: os.getenv(
            "SCRAPER_URL_NEW_AI_MODE",
            "https://discerning-dream-production-a744.up.railway.app/api/v1/scrape",
        )
    )

    # Shared scraper settings
    scraper_api_key: str | None = Field(default_factory=lambda: os.getenv("SCRAPER_API_KEY"))
    scraper_poll_initial_interval: float = Field(default_factory=lambda: float(os.getenv("SCRAPER_POLL_INITIAL_INTERVAL", "3.0")))
    scraper_poll_max_interval: float = Field(default_factory=lambda: float(os.getenv("SCRAPER_POLL_MAX_INTERVAL", "3.0")))
    scraper_poll_backoff_multiplier: float = Field(default_factory=lambda: float(os.getenv("SCRAPER_POLL_BACKOFF_MULTIPLIER", "1.0")))
    scraper_poll_max_attempts: int = Field(default_factory=lambda: int(os.getenv("SCRAPER_POLL_MAX_ATTEMPTS", "100")))
    scraper_request_timeout: float = Field(default_factory=lambda: float(os.getenv("SCRAPER_REQUEST_TIMEOUT", "30.0")))

    @property
    def scraper_pipelines(self) -> Dict[str, str]:
        """
        Registry mapping pipeline name → scrape endpoint URL.

        - ``"perplexity"`` → always the Perplexity scraper
        - ``"google_overview"`` → resolved based on ``google_ai_mode``:
            - ``"google_overview"`` → Google Overview direct scraper
            - ``"new_ai_mode"`` → New AI Mode job-polling scraper
        """
        # Resolve which URL to use for google queries
        if self.google_ai_mode == "new_ai_mode":
            google_url = self.scraper_url_new_ai_mode
        else:
            google_url = self.scraper_url_google_overview

        return {
            "perplexity": self.scraper_url_perplexity,
            "google_overview": google_url,
        }

    @property
    def active_google_scraper_mode(self) -> GoogleAIMode:
        """Return the currently active Google AI scraper mode."""
        return self.google_ai_mode

    def get_scraper_url(self, pipeline: str) -> str:
        """Resolve scraper URL for a given pipeline, raising on unknown."""
        url = self.scraper_pipelines.get(pipeline)
        if not url:
            supported = ", ".join(sorted(self.scraper_pipelines.keys()))
            raise ValueError(
                f"Unknown scraper pipeline '{pipeline}'. "
                f"Supported: {supported}"
            )
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

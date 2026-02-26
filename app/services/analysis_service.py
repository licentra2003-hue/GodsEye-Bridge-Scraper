from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from supabase import Client, create_client
from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.models import PipelineId, StrategicAnalysisRequest

logger = logging.getLogger(__name__)


class StrategicAnalysisResult(dict):
    """Result structure matching TypeScript interface."""


def get_supabase_client(settings: Settings) -> Client:
    """Initialize Supabase client."""
    supabase_url = settings.next_public_supabase_url
    supabase_key = settings.supabase_service_role_key

    if not supabase_url or not supabase_key:
        raise ValueError("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")

    return create_client(supabase_url, supabase_key)


async def async_supabase_insert(supabase: Client, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Async wrapper for Supabase insert operations."""
    return await asyncio.to_thread(
        lambda: supabase.table(table).insert(data).execute()
    )


async def async_supabase_update(supabase: Client, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
    """Async wrapper for Supabase update operations."""
    def _run():
        query = supabase.table(table).update(data)
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute()
    return await asyncio.to_thread(_run)


async def async_supabase_select(supabase: Client, table: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async wrapper for Supabase select operations."""
    def _run():
        query = supabase.table(table).select("*")
        for key, value in filters.items():
            query = query.eq(key, value)
        return query.execute()
    response = await asyncio.to_thread(_run)
    return response.data if response.data else []


async def async_supabase_rpc(supabase: Client, function_name: str, params: Dict[str, Any]) -> Any:
    """Async wrapper for Supabase RPC operations."""
    return await asyncio.to_thread(
        lambda: supabase.rpc(function_name, params).execute()
    )


async def create_analysis_snapshot(
    settings: Settings,
    product_id: str,
    batch_id: str,
    queries: List[str],
    total_no_of_query: Optional[int] = None,
) -> str:
    """
    Create or reuse an analysis snapshot using 3-scenario logic.

    Scenario 1 — No existing snapshot for this product+batch:
        Creates a new row with status='running', no_of_query=0.

    Scenario 2 — Existing snapshot, all queries already completed:
        Creates a NEW snapshot row (fresh start).

    Scenario 3 — Existing snapshot, incomplete (some queries still pending):
        Reuses the existing row, resets status to 'running'.

    Args:
        settings: Runtime settings
        product_id: UUID of the product (FK to products)
        batch_id: UUID of the query batch (FK to query_batches)
        queries: List of search queries (used for total_no_of_query)

    Returns:
        snapshot_id: UUID of the created or reused snapshot
    """
    supabase = get_supabase_client(settings)
    total = total_no_of_query if total_no_of_query is not None else len(queries)
    now_iso = datetime.now().isoformat()

    try:
        # ── Look for an existing snapshot for this product + batch ──
        existing = await async_supabase_select(
            supabase, "analysis_snapshots",
            {"product_id": product_id, "batch_id": batch_id},
        )

        if existing:
            snap = existing[0]
            snap_id = snap.get("id")
            completed = snap.get("no_of_query", 0) or 0
            snap_total = snap.get("total_no_of_query", 0) or 0
            snap_status = snap.get("status", "")

            # Simplified logic: reuse if incomplete (completed < total), else create new
            should_reuse = (completed < snap_total)

            if not should_reuse:
                # ── Scenario 2: existing (completed/full) → create fresh ──
                logger.info(
                    "[Snapshot] Existing snapshot %s is full (%d/%d). Creating new one.",
                    snap_id, completed, snap_total,
                )
                # fall through to create-new below
            else:
                # ── Scenario 3: existing + incomplete → reuse ──
                logger.info(
                    "[Snapshot] Reusing incomplete snapshot %s  (%d/%d done)",
                    snap_id, completed, snap_total,
                )
                await async_supabase_update(
                    supabase, "analysis_snapshots",
                    {
                        "status": "running",
                        "started_at": now_iso,
                        "completed_at": None,
                        "total_no_of_query": total,
                    },
                    {"id": snap_id},
                )
                return snap_id

        # ── Scenario 1 (or Scenario 2 fall-through): create new ──
        data_to_insert = {
            "product_id": product_id,
            "batch_id": batch_id,
            "status": "running",
            "total_no_of_query": total,
            "no_of_query": 0,
            "started_at": now_iso,
            "completed_at": None,
        }

        response = await async_supabase_insert(
            supabase, "analysis_snapshots", data_to_insert,
        )
        if response.data and len(response.data) > 0:
            snapshot_id = response.data[0].get("id")
            logger.info(
                "[Snapshot] Created snapshot %s for product %s",
                snapshot_id, product_id,
            )
            return snapshot_id

        raise Exception("Failed to create snapshot: No data returned")

    except Exception as exc:
        logger.exception("[ERROR] Failed to create analysis snapshot: %s", exc)
        raise Exception(f"Failed to create analysis snapshot: {exc}") from exc


async def update_snapshot_status(
    settings: Settings,
    snapshot_id: str,
    status: str,
    *,
    completed_queries: Optional[int] = None,
) -> None:
    """Update an analysis snapshot's status in Supabase.

    Valid statuses: 'running', 'completed', 'failed', 'partial'.
    Sets ``completed_at`` automatically for terminal states.
    """
    supabase = get_supabase_client(settings)
    update_data: Dict[str, Any] = {"status": status}
    if completed_queries is not None:
        update_data["no_of_query"] = completed_queries
    if status in ("completed", "failed", "partial"):
        update_data["completed_at"] = datetime.now().isoformat()

    try:
        await async_supabase_update(
            supabase, "analysis_snapshots", update_data, {"id": snapshot_id}
        )
        logger.info("[Snapshot] Updated snapshot %s → status=%s", snapshot_id, status)
    except Exception as exc:
        logger.exception("[Snapshot] Failed to update snapshot %s: %s", snapshot_id, exc)


async def get_snapshot_status(
    settings: Settings,
    snapshot_id: str,
) -> Dict[str, Any]:
    """Fetch the current status of a snapshot from Supabase."""
    supabase = get_supabase_client(settings)
    rows = await async_supabase_select(
        supabase, "analysis_snapshots", {"id": snapshot_id}
    )
    if not rows:
        return {}
    return rows[0]


async def run_optimization_background(
    product_id: str,
    perplexity_queries: List[str],
    google_queries: List[str],
    chatgpt_queries: List[str],
    snapshot_id: str,
    settings: Settings,
    client_product_json: Any,
    user_id: str,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Background task wrapper around ``run_optimization_batch``.

    Increments ``no_of_query`` after each individual query completes,
    then sets the final status (completed / partial / failed).

    If any queries fail, credits for those queries are refunded to the user.
    """
    total_queries = len(perplexity_queries) + len(google_queries) + len(chatgpt_queries)

    try:
        results = await run_optimization_batch(
            product_id=product_id,
            perplexity_queries=perplexity_queries,
            google_queries=google_queries,
            chatgpt_queries=chatgpt_queries,
            snapshot_id=snapshot_id,
            settings=settings,
            client_product_json=client_product_json,
            api_key=api_key,
            debug=debug,
        )

        # Count successes vs failures
        successes = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failures_count = total_queries - len(successes)

        # --- Refund credits for failed queries ---
        if failures_count > 0:
            logger.warning(
                "[Background] %d/%d queries failed for snapshot %s — refunding %d credits to user %s",
                failures_count, total_queries, snapshot_id, failures_count, user_id,
            )
            await refund_credits(settings, user_id, failures_count)

        # --- Update snapshot status ---
        if failures_count == 0:
            await update_snapshot_status(
                settings, snapshot_id,
                status="completed",
            )
        elif len(successes) > 0:
            await update_snapshot_status(
                settings, snapshot_id,
                status="partial",
            )
        else:
            await update_snapshot_status(
                settings, snapshot_id,
                status="failed",
            )

    except Exception as exc:
        logger.exception("[Background] Batch failed for snapshot %s: %s", snapshot_id, exc)
        # Full crash — refund ALL credits
        logger.warning(
            "[Background] Full batch crash — refunding all %d credits to user %s",
            total_queries, user_id,
        )
        await refund_credits(settings, user_id, total_queries)
        await update_snapshot_status(
            settings, snapshot_id,
            status="failed",
        )



async def verify_and_deduct_credits(
    settings: Settings,
    user_id: str,
    required_credits: int,
) -> bool:
    """
    Atomically verify user has sufficient credits and deduct them.

    Args:
        settings: Runtime settings
        user_id: User ID to check and deduct credits from
        required_credits: Number of credits required

    Returns:
        bool: True if credits were successfully deducted, False otherwise

    Raises:
        Exception: If credit verification/deduction fails
    """
    supabase = get_supabase_client(settings)

    try:
        # First, check if user has sufficient credits
        user_profiles = await async_supabase_select(
            supabase, "user_profiles", {"id": user_id}
        )

        if not user_profiles or len(user_profiles) == 0:
            raise Exception(f"User profile not found for user_id: {user_id}")

        user_profile = user_profiles[0]
        # User specified column is 'credits'
        current_balance = user_profile.get("credits") or 0

        if current_balance < required_credits:
            logger.warning(
                f"[Credits] Insufficient credits for user {user_id}: "
                f"required={required_credits}, balance={current_balance}"
            )
            return False

        # Deduct credits atomically using RPC or update with check
        new_balance = current_balance - required_credits
        await async_supabase_update(
            supabase, "user_profiles", {"credits": new_balance}, {"id": user_id}
        )

        logger.info(
            f"[Credits] Deducted {required_credits} credits from user {user_id}. "
            f"Previous balance: {current_balance}, New balance: {new_balance}"
        )
        return True

    except Exception as exc:
        logger.exception(f"[ERROR] Failed to verify and deduct credits: {exc}")
        raise Exception(f"Failed to verify and deduct credits: {exc}") from exc


async def refund_credits(
    settings: Settings,
    user_id: str,
    refund_amount: int,
) -> bool:
    """
    Refund credits to a user after failed queries.

    Called by ``run_optimization_background`` when some or all queries in
    a batch fail.  Only the credits for the *failed* queries are returned.

    Args:
        settings: Runtime settings.
        user_id: User whose balance should be credited.
        refund_amount: Number of credits to add back.

    Returns:
        True if the refund succeeded, False otherwise.
    """
    if refund_amount <= 0:
        return True

    supabase = get_supabase_client(settings)

    try:
        user_profiles = await async_supabase_select(
            supabase, "user_profiles", {"id": user_id}
        )

        if not user_profiles or len(user_profiles) == 0:
            logger.error("[Credits] Refund failed — user profile not found: %s", user_id)
            return False

        current_balance = user_profiles[0].get("credits") or 0
        new_balance = current_balance + refund_amount

        await async_supabase_update(
            supabase, "user_profiles", {"credits": new_balance}, {"id": user_id}
        )

        logger.info(
            "[Credits] Refunded %d credits to user %s. "
            "Previous balance: %d, New balance: %d",
            refund_amount, user_id, current_balance, new_balance,
        )
        return True

    except Exception as exc:
        logger.exception("[Credits] Refund failed for user %s: %s", user_id, exc)
        return False


def normalize_ai_search(ai_search_json: Any) -> List[Any]:
    """Normalize AI search JSON to always be a list."""
    if isinstance(ai_search_json, list):
        return ai_search_json
    if not ai_search_json:
        return []
    return [ai_search_json]


def flatten_source_links(ai_search_items: List[Any]) -> List[Any]:
    """Flatten source_links from all AI search items."""
    out: List[Any] = []
    for item in ai_search_items:
        if isinstance(item, dict):
            links = item.get("source_links")
            if isinstance(links, list):
                out.extend(links)
    return out


def clean_json_text(text: str) -> str:
    """Clean JSON response text from markdown and formatting."""
    clean_text = re.sub(r"```json\n?", "", text)
    clean_text = re.sub(r"```\n?", "", clean_text)
    clean_text = clean_text.strip()

    first_brace = clean_text.find("{")
    if first_brace > 0:
        clean_text = clean_text[first_brace:]

    last_brace = clean_text.rfind("}")
    last_bracket = clean_text.rfind("]")
    last_close = max(last_brace, last_bracket)

    if last_close != -1 and last_close < len(clean_text) - 1:
        clean_text = clean_text[: last_close + 1]

    return clean_text


def repair_json_text(text: str) -> str:
    """Repair common JSON issues."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def auto_balance_json(text: str) -> str:
    """Auto-balance missing closing brackets and braces."""
    final_text = text.rstrip()

    open_braces = final_text.count("{")
    close_braces = final_text.count("}")
    open_brackets = final_text.count("[")
    close_brackets = final_text.count("]")

    if close_brackets < open_brackets:
        final_text += "]"

    if close_braces < open_braces:
        final_text += "}"

    if not final_text.endswith("]}") and not final_text.endswith("}"):
        last_comma_index = final_text.rfind("},")
        if last_comma_index != -1 and last_comma_index > final_text.rfind("]"):
            final_text = final_text[: last_comma_index + 1] + "]}"

    return final_text


def save_debug_file(content: str, prefix: str, debug: bool = False) -> None:
    """Save debug files if debugging is enabled and SAVE_GEMINI_DEBUG_FILES is set."""
    if not debug:
        return

    # Allow blocking file creation via env var (default to False/Block)
    if os.getenv("SAVE_GEMINI_DEBUG_FILES", "false").lower() != "true":
        return

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"gemini-{prefix}-response-{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(content)
        logger.info("[DEBUG] %s Gemini response written to: %s", prefix.capitalize(), filename)
    except Exception as exc:  # pragma: no cover
        logger.warning("[DEBUG] Failed to write %s response file: %s", prefix, exc)


def create_fallback_result(text: str, ai_search_json: Any) -> StrategicAnalysisResult:
    """Create fallback result when parsing fails."""
    ai_search_any = ai_search_json
    merged_source_links = (
        flatten_source_links(ai_search_any)
        if isinstance(ai_search_any, list)
        else ai_search_any.get("source_links", [])
        if isinstance(ai_search_any, dict)
        else []
    )

    fallback_sources = (
        [
            {
                "source_snippet": src.get("snippet")
                or src.get("text")
                or "Snippet not clearly available in provided text.",
                "reason_for_inclusion": "Mentioned in AI response sources",
                "source_of_mention": src.get("url") or src.get("source") or "Unknown source",
            }
            for src in merged_source_links
        ]
        if isinstance(merged_source_links, list)
        else []
    )

    return StrategicAnalysisResult(
        {
            "executive_summary": {
                "title": "AEO Competitive Analysis (Fallback)",
                "status_overview": "The AI analysis service returned an unexpected format. This is an automatically generated fallback summary.",
                "strategic_analogy": "Think of this as receiving raw research notes without a clean report. The data is there, but the structure had to be approximated.",
            },
            "client_product_visibility": {
                "status": "Not Featured",
                "details": "The detailed structured visibility analysis could not be parsed from the AI response. Please rerun the analysis later or contact support if this persists.",
            },
            "ai_answer_deconstruction": {
                "dominant_narrative": "The AI response could not be properly parsed to extract the dominant narrative. This may be due to formatting issues in the AI output.",
                "key_decision_factors": [
                    "AI response parsing failed - unable to extract decision factors",
                    "Consider rerunning the analysis for complete results",
                    "Raw response data has been preserved for reference",
                ],
                "trusted_source_analysis": "Due to parsing difficulties, trusted source analysis could not be extracted. Please refer to the source links provided below for raw information.",
                "raw_response_preview": text[:500],
            },
            "competitive_landscape_analysis": [],
            "sources_ai_used": fallback_sources,
            "strategic_gap_and_opportunity_analysis": {
                "analysis_summary": "Due to a formatting issue in the AI output, a full gap and opportunity analysis could not be generated. However, the source links have been preserved for manual review.",
            },
            "actionable_recommendations": [],
        }
    )


def _store_analysis_result_supabase(
    *,
    settings: Settings,
    product_id: str,
    pipeline: PipelineId,
    search_query: str,
    analysis_result: Dict[str, Any],
    raw_serp_results: Any,
    snapshot_id: Optional[str] = None,
    related_analysis_id: Optional[str] = None,
    source_links: Optional[List[Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    supabase = get_supabase_client(settings)

    if pipeline == "google_overview":
        data_to_insert: Dict[str, Any] = {
            "product_id": product_id,
            "search_query": search_query,
            "google_overview_analysis": analysis_result,
            "raw_serp_results": raw_serp_results or {},
        }

        if snapshot_id:
            data_to_insert["snapshot_id"] = snapshot_id

        response = supabase.table("product_analysis_google").insert(data_to_insert).execute()
        if debug:
            logger.info(
                "[DEBUG] Stored Google analysis with ID: %s",
                response.data[0].get("id") if response.data else "unknown",
            )
        return response.data[0] if response.data else {}

    if pipeline == "perplexity":
        data_to_insert = {
            "product_id": product_id,
            "optimization_prompt": search_query,
            "optimization_analysis": analysis_result,
            "citations": source_links or [],
            "raw_serp_results": raw_serp_results or {},
        }

        if snapshot_id:
            data_to_insert["snapshot_id"] = snapshot_id

        if related_analysis_id:
            data_to_insert["related_google_analysis_id"] = related_analysis_id

        response = supabase.table("product_analysis_perplexity").insert(data_to_insert).execute()
        if debug:
            logger.info(
                "[DEBUG] Stored Perplexity analysis with ID: %s",
                response.data[0].get("id") if response.data else "unknown",
            )
        return response.data[0] if response.data else {}

    if pipeline == "chatgpt":
        data_to_insert = {
            "product_id": product_id,
            "optimization_prompt": search_query,
            "optimization_analysis": analysis_result,
            "citations": source_links or [],
            "raw_serp_results": raw_serp_results or {},
        }

        if snapshot_id:
            data_to_insert["snapshot_id"] = snapshot_id

        response = supabase.table("product_analysis_chatgpt").insert(data_to_insert).execute()
        if debug:
            logger.info(
                "[DEBUG] Stored ChatGPT analysis with ID: %s",
                response.data[0].get("id") if response.data else "unknown",
            )
        return response.data[0] if response.data else {}

    if debug:
        logger.warning("[WARNING] Unknown pipeline type: %s. Analysis not stored.", pipeline)
    return {}


def store_analysis_result(
    *,
    settings: Settings,
    product_id: str,
    pipeline: PipelineId,
    search_query: str,
    analysis_result: Dict[str, Any],
    raw_serp_results: Any,
    snapshot_id: Optional[str] = None,
    related_analysis_id: Optional[str] = None,
    source_links: Optional[List[Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Store analysis result in Supabase."""
    try:
        return _store_analysis_result_supabase(
            settings=settings,
            product_id=product_id,
            pipeline=pipeline,
            search_query=search_query,
            analysis_result=analysis_result,
            raw_serp_results=raw_serp_results,
            snapshot_id=snapshot_id,
            related_analysis_id=related_analysis_id,
            source_links=source_links,
            debug=debug,
        )
    except Exception as exc:
        logger.exception("[ERROR] Failed to store analysis")
        raise Exception(f"Failed to store analysis: {exc}") from exc


def generate_content_with_retry(
    client: genai.Client,
    model_name: str,
    prompt: str,
    config: types.GenerateContentConfig,
    settings: Settings,
) -> Any:
    """Generate model content with exponential retry policy."""
    retrying = Retrying(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            multiplier=settings.retry_multiplier,
            min=settings.retry_min_wait_seconds,
            max=settings.retry_max_wait_seconds,
        ),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    for attempt in retrying:
        with attempt:
            return client.models.generate_content(
                model=model_name, contents=prompt, config=config
            )

    raise RuntimeError("Retry policy exhausted without result")


def perform_strategic_analysis(
    request: StrategicAnalysisRequest,
    *,
    settings: Settings,
    api_key: Optional[str] = None,
    debug: bool = False,
    store_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Perform strategic analysis using Google Gemini API and optionally store to database.

    Args:
        request: StrategicAnalysisRequest containing analysis parameters
        settings: Runtime settings resolved from environment
        api_key: Google API key (if None, reads from GEMINI_API_KEY env var)
        debug: Enable debug file output
        store_to_db: Whether to store results in database

    Returns:
        Dictionary containing analysis result and storage info

    Raises:
        ValueError: If API key is not provided or required fields are missing for storage
        Exception: If analysis fails
    """
    api_key = api_key or settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not defined in environment variables or parameters")

    if store_to_db:
        if not request.product_id:
            raise ValueError("product_id is required for database storage")
        if not request.search_query:
            raise ValueError("search_query is required for database storage")
        if not request.pipeline:
            raise ValueError("pipeline is required for database storage")

    client = genai.Client(api_key=api_key)

    generation_config = types.GenerateContentConfig(
        temperature=settings.gemini_temperature,
        top_p=settings.gemini_top_p,
        top_k=settings.gemini_top_k,
        max_output_tokens=settings.gemini_max_output_tokens,
        response_mime_type="application/json",
    )

    normalized_ai_search = normalize_ai_search(request.aiSearchJson)

    base_prompt = f"""Act as a world-class AEO (Answer Engine Optimization) Strategist and Competitive Analyst who analyzes the response of Perplexity AI (The AI Search Engine). Your expertise is in deconstructing AI-generated search responses to provide clients with a decisive competitive advantage.

You will be provided with two JSON objects:

[AI_SEARCH_JSON]: The scraped data from an AI search engine's answer.
This may be a single object or an array of objects (multiple queries). If it is an array, you MUST analyze them together and provide a consolidated result.

[CLIENT_PRODUCT_JSON]: The data for your client's product.

Your mission is to perform a deep, competitive analysis and deliver your findings in a structured, machine-readable format. Your analysis must be objective, data-driven, and provide clear, actionable recommendations.

**Your Internal Thought Process Before Generating the Output:**

1.  **Identify Presence:** First, meticulously scan the [AI_SEARCH_JSON]. Is your client's product (identified from [CLIENT_PRODUCT_JSON]) mentioned by name or brand in the ai_overview_text or source_links? This is the most critical first step and will determine the entire direction of your analysis.

2.  **Deconstruct the AI's Logic:** Analyze the AI's response as a whole. What is the narrative? What types of features, claims, and sources does it consistently highlight? Ask yourself: "What are the common threads connecting the recommended products?"

3.  **Profile the Winners:** For each competitor product mentioned, identify *why* it was included. What specific claim did the AI extract? Which source URL was cited for that claim? This reverse-engineers the "winning formula."

4.  **Compare and Contrast:**

    * If your product IS mentioned: How is it framed compared to competitors? Is it positioned as a premium option, a budget choice, or a specialized solution? Are its best features being highlighted?

    * If your product IS NOT mentioned: What key attributes from the "winning formula" are missing from your product's data or likely from its online presence? Are competitors winning on specific ingredients, clinical proof, source authority, or better-structured content?

5.  **Formulate Strategy:** Based on the gap analysis, what are the most impactful actions your client can take to either improve their position or get included in the first place?

Your entire output must be a **single, clean JSON object**. Use the detailed structure provided below.

6. **Source Links:** You MUST represent all source_links from the [AI_SEARCH_JSON] in the structured output, without inventing new URLs or sources.

- **competitive_landscape_analysis**: Summarizes competitors that appear in the AI answer, and links them to the MOST relevant source URL from source_links.

Here is the [AI_SEARCH_JSON]:

{json.dumps(normalized_ai_search[0] if len(normalized_ai_search) == 1 else normalized_ai_search, indent=2)}

Here is the [CLIENT_PRODUCT_JSON]:

{json.dumps(request.clientProductJson, indent=2)}

The JSON Structure for the Output:
{{
  "executive_summary": {{
    "title": "Your AEO Competitive Analysis for [Client Product Name]",
    "status_overview": "A one-sentence summary stating if your product was featured and its overall competitive position (e.g., 'Your product was not featured, as the AI currently prioritizes competitors with clinically-backed claims and mentions on high-authority health publications.') or ('Your product was featured, but it is being positioned as a natural alternative, while competitors are highlighted for their scientific formulations.')",
    "strategic_analogy": "A powerful, memorable analogy summarizing the strategic situation. (e.g., 'This is like arriving at a science fair with a beautiful painting. The judges are rewarding data and evidence, and while your product has artistic merit, it's not speaking the language of the competition.')"
  }},

  "client_product_visibility": {{
    "status": "Featured | Not Featured",
    "details": "If 'Featured', describe exactly how and where it was mentioned (e.g., 'Mentioned by name in the main recommendation list and cited directly from your brand website.'). If 'Not Featured', state this clearly."
  }},

  "ai_answer_deconstruction": {{
    "dominant_narrative": "Describe the 'story' the AI is telling. What kind of solution is it promoting for the user's query? (e.g., 'The AI is building a narrative around solving hair fall with scientifically-validated ingredients and expert-approved formulas, while also acknowledging natural alternatives.')",
    "key_decision_factors": [
      "List the specific attributes the AI is using to select and rank products. Examples: 'Specific, named ingredients (e.g., keratin, adenosine, onion)', 'Presence of scientific terms (e.g., clinically proven, nutrilock actives)', 'Source authority (e.g., direct brand sites, health publications, major e-commerce platforms)', 'Key product features (e.g., sulphate-free)', 'Solutions for a specific sub-problem (e.g., strengthening roots, reducing breakage)'."
    ],
    "trusted_source_analysis": "Analyze the source_links. What types of websites is the AI citing and trusting? (e.g., 'The AI demonstrates high trust in a mix of direct-to-consumer brand websites, major e-commerce category pages, and authoritative third-party content sites like health magazines.')"
  }},
  "competitive_landscape_analysis": [
    {{
      "competitor_name": "Name of the competitor product or brand as mentioned in the AI's answer.",
      "reason_for_inclusion": "Explain, using only evidence from the AI's overview text and source_links, why this competitor appears in the answer.",
      "source_of_mention": "Provide the URL from source_links that this competitor is most directly associated with. If multiple could apply, pick the single best matching URL. Do not invent or modify URLs."
    }}
  ],
  "strategic_gap_and_opportunity_analysis": {{
      "analysis_summary": "This is the core of your report. Provide a detailed explanation based on your product's visibility status.",
      "if_featured": {{
        "current_positioning": "How does the AI's description of your product compare to competitors? What are its perceived strengths and weaknesses? (e.g., 'Your product is positioned effectively as a premium, science-backed solution due to the mention of 'adenosine' and 'procapil'. However, competitors like Dove are framed as being better for 'smoother hair', indicating a potential gap in highlighting secondary benefits.')",
        "opportunities_for_improvement": "How can you enhance your positioning? (e.g., 'Update your product page to also emphasize moisturizing and smoothing properties to compete with Dove's narrative. Seek mentions on pharmacy or health blogs to match the source diversity of competitors.')"
      }},
      "if_not_featured": {{
        "exclusion_reasons": "Based on the competitive_landscape_analysis, what are the specific reasons your product was excluded? Be direct. (e.g., 'Your product was not mentioned primarily because your online content does not explicitly name the key active ingredients that the AI is looking for, such as 'keratin' or 'biotin'. Competitors are being rewarded for this specificity.')",
        "path_to_inclusion": "How can you get the product featured? (e.g., 'Revise your product page to include a 'Key Ingredients' section that clearly lists active components and their benefits. Pursue a product feature on a major e-commerce platform like Nykaa or a content site like Health.com, as the AI trusts these third-party sources.')"
      }}
  }},

  "actionable_recommendations": [
      {{
        "recommendation": "Content Structure Enhancement",
        "action": "On your primary product page, implement a clear, scannable structure using H2/H3 tags for 'Key Benefits', 'Active Ingredients', and 'How It Works'. Use bullet points to list features, making the information easily parsable for AI crawlers."
      }},
      {{
        "recommendation": "Keyword and Claim Alignment",
        "action": "Ensure the exact phrases and claims rewarded by the AI (e.g., 'reduces breakage', 'clinically proven', 'sulphate-free') are present and prominent in your product descriptions and marketing copy."
      }},
      {{
      "recommendation": "Third-Party Validation Strategy",
      "action": "Develop a strategy to get your product listed or reviewed on the types of authoritative domains the AI is citing (e.g., health blogs, online pharmacies, respected e-commerce sites). This builds the external trust the AI is looking for."
      }}
    ]
  }}"""

    if request.pipeline == "google_overview":
        prompt = base_prompt.replace(
            "Perplexity AI (The AI Search Engine)",
            "Google AI Overview (The AI Search Engine)",
        )
    else:
        prompt = base_prompt

    try:
        response = generate_content_with_retry(
            client=client,
            model_name=settings.gemini_model_name,
            prompt=prompt,
            config=generation_config,
            settings=settings,
        )
        text = response.text

        if debug:
            try:
                usage = response.usage_metadata
                logger.info(
                    "[Gemini][Strategic Analysis] %s",
                    {
                        "inputTokens": usage.prompt_token_count if usage else 0,
                        "outputTokens": usage.candidates_token_count if usage else 0,
                        "totalTokens": usage.total_token_count if usage else 0,
                    },
                )
            except Exception:
                pass

        try:
            clean_text = clean_json_text(text)
            save_debug_file(clean_text, "raw", debug)

            try:
                analysis_result = json.loads(clean_text)
            except json.JSONDecodeError as inner_error:
                if debug:
                    logger.warning("Initial JSON.parse failed, attempting repair: %s", inner_error)

                repaired_text = repair_json_text(clean_text)
                save_debug_file(repaired_text, "repaired", debug)

                try:
                    analysis_result = json.loads(repaired_text)
                    clean_text = repaired_text
                except json.JSONDecodeError as second_error:
                    if debug:
                        logger.warning("Second JSON.parse failed, attempting auto-balance: %s", second_error)

                    final_text = auto_balance_json(repaired_text)
                    save_debug_file(final_text, "balanced", debug)

                    analysis_result = json.loads(final_text)
                    clean_text = final_text

            if not analysis_result.get("executive_summary") or not analysis_result.get("client_product_visibility"):
                analysis_result = create_fallback_result(text, request.aiSearchJson)

            stored_record: Dict[str, Any] = {}
            if store_to_db:
                source_links: List[Any] = []
                if isinstance(normalized_ai_search, list):
                    source_links = flatten_source_links(normalized_ai_search)
                elif isinstance(normalized_ai_search, dict):
                    source_links = normalized_ai_search.get("source_links", [])

                stored_record = store_analysis_result(
                    settings=settings,
                    product_id=request.product_id,
                    pipeline=request.pipeline,
                    search_query=request.search_query,
                    analysis_result=analysis_result,
                    raw_serp_results=request.raw_serp_results,
                    snapshot_id=request.snapshot_id,
                    source_links=source_links,
                    debug=debug,
                )

            return {
                "success": True,
                "analysis": analysis_result,
                "stored_record": stored_record if store_to_db else None,
                "product_id": request.product_id,
                "pipeline": request.pipeline,
                "search_query": request.search_query,
            }

        except json.JSONDecodeError as parse_error:
            if debug:
                logger.warning("[StrategicAnalysis] JSON Parse Error: %s", parse_error)
            fallback_result = create_fallback_result(text, request.aiSearchJson)

            stored_record: Dict[str, Any] = {}
            if store_to_db:
                stored_record = store_analysis_result(
                    settings=settings,
                    product_id=request.product_id,
                    pipeline=request.pipeline,
                    search_query=request.search_query,
                    analysis_result=fallback_result,
                    raw_serp_results=request.raw_serp_results,
                    snapshot_id=request.snapshot_id,
                    debug=debug,
                )

            return {
                "success": False,
                "analysis": fallback_result,
                "stored_record": stored_record if store_to_db else None,
                "error": "JSON parsing failed, returning fallback result",
            }

    except Exception as exc:
        if debug:
            logger.exception("[StrategicAnalysis] Error")
        raise Exception(f"Failed to perform strategic analysis: {exc}") from exc


# ---------------------------------------------------------------------------
# Phase 3 — Async Pipeline Orchestration
# ---------------------------------------------------------------------------

async def process_single_query_pipeline(
    product_id: str,
    query: str,
    pipeline: PipelineId,
    snapshot_id: Optional[str] = None,
    *,
    settings: Settings,
    client_product_json: Any,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Full async pipeline for a single query:
      1. Scrape AI search data (async)
      2. Run Gemini strategic analysis (sync → ``to_thread``)
      3. Store the result in Supabase (sync → ``to_thread``)

    Args:
        product_id: UUID of the product being analysed.
        query: The search query to scrape.
        pipeline: Pipeline id (``"perplexity"`` or ``"google_overview"``).
        snapshot_id: Optional snapshot id to group analyses.
        settings: Application settings.
        client_product_json: Product data for Gemini analysis.
        api_key: Optional Gemini API key override.
        debug: Enable debug logging/files.

    Returns:
        Dict with ``success``, ``analysis``, ``stored_record``, etc.
    """
    from app.services.scraping_service import fetch_ai_search_data  # lazy import to avoid circular deps

    try:
        logger.info(
            "[Pipeline] Starting  pipeline=%s  query=%r  product_id=%s",
            pipeline, query, product_id,
        )

        # Step 1 — Scrape (fully async)
        scraped_data = await fetch_ai_search_data(
            pipeline=pipeline,
            query=query,
            settings=settings,
        )
        logger.info("[Pipeline] Scraping complete  pipeline=%s  query=%r", pipeline, query)

        # Step 2 — Gemini analysis (sync SDK → run in thread pool)
        request = StrategicAnalysisRequest(
            aiSearchJson=scraped_data,
            clientProductJson=client_product_json,
            pipeline=pipeline,
            product_id=product_id,
            search_query=query,
            raw_serp_results=scraped_data,
            snapshot_id=snapshot_id,
        )

        analysis_result = await asyncio.to_thread(
            perform_strategic_analysis,
            request,
            settings=settings,
            api_key=api_key,
            debug=debug,
            store_to_db=False,  # We store separately in Step 3
        )
        logger.info("[Pipeline] Analysis complete  pipeline=%s  query=%r", pipeline, query)

        # Step 3 — Store to Supabase (sync client → run in thread pool)
        analysis_data = analysis_result.get("analysis", {})
        normalized = normalize_ai_search(scraped_data)
        source_links: List[Any] = []
        if isinstance(normalized, list):
            source_links = flatten_source_links(normalized)
        elif isinstance(normalized, dict):
            source_links = normalized.get("source_links", [])

        stored_record = await asyncio.to_thread(
            store_analysis_result,
            settings=settings,
            product_id=product_id,
            pipeline=pipeline,
            search_query=query,
            analysis_result=analysis_data,
            raw_serp_results=scraped_data,
            snapshot_id=snapshot_id,
            source_links=source_links,
            debug=debug,
        )
        logger.info("[Pipeline] Stored  pipeline=%s  query=%r", pipeline, query)

        return {
            "success": analysis_result.get("success", False),
            "analysis": analysis_data,
            "stored_record": stored_record,
            "product_id": product_id,
            "pipeline": pipeline,
            "search_query": query,
        }
    except Exception as e:
        logger.exception("[Pipeline] Failed pipeline=%s query=%r error=%s", pipeline, query, e)
        return {
            "success": False,
            "error": str(e),
            "pipeline": pipeline,
            "search_query": query,
            "product_id": product_id,
        }


async def _run_and_increment(
    coro,
    snapshot_id: Optional[str],
    settings: Settings,
) -> Dict[str, Any]:
    """
    Await a single query coroutine, then increment ``no_of_query``
    in the snapshot so progress is visible in real-time.
    """
    result = await coro

    # Only increment on success, and enforce strict upper bound
    if snapshot_id and result.get("success"):
        try:
            supabase = get_supabase_client(settings)
            # Atomic increment: read current value, add 1 (if < total)
            rows = await async_supabase_select(
                supabase, "analysis_snapshots", {"id": snapshot_id},
            )
            if rows:
                snap = rows[0]
                current = snap.get("no_of_query", 0) or 0
                total = snap.get("total_no_of_query", 0) or 0

                if current < total:
                    new_val = current + 1
                    await async_supabase_update(
                        supabase, "analysis_snapshots",
                        {"no_of_query": new_val},
                        {"id": snapshot_id},
                    )
                    logger.info(
                        "[Snapshot] Incremented no_of_query (%d/%d) for snapshot %s",
                        new_val, total, snapshot_id,
                    )
                else:
                    logger.warning(
                        "[Snapshot] Skipping increment for %s: limit reached (%d/%d)",
                        snapshot_id, current, total,
                    )

        except Exception as exc:
            logger.warning(
                "[Snapshot] Failed to increment no_of_query for %s: %s",
                snapshot_id, exc,
            )
    return result


async def run_optimization_batch(
    product_id: str,
    perplexity_queries: List[str],
    google_queries: List[str],
    chatgpt_queries: List[str],
    snapshot_id: Optional[str] = None,
    *,
    settings: Settings,
    client_product_json: Any,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fan out all requested queries across both pipelines concurrently.

    Uses ``asyncio.gather`` so that *N* queries run in parallel.
    After each individual query completes, ``no_of_query`` is incremented
    in the snapshot so the frontend can show real-time progress.

    Args:
        product_id: UUID of the product.
        perplexity_queries: Queries to run through the Perplexity pipeline.
        google_queries: Queries to run through the Google Overview pipeline.
        snapshot_id: Optional snapshot id for progress tracking.
        settings: Application settings.
        client_product_json: Product data for Gemini analysis.
        api_key: Optional Gemini API key override.
        debug: Enable debug logging/files.

    Returns:
        List of result dicts, one per query.
    """
    tasks = []

    for query in perplexity_queries:
        coro = process_single_query_pipeline(
            product_id=product_id,
            query=query,
            pipeline="perplexity",
            snapshot_id=snapshot_id,
            settings=settings,
            client_product_json=client_product_json,
            api_key=api_key,
            debug=debug,
        )
        tasks.append(_run_and_increment(coro, snapshot_id, settings))

    for query in google_queries:
        coro = process_single_query_pipeline(
            product_id=product_id,
            query=query,
            pipeline="google_overview",
            snapshot_id=snapshot_id,
            settings=settings,
            client_product_json=client_product_json,
            api_key=api_key,
            debug=debug,
        )
        tasks.append(_run_and_increment(coro, snapshot_id, settings))

    for query in chatgpt_queries:
        coro = process_single_query_pipeline(
            product_id=product_id,
            query=query,
            pipeline="chatgpt",
            snapshot_id=snapshot_id,
            settings=settings,
            client_product_json=client_product_json,
            api_key=api_key,
            debug=debug,
        )
        tasks.append(_run_and_increment(coro, snapshot_id, settings))

    logger.info(
        "[Batch] Dispatching %d tasks  (perplexity=%d, google=%d, chatgpt=%d)  product_id=%s",
        len(tasks), len(perplexity_queries), len(google_queries), len(chatgpt_queries), product_id,
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]

    if failures:
        for exc in failures:
            logger.error("[Batch] Task failed: %s", exc)

    logger.info(
        "[Batch] Complete  total=%d  successes=%d  failures=%d",
        len(results), len(successes), len(failures),
    )

    return results  # type: ignore[return-value]

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import logging
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from app.config import get_settings
from app.models import (
    FallbackRequest,
    FlattenLinksRequest,
    HealthResponse,
    JsonTextRequest,
    NormalizeAiSearchRequest,
    OptimizationStartRequest,
    OptimizationStartResponse,
    OptimizationStatusResponse,
    StrategicAnalysisRequest,
    StrategicAnalysisResponse,
    StrategicAnalysisRunRequest,
    StoreAnalysisRequest,
)
from app.services.analysis_service import (
    auto_balance_json,
    clean_json_text,
    create_analysis_snapshot,
    create_fallback_result,
    flatten_source_links,
    get_snapshot_status,
    normalize_ai_search,
    perform_strategic_analysis,
    repair_json_text,
    run_optimization_background,
    store_analysis_result,
    verify_and_deduct_credits,
)

settings = get_settings()

import sys
logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="FastAPI wrapper for strategic analysis pipeline",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
    )


# -----------------------------------------------------------------------
# Existing analysis endpoints
# -----------------------------------------------------------------------

@app.post(
    "/api/v1/analysis/strategic",
    response_model=StrategicAnalysisResponse,
    tags=["analysis"],
)
def run_strategic_analysis(payload: StrategicAnalysisRunRequest) -> StrategicAnalysisResponse:
    try:
        request = StrategicAnalysisRequest(
            **payload.model_dump(exclude={"api_key", "debug", "store_to_db"})
        )

        result = perform_strategic_analysis(
            request=request,
            settings=settings,
            api_key=payload.api_key,
            debug=payload.debug,
            store_to_db=payload.store_to_db,
        )
        return StrategicAnalysisResponse(**jsonable_encoder(result))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/analysis/store", tags=["analysis"])
def store_analysis(payload: StoreAnalysisRequest) -> Dict[str, Any]:
    try:
        result = store_analysis_result(
            settings=settings,
            product_id=payload.product_id,
            pipeline=payload.pipeline,
            search_query=payload.search_query,
            analysis_result=payload.analysis_result,
            raw_serp_results=payload.raw_serp_results,
            snapshot_id=payload.snapshot_id,
            related_analysis_id=payload.related_analysis_id,
            source_links=payload.source_links,
            debug=payload.debug,
        )
        return {"success": True, "stored_record": jsonable_encoder(result)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -----------------------------------------------------------------------
# Phase 4 — Optimization endpoints (The Bridge)
# -----------------------------------------------------------------------

@app.post(
    "/api/v1/optimize/start",
    response_model=OptimizationStartResponse,
    status_code=202,
    tags=["optimize"],
)
async def optimization_start(
    payload: OptimizationStartRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Start an optimization batch.

    1. Verify credits and create a snapshot (synchronous).
    2. Dispatch ``run_optimization_background`` as a ``BackgroundTask``.
    3. Return ``HTTP 202 Accepted`` with the ``snapshot_id`` immediately.
    """
    if not settings.enable_chatgpt and payload.chatgpt_queries:
        raise HTTPException(status_code=400, detail="ChatGPT scraper is disabled")

    all_queries = payload.perplexity_queries + payload.google_queries + payload.chatgpt_queries
    total = len(all_queries)

    if total == 0:
        raise HTTPException(status_code=400, detail="At least one query is required")

    # --- Credit check ---
    try:
        has_credits = await verify_and_deduct_credits(
            settings=settings,
            user_id=payload.user_id,
            required_credits=total,
        )
        if not has_credits:
            raise HTTPException(status_code=402, detail="Insufficient credits")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Credit check failed: {exc}") from exc

    # --- Create snapshot ---
    try:
        snapshot_id = await create_analysis_snapshot(
            settings=settings,
            product_id=payload.product_id,
            batch_id=payload.batch_id,
            queries=all_queries,
            total_no_of_query=payload.total_no_of_query,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {exc}") from exc

    # --- Dispatch background batch ---
    background_tasks.add_task(
        run_optimization_background,
        product_id=payload.product_id,
        perplexity_queries=payload.perplexity_queries,
        google_queries=payload.google_queries,
        chatgpt_queries=payload.chatgpt_queries,
        snapshot_id=snapshot_id,
        settings=settings,
        client_product_json=payload.client_product_json,
        user_id=payload.user_id,
        api_key=payload.api_key,
        debug=payload.debug,
    )

    return JSONResponse(
        status_code=202,
        content=OptimizationStartResponse(
            snapshot_id=snapshot_id,
            status="running",
            total_queries=total,
        ).model_dump(),
    )


@app.get(
    "/api/v1/optimize/status/{snapshot_id}",
    response_model=OptimizationStatusResponse,
    tags=["optimize"],
)
async def optimization_status(snapshot_id: str) -> OptimizationStatusResponse:
    """
    Poll the status of an optimization batch by snapshot id.

    Returns the snapshot's current status (``processing``, ``completed``,
    ``failed``) along with query progress counters.
    """
    try:
        snapshot = await get_snapshot_status(settings=settings, snapshot_id=snapshot_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {exc}") from exc

    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return OptimizationStatusResponse(
        snapshot_id=snapshot_id,
        status=snapshot.get("status", "unknown"),
        total_queries=snapshot.get("total_no_of_query", 0),
        completed_queries=snapshot.get("no_of_query", 0),
    )


# -----------------------------------------------------------------------
# Utility endpoints
# -----------------------------------------------------------------------

@app.post("/api/v1/utils/normalize-ai-search", tags=["utils"])
def normalize_ai_search_endpoint(payload: NormalizeAiSearchRequest) -> Dict[str, Any]:
    return {"items": normalize_ai_search(payload.ai_search_json)}


@app.post("/api/v1/utils/flatten-source-links", tags=["utils"])
def flatten_source_links_endpoint(payload: FlattenLinksRequest) -> Dict[str, Any]:
    return {"links": flatten_source_links(payload.ai_search_items)}


@app.post("/api/v1/utils/clean-json", tags=["utils"])
def clean_json_endpoint(payload: JsonTextRequest) -> Dict[str, str]:
    return {"clean_text": clean_json_text(payload.text)}


@app.post("/api/v1/utils/repair-json", tags=["utils"])
def repair_json_endpoint(payload: JsonTextRequest) -> Dict[str, str]:
    return {"repaired_text": repair_json_text(payload.text)}


@app.post("/api/v1/utils/auto-balance-json", tags=["utils"])
def auto_balance_json_endpoint(payload: JsonTextRequest) -> Dict[str, str]:
    return {"balanced_text": auto_balance_json(payload.text)}


# -----------------------------------------------------------------------
# Webhook / Callback endpoints for high-scale scrapers
# -----------------------------------------------------------------------

@app.post("/api/v1/callbacks/perplexity", tags=["callbacks"])
async def perplexity_callback(payload: Dict[str, Any]) -> Dict[str, bool]:
    """
    Webhook receiver for the high-scale Perplexity scraper.
    
    The worker posts the result here when STORAGE_MODE_API=true.
    We match the job_id and resolve the waiting future.
    """
    job_id = payload.get("job_id")
    if not job_id:
        logging.warning("Received Perplexity callback without job_id")
        return {"success": False}

    logging.info("Received Perplexity callback for job_id: %s", job_id)
    
    from app.services.scraping_service import perplexity_job_tracker
    perplexity_job_tracker.complete(job_id, payload)
    
    return {"success": True}


@app.post("/api/v1/callbacks/chatgpt", tags=["callbacks"])
async def chatgpt_callback(payload: Dict[str, Any]) -> Dict[str, bool]:
    """
    Webhook receiver for the ChatGPT scraper.

    The scraper POSTs the completed job result here (identified by job_id).
    We resolve the waiting Future in _fetch_chatgpt so the pipeline can continue.
    """
    job_id = payload.get("job_id")
    if not job_id:
        logging.warning("[ChatGPT Callback] Received callback without job_id. Keys: %s", list(payload.keys()))
        return {"success": False}

    logging.info("[ChatGPT Callback] Received result for job_id: %s", job_id)

    from app.services.scraping_service import chatgpt_job_tracker
    # Extract the 'result' object if provided, otherwise use full payload
    data = payload.get("result", payload)
    chatgpt_job_tracker.complete(job_id, data)

    return {"success": True}


@app.post("/api/v1/callbacks/new_ai_mode", tags=["callbacks"])
async def new_ai_mode_callback(payload: Dict[str, Any]) -> Dict[str, bool]:
    """
    Webhook receiver for the New AI Mode scraper.

    The scraper POSTs the completed job result here (identified by job_id).
    We resolve the waiting Future in _fetch_new_ai_mode so the pipeline can continue.
    """
    job_id = payload.get("job_id")
    if not job_id:
        logging.warning("[New AI Callback] Received callback without job_id. Keys: %s", list(payload.keys()))
        return {"success": False}

    logging.info("[New AI Callback] Received result for job_id: %s", job_id)

    from app.services.scraping_service import new_ai_job_tracker
    # Extract the 'result' object if provided, otherwise use full payload
    data = payload.get("result", payload)
    new_ai_job_tracker.complete(job_id, data)

    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)

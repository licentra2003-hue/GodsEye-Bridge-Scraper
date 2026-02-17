from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PipelineId = Literal["perplexity", "google_overview", "new_ai_mode", "chatgpt", "gemini"]


class StrategicAnalysisRequest(BaseModel):
    aiSearchJson: Any
    clientProductJson: Any
    analysisId: Optional[str] = None
    pipeline: Optional[PipelineId] = None
    product_id: Optional[str] = None
    search_query: Optional[str] = None
    raw_serp_results: Optional[Any] = None
    snapshot_id: Optional[str] = None


class StrategicAnalysisRunRequest(StrategicAnalysisRequest):
    api_key: Optional[str] = None
    debug: bool = False
    store_to_db: bool = True


class StoreAnalysisRequest(BaseModel):
    product_id: str
    pipeline: PipelineId
    search_query: str
    analysis_result: Dict[str, Any]
    raw_serp_results: Any
    snapshot_id: Optional[str] = None
    related_analysis_id: Optional[str] = None
    source_links: Optional[List[Any]] = None
    debug: bool = False


class StrategicAnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    stored_record: Optional[Dict[str, Any]] = None
    product_id: Optional[str] = None
    pipeline: Optional[PipelineId] = None
    search_query: Optional[str] = None
    error: Optional[str] = None


class JsonTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class NormalizeAiSearchRequest(BaseModel):
    ai_search_json: Any


class FlattenLinksRequest(BaseModel):
    ai_search_items: List[Any]


class FallbackRequest(BaseModel):
    text: str
    ai_search_json: Any


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class OptimizationStartRequest(BaseModel):
    """Payload for ``POST /api/v1/optimize/start``."""
    product_id: str
    user_id: str              # Used for credit checks only, NOT stored in snapshot
    batch_id: str             # Must already exist in query_batches table
    perplexity_queries: List[str] = Field(default_factory=list)
    google_queries: List[str] = Field(default_factory=list)
    client_product_json: Any = None
    api_key: Optional[str] = None
    debug: bool = False
    total_no_of_query: Optional[int] = None  # Allow frontend to override total count


class OptimizationStartResponse(BaseModel):
    """Response for ``POST /api/v1/optimize/start``."""
    snapshot_id: str
    status: str = "running"
    total_queries: int


class OptimizationStatusResponse(BaseModel):
    """Response for ``GET /api/v1/optimize/status/{snapshot_id}``."""
    snapshot_id: str
    status: str
    total_queries: int = 0
    completed_queries: int = 0

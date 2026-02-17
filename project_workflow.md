# GodsEye Project Workflow Documentation

This document outlines the end-to-end workflow of the GodsEye backend system, detailing the input parameters, processing stages, and final output structure.

## 1. High-Level Overview

The GodsEye backend is a FastAPI-based application designed to:
1.  **Receive** a product optimization request.
2.  **Scrape** search engine results (Perplexity, Google Overview) for specific queries.
3.  **Analyze** the scraped data using Google Gemini to identify competitive gaps and opportunities.
4.  **Store** the analysis results in a Supabase database.
5.  **Expose** the status and results via API endpoints.

## 2. Input Phase

The workflow begins with a `POST` request to the `/api/v1/optimize/start` endpoint.

### Endpoint: `POST /api/v1/optimize/start`

**Payload (`OptimizationStartRequest`):**

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `product_id` | `uuid` | Yes | Unique identifier for the product being analyzed. |
| `user_id` | `uuid` | Yes | ID of the user initiating the request (used for credit deduction). |
| `batch_id` | `uuid` | Yes | ID linking the request to a specific batch of queries. |
| `perplexity_queries` | `List[str]` | No* | List of search queries to run on Perplexity AI. (*At least one query list must be non-empty*) |
| `google_queries` | `List[str]` | No* | List of search queries to run on Google Overview (AI Mode). (*At least one query list must be non-empty*) |
| `client_product_json` | `Any` | Yes | JSON object containing product details (name, description, features) used for Gemini analysis. |
| `api_key` | `str` | No | Optional API key override for Gemini. |
| `debug` | `bool` | No | If `true`, enables verbose logging (and potential debug file creation if enabled via env). |

**Response (`202 Accepted`):**
Returns a `snapshot_id` immediately, indicating the background job has started.

```json
{
  "snapshot_id": "uuid-string",
  "status": "running",
  "total_queries": 5
}
```

## 3. Orchestration Phase

Upon receiving the request, `app/main.py`:
1.  **Validates Credits:** Calls `verify_and_deduct_credits` to ensure the user has enough balance for the total number of queries.
2.  **Creates Snapshot:** Inserts a new record into the `query_snapshots` table in Supabase via `create_analysis_snapshot`. This tracks the overall progress of the batch.
3.  **Dispatches Task:** Launches `run_optimization_background` as a FastAPI BackgroundTask. This allows the API to respond immediately while processing continues asynchronously.

## 4. Processing Phase

The core logic resides in `app/services/analysis_service.py` within `run_optimization_background`. This function iterates through all provided queries (Perplexity and Google) and processes them **concurrently**.

### 4.1. The Pipeline (`process_single_query_pipeline`)

For each query, the following steps are executed:

#### Step A: Scraping (`fetch_ai_search_data`)
*   **Source:** `app/services/scraping_service.py`
*   **Action:** Sends requests to the configured scraper endpoints.
*   **Retries:** Implements robust retry logic (via `tenacity`) for handling network errors, bot detection, or empty responses.
*   **Pipelines:**
    *   **Perplexity:** Direct scraping or API call to Perplexity.
    *   **Google Overview:** Uses "New AI Mode" which involves a Submit → Poll architecture (`_submit_new_ai_job` followed by `_poll_job_result`).

#### Step B: Strategic Analysis (`perform_strategic_analysis`)
*   **Source:** `app/services/analysis_service.py` (logic shared with `strategic-analysis-2.py`)
*   **Engine:** Google Gemini (via `google-genai` SDK).
*   **Input:**
    *   `aiSearchJson`: The raw data scraped in Step A.
    *   `clientProductJson`: The product details provided in the initial input.
*   **Prompting:** Uses a complex prompt to act as an "AEO Strategist", analyzing visibility, competitive landscape, and strategic gaps.
*   **Output:** A structured JSON object containing:
    *   `executive_summary`
    *   `client_product_visibility`
    *   `ai_answer_deconstruction`
    *   `competitive_landscape_analysis`
    *   `strategic_gap_and_opportunity_analysis`
    *   `actionable_recommendations`

#### Step C: Storage (`store_analysis_result`)
*   **Source:** `app/services/analysis_service.py`
*   **Destination:** Supabase Database.
*   **Tables:**
    *   `product_analysis_google`: Stores results for Google Overview queries.
    *   `product_analysis_perplexity`: Stores results for Perplexity queries.
*   **Data Stored:** Includes the full analysis JSON, the raw SERP results (scraping output), the search query, and the `snapshot_id`.

## 5. Output Phase

The frontend polls for the status of the batch using the `snapshot_id`.

### Endpoint: `GET /api/v1/optimize/status/{snapshot_id}`

**Response (`OptimizationStatusResponse`):**

```json
{
  "snapshot_id": "uuid-string",
  "status": "processing",
  "total_queries": 5,
  "completed_queries": 3
}
```

Once `status` is `completed`, the frontend retrieves the detailed results directly from the Supabase tables (`product_analysis_google` / `product_analysis_perplexity`) using the `snapshot_id`.

## 6. Error Handling

*   **Pipeline Failures:** If a single query fails (scraping error, analysis error), it is caught within `process_single_query_pipeline`. The error is logged, and a failure record is returned to the batch processor. This ensures one bad query does not crash the entire batch.
*   **Batch Status:** If individual queries fail, the snapshot status may be marked as `partial` or `failed` depending on the aggregate result.
*   **Debug Files:** If `SAVE_GEMINI_DEBUG_FILES` is set to `true` in `.env` AND `debug=true` is passed in the request, raw Gemini responses are saved to disk for debugging.

## 7. Infrastructure Components

*   **FastAPI:** Web framework.
*   **Supabase:** Database and Auth.
*   **Google Gemini:** LLM for analysis.
*   **Tenacity:** Retry library.
*   **Httpx:** Async HTTP client.

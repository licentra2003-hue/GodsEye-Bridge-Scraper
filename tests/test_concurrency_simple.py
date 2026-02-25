import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.analysis_service import run_optimization_batch
from app.config import Settings

@pytest.mark.asyncio
async def test_simple_concurrency_modes():
    print("\n\n=== STARTING SIMPLE CONCURRENCY TEST ===")
    
    # 1. Setup: Define 5 queries across 2 modes
    perplexity_input = ["perp_q1", "perp_q2", "perp_q3"]
    google_input = ["goog_q1", "goog_q2"]
    total_queries = len(perplexity_input) + len(google_input)
    
    # Mock settings
    settings = MagicMock(spec=Settings)
    
    # Delay for each query (simulating network/AI work)
    QUERY_DELAY = 1.0
    
    # Mock the processor to simply sleep
    async def mock_processor(pipeline, query, **kwargs):
        print(f"  [START] {pipeline.upper()}: {query}")
        await asyncio.sleep(QUERY_DELAY)
        print(f"  [ END ] {pipeline.upper()}: {query}")
        return {"success": True, "pipeline": pipeline, "query": query}

    # Mock DB to prevent errors in progress tracking
    mock_supabase = MagicMock()
    
    # Patch dependencies
    with patch("app.services.analysis_service.process_single_query_pipeline", side_effect=mock_processor), \
         patch("app.services.analysis_service.get_supabase_client", return_value=mock_supabase), \
         patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select, \
         patch("app.services.analysis_service.async_supabase_update", new_callable=AsyncMock) as mock_update:
        
        # Mock DB return for increment logic
        mock_select.return_value = [{"no_of_query": 0}]
        
        print(f"\nDisptaching {total_queries} queries (3 Perplexity, 2 Google)...")
        print(f"Each query takes {QUERY_DELAY}s. Sequential would take {total_queries * QUERY_DELAY}s.")
        
        start_time = time.time()
        
        # 2. Execute Batch
        results = await run_optimization_batch(
            product_id="test-prod-123",
            perplexity_queries=perplexity_input,
            google_queries=google_input,
            chatgpt_queries=[],
            snapshot_id="snap-test-123",  # triggers DB increment logic
            settings=settings,
            client_product_json={},
        )
        
        duration = time.time() - start_time
        
    # 3. Validation
    print(f"\n=== RESULTS ===")
    print(f"Total Results: {len(results)}")
    print(f"Execution Time: {duration:.3f} seconds")
    
    # Assertions
    assert len(results) == total_queries, f"Expected {total_queries} results, got {len(results)}"
    
    # Verify concurrency: Duration should be roughly equal to single query delay (plus overhead), 
    # definitely much less than sequential sum.
    assert duration < (total_queries * QUERY_DELAY * 0.5), \
        f"Too slow! Took {duration}s, expected < {total_queries * QUERY_DELAY * 0.5}s"
        
    print(f"[OK] PASSED: Execution was concurrent (took {duration:.3f}s, much less than {total_queries * QUERY_DELAY}s)")

if __name__ == "__main__":
    asyncio.run(test_simple_concurrency_modes())

import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load real environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

from app.config import Settings
from app.services.analysis_service import (
    create_analysis_snapshot,
    run_optimization_background,
    get_snapshot_status,
)

# ---------------------------------------------------------------------------
# HARDCODED TEST DATA (Required by User)
# ---------------------------------------------------------------------------
# These UUIDs must exist in your database or be valid references.
# Adjust if your local DB has different specific records.
TEST_PRODUCT_ID = "0630c72a-fcf3-4c76-b373-372a1fc67402"
TEST_USER_ID = "3c451d93-1287-4b20-9d08-a0eaa8f953e9"
TEST_BATCH_ID = "facd4de3-e360-4623-8a8a-ea73d54cacff"

PERPLEXITY_QUERIES = [
    "latest advancements in quantum computing 2024", 
    "best practices for python asyncio concurrency"
]
GOOGLE_QUERIES = [
    "overview of generative ai market trends",
    "impact of AI on software engineering jobs"
]

async def main():
    print("\n=== STARTING HARDCODED REAL CONCURRENCY TEST ===")
    
    settings = Settings()
    
    # 1. Verify Credentials
    if not settings.gemini_api_key or not settings.next_public_supabase_url:
        print("ERROR: Missing GEMINI_API_KEY or SUPABASE credentials in .env")
        return

    # 2. Create Snapshot (Simulating API /optimize/start)
    print(f"\n[1/3] Creating Snapshot...")
    all_queries = PERPLEXITY_QUERIES + GOOGLE_QUERIES
    try:
        snapshot_id = await create_analysis_snapshot(
            settings=settings,
            product_id=TEST_PRODUCT_ID,
            batch_id=TEST_BATCH_ID,
            queries=all_queries,
        )
        print(f"      Snapshot ID Created: {snapshot_id}")
    except Exception as e:
        print(f"FAILED to create snapshot: {e}")
        return

    # 3. Run Optimization Batch (Simulating Background Task)
    print(f"\n[2/3] Running Optimization Batch (Concurrent)...")
    start_time = time.time()
    
    # This runs the actual scraping + Gemini analysis + DB storage
    await run_optimization_background(
        product_id=TEST_PRODUCT_ID,
        perplexity_queries=PERPLEXITY_QUERIES,
        google_queries=GOOGLE_QUERIES,
        snapshot_id=snapshot_id,
        settings=settings,
        client_product_json={"name": "Test Product", "description": "Hardcoded test"},
        debug=True
    )
    
    duration = time.time() - start_time
    print(f"      Batch execution completed in {duration:.2f} seconds")

    # 4. Final Status Check
    print(f"\n[3/3] Verifying Final Status...")
    status_row = await get_snapshot_status(settings, snapshot_id)
    status = status_row.get("status")
    completed = status_row.get("no_of_query")
    total = status_row.get("total_no_of_query")
    
    print(f"      Final Status: {status}")
    print(f"      Progress: {completed}/{total} queries")
    
    if status == "completed":
        print("\n✅ SUCCESS: Full real-world concurrent workflow completed.")
    else:
        print(f"\n❌ FAILED: Status is {status}")

if __name__ == "__main__":
    asyncio.run(main())

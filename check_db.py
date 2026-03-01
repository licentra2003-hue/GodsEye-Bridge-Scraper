import asyncio
from dotenv import load_dotenv
import os
import sys

# Ensure app directory is in path
sys.path.append(os.getcwd())

load_dotenv()

from app.config import get_settings
from app.services.analysis_service import get_supabase_client, async_supabase_select

async def main():
    settings = get_settings()
    supabase = get_supabase_client(settings)
    
    batch_id = "ddcf10ae-5e25-4b06-9517-c7cc0a3abbb9"
    
    rows = await async_supabase_select(
        supabase, "analysis_snapshots", {"batch_id": batch_id}
    )
    
    print(f"Found {len(rows)} snapshots for batch {batch_id}:")
    for r in sorted(rows, key=lambda x: x.get('created_at', '')):
        print(f"ID: {r.get('id')} | total: {r.get('total_no_of_query')} | current: {r.get('no_of_query')} | status: {r.get('status')} | created: {r.get('created_at')} | started: {r.get('started_at')} | completed: {r.get('completed_at')}")

if __name__ == "__main__":
    asyncio.run(main())

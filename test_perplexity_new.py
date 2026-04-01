
import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# Re-point to the new gateway for testing
NEW_PERPLEXITY_URL = "https://gateway-production-957e.up.railway.app/api/v1/scrape"

async def test_perplexity_query(query: str):
    """
    Test script to verify the new Perplexity scraper.
    Checks if it's a direct response or a job submission.
    """
    print(f"\n--- Testing Perplexity Scraper ---")
    print(f"URL: {NEW_PERPLEXITY_URL}")
    print(f"Query: {query}")
    print("-" * 34)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            payload = {
                "query": query,
                "location": "India"
            }
            
            # 1. Try to submit job/query
            print("\n[Action] Sending POST request...")
            response = await client.post(NEW_PERPLEXITY_URL, json=payload)
            
            print(f"[Status] Status Code: {response.status_code}")
            
            data = response.json()
            
            # If it's a job-based scraper (returning job_id), we need to poll
            if "job_id" in data:
                job_id = data["job_id"]
                print(f"[Info] Job successfully submitted. Job ID: {job_id}")
                print("[Action] Starting polling...")
                
                # Derive poll URL (standard pattern: /api/v1/scrape -> /api/job-result)
                poll_url = NEW_PERPLEXITY_URL.replace("/api/v1/scrape", "/api/job-result")
                if poll_url == NEW_PERPLEXITY_URL:
                   # fallback if pattern doesn't match
                   poll_url = "https://gateway-production-957e.up.railway.app/api/job-result"
                
                poll_endpoint = f"{poll_url}/{job_id}"
                print(f"[Info] Poll Endpoint: {poll_endpoint}")

                # Polling loop
                for attempt in range(1, 51):  # Max 50 attempts
                    poll_response = await client.get(poll_endpoint)
                    poll_data = poll_response.json()
                    status = poll_data.get("status", "").lower()
                    
                    print(f"Attempt {attempt:02d}: Status is '{status}'")
                    
                    if status == "completed":
                        print("\n[SUCCESS] Scrape completed!")
                        # Extraction varies based on scraper version, 
                        # but normally results are in "data" or "result"
                        scrape_result = poll_data.get("data") or poll_data.get("result")
                        print("\n--- Result Data ---")
                        print(json.dumps(scrape_result, indent=2)[:2000])
                        return
                    elif status == "failed":
                        print(f"\n[FAILURE] Scraper job failed: {poll_data.get('error_message')}")
                        return
                    
                    await asyncio.sleep(5)  # Wait 5 seconds between polls
                
                print("\n[TIMEOUT] Polling exceeded maximum attempts.")
            
            # If it's a direct response (returning results immediately)
            else:
                print("\n[Info] Detected direct response (no job_id).")
                print("\n--- Scraper Result ---")
                print(json.dumps(data, indent=2)[:5000])

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    load_dotenv()
    test_query = "What are the core features of the GodsEye AI optimization platform?"
    asyncio.run(test_perplexity_query(test_query))

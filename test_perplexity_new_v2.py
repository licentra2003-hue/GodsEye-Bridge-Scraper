
import asyncio
import httpx
import json
import uuid

# The new gateway endpoint
GATEWAY_URL = "https://gateway-production-957e.up.railway.app/api/v1/scrape"

async def test_perplexity_gateway():
    """
    Test the new Perplexity scraper gateway which expects job_id and callback_url.
    """
    job_id = str(uuid.uuid4())
    query = "What are the core features of the GodsEye AI optimization platform?"
    callback_url = "https://example.com/callback"
    
    print(f"--- Gateway Test (v2) ---")
    print(f"URL: {GATEWAY_URL}")
    print(f"Job ID: {job_id}")
    print(f"Query: {query}")
    print(f"Callback URL: {callback_url}")
    print("-" * 25)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # 1. Try to post with the required parameters
            payload = {
                "job_id": job_id,
                "query": query,
                "callback_url": callback_url
            }
            
            print("\n[Action] Sending POST request with job_id and callback_url...")
            response = await client.post(GATEWAY_URL, json=payload)
            
            print(f"[Status] Status Code: {response.status_code}")
            
            # The Gateway usually returns success immediately if job is accepted
            data = response.json()
            print("\n[Response] Status Content:")
            print(json.dumps(data, indent=2))

            if response.status_code in [200, 201, 202]:
                print("\n[Success] Scraper job accepted by gateway.")
                
                # Now try to poll for the result.
                # Per standard convention: /api/v1/scrape -> /api/job-result
                poll_url = GATEWAY_URL.replace("/api/v1/scrape", "/api/job-result")
                poll_endpoint = f"{poll_url}/{job_id}"
                print(f"\n[Action] Polling for result at: {poll_endpoint}")

                for attempt in range(1, 21):
                    # We need to send job_id even in the poll probably or it's in the path
                    poll_response = await client.get(poll_endpoint)
                    if poll_response.status_code != 200:
                        print(f"Attempt {attempt}: HTTP {poll_response.status_code}")
                        await asyncio.sleep(5)
                        continue
                    
                    poll_data = poll_response.json()
                    status = poll_data.get("status", "").lower()
                    print(f"Attempt {attempt}: Status is '{status}'")

                    if status == "completed":
                        print("\n[RESULT FOUND]")
                        print(json.dumps(poll_data.get("data") or poll_data.get("result"), indent=2)[:5000])
                        return
                    elif status == "failed":
                        print(f"\n[FAILURE] Job failed: {poll_data}")
                        return
                    
                    await asyncio.sleep(5)
                
                print("\n[TIMEOUT] Still pending after 20 attempts.")
            else:
                print(f"\n[ERROR] Gateway rejected request: {data}")

        except Exception as e:
            print(f"\n[EXCEPTION] {e}")

if __name__ == "__main__":
    asyncio.run(test_perplexity_gateway())

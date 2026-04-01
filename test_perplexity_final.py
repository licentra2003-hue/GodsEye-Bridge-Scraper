
import asyncio
import os
import json
from dotenv import load_dotenv
from app.config import Settings
from app.services.scraping_service import fetch_ai_search_data

async def test_perplexity_pipeline():
    """
    Test the integrated 'Smart' Perplexity scraper within the application flow.
    """
    load_dotenv()
    settings = Settings()
    
    # Verify the new URL is loaded from .env
    print(f"\n--- Testing Integrated Perplexity Pipeline ---")
    print(f"Endpoint: {settings.scraper_url_perplexity}")
    
    test_query = "What are the key benefits of AI SEO for startups in 2024?"
    print(f"Query: {test_query}")
    print("-" * 45)

    try:
        # This calls the newly updated _fetch_perplexity with automatic gateway detection
        result = await fetch_ai_search_data(
            pipeline="perplexity",
            query=test_query,
            settings=settings
        )
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        print("\n--- Final Scraped Result (First 1000 chars) ---")
        # Pretty print the result keys and a snippet of content
        print(f"Keys returned: {list(result.keys())}")
        
        # Look for the primary content field (ai_overview_text or similar)
        content = result.get("ai_overview_text") or result.get("answer_text") or result.get("response") or "No text content found"
        print(f"\nContent Snippet:\n{str(content)[:1000]}...")
        
    except Exception as e:
        print(f"\n[FAILURE] Pipeline errored: {e}")

if __name__ == "__main__":
    asyncio.run(test_perplexity_pipeline())

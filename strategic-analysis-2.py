import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from google import genai
from google.genai import types
from supabase import create_client, Client

from dotenv import load_dotenv

load_dotenv()

# Type definitions
PipelineId = Literal['perplexity', 'google_overview', 'chatgpt', 'gemini']


@dataclass
class StrategicAnalysisRequest:
    aiSearchJson: Any
    clientProductJson: Any
    analysisId: Optional[str] = None
    pipeline: Optional[PipelineId] = None
    # Storage-related fields
    product_id: Optional[str] = None
    search_query: Optional[str] = None
    raw_serp_results: Optional[Any] = None
    snapshot_id: Optional[str] = None


@dataclass
class ExecutiveSummary:
    title: str
    status_overview: str
    strategic_analogy: str


@dataclass
class ClientProductVisibility:
    status: str
    details: str


class StrategicAnalysisResult(dict):
    """Result structure matching TypeScript interface"""
    pass


def get_supabase_client() -> Client:
    """Initialize Supabase client"""
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")
    
    return create_client(supabase_url, supabase_key)


def normalize_ai_search(ai_search_json: Any) -> List[Any]:
    """Normalize AI search JSON to always be a list"""
    if isinstance(ai_search_json, list):
        return ai_search_json
    if not ai_search_json:
        return []
    return [ai_search_json]


def flatten_source_links(ai_search_items: List[Any]) -> List[Any]:
    """Flatten source_links from all AI search items"""
    out = []
    for item in ai_search_items:
        if isinstance(item, dict):
            links = item.get('source_links')
            if isinstance(links, list):
                out.extend(links)
    return out


def clean_json_text(text: str) -> str:
    """Clean JSON response text from markdown and formatting"""
    # Remove markdown code blocks
    clean_text = re.sub(r'```json\n?', '', text)
    clean_text = re.sub(r'```\n?', '', clean_text)
    clean_text = clean_text.strip()
    
    # Remove text before first opening brace
    first_brace = clean_text.find('{')
    if first_brace > 0:
        clean_text = clean_text[first_brace:]
    
    # Remove trailing junk after last closing brace or bracket
    last_brace = clean_text.rfind('}')
    last_bracket = clean_text.rfind(']')
    last_close = max(last_brace, last_bracket)
    
    if last_close != -1 and last_close < len(clean_text) - 1:
        clean_text = clean_text[:last_close + 1]
    
    return clean_text


def repair_json_text(text: str) -> str:
    """Repair common JSON issues"""
    # Remove trailing commas before closing braces/brackets
    repaired = re.sub(r',\s*([}\]])', r'\1', text)
    return repaired


def auto_balance_json(text: str) -> str:
    """Auto-balance missing closing brackets and braces"""
    final_text = text.rstrip()
    
    # Count braces and brackets
    open_braces = final_text.count('{')
    close_braces = final_text.count('}')
    open_brackets = final_text.count('[')
    close_brackets = final_text.count(']')
    
    # Close open arrays
    if close_brackets < open_brackets:
        final_text += ']'
    
    # Close open objects
    if close_braces < open_braces:
        final_text += '}'
    
    # Special case: handle incomplete array elements
    if not final_text.endswith(']}') and not final_text.endswith('}'):
        last_comma_index = final_text.rfind('},')
        if last_comma_index != -1 and last_comma_index > final_text.rfind(']'):
            final_text = final_text[:last_comma_index + 1] + ']}'
    
    return final_text


def save_debug_file(content: str, prefix: str, debug: bool = False) -> None:
    """Save debug files if debugging is enabled"""
    if not debug:
        return
    
    # Allow blocking file creation via env var (default to False/Block)
    if os.getenv("SAVE_GEMINI_DEBUG_FILES", "false").lower() != "true":
        return
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"gemini-{prefix}-response-{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[DEBUG] {prefix.capitalize()} Gemini response written to: {filename}")
    except Exception as e:
        print(f"[DEBUG] Failed to write {prefix} response file: {e}")


def create_fallback_result(
    text: str,
    ai_search_json: Any
) -> StrategicAnalysisResult:
    """Create fallback result when parsing fails"""
    ai_search_any = ai_search_json
    merged_source_links = (
        flatten_source_links(ai_search_any)
        if isinstance(ai_search_any, list)
        else ai_search_any.get('source_links', [])
        if isinstance(ai_search_any, dict)
        else []
    )
    
    fallback_sources = [
        {
            'source_snippet': src.get('snippet') or src.get('text') or 'Snippet not clearly available in provided text.',
            'reason_for_inclusion': 'Mentioned in AI response sources',
            'source_of_mention': src.get('url') or src.get('source') or 'Unknown source',
        }
        for src in merged_source_links
    ] if isinstance(merged_source_links, list) else []
    
    return StrategicAnalysisResult({
        'executive_summary': {
            'title': 'AEO Competitive Analysis (Fallback)',
            'status_overview': 'The AI analysis service returned an unexpected format. This is an automatically generated fallback summary.',
            'strategic_analogy': 'Think of this as receiving raw research notes without a clean report. The data is there, but the structure had to be approximated.',
        },
        'client_product_visibility': {
            'status': 'Not Featured',
            'details': 'The detailed structured visibility analysis could not be parsed from the AI response. Please rerun the analysis later or contact support if this persists.',
        },
        'ai_answer_deconstruction': {
            'dominant_narrative': 'The AI response could not be properly parsed to extract the dominant narrative. This may be due to formatting issues in the AI output.',
            'key_decision_factors': [
                'AI response parsing failed - unable to extract decision factors',
                'Consider rerunning the analysis for complete results',
                'Raw response data has been preserved for reference',
            ],
            'trusted_source_analysis': 'Due to parsing difficulties, trusted source analysis could not be extracted. Please refer to the source links provided below for raw information.',
            'raw_response_preview': text[:500],
        },
        'competitive_landscape_analysis': [],
        'sources_ai_used': fallback_sources,
        'strategic_gap_and_opportunity_analysis': {
            'analysis_summary': 'Due to a formatting issue in the AI output, a full gap and opportunity analysis could not be generated. However, the source links have been preserved for manual review.',
        },
        'actionable_recommendations': [],
    })


def store_analysis_result(
    product_id: str,
    pipeline: PipelineId,
    search_query: str,
    analysis_result: Dict[str, Any],
    raw_serp_results: Any,
    snapshot_id: Optional[str] = None,
    related_analysis_id: Optional[str] = None,
    source_links: Optional[List[Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Store analysis result in Supabase database
    
    Args:
        product_id: UUID of the product
        pipeline: Pipeline type (google_overview or perplexity)
        search_query: The search query used
        analysis_result: The generated analysis result
        raw_serp_results: The raw SERP data
        snapshot_id: Optional snapshot ID
        related_analysis_id: Optional related analysis ID (for linking Perplexity to Google)
        source_links: Optional source links (for Perplexity)
        debug: Enable debug output
    
    Returns:
        Dictionary containing the stored record
    """
    try:
        supabase = get_supabase_client()
        
        # Pass dictionaries directly for JSONB columns (don't convert to string)
        # Supabase will handle the JSON serialization automatically
        
        if pipeline == 'google_overview':
            # Store in product_analysis_google table
            data_to_insert = {
                'product_id': product_id,
                'search_query': search_query,
                'google_overview_analysis': analysis_result,  # Pass as dict, not string
                'raw_serp_results': raw_serp_results or {},   # Pass as dict, not string
            }
            
            if snapshot_id:
                data_to_insert['snapshot_id'] = snapshot_id
            
            response = supabase.table('product_analysis_google').insert(data_to_insert).execute()
            
            if debug:
                print(f"[DEBUG] Stored Google analysis with ID: {response.data[0].get('id') if response.data else 'unknown'}")
            
            return response.data[0] if response.data else {}
            
        elif pipeline == 'perplexity':
            # Store in product_analysis_perplexity table
            data_to_insert = {
                'product_id': product_id,
                'optimization_prompt': search_query,
                'optimization_analysis': analysis_result,  # Pass as dict, not string
                'citations': source_links or [],
                'raw_serp_results': raw_serp_results or {},  # Pass as dict, not string
            }
            
            if snapshot_id:
                data_to_insert['snapshot_id'] = snapshot_id
            
            if related_analysis_id:
                data_to_insert['related_google_analysis_id'] = related_analysis_id
            
            response = supabase.table('product_analysis_perplexity').insert(data_to_insert).execute()
            
            if debug:
                print(f"[DEBUG] Stored Perplexity analysis with ID: {response.data[0].get('id') if response.data else 'unknown'}")
            
            return response.data[0] if response.data else {}
        
        else:
            if debug:
                print(f"[WARNING] Unknown pipeline type: {pipeline}. Analysis not stored.")
            return {}
            
    except Exception as e:
        print(f"[ERROR] Failed to store analysis: {e}")
        raise


def perform_strategic_analysis(
    request: StrategicAnalysisRequest,
    api_key: Optional[str] = None,
    debug: bool = False,
    store_to_db: bool = True
) -> Dict[str, Any]:
    """
    Perform strategic analysis using Google Gemini API and optionally store to database
    
    Args:
        request: StrategicAnalysisRequest containing analysis parameters
        api_key: Google API key (if None, reads from GEMINI_API_KEY env var)
        debug: Enable debug file output
        store_to_db: Whether to store results in database
    
    Returns:
        Dictionary containing analysis result and storage info
    
    Raises:
        ValueError: If API key is not provided or required fields are missing for storage
        Exception: If analysis fails
    """
    # Get API key
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not defined in environment variables or parameters")
    
    # Validate required fields for storage
    if store_to_db:
        if not request.product_id:
            raise ValueError("product_id is required for database storage")
        if not request.search_query:
            raise ValueError("search_query is required for database storage")
        if not request.pipeline:
            raise ValueError("pipeline is required for database storage")
    
    # Configure Client
    client = genai.Client(api_key=api_key)

    # Create config
    generation_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    # Normalize AI search data
    normalized_ai_search = normalize_ai_search(request.aiSearchJson)

    # Build the base prompt
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

    # Adjust prompt based on pipeline
    if request.pipeline == 'google_overview':
        prompt = base_prompt.replace(
            'Perplexity AI (The AI Search Engine)',
            'Google AI Overview (The AI Search Engine)'
        )
    else:
        prompt = base_prompt

    try:
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=generation_config
        )
        text = response.text
        
        # Log token usage if debug enabled
        if debug:
            try:
                usage = response.usage_metadata
                print(f"[Gemini][Strategic Analysis]", {
                    'inputTokens': usage.prompt_token_count if usage else 0,
                    'outputTokens': usage.candidates_token_count if usage else 0,
                    'totalTokens': usage.total_token_count if usage else 0,
                })
            except:
                pass
        
        # Parse JSON with enhanced error handling
        try:
            # Clean the text
            clean_text = clean_json_text(text)
            
            # Save raw response if debugging
            save_debug_file(clean_text, 'raw', debug)
            
            # First parse attempt
            try:
                analysis_result = json.loads(clean_text)
            except json.JSONDecodeError as inner_error:
                if debug:
                    print(f"Initial JSON.parse failed, attempting repair: {inner_error}")
                
                # Repair JSON
                repaired_text = repair_json_text(clean_text)
                save_debug_file(repaired_text, 'repaired', debug)
                
                # Second parse attempt
                try:
                    analysis_result = json.loads(repaired_text)
                    clean_text = repaired_text
                except json.JSONDecodeError as second_error:
                    if debug:
                        print(f"Second JSON.parse failed, attempting auto-balance: {second_error}")
                    
                    # Auto-balance JSON
                    final_text = auto_balance_json(repaired_text)
                    save_debug_file(final_text, 'balanced', debug)
                    
                    analysis_result = json.loads(final_text)
                    clean_text = final_text
            
            # Validate required structure
            if not analysis_result.get('executive_summary') or not analysis_result.get('client_product_visibility'):
                analysis_result = create_fallback_result(text, request.aiSearchJson)
            
            # Store to database if requested
            stored_record = {}
            if store_to_db:
                # Extract source links from normalized AI search
                source_links = []
                if isinstance(normalized_ai_search, list):
                    source_links = flatten_source_links(normalized_ai_search)
                elif isinstance(normalized_ai_search, dict):
                    source_links = normalized_ai_search.get('source_links', [])
                
                stored_record = store_analysis_result(
                    product_id=request.product_id,
                    pipeline=request.pipeline,
                    search_query=request.search_query,
                    analysis_result=analysis_result,
                    raw_serp_results=request.raw_serp_results,
                    snapshot_id=request.snapshot_id,
                    source_links=source_links,
                    debug=debug
                )
            
            # Return combined result
            return {
                'success': True,
                'analysis': analysis_result,
                'stored_record': stored_record if store_to_db else None,
                'product_id': request.product_id,
                'pipeline': request.pipeline,
                'search_query': request.search_query,
            }
            
        except json.JSONDecodeError as parse_error:
            if debug:
                print(f"[StrategicAnalysis] JSON Parse Error: {parse_error}")
            fallback_result = create_fallback_result(text, request.aiSearchJson)
            
            # Still try to store fallback if requested
            stored_record = {}
            if store_to_db:
                stored_record = store_analysis_result(
                    product_id=request.product_id,
                    pipeline=request.pipeline,
                    search_query=request.search_query,
                    analysis_result=fallback_result,
                    raw_serp_results=request.raw_serp_results,
                    snapshot_id=request.snapshot_id,
                    debug=debug
                )
            
            return {
                'success': False,
                'analysis': fallback_result,
                'stored_record': stored_record if store_to_db else None,
                'error': 'JSON parsing failed, returning fallback result',
            }
        
    except Exception as e:
        if debug:
            print(f"[StrategicAnalysis] Error: {e}")
        raise Exception(f"Failed to perform strategic analysis: {e}")


def main():
    """
    Main execution function
    
    USER: PROVIDE THESE VARIABLES BELOW
    """
    
    # ============================================================================
    # USER-PROVIDED VARIABLES - FILL THESE IN
    # ============================================================================
    
    # 1. Product ID (UUID) - REQUIRED for database storage
    PRODUCT_ID = "0630c72a-fcf3-4c76-b373-372a1fc67402"
    
    # 2. Search Query - REQUIRED for database storage
    SEARCH_QUERY = "top AI product optimization solutions for businesses wanting their products in AI recommendations"
    
    # 3. Pipeline Type - REQUIRED for database storage ('google_overview' or 'perplexity')
    PIPELINE = "google_overview"
    
    # 4. Snapshot ID (Optional)
    SNAPSHOT_ID = "f462ee52-235a-45fd-a282-ce0129eda748"
    
    # 5. Product Data (JSON) - Your client's product information
    PRODUCT_DATA = {
        "name": "GodsEye",
        "description": "GodsEye is an AI-powered product optimization platform designed to enhance product visibility in Generative AI search environments. It functions similarly to SEO but is specifically tailored for AI, analyzing how AI 'perceives' a product and providing strategies to ensure it becomes a top recommendation for users seeking AI-driven advice or comparisons.",
        "specifications": {
          "general_product_type": "Software Platform",
          "specific_product_type": "AI Product Optimization Platform"
        },
        "features": [
          {
            "name": "Smart Product Analysis",
            "description": "Extracts product data directly from a URL and identifies the unique value proposition that an AI model would find relevant."
          },
          {
            "name": "AI Search Visibility Tracking",
            "description": "Evaluates a product's current ranking in AI search results and identifies which competitors are being mentioned instead."
          },
          {
            "name": "AEO Recommendations",
            "description": "Provides tailored, actionable steps to adjust web content, metadata, and product descriptions to better align with AI training data and retrieval patterns."
          },
          {
            "name": "Competitive Intelligence",
            "description": "Compares a product's 'AI footprint' against competitors to highlight gaps in market positioning."
          },
          {
            "name": "Automated Insights & Reports",
            "description": "Generates strategic reports that can be exported for marketing teams to track optimization progress."
          }
        ],
        "targeted_market": "E-commerce brands, SaaS and Tech Startups, and Marketing Agencies looking to ensure their products are visible and recommended in AI-driven search results and shopping recommendations. It also targets Product Managers seeking to understand AI search.",
        "problem_product_is_solving": "This product solves the problem of products not being visible in AI-generated search results, a challenge that traditional SEO does not address. It helps brands ensure their products are part of AI conversations, especially when users ask AI for product recommendations or comparisons, which is becoming increasingly common.",
        "general_product_type": "Software Platform",
        "specific_product_type": "AI Product Optimization Platform",
    }
    
    # 6. Raw SERP Results - The scraped data from the AI search engine
    # This should be the complete response from your scraper
    RAW_SERP_RESULTS = {
      "query": "which AI product optimization platforms help product managers understand AI search trends?",
      "job_id": "6821bf8e-1200-47c6-9401-b62c5b4bedbc",
      "success": True,
      "location": "India",
      "timestamp": "2026-01-27T10:35:51.666621",
      "source_links": [
        {
          "url": "https://visible.seranking.com/blog/best-ai-visibility-tools/#:~:text=3.-,Profound%20AI,user%20interfaces%20of%20AI%20engines.",
          "date": "16 Dec 2025",
          "title": ""
        },
        {
          "url": "https://visible.seranking.com/blog/best-ai-visibility-tools/",
          "date": "16 Dec 2025",
          "title": ""
        },
        {
          "url": "https://www.tryprofound.com/blog/best-generative-engine-optimization-tools",
          "date": "15 Nov 2025",
          "title": ""
        },
        {
          "url": "https://llmrefs.com/blog/top-ai-visibility-products-for-generative-engine-optimization#:~:text=Semrush%20earns%20its%20high%20ranking,Position%20Tracking%20and%20Organic%20Research.",
          "date": "20 Dec 2025",
          "title": ""
        },
        {
          "url": "https://chad-wyatt.com/ai/best-generative-engine-optimization-tools/#:~:text=1.,for%20crawler%20and%20parsing%20issues",
          "date": "9 Dec 2025",
          "title": ""
        },
        {
          "url": "https://monday.com/blog/rnd/ai-for-product-managers/#:~:text=ProdPad:%20Uses%20AI%20to%20recommend,%2C%20paid%20from%20$20/month.",
          "date": "21 Sept 2025",
          "title": ""
        },
        {
          "url": "https://www.wixseoexpert.com/post/seo-and-ai-search-trends-to-watch-in-2026#:~:text=Dhruv%20Nimbark,signal%20for%20any%20search%20engine.",
          "date": "21 Jan 2026",
          "title": ""
        },
        {
          "url": "https://visible.seranking.com/blog/best-generative-engine-optimization-tools-2026/#:~:text=buck%20right%20now.%E2%80%9D-,Pricing,based%20on%20real%2Dworld%20data.",
          "date": "23 Jan 2026",
          "title": ""
        },
        {
          "url": "https://socialchamps.com/top-geo-tools-2026-platforms-for-brand-tracking-monitoring-key-features-and-pricing/#:~:text=Citation%20Monitoring:%20See%20where%20your,exports%20for%20marketing%20&%20PR%20teams.",
          "date": "17 Sept 2025",
          "title": ""
        },
        {
          "url": "https://www.plandigi.com/blog/why-generative-engine-optimization-in-2026/#:~:text=What%20Is%20Generative%20Engine%20Optimization,pick%20your%20pages%20more%20frequently.",
          "date": "10 Dec 2025",
          "title": ""
        },
        {
          "url": "https://launchlify.com/ai-tools-for-product-managers/#:~:text=Mixpanel%20is%20an%20AI%2Dbased,them%20in%20front%20of%20data.",
          "date": "9 Jan 2026",
          "title": ""
        },
        {
          "url": "https://www.data-mania.com/blog/best-ai-search-tools-2026/#:~:text=%E2%80%9CPerplexity%20is%20the%20best%20AI,confidence.%E2%80%9D%20%E2%80%93%20searchatlas.com",
          "date": "11 Jan 2026",
          "title": ""
        },
        {
          "url": "https://gopractice.io/skills/using-ai-for-product-and-market-research/#:~:text=Currently%2C%20OpenAI%2C%20Grok%2C%20Perplexity%2C%20and%20Gemini%20provide,and%20research%2Dfocused%20products%20such%20as%20Manus%20AI).",
          "title": ""
        },
        {
          "url": "https://www.sitepoint.com/best-ai-mode-tracking-tools/#:~:text=seoClarity%20is%20an%20enterprise%2Dlevel,comparison%20and%20content%20performance%20analysis.",
          "date": "21 Nov 2025",
          "title": ""
        },
        {
          "url": "https://www.youtube.com/watch?v=0xw4xCxpXYg&t=28",
          "date": "29 Dec 2025",
          "title": ""
        },
        {
          "url": "https://www.therankmasters.com/insights/ai-visibility/best-tools-tracking-brand-visibility-ai-search",
          "date": "3 Sept 2025",
          "title": ""
        },
      ],
      "ai_overview_text": "In 2026, AI product optimization platforms have evolved into a specialized category known as Generative Engine Optimization (GEO) tools. These platforms help product managers (PMs) track brand visibility in AI-generated answers, identify trending conversational prompts, and understand the source data influencing Large Language Models (LLMs). ### Specialized AI Search Visibility & Trend Platforms • Profound AI: A high-end enterprise platform that monitors brand mentions and sentiment across various AI engine user interfaces. It features Prompt Volume Research, allowing PMs to identify the specific high-impact questions users are asking AI. • Writesonic GEO: An all-in-one platform that combines content creation with AI-visibility monitoring. Its Action Center identifies citation gaps and suggests specific content updates to outrank competitors in AI-driven search results. • Gumshoe.AI: Unique for its persona-driven approach, this platform models how different audience segments ask and consume AI answers, providing insight into user intent patterns that prompt-centric testing might miss. • Peec AI: Focuses on Source Identification, helping PMs understand which third-party websites or forums (like Reddit or niche communities) are influencing the answers AI gives about their product. • Addlly AI: Uses customizable AI agents to monitor and improve citation opportunities in real-time, automating the workflow of flagging missed brand mentions across engines. ### Enterprise SEO Suites with AI Trend Modules Traditional SEO leaders have integrated dedicated AI Mode or AIO trackers for 2026: • Semrush AIO/One: Provides an AI Visibility Toolkit that tracks brand share-of-voice in AI answers and identifies which prompts trigger citations. • seoClarity: Offers enterprise-scale tracking for Perplexity and Google AI Overviews, measuring unlinked mentions and citations across millions of keywords to quantify business impact. • Conductor: Features an AI Search Performance module that ties AI answer visibility and sentiment to existing keyword and content data for cross-functional reporting. • SISTRIX: Maintains an extensive SERP archive that stores the full text of AI Overviews over time, enabling longitudinal analysis of how AI responses for a product category evolve. ### Consumer Research & Market Discovery Tools • Perplexity AI (Deep Research): PMs use Perplexity to perform real-time market validation and synthesize complex trends from source-backed answers. • Dovetail: Analyzes user interviews and support tickets using AI to uncover recurring themes and pain points that may indicate emerging search behaviors or feature needs. • Monterey AI: Aggregates feedback from multiple sources—social media, reviews, and surveys—to track sentiment trends and discover product issues before they escalate.",
    }
    
    # 7. AI Search JSON - This is typically the same as RAW_SERP_RESULTS or a subset
    AI_SEARCH_JSON = RAW_SERP_RESULTS
    
    # ============================================================================
    # END OF USER-PROVIDED VARIABLES
    # ============================================================================
    
    # Create request object
    request = StrategicAnalysisRequest(
        aiSearchJson=AI_SEARCH_JSON,
        clientProductJson=PRODUCT_DATA,
        analysisId="analysis-001",
        pipeline=PIPELINE,
        product_id=PRODUCT_ID,
        search_query=SEARCH_QUERY,
        raw_serp_results=RAW_SERP_RESULTS,
        snapshot_id=SNAPSHOT_ID
    )
    
    # Perform analysis with database storage
    try:
        print("\n" + "="*80)
        print("STARTING STRATEGIC ANALYSIS")
        print("="*80)
        print(f"Product ID: {PRODUCT_ID}")
        print(f"Pipeline: {PIPELINE}")
        print(f"Search Query: {SEARCH_QUERY}")
        print("="*80 + "\n")
        
        result = perform_strategic_analysis(
            request=request,
            debug=True,
            store_to_db=True  # Set to False if you don't want to store in database
        )
        
        # Print results
        print("\n" + "="*80)
        print("STRATEGIC ANALYSIS RESULT")
        print("="*80)
        print(json.dumps(result['analysis'], indent=2))
        
        if result.get('stored_record'):
            print("\n" + "="*80)
            print("DATABASE STORAGE INFO")
            print("="*80)
            print(json.dumps(result['stored_record'], indent=2, default=str))
        
        # Save result to local file
        output_filename = f"strategic-analysis-{PIPELINE}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[SUCCESS] Full result saved to: {output_filename}")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
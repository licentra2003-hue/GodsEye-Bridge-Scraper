# GodsEye Strategic Analysis Backend - Project Documentation

## Overview

The GodsEye Strategic Analysis Backend is a sophisticated FastAPI-based service designed to perform competitive analysis and Answer Engine Optimization (AEO) for products using AI-powered search results. The system analyzes how products are positioned in AI search engine responses (particularly Perplexity AI and Google AI Overview) and provides strategic recommendations to improve visibility and competitive positioning.

## Architecture

### Technology Stack
- **Backend Framework**: FastAPI 0.116.1
- **AI Integration**: Google Generative AI (Gemini 2.5-flash)
- **Database**: Supabase (PostgreSQL)
- **Containerization**: Docker & Docker Compose
- **Language**: Python 3.12
- **Retry Mechanism**: Tenacity for resilient API calls

### Project Structure
```
Pavan-main/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and API endpoints
│   ├── models.py            # Pydantic models for request/response validation
│   ├── config.py            # Configuration management with environment variables
│   └── services/
│       ├── __init__.py
│       └── analysis_service.py  # Core analysis logic and AI integration
├── strategic-analysis-2.py  # Standalone analysis script (legacy)
├── Database.sql             # Database schema reference
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── .gitignore            # Git ignore rules
```

## Core Functionality

### 1. Strategic Analysis Pipeline

The system performs comprehensive competitive analysis through the following workflow:

#### Input Data Processing
- **AI Search JSON**: Raw search results from AI engines (Perplexity, Google AI Overview)
- **Client Product JSON**: Product information including features, ingredients, and marketing claims
- **Pipeline Selection**: Supports multiple analysis pipelines (`perplexity`, `google_overview`, `chatgpt`, `gemini`)

#### AI-Powered Analysis
The system uses Google Gemini AI to analyze the competitive landscape with a sophisticated prompt engineering approach:

1. **Presence Detection**: Identifies if the client's product is mentioned in AI search results
2. **Logic Deconstruction**: Analyzes the AI's narrative and decision-making patterns
3. **Competitor Profiling**: Reverse-engineers why competitors are being featured
4. **Gap Analysis**: Identifies missing attributes and positioning opportunities
5. **Strategy Formulation**: Provides actionable recommendations

#### Output Structure
The analysis returns a structured JSON response containing:

```json
{
  "executive_summary": {
    "title": "Analysis title",
    "status_overview": "Product visibility status",
    "strategic_analogy": "Memorable strategic analogy"
  },
  "client_product_visibility": {
    "status": "Featured | Not Featured",
    "details": "Detailed visibility information"
  },
  "ai_answer_deconstruction": {
    "dominant_narrative": "AI's narrative analysis",
    "key_decision_factors": ["Factor 1", "Factor 2"],
    "trusted_source_analysis": "Source authority analysis"
  },
  "competitive_landscape_analysis": [
    {
      "competitor_name": "Competitor name",
      "reason_for_inclusion": "Why they're featured",
      "source_of_mention": "Associated source URL"
    }
  ],
  "strategic_gap_and_opportunity_analysis": {
    "analysis_summary": "Core analysis findings",
    "if_featured": { /* Analysis when product is featured */ },
    "if_not_featured": { /* Analysis when product is not featured */ }
  },
  "actionable_recommendations": [
    {
      "recommendation": "Recommendation title",
      "action": "Specific action to take"
    }
  ]
}
```

### 2. Database Integration

The system stores analysis results in Supabase with support for multiple pipeline types:

#### Google Overview Pipeline
- Table: `product_analysis_google`
- Stores: Google AI Overview analysis results, SERP data, snapshots

#### Perplexity Pipeline  
- Table: `product_analysis_perplexity`
- Stores: Perplexity analysis results, citations, related Google analysis

#### Analysis History Tracking
- Table: `analysis_history`
- Tracks: User analysis requests, credits used, status, related analysis IDs

### 3. API Endpoints

#### Health Check
```
GET /health
```
Returns service status and environment information.

#### Strategic Analysis
```
POST /api/v1/analysis/strategic
```
Performs comprehensive strategic analysis with optional database storage.

**Request Model**: `StrategicAnalysisRunRequest`
- Inherits from `StrategicAnalysisRequest`
- Adds: `api_key`, `debug`, `store_to_db` parameters

#### Analysis Storage
```
POST /api/v1/analysis/store
```
Stores analysis results separately from the analysis execution.

#### Utility Endpoints
```
POST /api/v1/utils/normalize-ai-search     # Normalizes AI search JSON to list format
POST /api/v1/utils/flatten-source-links     # Extracts source links from AI search items
POST /api/v1/utils/clean-json               # Removes markdown formatting from JSON text
POST /api/v1/utils/repair-json              # Fixes common JSON syntax issues
POST /api/v1/utils/auto-balance-json        # Auto-balances missing brackets/braces
POST /api/v1/utils/fallback-result          # Creates fallback analysis result
```

## Configuration Management

### Environment Variables
The system uses a comprehensive configuration system via `app/config.py`:

#### Application Settings
- `APP_NAME`: Service identifier
- `APP_ENV`: Environment (development/production)
- `APP_DEBUG`: Debug mode flag
- `HOST/PORT`: Server binding configuration

#### Gemini AI Configuration
- `GEMINI_API_KEY`: Google AI API authentication
- `GEMINI_MODEL_NAME`: Model selection (default: gemini-2.5-flash)
- `GEMINI_TEMPERATURE`: Response creativity (0.0-1.0)
- `GEMINI_TOP_P/Top_K`: Nucleus sampling parameters
- `GEMINI_MAX_OUTPUT_TOKENS`: Response length limit

#### Retry Policy
- `RETRY_MAX_ATTEMPTS`: Maximum retry attempts (default: 5)
- `RETRY_MIN/MAX_WAIT_SECONDS`: Exponential backoff bounds
- `RETRY_MULTIPLIER`: Backoff multiplier

#### Database Configuration
- `NEXT_PUBLIC_SUPABASE_URL`: Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY`: Database access key

## Error Handling & Resilience

### Multi-Layer JSON Parsing
The system implements robust JSON parsing with fallback mechanisms:

1. **Initial Parse**: Attempts direct JSON parsing after cleaning markdown
2. **Repair Attempt**: Fixes common issues (trailing commas, etc.)
3. **Auto-Balance**: Adds missing brackets/braces if needed
4. **Fallback Result**: Generates structured fallback when all parsing fails

### Retry Strategy
Uses Tenacity library for exponential backoff:
- Configurable maximum attempts
- Exponential wait with jitter
- Exception-specific retry logic
- Comprehensive logging

### Debug Support
When debug mode is enabled:
- Saves raw AI responses to timestamped files
- Logs detailed token usage metrics
- Provides intermediate parsing states
- Enhanced error reporting

## Deployment

### Docker Configuration
The system is containerized with:
- **Base Image**: Python 3.12-slim
- **Security**: Non-root user execution
- **Health Check**: Automated endpoint monitoring
- **Environment**: Production-ready configuration

### Docker Compose
Multi-container setup with:
- Backend service exposure
- Environment variable injection
- Health monitoring
- Restart policies

## Development Workflow

### Local Development
1. Copy `.env.example` to `.env` and configure variables
2. Install dependencies: `pip install -r requirements.txt`
3. Run server: `uvicorn app.main:app --reload`
4. Access API docs: `http://localhost:8000/docs`

### Production Deployment
1. Configure environment variables
2. Build container: `docker-compose build`
3. Start services: `docker-compose up -d`
4. Monitor health: Check `/health` endpoint

## Key Features

### 1. Multi-Pipeline Support
- Perplexity AI analysis
- Google AI Overview analysis  
- Extensible pipeline architecture for future AI engines

### 2. Intelligent Fallback Mechanisms
- Graceful degradation when AI responses are malformed
- Preserves source data even when structured analysis fails
- Maintains system reliability

### 3. Comprehensive Competitive Intelligence
- Identifies why competitors are featured
- Analyzes AI decision-making patterns
- Provides specific, actionable recommendations

### 4. Enterprise-Grade Architecture
- Scalable FastAPI foundation
- Robust error handling and retry logic
- Comprehensive logging and monitoring
- Containerized deployment

## Business Value

### For Product Teams
- **Competitive Intelligence**: Understand why competitors win AI search placements
- **Optimization Roadmap**: Get specific recommendations to improve AI search visibility
- **Performance Tracking**: Monitor changes in AI search positioning over time

### For Marketing Teams  
- **AEO Strategy**: Optimize content for Answer Engine Optimization
- **Source Authority**: Identify which publications and domains AI engines trust
- **Content Alignment**: Ensure marketing claims match AI decision factors

### For Business Strategy
- **Market Positioning**: Understand how AI engines categorize and rank products
- **Opportunity Identification**: Discover gaps in competitive landscape
- **ROI Measurement**: Track impact of optimization efforts on AI search visibility

## Future Extensibility

The architecture supports easy expansion:
- Additional AI engine pipelines
- Enhanced analysis dimensions
- Advanced reporting and visualization
- Real-time monitoring and alerting
- Integration with marketing automation platforms

This system represents a sophisticated approach to modern search engine optimization, specifically designed for the era of AI-powered search results and answer engines.

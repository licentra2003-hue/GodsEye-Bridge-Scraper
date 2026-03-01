import pytest
from unittest.mock import MagicMock, patch
from app.services.analysis_service import _store_analysis_result_supabase
from app.config import get_settings

@pytest.fixture
def mock_supabase():
    with patch("app.services.analysis_service.get_supabase_client") as mock_client:
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        yield mock_db

def test_store_chatgpt_analysis_result(mock_supabase):
    settings = get_settings()
    
    # Setup mock response
    mock_table = MagicMock()
    mock_insert = MagicMock()
    mock_execute = MagicMock()
    
    mock_supabase.table.return_value = mock_table
    mock_table.insert.return_value = mock_insert
    mock_execute.data = [{"id": "mock-uuid"}]
    mock_insert.execute.return_value = mock_execute

    result = _store_analysis_result_supabase(
        settings=settings,
        product_id="test-product-id",
        pipeline="chatgpt",
        search_query="test query",
        analysis_result={"some": "analysis"},
        raw_serp_results={"raw": "data"},
        snapshot_id="test-snapshot-id",
        source_links=[{"link": "url"}],
        debug=True,
    )

    # Verify table name
    mock_supabase.table.assert_called_once_with("product_analysis_chatgpt")
    
    # Verify exact insertion data
    mock_table.insert.assert_called_once_with({
        "product_id": "test-product-id",
        "optimization_prompt": "test query",
        "optimization_analysis": {"some": "analysis"},
        "citations": [{"link": "url"}],
        "raw_serp_results": {"raw": "data"},
        "snapshot_id": "test-snapshot-id",
    })

    assert result == {"id": "mock-uuid"}

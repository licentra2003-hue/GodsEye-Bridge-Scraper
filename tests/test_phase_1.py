"""Tests for Phase 1: Async Foundations & State Management"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.config import Settings
from app.services.analysis_service import (
    async_supabase_insert,
    async_supabase_select,
    async_supabase_update,
    create_analysis_snapshot,
    verify_and_deduct_credits,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Settings(
        next_public_supabase_url="https://test.supabase.co",
        supabase_service_role_key="test-key",
    )
    return settings


@pytest.fixture
def mock_supabase_client():
    """Create mock Supabase client."""
    client = MagicMock()
    return client


class TestAsyncSupabaseWrappers:
    """Test async Supabase wrapper functions."""

    @pytest.mark.asyncio
    async def test_async_supabase_insert(self, mock_supabase_client):
        """Test async insert wrapper."""
        mock_response = MagicMock()
        mock_response.data = [{"id": "test-id"}]
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = mock_response

        result = await async_supabase_insert(mock_supabase_client, "test_table", {"key": "value"})

        assert result.data[0]["id"] == "test-id"
        mock_supabase_client.table.assert_called_once_with("test_table")

    @pytest.mark.asyncio
    async def test_async_supabase_update(self, mock_supabase_client):
        """Test async update wrapper."""
        mock_response = MagicMock()
        mock_response.data = [{"id": "test-id", "updated": True}]
        mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_response

        result = await async_supabase_update(
            mock_supabase_client, "test_table", {"updated": True}, {"id": "test-id"}
        )

        assert result.data[0]["updated"] is True

    @pytest.mark.asyncio
    async def test_async_supabase_select(self, mock_supabase_client):
        """Test async select wrapper."""
        mock_response = MagicMock()
        mock_response.data = [{"id": "test-id", "name": "test"}]
        mock_supabase_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_response

        result = await async_supabase_select(mock_supabase_client, "test_table", {"id": "test-id"})

        assert len(result) == 1
        assert result[0]["id"] == "test-id"


class TestCreateAnalysisSnapshot:
    """Test create_analysis_snapshot function."""

    @pytest.mark.asyncio
    async def test_create_snapshot_success(self, mock_settings):
        """Test successful snapshot creation (Scenario 1: no existing snapshot)."""
        snapshot_id = str(uuid4())
        product_id = str(uuid4())
        batch_id = str(uuid4())
        queries = ["test query 1", "test query 2"]

        mock_response = MagicMock()
        mock_response.data = [{"id": snapshot_id}]

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = []  # No existing snapshot

                with patch("app.services.analysis_service.async_supabase_insert", new_callable=AsyncMock) as mock_insert:
                    mock_insert.return_value = mock_response

                    result = await create_analysis_snapshot(
                        settings=mock_settings,
                        product_id=product_id,
                        batch_id=batch_id,
                        queries=queries,
                    )

                    assert result == snapshot_id
                    mock_select.assert_called_once()  # checked for existing
                    mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_snapshot_no_data_returned(self, mock_settings):
        """Test snapshot creation when no data is returned."""
        product_id = str(uuid4())
        batch_id = str(uuid4())
        queries = ["test query"]

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = []  # No existing snapshot

                with patch("app.services.analysis_service.async_supabase_insert", new_callable=AsyncMock) as mock_insert:
                    mock_response = MagicMock()
                    mock_response.data = []
                    mock_insert.return_value = mock_response

                    with pytest.raises(Exception, match="Failed to create snapshot"):
                        await create_analysis_snapshot(
                            settings=mock_settings,
                            product_id=product_id,
                            batch_id=batch_id,
                            queries=queries,
                        )

    @pytest.mark.asyncio
    async def test_create_snapshot_exception(self, mock_settings):
        """Test snapshot creation with exception."""
        product_id = str(uuid4())
        batch_id = str(uuid4())
        queries = ["test query"]

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.side_effect = Exception("Database error")

                with pytest.raises(Exception, match="Failed to create analysis snapshot"):
                    await create_analysis_snapshot(
                        settings=mock_settings,
                        product_id=product_id,
                        batch_id=batch_id,
                        queries=queries,
                    )


class TestVerifyAndDeductCredits:
    """Test verify_and_deduct_credits function."""

    @pytest.mark.asyncio
    async def test_verify_and_deduct_credits_success(self, mock_settings):
        """Test successful credit verification and deduction."""
        user_id = str(uuid4())
        required_credits = 5
        current_balance = 10

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = [{"id": user_id, "credits": current_balance}]

                with patch("app.services.analysis_service.async_supabase_update", new_callable=AsyncMock) as mock_update:
                    result = await verify_and_deduct_credits(
                        settings=mock_settings,
                        user_id=user_id,
                        required_credits=required_credits,
                    )

                    assert result is True
                    mock_update.assert_called_once()
                    call_args = mock_update.call_args
                    assert call_args[0][2]["credits"] == current_balance - required_credits

    @pytest.mark.asyncio
    async def test_verify_and_deduct_credits_insufficient(self, mock_settings):
        """Test credit verification with insufficient balance."""
        user_id = str(uuid4())
        required_credits = 15
        current_balance = 10

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = [{"id": user_id, "credits": current_balance}]

                result = await verify_and_deduct_credits(
                    settings=mock_settings,
                    user_id=user_id,
                    required_credits=required_credits,
                )

                assert result is False

    @pytest.mark.asyncio
    async def test_verify_and_deduct_credits_user_not_found(self, mock_settings):
        """Test credit verification when user not found."""
        user_id = str(uuid4())

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = []

                with pytest.raises(Exception, match="User profile not found"):
                    await verify_and_deduct_credits(
                        settings=mock_settings,
                        user_id=user_id,
                        required_credits=5,
                    )

    @pytest.mark.asyncio
    async def test_verify_and_deduct_credits_exception(self, mock_settings):
        """Test credit verification with exception."""
        user_id = str(uuid4())

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.side_effect = Exception("Database error")

                with pytest.raises(Exception, match="Failed to verify and deduct credits"):
                    await verify_and_deduct_credits(
                        settings=mock_settings,
                        user_id=user_id,
                        required_credits=5,
                    )

    @pytest.mark.asyncio
    async def test_verify_and_deduct_credits_exact_balance(self, mock_settings):
        """Test credit verification with exact balance."""
        user_id = str(uuid4())
        required_credits = 10
        current_balance = 10

        with patch("app.services.analysis_service.get_supabase_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            with patch("app.services.analysis_service.async_supabase_select", new_callable=AsyncMock) as mock_select:
                mock_select.return_value = [{"id": user_id, "credits": current_balance}]

                with patch("app.services.analysis_service.async_supabase_update", new_callable=AsyncMock) as mock_update:
                    result = await verify_and_deduct_credits(
                        settings=mock_settings,
                        user_id=user_id,
                        required_credits=required_credits,
                    )

                    assert result is True
                    mock_update.assert_called_once()
                    call_args = mock_update.call_args
                    assert call_args[0][2]["credits"] == 0

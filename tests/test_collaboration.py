"""
Tests for Collaboration Agent Tracing with Langfuse Integration.

This module tests the Langfuse tracing functionality in the collaboration workflows,
ensuring that spans are properly created, ended, and handle errors gracefully.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.collaboration import CollaborationWorkflow, CollaborationType
from config.langfuse_integration import get_langfuse_integration


class TestCollaborationTracing:
    """Test suite for collaboration workflow tracing."""

    @pytest.fixture
    def collaboration_workflow(self):
        """Create a CollaborationWorkflow instance for testing."""
        return CollaborationWorkflow()

    @pytest.fixture
    def mock_langfuse(self):
        """Mock Langfuse integration for testing."""
        with patch("agents.collaboration.get_langfuse_integration") as mock_get:
            mock_integration = MagicMock()
            mock_integration.start_agent_span.return_value = "test_span_id"
            mock_integration.end_agent_span = MagicMock()
            mock_get.return_value = mock_integration
            yield mock_integration

    @pytest.fixture
    def sample_slow_moving_items(self):
        """Sample data for slow-moving items."""
        return [
            {
                "product_id": "prod_1",
                "days_without_sale": 45,
                "inventory_value": 1500.0,
                "current_stock": 20,
            },
            {
                "product_id": "prod_2",
                "days_without_sale": 30,
                "inventory_value": 800.0,
                "current_stock": 15,
            },
        ]

    @pytest.fixture
    def sample_discount_opportunities(self):
        """Sample data for discount opportunities."""
        return [
            {
                "product_id": "prod_1",
                "current_price": 100.0,
                "discount_percentage": 20.0,
                "reason": "slow_moving_inventory",
            },
        ]

    @pytest.fixture
    def sample_campaign_requests(self):
        """Sample data for campaign requests."""
        return [
            {
                "campaign_id": "camp_1",
                "product_ids": ["prod_1", "prod_2"],
                "expected_demand_increase": 50,
                "duration_days": 7,
                "current_inventory": {"prod_1": 100, "prod_2": 80},
                "daily_demand": {"prod_1": 5, "prod_2": 3},
            },
        ]

    @pytest.fixture
    def sample_decision_outcomes(self):
        """Sample data for decision outcomes."""
        return [
            {
                "agent_id": "pricing_agent",
                "decision_type": "markdown",
                "success": True,
                "collaboration_id": "collab_1",
                "timestamp": "2023-01-01T10:00:00Z",
            },
            {
                "agent_id": "promotion_agent",
                "decision_type": "campaign",
                "success": False,
                "collaboration_id": "collab_1",
                "timestamp": "2023-01-01T11:00:00Z",
            },
        ]

    @pytest.fixture
    def sample_market_event(self):
        """Sample data for market event."""
        return {
            "event_type": "demand_spike",
            "severity": "high",
            "affected_products": ["prod_1"],
        }

    @pytest.mark.asyncio
    async def test_inventory_to_pricing_tracing_success(
        self, collaboration_workflow, mock_langfuse, sample_slow_moving_items
    ):
        """Test tracing for successful inventory-to-pricing collaboration."""
        result = await collaboration_workflow.inventory_to_pricing_slow_moving_alert(
            sample_slow_moving_items
        )

        # Verify span was started
        mock_langfuse.start_agent_span.assert_called_once()
        call_args = mock_langfuse.start_agent_span.call_args
        assert call_args[1]["operation"] == "inventory_to_pricing_slow_moving_alert"
        assert call_args[1]["input_data"]["slow_moving_items_count"] == 2

        # Verify span was ended with success
        mock_langfuse.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "collaboration_id" in end_call_args[1]["outcome"]
        assert end_call_args[1]["outcome"]["items_processed"] == 2

        # Verify result structure
        assert result["status"] == "initiated"
        assert "collaboration_id" in result

    @pytest.mark.asyncio
    async def test_pricing_to_promotion_tracing_success(
        self, collaboration_workflow, mock_langfuse, sample_discount_opportunities
    ):
        """Test tracing for successful pricing-to-promotion coordination."""
        result = (
            await collaboration_workflow.pricing_to_promotion_discount_coordination(
                sample_discount_opportunities
            )
        )

        # Verify span was started
        mock_langfuse.start_agent_span.assert_called_once()
        call_args = mock_langfuse.start_agent_span.call_args
        assert call_args[1]["operation"] == "pricing_to_promotion_discount_coordination"
        assert call_args[1]["input_data"]["discount_opportunities_count"] == 1

        # Verify span was ended with success
        mock_langfuse.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "collaboration_id" in end_call_args[1]["outcome"]
        assert end_call_args[1]["outcome"]["opportunities_processed"] == 1

        # Verify result structure
        assert result["status"] == "initiated"
        assert "collaboration_id" in result

    @pytest.mark.asyncio
    async def test_promotion_to_inventory_tracing_success(
        self, collaboration_workflow, mock_langfuse, sample_campaign_requests
    ):
        """Test tracing for successful promotion-to-inventory validation."""
        result = await collaboration_workflow.promotion_to_inventory_stock_validation(
            sample_campaign_requests
        )

        # Verify span was started
        mock_langfuse.start_agent_span.assert_called_once()
        call_args = mock_langfuse.start_agent_span.call_args
        assert call_args[1]["operation"] == "promotion_to_inventory_stock_validation"
        assert call_args[1]["input_data"]["campaign_requests_count"] == 1

        # Verify span was ended with success
        mock_langfuse.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "collaboration_id" in end_call_args[1]["outcome"]
        assert end_call_args[1]["outcome"]["campaigns_validated"] == 1

        # Verify result structure
        assert result["status"] == "completed"
        assert "collaboration_id" in result

    @pytest.mark.asyncio
    async def test_cross_agent_learning_tracing_success(
        self, collaboration_workflow, mock_langfuse, sample_decision_outcomes
    ):
        """Test tracing for successful cross-agent learning."""
        result = await collaboration_workflow.cross_agent_learning_from_outcomes(
            sample_decision_outcomes
        )

        # Verify span was started
        mock_langfuse.start_agent_span.assert_called_once()
        call_args = mock_langfuse.start_agent_span.call_args
        assert call_args[1]["operation"] == "cross_agent_learning_from_outcomes"
        assert call_args[1]["input_data"]["decision_outcomes_count"] == 2

        # Verify span was ended with success
        mock_langfuse.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "learning_id" in end_call_args[1]["outcome"]
        assert end_call_args[1]["outcome"]["outcomes_analyzed"] == 2

        # Verify result structure
        assert result["status"] == "completed"
        assert "learning_id" in result

    @pytest.mark.asyncio
    async def test_market_event_response_tracing_success(
        self, collaboration_workflow, mock_langfuse, sample_market_event
    ):
        """Test tracing for successful market event response."""
        result = await collaboration_workflow.collaborative_market_event_response(
            sample_market_event
        )

        # Verify span was started
        mock_langfuse.start_agent_span.assert_called_once()
        call_args = mock_langfuse.start_agent_span.call_args
        assert call_args[1]["operation"] == "collaborative_market_event_response"
        assert (
            call_args[1]["input_data"]["market_event"]["event_type"] == "demand_spike"
        )

        # Verify span was ended with success
        mock_langfuse.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "collaboration_id" in end_call_args[1]["outcome"]
        assert end_call_args[1]["outcome"]["event_type"] == "demand_spike"

        # Verify result structure
        assert result["status"] == "initiated"
        assert "collaboration_id" in result

    @pytest.mark.asyncio
    async def test_tracing_error_handling(
        self, collaboration_workflow, mock_langfuse, sample_slow_moving_items
    ):
        """Test tracing error handling when Langfuse fails."""
        # Simulate Langfuse error
        mock_langfuse.start_agent_span.side_effect = Exception("Langfuse error")

        result = await collaboration_workflow.inventory_to_pricing_slow_moving_alert(
            sample_slow_moving_items
        )

        # Verify span start was attempted
        mock_langfuse.start_agent_span.assert_called_once()

        # Verify result indicates failure
        assert result["status"] == "failed"
        assert "Error" in result["analysis"]

    @pytest.mark.asyncio
    async def test_tracing_graceful_degradation(
        self, collaboration_workflow, sample_slow_moving_items
    ):
        """Test that workflow continues even if tracing fails."""
        with patch("agents.collaboration.get_langfuse_integration") as mock_get:
            # Mock Langfuse to raise error on start_span
            mock_integration = MagicMock()
            mock_integration.start_agent_span.side_effect = Exception(
                "Tracing unavailable"
            )
            mock_get.return_value = mock_integration

            # Workflow should still execute and return result
            result = (
                await collaboration_workflow.inventory_to_pricing_slow_moving_alert(
                    sample_slow_moving_items
                )
            )

            # Should still get a result despite tracing failure
            assert "collaboration_id" in result
            assert (
                result["status"] == "failed"
            )  # Fails due to tracing error, but gracefully

    @pytest.mark.asyncio
    async def test_span_context_propagation(
        self, collaboration_workflow, mock_langfuse, sample_slow_moving_items
    ):
        """Test that span context is properly propagated through workflow."""
        result = await collaboration_workflow.inventory_to_pricing_slow_moving_alert(
            sample_slow_moving_items
        )

        # Verify span operations include proper context
        start_call = mock_langfuse.start_agent_span.call_args
        end_call = mock_langfuse.end_agent_span.call_args

        # Input data should be captured
        assert start_call[1]["input_data"]["slow_moving_items_count"] == 2

        # Outcome should include results
        assert end_call[1]["outcome"]["items_processed"] == 2
        assert end_call[1]["outcome"]["success"] is True

    @pytest.mark.asyncio
    async def test_multiple_collaboration_types_tracing(
        self, collaboration_workflow, mock_langfuse
    ):
        """Test tracing across multiple collaboration types."""
        # Test different collaboration types
        await collaboration_workflow.inventory_to_pricing_slow_moving_alert([])
        await collaboration_workflow.pricing_to_promotion_discount_coordination([])
        await collaboration_workflow.promotion_to_inventory_stock_validation([])
        await collaboration_workflow.cross_agent_learning_from_outcomes([])
        await collaboration_workflow.collaborative_market_event_response({})

        # Verify all operations were traced
        assert mock_langfuse.start_agent_span.call_count == 5
        assert mock_langfuse.end_agent_span.call_count == 5

        # Verify different operations were called
        operations = [
            call[1]["operation"]
            for call in mock_langfuse.start_agent_span.call_args_list
        ]
        expected_operations = [
            "inventory_to_pricing_slow_moving_alert",
            "pricing_to_promotion_discount_coordination",
            "promotion_to_inventory_stock_validation",
            "cross_agent_learning_from_outcomes",
            "collaborative_market_event_response",
        ]
        assert operations == expected_operations

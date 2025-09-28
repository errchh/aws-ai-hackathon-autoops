"""
Unit tests for the Inventory Agent implementation.

This module tests the inventory agent's demand forecasting, safety buffer calculations,
restock alert generation, and slow-moving inventory identification functionality.
"""

import pytest
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from agents.inventory_agent import InventoryAgent
from models.core import AgentDecision, ActionType
from config.langfuse_integration import get_langfuse_integration


class TestInventoryAgent:
    """Test suite for the Inventory Agent."""

    @pytest.fixture
    def inventory_agent(self):
        """Create an inventory agent instance for testing."""
        with (
            patch("agents.inventory_agent.BedrockModel"),
            patch("agents.inventory_agent.Agent"),
            patch("agents.inventory_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = InventoryAgent()
            agent.agent = Mock()  # Mock the Strands agent
            return agent

    @pytest.fixture
    def sample_product_data(self):
        """Sample product data for testing."""
        return {
            "id": "TEST_SKU_001",
            "name": "Test Product",
            "inventory_level": 45,
            "reorder_point": 25,
            "daily_expected_demand": 8.5,
            "demand_variance": 4.2,
            "supplier_lead_time": 7,
            "cost": 12.50,
            "current_price": 24.99,
            "order_cost": 50,
            "carrying_cost_rate": 0.20,
            "historical_sales": [
                {"date": "2024-01-01", "quantity": 8},
                {"date": "2024-01-02", "quantity": 9},
                {"date": "2024-01-03", "quantity": 7},
                {"date": "2024-01-04", "quantity": 10},
                {"date": "2024-01-05", "quantity": 8},
                {"date": "2024-01-06", "quantity": 12},
                {"date": "2024-01-07", "quantity": 6},
                {"date": "2024-01-08", "quantity": 9},
                {"date": "2024-01-09", "quantity": 8},
                {"date": "2024-01-10", "quantity": 11},
                {"date": "2024-01-11", "quantity": 7},
                {"date": "2024-01-12", "quantity": 9},
                {"date": "2024-01-13", "quantity": 8},
                {"date": "2024-01-14", "quantity": 10},
            ],
        }

    def test_forecast_demand_probabilistic_with_sufficient_data(
        self, inventory_agent, sample_product_data
    ):
        """Test probabilistic demand forecasting with sufficient historical data."""
        result = inventory_agent.forecast_demand_probabilistic(
            sample_product_data, forecast_days=30
        )

        assert result["product_id"] == "TEST_SKU_001"
        assert result["forecast_days"] == 30
        assert result["total_expected_demand"] > 0
        assert result["daily_expected_demand"] > 0
        assert result["demand_variance"] > 0
        assert result["confidence_level"] >= 0.8
        assert result["forecast_method"] == "probabilistic_with_trend_seasonality"
        assert "forecast_points" in result
        assert len(result["forecast_points"]) == 7  # First week

        # Verify forecast points structure
        for point in result["forecast_points"]:
            assert "day" in point
            assert "date" in point
            assert "expected_demand" in point
            assert "lower_bound" in point
            assert "upper_bound" in point
            assert (
                point["lower_bound"] <= point["expected_demand"] <= point["upper_bound"]
            )

    def test_forecast_demand_probabilistic_insufficient_data(self, inventory_agent):
        """Test probabilistic demand forecasting with insufficient historical data."""
        product_data = {
            "id": "TEST_SKU_002",
            "average_daily_sales": 5.0,
            "historical_sales": [
                {"date": "2024-01-01", "quantity": 5},
                {"date": "2024-01-02", "quantity": 4},
            ],
        }

        result = inventory_agent.forecast_demand_probabilistic(
            product_data, forecast_days=15
        )

        assert result["product_id"] == "TEST_SKU_002"
        assert result["forecast_method"] == "insufficient_data_fallback"
        assert result["confidence_level"] == 0.4
        assert result["total_expected_demand"] == 75.0  # 5.0 * 15
        assert result["daily_expected_demand"] == 5.0

    def test_calculate_safety_buffer(self, inventory_agent, sample_product_data):
        """Test safety buffer calculation with various service levels."""
        # Test with 95% service level
        result = inventory_agent.calculate_safety_buffer(
            sample_product_data, service_level=0.95
        )

        assert result["product_id"] == "TEST_SKU_001"
        assert result["service_level"] == 0.95
        assert result["safety_stock"] > 0
        assert result["reorder_point"] > 0
        assert result["economic_order_quantity"] > 0
        assert result["z_score"] == 1.65  # 95% service level
        assert result["lead_time_days"] == 7

        # Verify safety stock calculation logic
        expected_safety_stock = 1.65 * math.sqrt(7) * math.sqrt(4.2)
        assert abs(result["safety_stock"] - expected_safety_stock) < 1

        # Test with 99% service level
        result_99 = inventory_agent.calculate_safety_buffer(
            sample_product_data, service_level=0.99
        )
        assert result_99["z_score"] == 2.33
        assert (
            result_99["safety_stock"] > result["safety_stock"]
        )  # Higher service level = more safety stock

    def test_generate_restock_alert(self, inventory_agent, sample_product_data):
        """Test restock alert generation with different urgency levels."""
        # Test medium urgency
        result = inventory_agent.generate_restock_alert(
            sample_product_data, urgency_level="medium"
        )

        assert result["product_id"] == "TEST_SKU_001"
        assert result["urgency_level"] == "medium"
        assert result["current_stock"] == 45
        assert result["reorder_point"] == 25
        assert result["recommended_quantity"] > 0
        assert result["total_cost"] > 0
        assert result["priority_score"] >= 0
        assert "alert_id" in result
        assert "expected_delivery" in result

        # Test critical urgency - should recommend more stock
        result_critical = inventory_agent.generate_restock_alert(
            sample_product_data, urgency_level="critical"
        )
        assert result_critical["recommended_quantity"] > result["recommended_quantity"]
        assert result_critical["priority_score"] >= result["priority_score"]

    def test_identify_slow_moving_inventory(self, inventory_agent):
        """Test slow-moving inventory identification."""
        inventory_data = [
            {
                "id": "FAST_MOVING",
                "inventory_level": 50,
                "days_without_sale": 5,
                "daily_expected_demand": 10.0,
                "cost": 10.0,
                "current_price": 20.0,
            },
            {
                "id": "SLOW_MOVING_1",
                "inventory_level": 100,
                "days_without_sale": 45,
                "daily_expected_demand": 2.0,
                "cost": 15.0,
                "current_price": 30.0,
            },
            {
                "id": "SLOW_MOVING_2",
                "inventory_level": 200,
                "days_without_sale": 90,
                "daily_expected_demand": 1.0,
                "cost": 8.0,
                "current_price": 16.0,
            },
        ]

        result = inventory_agent.identify_slow_moving_inventory(
            inventory_data, threshold_days=30
        )

        assert result["total_products_analyzed"] == 3
        assert result["slow_moving_count"] == 2  # Two items exceed 30-day threshold
        assert result["slow_moving_percentage"] > 0
        assert result["total_slow_moving_value"] > 0
        assert len(result["slow_moving_items"]) == 2

        # Verify slow-moving items are correctly identified
        slow_moving_ids = [item["product_id"] for item in result["slow_moving_items"]]
        assert "SLOW_MOVING_1" in slow_moving_ids
        assert "SLOW_MOVING_2" in slow_moving_ids
        assert "FAST_MOVING" not in slow_moving_ids

        # Verify recommendations are generated
        for item in result["slow_moving_items"]:
            assert len(item["recommended_actions"]) > 0
            assert item["urgency_score"] > 0

    def test_analyze_demand_patterns(self, inventory_agent, sample_product_data):
        """Test demand pattern analysis."""
        result = inventory_agent.analyze_demand_patterns(sample_product_data)

        assert result["product_id"] == "TEST_SKU_001"
        assert result["analysis_period_days"] == 14
        assert result["mean_daily_demand"] > 0
        assert result["median_daily_demand"] > 0
        assert result["demand_std_dev"] >= 0
        assert result["coefficient_of_variation"] >= 0
        assert result["demand_stability"] in [
            "very_stable",
            "stable",
            "moderate",
            "volatile",
        ]
        assert "trend_analysis" in result
        assert "seasonality_analysis" in result
        assert "forecasting_recommendations" in result

        # Verify trend analysis structure
        trend = result["trend_analysis"]
        assert "direction" in trend
        assert "strength" in trend
        assert "confidence" in trend

        # Verify seasonality analysis structure
        seasonality = result["seasonality_analysis"]
        assert "detected" in seasonality
        assert "pattern" in seasonality

    def test_analyze_demand_patterns_insufficient_data(self, inventory_agent):
        """Test demand pattern analysis with insufficient data."""
        product_data = {
            "id": "TEST_SKU_003",
            "historical_sales": [{"date": "2024-01-01", "quantity": 5}],
        }

        result = inventory_agent.analyze_demand_patterns(product_data)

        assert result["product_id"] == "TEST_SKU_003"
        assert result["pattern_analysis"] == "insufficient_data"
        assert "Need at least 14 days" in result["analysis"]

    @patch("agents.inventory_agent.agent_memory")
    def test_retrieve_inventory_history(self, mock_memory, inventory_agent):
        """Test inventory history retrieval."""
        # Mock memory responses
        mock_memory.retrieve_similar_decisions.return_value = [
            (
                {
                    "decision": {
                        "id": "decision_1",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "parameters": {"recommended_quantity": 100},
                    },
                    "outcome": {
                        "success": True,
                        "actual_quantity": 100,
                        "stockout_prevented": True,
                    },
                },
                0.85,
            )
        ]

        mock_memory.get_agent_decision_history.return_value = [
            {
                "decision": {"id": "decision_1", "timestamp": "2024-01-01T00:00:00Z"},
                "outcome": {"success": True, "forecast_accuracy": 0.85},
            }
        ]

        result = inventory_agent.retrieve_inventory_history("TEST_SKU_001", days=30)

        assert result["product_id"] == "TEST_SKU_001"
        assert result["total_decisions"] == 1
        assert result["similar_decisions"] == 1
        assert result["success_rate"] == 1.0
        assert result["forecast_accuracy"] >= 0
        assert len(result["restock_decisions"]) == 1
        assert "insights" in result
        assert "learning_recommendations" in result

    def test_helper_methods(self, inventory_agent):
        """Test helper methods for analysis."""
        daily_sales = [8, 9, 7, 10, 8, 12, 6, 9, 8, 11, 7, 9, 8, 10]

        # Test seasonality detection
        seasonality = inventory_agent._detect_seasonality(daily_sales)
        assert isinstance(seasonality, dict)
        assert len(seasonality) <= 7  # Max 7 days of week

        # Test trend calculation
        trend = inventory_agent._calculate_trend(daily_sales)
        assert isinstance(trend, float)

        # Test forecast accuracy calculation
        mean_demand = 8.5
        std_dev = 1.5
        accuracy = inventory_agent._calculate_forecast_accuracy(
            daily_sales, mean_demand, std_dev
        )
        assert 0.1 <= accuracy <= 0.95

    @patch("agents.inventory_agent.agent_memory")
    def test_make_inventory_decision(
        self, mock_memory, inventory_agent, sample_product_data
    ):
        """Test comprehensive inventory decision making."""
        # Mock the Strands agent response
        inventory_agent.agent.return_value = "Comprehensive inventory analysis completed. Recommend restocking 150 units based on demand forecast and safety buffer calculations."

        # Mock memory storage
        mock_memory.store_decision.return_value = "memory_id_123"

        context = {
            "analysis_type": "routine_check",
            "urgency": "medium",
            "market_conditions": "normal",
        }

        decision = inventory_agent.make_inventory_decision(sample_product_data, context)

        assert isinstance(decision, AgentDecision)
        assert decision.agent_id == "inventory_agent"
        assert decision.action_type == ActionType.INVENTORY_RESTOCK
        assert decision.confidence_score == 0.88
        assert "product_id" in decision.parameters
        assert decision.parameters["analysis_performed"] is True
        assert len(decision.parameters["tools_used"]) > 0

        # Verify memory storage was called
        mock_memory.store_decision.assert_called_once()

    @patch("agents.inventory_agent.agent_memory")
    def test_update_decision_outcome(self, mock_memory, inventory_agent):
        """Test updating decision outcomes for learning."""
        # Mock decision history
        mock_memory.get_agent_decision_history.return_value = [
            {
                "decision": {"id": "test_decision_id"},
                "metadata": {"memory_id": "memory_123"},
            }
        ]

        outcome_data = {
            "success": True,
            "actual_quantity": 150,
            "stockout_prevented": True,
            "forecast_accuracy": 0.87,
        }

        inventory_agent.update_decision_outcome("test_decision_id", outcome_data)

        # Verify memory update was called
        mock_memory.update_outcome.assert_called_once_with("memory_123", outcome_data)

    def test_error_handling(self, inventory_agent):
        """Test error handling in various methods."""
        # Test with invalid product data
        invalid_data = {"id": "INVALID"}

        # Should not raise exceptions, but return fallback responses
        forecast_result = inventory_agent.forecast_demand_probabilistic(invalid_data)
        assert forecast_result["forecast_method"] == "insufficient_data_fallback"

        safety_result = inventory_agent.calculate_safety_buffer(invalid_data)
        assert safety_result["safety_stock"] >= 0  # Uses default values

        alert_result = inventory_agent.generate_restock_alert(invalid_data)
        assert alert_result["recommended_quantity"] >= 0

    def test_edge_cases(self, inventory_agent):
        """Test edge cases and boundary conditions."""
        # Test with zero demand
        zero_demand_data = {
            "id": "ZERO_DEMAND",
            "inventory_level": 100,
            "daily_expected_demand": 0,
            "demand_variance": 0,
            "supplier_lead_time": 7,
            "cost": 10,
        }

        safety_result = inventory_agent.calculate_safety_buffer(zero_demand_data)
        assert safety_result["safety_stock"] >= 0

        # Test with very high variance
        high_variance_data = {
            "id": "HIGH_VARIANCE",
            "inventory_level": 50,
            "daily_expected_demand": 10,
            "demand_variance": 100,  # Very high variance
            "supplier_lead_time": 14,
            "cost": 5,
        }

        safety_result = inventory_agent.calculate_safety_buffer(high_variance_data)
        assert safety_result["safety_stock"] > 0
        assert safety_result["reorder_point"] > safety_result["safety_stock"]

    @patch("config.langfuse_integration.get_langfuse_integration")
    def test_inventory_agent_tracing(self, mock_get_langfuse):
        """Test that inventory agent methods are properly traced with Langfuse."""
        # Mock Langfuse integration service
        mock_langfuse_service = Mock()
        mock_span_id = "test_span_123"
        mock_langfuse_service.start_agent_span.return_value = mock_span_id
        mock_get_langfuse.return_value = mock_langfuse_service

        # Create inventory agent
        with (
            patch("agents.inventory_agent.BedrockModel"),
            patch("agents.inventory_agent.Agent"),
            patch("agents.inventory_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = InventoryAgent()

            # Test forecast_demand_probabilistic tracing
            product_data = {
                "id": "TEST_PRODUCT",
                "historical_sales": [
                    {"date": "2024-01-01", "quantity": 8},
                    {"date": "2024-01-02", "quantity": 9},
                    {"date": "2024-01-03", "quantity": 7},
                    {"date": "2024-01-04", "quantity": 10},
                    {"date": "2024-01-05", "quantity": 8},
                    {"date": "2024-01-06", "quantity": 12},
                    {"date": "2024-01-07", "quantity": 6},
                    {"date": "2024-01-08", "quantity": 9},
                    {"date": "2024-01-09", "quantity": 8},
                    {"date": "2024-01-10", "quantity": 11},
                ],
            }

            result = agent.forecast_demand_probabilistic(product_data, forecast_days=7)

            # Verify tracing was called
            mock_langfuse_service.start_agent_span.assert_called()
            assert mock_langfuse_service.end_agent_span.call_count >= 1

            # Test calculate_safety_buffer tracing
            safety_result = agent.calculate_safety_buffer(product_data)

            # Verify tracing was called for safety buffer calculation
            assert mock_langfuse_service.start_agent_span.call_count >= 2
            assert mock_langfuse_service.end_agent_span.call_count >= 2

            # Test generate_restock_alert tracing
            alert_result = agent.generate_restock_alert(
                product_data, urgency_level="high"
            )

            # Verify tracing was called for restock alert
            assert mock_langfuse_service.start_agent_span.call_count >= 3
            assert mock_langfuse_service.end_agent_span.call_count >= 3

            # Test identify_slow_moving_inventory tracing
            inventory_data = [
                {
                    "id": "SLOW_PRODUCT",
                    "inventory_level": 100,
                    "days_without_sale": 45,
                    "daily_expected_demand": 2.0,
                    "cost": 15.0,
                }
            ]

            slow_result = agent.identify_slow_moving_inventory(
                inventory_data, threshold_days=30
            )

            # Verify tracing was called for slow-moving inventory identification
            assert mock_langfuse_service.start_agent_span.call_count >= 4
            assert mock_langfuse_service.end_agent_span.call_count >= 4


if __name__ == "__main__":
    pytest.main([__file__])

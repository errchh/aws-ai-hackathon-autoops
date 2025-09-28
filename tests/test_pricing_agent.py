"""
Unit tests for the Pricing Agent implementation.
"""

import pytest
from unittest.mock import Mock, patch
from agents.pricing_agent import PricingAgent
from models.core import AgentDecision, ActionType
from config.langfuse_integration import get_langfuse_integration


class TestPricingAgent:
    """Test suite for PricingAgent class."""

    @pytest.fixture
    def pricing_agent(self):
        """Create a PricingAgent instance for testing."""
        with patch("agents.pricing_agent.get_settings") as mock_settings:
            mock_settings.return_value.bedrock.model_id = (
                "anthropic.claude-3-sonnet-20240229-v1:0"
            )
            mock_settings.return_value.bedrock.temperature = 0.1
            mock_settings.return_value.bedrock.max_tokens = 4096

            with patch("agents.pricing_agent.BedrockModel"):
                with patch("agents.pricing_agent.Agent"):
                    return PricingAgent()

    def test_analyze_demand_elasticity_with_sufficient_data(self, pricing_agent):
        """Test demand elasticity analysis with sufficient historical data."""
        price_history = [
            {"price": 25.00, "quantity": 100},
            {"price": 24.00, "quantity": 110},
            {"price": 23.00, "quantity": 125},
        ]

        result = pricing_agent.analyze_demand_elasticity("SKU123", price_history)

        assert "elasticity_coefficient" in result
        assert "confidence" in result
        assert "category" in result
        assert "recommendation" in result
        assert result["data_points"] == 3
        assert result["confidence"] > 0.3

    def test_calculate_optimal_price_basic(self, pricing_agent):
        """Test basic optimal price calculation."""
        product_data = {
            "id": "SKU123",
            "current_price": 24.99,
            "cost": 12.50,
            "inventory_level": 100,
            "reorder_point": 25,
        }

        market_conditions = {
            "elasticity_coefficient": -1.2,
            "competitor_average_price": 25.50,
        }

        result = pricing_agent.calculate_optimal_price(product_data, market_conditions)

        assert "optimal_price" in result
        assert "current_price" in result
        assert "price_change_percentage" in result
        assert "profit_impact_estimate" in result
        assert result["optimal_price"] > product_data["cost"] * 1.1
        assert result["confidence_score"] > 0

    def test_apply_markdown_strategy_low_urgency(self, pricing_agent):
        """Test markdown strategy with low urgency."""
        product_data = {
            "id": "SKU123",
            "current_price": 24.99,
            "cost": 12.50,
            "inventory_level": 150,
            "days_without_sale": 7,
        }

        result = pricing_agent.apply_markdown_strategy(product_data, "low")

        assert "markdown_percentage" in result
        assert "markdown_price" in result
        assert result["markdown_percentage"] >= 10
        assert result["markdown_price"] < product_data["current_price"]
        assert result["markdown_price"] > product_data["cost"] * 1.05

    @patch("agents.pricing_agent.agent_memory")
    def test_make_pricing_decision(self, mock_memory, pricing_agent):
        """Test comprehensive pricing decision making."""
        product_data = {
            "id": "SKU123",
            "current_price": 24.99,
            "cost": 12.50,
            "inventory_level": 150,
            "days_without_sale": 5,
        }

        market_context = {
            "demand_trend": "stable",
            "competitor_activity": "normal",
            "seasonal_factor": 1.0,
        }

        mock_memory.store_decision.return_value = "memory_123"

        decision = pricing_agent.make_pricing_decision(product_data, market_context)

        assert isinstance(decision, AgentDecision)
        assert decision.agent_id == "pricing_agent"
        assert decision.action_type == ActionType.PRICE_ADJUSTMENT
        assert decision.confidence_score > 0
        assert "product_data" in decision.context

        mock_memory.store_decision.assert_called_once()

    @patch("config.langfuse_integration.get_langfuse_integration")
    def test_pricing_agent_tracing_comprehensive(self, mock_get_langfuse):
        """Test comprehensive tracing for all pricing agent methods."""
        # Mock Langfuse integration service
        mock_langfuse_service = Mock()
        mock_span_id = "test_span_123"
        mock_langfuse_service.start_agent_span.return_value = mock_span_id
        mock_get_langfuse.return_value = mock_langfuse_service

        # Create pricing agent
        with (
            patch("agents.pricing_agent.BedrockModel"),
            patch("agents.pricing_agent.Agent"),
            patch("agents.pricing_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = PricingAgent()

            # Test analyze_demand_elasticity tracing
            price_history = [
                {"price": 25.00, "quantity": 100},
                {"price": 24.00, "quantity": 110},
                {"price": 23.00, "quantity": 125},
            ]

            result = agent.analyze_demand_elasticity("SKU123", price_history)

            # Verify span was started and ended
            mock_langfuse_service.start_agent_span.assert_called()
            assert mock_langfuse_service.end_agent_span.call_count >= 1
            mock_langfuse_service.log_agent_decision.assert_called()

            # Test calculate_optimal_price tracing
            product_data = {
                "id": "SKU123",
                "current_price": 24.99,
                "cost": 12.50,
                "inventory_level": 100,
                "reorder_point": 25,
            }

            market_conditions = {
                "elasticity_coefficient": -1.2,
                "competitor_average_price": 25.50,
            }

            result = agent.calculate_optimal_price(product_data, market_conditions)

            # Verify additional span calls
            assert mock_langfuse_service.start_agent_span.call_count >= 2
            assert mock_langfuse_service.end_agent_span.call_count >= 2
            assert mock_langfuse_service.log_agent_decision.call_count >= 2

            # Test apply_markdown_strategy tracing
            markdown_data = {
                "id": "SKU123",
                "current_price": 24.99,
                "cost": 12.50,
                "inventory_level": 150,
                "days_without_sale": 7,
            }

            result = agent.apply_markdown_strategy(markdown_data, "medium")

            # Verify more span calls
            assert mock_langfuse_service.start_agent_span.call_count >= 3
            assert mock_langfuse_service.end_agent_span.call_count >= 3
            assert mock_langfuse_service.log_agent_decision.call_count >= 3

    @patch("config.langfuse_integration.get_langfuse_integration")
    def test_pricing_agent_error_tracing(self, mock_get_langfuse):
        """Test that errors are properly traced in pricing agent methods."""
        # Mock Langfuse integration service
        mock_langfuse_service = Mock()
        mock_span_id = "test_span_123"
        mock_langfuse_service.start_agent_span.return_value = mock_span_id
        mock_get_langfuse.return_value = mock_langfuse_service

        # Create pricing agent
        with (
            patch("agents.pricing_agent.BedrockModel"),
            patch("agents.pricing_agent.Agent"),
            patch("agents.pricing_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = PricingAgent()

            # Test error in analyze_demand_elasticity
            with patch.object(agent, "analyze_demand_elasticity") as mock_method:
                mock_method.side_effect = Exception("Test error")

                try:
                    agent.analyze_demand_elasticity("SKU123", [])
                except Exception:
                    pass  # Expected to fail

                # Verify error was passed to end_agent_span
                mock_langfuse_service.end_agent_span.assert_called()
                call_args = mock_langfuse_service.end_agent_span.call_args
                assert "error" in call_args.kwargs

    @patch("config.langfuse_integration.get_langfuse_integration")
    def test_pricing_agent_decision_logging(self, mock_get_langfuse):
        """Test that pricing decisions are properly logged to Langfuse."""
        # Mock Langfuse integration service
        mock_langfuse_service = Mock()
        mock_span_id = "test_span_123"
        mock_langfuse_service.start_agent_span.return_value = mock_span_id
        mock_get_langfuse.return_value = mock_langfuse_service

        # Create pricing agent
        with (
            patch("agents.pricing_agent.BedrockModel"),
            patch("agents.pricing_agent.Agent"),
            patch("agents.pricing_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = PricingAgent()

            # Test decision logging in calculate_optimal_price
            product_data = {
                "id": "SKU123",
                "current_price": 24.99,
                "cost": 12.50,
                "inventory_level": 100,
                "reorder_point": 25,
            }

            market_conditions = {
                "elasticity_coefficient": -1.2,
                "competitor_average_price": 25.50,
            }

            result = agent.calculate_optimal_price(product_data, market_conditions)

            # Verify log_agent_decision was called with correct parameters
            mock_langfuse_service.log_agent_decision.assert_called()
            call_args = mock_langfuse_service.log_agent_decision.call_args
            decision_data = call_args.kwargs["decision_data"]

            assert decision_data["type"] == "optimal_price_calculation"
            assert "inputs" in decision_data
            assert "outputs" in decision_data
            assert "confidence" in decision_data
            assert "reasoning" in decision_data

    @patch("config.langfuse_integration.get_langfuse_integration")
    def test_pricing_agent_span_context_propagation(self, mock_get_langfuse):
        """Test that span context is properly propagated between method calls."""
        # Mock Langfuse integration service
        mock_langfuse_service = Mock()
        mock_span_id = "test_span_123"
        mock_langfuse_service.start_agent_span.return_value = mock_span_id
        mock_get_langfuse.return_value = mock_langfuse_service

        # Create pricing agent
        with (
            patch("agents.pricing_agent.BedrockModel"),
            patch("agents.pricing_agent.Agent"),
            patch("agents.pricing_agent.get_settings") as mock_settings,
        ):
            mock_settings.return_value = Mock(
                bedrock=Mock(model_id="test-model", temperature=0.1, max_tokens=4096)
            )

            agent = PricingAgent()

            # Test that each method gets its own span
            product_data = {
                "id": "SKU123",
                "current_price": 24.99,
                "cost": 12.50,
                "inventory_level": 100,
                "reorder_point": 25,
            }

            market_conditions = {
                "elasticity_coefficient": -1.2,
                "competitor_average_price": 25.50,
            }

            # Call multiple methods
            agent.calculate_optimal_price(product_data, market_conditions)
            agent.get_competitor_prices("SKU123", "vitamins")
            agent.evaluate_price_impact(
                "SKU123", 26.00, {"current_price": 24.99, "cost": 12.50}
            )

            # Verify each method started its own span
            assert mock_langfuse_service.start_agent_span.call_count == 3
            assert mock_langfuse_service.end_agent_span.call_count == 3
            assert mock_langfuse_service.log_agent_decision.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])

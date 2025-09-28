"""
Unit tests for the Promotion Agent implementation.

This module tests the Promotion Agent's functionality including flash sale creation,
bundle recommendations, social sentiment analysis, and campaign orchestration.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

# Mock strands module before importing PromotionAgent
import sys
from unittest.mock import MagicMock

# Create mock strands module
mock_strands = MagicMock()
mock_strands_models = MagicMock()


class MockAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools

    def __call__(self, prompt):
        return f"Mock agent response to: {prompt[:50]}..."


class MockBedrockModel:
    def __init__(self, model_id, temperature, max_tokens, streaming):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming


def mock_tool(func):
    return func


mock_strands.Agent = MockAgent
mock_strands.tool = mock_tool
mock_strands_models.BedrockModel = MockBedrockModel

sys.modules["strands"] = mock_strands
sys.modules["strands.models"] = mock_strands_models

from agents.promotion_agent import PromotionAgent
from config.langfuse_integration import LangfuseIntegrationService


class TestPromotionAgent:
    """Test suite for Promotion Agent functionality."""

    @pytest.fixture
    def promotion_agent(self):
        """Create a promotion agent instance for testing."""
        return PromotionAgent()

    @pytest.fixture
    def sample_product(self):
        """Sample product data for testing."""
        return {
            "id": "PROD001",
            "name": "Test Product",
            "category": "electronics",
            "current_price": 100.0,
            "cost": 60.0,
            "inventory_level": 50,
        }

    @pytest.fixture
    def sample_products(self):
        """Sample products list for bundle testing."""
        return [
            {
                "id": "PROD001",
                "name": "Smartphone",
                "category": "electronics",
                "current_price": 800.0,
                "cost": 500.0,
                "inventory_level": 25,
            },
            {
                "id": "PROD002",
                "name": "Phone Case",
                "category": "accessories",
                "current_price": 25.0,
                "cost": 10.0,
                "inventory_level": 100,
            },
            {
                "id": "PROD003",
                "name": "Screen Protector",
                "category": "accessories",
                "current_price": 15.0,
                "cost": 5.0,
                "inventory_level": 200,
            },
            {
                "id": "PROD004",
                "name": "Wireless Charger",
                "category": "accessories",
                "current_price": 50.0,
                "cost": 25.0,
                "inventory_level": 75,
            },
        ]

    def test_create_flash_sale_general_audience(self, promotion_agent, sample_product):
        """Test flash sale creation for general audience."""
        result = promotion_agent.create_flash_sale(
            product_data=sample_product, duration_hours=24, target_audience="general"
        )

        assert "flash_sale_id" in result
        assert result["product_id"] == "PROD001"
        assert result["original_price"] == 100.0
        assert result["flash_sale_price"] < 100.0
        assert result["discount_percentage"] > 0
        assert result["target_audience"] == "general"
        assert result["duration_hours"] == 24
        assert result["expected_units_sold"] > 0
        assert "start_time" in result
        assert "end_time" in result
        assert "rationale" in result

    def test_create_flash_sale_budget_audience(self, promotion_agent, sample_product):
        """Test flash sale creation for budget-conscious audience."""
        result = promotion_agent.create_flash_sale(
            product_data=sample_product, duration_hours=12, target_audience="budget"
        )

        assert result["target_audience"] == "budget"
        assert (
            result["discount_percentage"] >= 20
        )  # Budget audience gets higher discounts
        assert result["duration_hours"] == 12
        assert "Massive Savings" in result["campaign_message"]

    def test_create_flash_sale_premium_audience(self, promotion_agent, sample_product):
        """Test flash sale creation for premium audience."""
        result = promotion_agent.create_flash_sale(
            product_data=sample_product, duration_hours=48, target_audience="premium"
        )

        assert result["target_audience"] == "premium"
        assert (
            result["discount_percentage"] <= 20
        )  # Premium audience gets lower discounts
        assert "Exclusive" in result["campaign_message"]

    def test_create_flash_sale_high_inventory(self, promotion_agent):
        """Test flash sale with high inventory levels."""
        high_inventory_product = {
            "id": "PROD002",
            "name": "Overstocked Item",
            "category": "general",
            "current_price": 50.0,
            "cost": 20.0,
            "inventory_level": 150,  # High inventory
        }

        result = promotion_agent.create_flash_sale(
            product_data=high_inventory_product, target_audience="general"
        )

        # High inventory should result in higher discount
        assert result["discount_percentage"] >= 20
        assert result["expected_units_sold"] > 0

    def test_create_flash_sale_maintains_margin(self, promotion_agent):
        """Test that flash sale maintains minimum profit margin."""
        low_margin_product = {
            "id": "PROD003",
            "name": "Low Margin Product",
            "category": "general",
            "current_price": 25.0,
            "cost": 23.0,  # Very high cost relative to price
            "inventory_level": 50,
        }

        result = promotion_agent.create_flash_sale(
            product_data=low_margin_product, target_audience="budget"
        )

        # Should not discount below cost + 5% margin (with floating point tolerance)
        expected_min_price = 23.0 * 1.05
        assert result["flash_sale_price"] >= expected_min_price - 0.01

    def test_generate_bundle_recommendation(self, promotion_agent, sample_products):
        """Test bundle recommendation generation."""
        anchor_product = sample_products[0]  # Smartphone
        available_products = sample_products[1:]  # Accessories

        result = promotion_agent.generate_bundle_recommendation(
            anchor_product=anchor_product, available_products=available_products
        )

        assert "bundle_id" in result
        assert result["anchor_product_id"] == "PROD001"
        assert result["bundle_feasible"] is True
        assert len(result["complementary_products"]) > 0
        assert result["bundle_price"] < result["individual_total_price"]
        assert result["discount_percentage"] > 0
        assert result["savings_amount"] > 0
        assert "bundle_name" in result
        assert "bundle_description" in result

    def test_generate_bundle_no_complementary_products(self, promotion_agent):
        """Test bundle generation when no suitable products are available."""
        anchor_product = {
            "id": "PROD001",
            "name": "Unique Product",
            "category": "unique",
            "current_price": 100.0,
            "cost": 50.0,
            "inventory_level": 10,
        }

        # No complementary products
        result = promotion_agent.generate_bundle_recommendation(
            anchor_product=anchor_product, available_products=[]
        )

        assert result["bundle_feasible"] is False
        assert len(result["complementary_products"]) == 0
        assert "No suitable complementary products" in result["analysis"]

    def test_analyze_social_sentiment_electronics(self, promotion_agent):
        """Test social sentiment analysis for electronics category."""
        result = promotion_agent.analyze_social_sentiment(
            product_category="electronics",
            keywords=["smartphone", "tech"],
            time_period_hours=24,
        )

        assert "analysis_timestamp" in result
        assert result["product_category"] == "electronics"
        assert "smartphone" in result["keywords_tracked"]
        assert "tech" in result["keywords_tracked"]
        assert result["analysis_period_hours"] == 24
        assert "overall_sentiment" in result
        assert "sentiment_trend" in result
        assert "total_mentions" in result
        assert len(result["platform_data"]) == 4  # twitter, instagram, facebook, tiktok
        assert "promotional_opportunities" in result
        assert "risk_factors" in result
        assert result["confidence_score"] > 0

    def test_analyze_social_sentiment_no_category(self, promotion_agent):
        """Test social sentiment analysis without specific category."""
        result = promotion_agent.analyze_social_sentiment(time_period_hours=12)

        assert result["product_category"] is None
        assert result["analysis_period_hours"] == 12
        assert result["total_mentions"] > 0
        assert len(result["platform_data"]) == 4
        assert len(result["promotional_opportunities"]) > 0

    def test_schedule_promotional_campaign(self, promotion_agent):
        """Test promotional campaign scheduling."""
        campaign_data = {
            "name": "Summer Sale Campaign",
            "type": "seasonal",
            "product_ids": ["PROD001", "PROD002"],
            "budget": 5000.0,
            "target_audience": "general",
        }

        result = promotion_agent.schedule_promotional_campaign(campaign_data)

        assert "campaign_id" in result
        assert result["campaign_name"] == "Summer Sale Campaign"
        assert result["campaign_type"] == "seasonal"
        assert result["status"] == "scheduled"
        assert result["total_budget"] == 5000.0
        assert result["product_ids"] == ["PROD001", "PROD002"]
        assert result["target_audience"] == "general"
        assert "start_date" in result
        assert "end_date" in result
        assert "phases" in result
        assert "timeline" in result
        assert "coordination_requirements" in result
        assert "expected_metrics" in result

    def test_schedule_campaign_with_dates(self, promotion_agent):
        """Test campaign scheduling with specific dates."""
        start_date = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        end_date = (datetime.now(timezone.utc) + timedelta(days=8)).isoformat()

        campaign_data = {
            "name": "Flash Week",
            "type": "flash_sale",
            "start_date": start_date,
            "end_date": end_date,
            "budget": 2000.0,
        }

        result = promotion_agent.schedule_promotional_campaign(campaign_data)

        assert result["duration_days"] == 7
        assert abs(result["daily_budget"] - (2000.0 / 7)) < 0.01  # Allow for rounding
        assert len(result["timeline"]) <= 7  # Shows first week

    def test_schedule_campaign_invalid_dates(self, promotion_agent):
        """Test campaign scheduling with invalid date range."""
        start_date = datetime.now(timezone.utc).isoformat()
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        campaign_data = {
            "name": "Invalid Campaign",
            "start_date": start_date,
            "end_date": end_date,
            "budget": 1000.0,
        }

        result = promotion_agent.schedule_promotional_campaign(campaign_data)

        assert result["status"] == "failed"
        assert "end date must be after start date" in result["analysis"]

    def test_evaluate_campaign_effectiveness(self, promotion_agent):
        """Test campaign effectiveness evaluation."""
        performance_data = {
            "impressions": 12000,
            "clicks": 360,
            "conversions": 18,
            "revenue": 4200.0,
            "cost": 1200.0,
        }

        result = promotion_agent.evaluate_campaign_effectiveness(
            campaign_id="CAMP001", performance_data=performance_data
        )

        assert "campaign_id" in result
        assert "effectiveness_score" in result
        assert "performance_metrics" in result
        assert "success_level" in result
        assert "recommendations" in result
        assert result["performance_metrics"]["ctr"] > 0
        assert result["performance_metrics"]["conversion_rate"] > 0
        assert result["performance_metrics"]["roi"] > 0

    def test_coordinate_with_pricing_agent(self, promotion_agent):
        """Test coordination with pricing agent."""
        coordination_request = {
            "type": "flash_sale_pricing",
            "product_ids": ["PROD001"],
            "campaign_details": {"requested_discount": 20.0, "duration_hours": 24},
        }

        result = promotion_agent.coordinate_with_pricing_agent(coordination_request)

        assert "coordination_id" in result
        assert result["request_type"] == "flash_sale_pricing"
        assert result["products_coordinated"] == 1
        assert "coordination_results" in result
        assert "overall_status" in result
        assert "success_rate" in result

    def test_validate_inventory_availability(self, promotion_agent):
        """Test inventory availability validation."""
        inventory_request = {
            "products": [
                {
                    "id": "PROD001",
                    "name": "Product 1",
                    "inventory_level": 100,
                    "daily_demand": 5,
                    "reorder_point": 25,
                },
                {
                    "id": "PROD002",
                    "name": "Product 2",
                    "inventory_level": 50,
                    "daily_demand": 3,
                    "reorder_point": 15,
                },
            ],
            "campaign_duration_days": 7,
            "demand_multiplier": 2.0,
        }

        result = promotion_agent.validate_inventory_availability(inventory_request)

        assert "validation_id" in result
        assert result["products_validated"] == 2
        assert len(result["validation_results"]) == 2
        assert "overall_availability" in result
        assert "overall_assessment" in result

    def test_retrieve_promotion_history(self, promotion_agent):
        """Test promotion history retrieval."""
        result = promotion_agent.retrieve_promotion_history(
            product_id="PROD001", campaign_type="flash_sale", days=30
        )

        assert "campaign_history" in result
        assert len(result["campaign_history"]) > 0
        assert "insights" in result
        assert "analysis" in result
        assert result["product_id"] == "PROD001"
        assert result["campaign_type"] == "flash_sale"
        assert result["analysis_period_days"] == 30

    def test_agent_initialization(self, promotion_agent):
        """Test that promotion agent initializes correctly."""
        assert promotion_agent.agent_id == "promotion_agent"
        assert promotion_agent.model is not None
        assert promotion_agent.agent is not None
        # Check that agent has tools (exact count may vary based on implementation)
        assert hasattr(promotion_agent.agent, "tool") or hasattr(
            promotion_agent.agent, "tools"
        )

    def test_make_promotional_decision(self, promotion_agent):
        """Test comprehensive promotional decision making."""
        campaign_request = {
            "campaign_type": "flash_sale",
            "product_id": "PROD001",
            "target_audience": "general",
            "duration_hours": 24,
        }

        market_context = {
            "competitor_activity": "moderate",
            "seasonal_factor": 1.2,
            "inventory_pressure": "medium",
        }

        result = promotion_agent.make_promotional_decision(
            campaign_request, market_context
        )

        assert hasattr(result, "id")
        assert hasattr(result, "agent_id")
        assert hasattr(result, "action_type")
        assert hasattr(result, "confidence_score")
        assert result.agent_id == "promotion_agent"

    @patch("agents.promotion_agent.agent_memory")
    def test_update_decision_outcome(self, mock_memory, promotion_agent):
        """Test decision outcome updating."""
        # Mock decision history
        mock_memory.get_agent_decision_history.return_value = [
            {"decision": {"id": "DEC001"}, "metadata": {"memory_id": "MEM001"}}
        ]

        outcome_data = {
            "actual_sales": 15,
            "revenue_generated": 1500.0,
            "effectiveness_score": 0.75,
        }

        promotion_agent.update_decision_outcome("DEC001", outcome_data)

        mock_memory.get_agent_decision_history.assert_called_once()
        mock_memory.update_outcome.assert_called_once_with("MEM001", outcome_data)


class TestPromotionAgentTracing:
    """Test suite for Promotion Agent Langfuse tracing functionality."""

    @pytest.fixture
    def promotion_agent(self):
        """Create a promotion agent instance for testing."""
        return PromotionAgent()

    @pytest.fixture
    def sample_product(self):
        """Sample product data for testing."""
        return {
            "id": "PROD001",
            "name": "Test Product",
            "category": "electronics",
            "current_price": 100.0,
            "cost": 60.0,
            "inventory_level": 50,
        }

    @pytest.fixture
    def sample_products(self):
        """Sample products list for bundle testing."""
        return [
            {
                "id": "PROD001",
                "name": "Smartphone",
                "category": "electronics",
                "current_price": 800.0,
                "cost": 500.0,
                "inventory_level": 25,
            },
            {
                "id": "PROD002",
                "name": "Phone Case",
                "category": "accessories",
                "current_price": 25.0,
                "cost": 10.0,
                "inventory_level": 100,
            },
            {
                "id": "PROD003",
                "name": "Screen Protector",
                "category": "accessories",
                "current_price": 15.0,
                "cost": 5.0,
                "inventory_level": 200,
            },
            {
                "id": "PROD004",
                "name": "Wireless Charger",
                "category": "accessories",
                "current_price": 50.0,
                "cost": 25.0,
                "inventory_level": 75,
            },
        ]

    @pytest.fixture
    def mock_langfuse_integration(self):
        """Mock Langfuse integration service."""
        mock_service = MagicMock(spec=LangfuseIntegrationService)
        mock_service.is_available = True
        mock_service.start_agent_span.return_value = "test_span_id"
        mock_service.end_agent_span.return_value = None
        return mock_service

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_promotion_agent_initialization_with_tracing(
        self, mock_get_integration, mock_langfuse_integration
    ):
        """Test that promotion agent initializes with Langfuse integration."""
        mock_get_integration.return_value = mock_langfuse_integration

        agent = PromotionAgent()

        assert hasattr(agent, "langfuse_integration")
        assert agent.langfuse_integration == mock_langfuse_integration
        mock_get_integration.assert_called_once()

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_create_flash_sale_with_tracing(
        self,
        mock_get_integration,
        mock_langfuse_integration,
        promotion_agent,
        sample_product,
    ):
        """Test that flash sale creation includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        result = promotion_agent.create_flash_sale(
            product_data=sample_product, duration_hours=24, target_audience="general"
        )

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "create_flash_sale"
        assert "product_id" in call_args[1]["input_data"]

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "flash_sale_id" in end_call_args[1]["outcome"]

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_create_flash_sale_tracing_error_handling(
        self, mock_get_integration, mock_langfuse_integration, promotion_agent
    ):
        """Test that flash sale creation handles tracing errors gracefully."""
        mock_get_integration.return_value = mock_langfuse_integration
        mock_langfuse_integration.start_agent_span.side_effect = Exception(
            "Tracing error"
        )

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        # Should not raise an exception even if tracing fails
        result = promotion_agent.create_flash_sale(
            product_data={"id": "PROD001", "name": "Test Product"},
            duration_hours=24,
            target_audience="general",
        )

        # Should still return a result
        assert "flash_sale_id" in result
        assert result["success"] is False

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_generate_bundle_recommendation_with_tracing(
        self,
        mock_get_integration,
        mock_langfuse_integration,
        promotion_agent,
        sample_products,
    ):
        """Test that bundle recommendation includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        result = promotion_agent.generate_bundle_recommendation(
            anchor_product=sample_products[0], available_products=sample_products[1:]
        )

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "generate_bundle_recommendation"

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_analyze_social_sentiment_with_tracing(
        self, mock_get_integration, mock_langfuse_integration, promotion_agent
    ):
        """Test that social sentiment analysis includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        result = promotion_agent.analyze_social_sentiment(
            product_category="electronics",
            keywords=["smartphone", "tech"],
            time_period_hours=24,
        )

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "analyze_social_sentiment"

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "overall_sentiment" in end_call_args[1]["outcome"]

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_schedule_promotional_campaign_with_tracing(
        self, mock_get_integration, mock_langfuse_integration, promotion_agent
    ):
        """Test that promotional campaign scheduling includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        campaign_data = {
            "name": "Test Campaign",
            "type": "flash_sale",
            "product_ids": ["PROD001", "PROD002"],
            "budget": 1000.0,
        }

        result = promotion_agent.schedule_promotional_campaign(campaign_data)

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "schedule_promotional_campaign"

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "campaign_id" in end_call_args[1]["outcome"]

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_evaluate_campaign_effectiveness_with_tracing(
        self, mock_get_integration, mock_langfuse_integration, promotion_agent
    ):
        """Test that campaign effectiveness evaluation includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        performance_data = {
            "impressions": 10000,
            "clicks": 500,
            "conversions": 50,
            "revenue": 2500.0,
            "cost": 1000.0,
        }

        result = promotion_agent.evaluate_campaign_effectiveness(
            "CAMP001", performance_data
        )

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "evaluate_campaign_effectiveness"

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "effectiveness_score" in end_call_args[1]["outcome"]

    @patch("agents.promotion_agent.get_langfuse_integration")
    @patch("agents.promotion_agent.agent_memory")
    def test_make_promotional_decision_with_tracing(
        self,
        mock_agent_memory,
        mock_get_integration,
        mock_langfuse_integration,
        promotion_agent,
    ):
        """Test that promotional decision making includes tracing."""
        mock_get_integration.return_value = mock_langfuse_integration
        mock_agent_memory.store_decision.return_value = "mock_memory_id"

        # Mock the agent to use our mock integration
        promotion_agent.langfuse_integration = mock_langfuse_integration

        campaign_request = {
            "type": "flash_sale",
            "product_ids": ["PROD001"],
            "budget": 500.0,
        }

        market_context = {"social_sentiment": "positive", "health_trends": "stable"}

        result = promotion_agent.make_promotional_decision(
            campaign_request, market_context
        )

        # Verify tracing was called
        mock_langfuse_integration.start_agent_span.assert_called_once()
        call_args = mock_langfuse_integration.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "promotion_agent"
        assert call_args[1]["operation"] == "make_promotional_decision"

        # Verify span was ended with success
        mock_langfuse_integration.end_agent_span.assert_called_once()
        end_call_args = mock_langfuse_integration.end_agent_span.call_args
        assert end_call_args[1]["outcome"]["success"] is True
        assert "decision_id" in end_call_args[1]["outcome"]

    @patch("agents.promotion_agent.get_langfuse_integration")
    def test_tracing_graceful_degradation_when_unavailable(
        self, mock_get_integration, promotion_agent
    ):
        """Test that agent continues to function when Langfuse is unavailable."""
        # Mock unavailable Langfuse integration
        mock_integration = MagicMock(spec=LangfuseIntegrationService)
        mock_integration.is_available = False
        mock_get_integration.return_value = mock_integration

        # Agent should still work even with unavailable tracing
        result = promotion_agent.create_flash_sale(
            product_data={"id": "PROD001", "name": "Test Product"},
            duration_hours=24,
            target_audience="general",
        )

        # Should return a result despite tracing being unavailable
        assert "flash_sale_id" in result


if __name__ == "__main__":
    pytest.main([__file__])

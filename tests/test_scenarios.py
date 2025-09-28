"""
Automated tests for demonstration scenarios.

This module contains comprehensive tests that validate the effectiveness
of each demonstration scenario and ensure they produce expected outcomes.
"""

import pytest
from datetime import datetime
from typing import List

from scenarios.base import ScenarioResult
from scenarios.demand_spike import DemandSpikeScenario
from scenarios.price_war import PriceWarScenario
from scenarios.seasonal_inventory import SeasonalInventoryScenario
from scenarios.flash_sale import FlashSaleScenario
from scenarios.waste_reduction import WasteReductionScenario
from scenarios.data_generator import ScenarioDataGenerator

from models.core import SystemMetrics


class TestScenarioBase:
    """Base test class with common scenario testing utilities."""

    def assert_scenario_result_valid(self, result: ScenarioResult):
        """Assert that a scenario result is valid and complete."""
        assert result.success, f"Scenario failed: {result.error_message}"
        assert result.duration_seconds > 0, "Scenario duration should be positive"
        assert result.initial_metrics is not None, "Initial metrics should be present"
        assert result.final_metrics is not None, "Final metrics should be present"
        assert len(result.decisions_made) > 0, "Scenario should produce agent decisions"
        assert len(result.key_insights) > 0, "Scenario should provide insights"

    def assert_metrics_improved(
        self,
        initial: SystemMetrics,
        final: SystemMetrics,
        expected_improvements: List[str],
    ):
        """Assert that key metrics improved as expected."""
        for improvement in expected_improvements:
            if improvement == "revenue":
                assert final.total_revenue >= initial.total_revenue, (
                    "Revenue should not decrease"
                )
            elif improvement == "profit":
                # Allow small profit decreases in some scenarios (e.g., promotional discounts)
                profit_change = (
                    final.total_profit - initial.total_profit
                ) / initial.total_profit
                assert profit_change > -0.2, "Profit decrease should be less than 20%"
            elif improvement == "inventory_turnover":
                assert final.inventory_turnover >= initial.inventory_turnover, (
                    "Inventory turnover should improve"
                )
            elif improvement == "waste_reduction":
                assert (
                    final.waste_reduction_percentage
                    >= initial.waste_reduction_percentage
                ), "Waste reduction should improve"
            elif improvement == "stockout_prevention":
                assert final.stockout_incidents <= initial.stockout_incidents, (
                    "Stockouts should not increase"
                )


class TestDemandSpikeScenario(TestScenarioBase):
    """Test the demand spike response scenario."""

    def test_scenario_execution(self):
        """Test that the demand spike scenario executes successfully."""
        scenario = DemandSpikeScenario()
        result = scenario.run()

        self.assert_scenario_result_valid(result)

        # Check specific demand spike outcomes
        assert result.final_metrics.total_revenue > result.initial_metrics.total_revenue
        assert (
            result.final_metrics.inventory_turnover
            > result.initial_metrics.inventory_turnover
        )
        assert (
            result.final_metrics.agent_collaboration_score > 0.9
        )  # High coordination expected

        # Check for key decisions
        decision_types = [d.action_type.value for d in result.decisions_made]
        assert "stock_alert" in decision_types
        assert "price_adjustment" in decision_types
        assert "promotion_creation" in decision_types
        assert "inventory_restock" in decision_types

    def test_scenario_insights(self):
        """Test that demand spike scenario provides meaningful insights."""
        scenario = DemandSpikeScenario()
        result = scenario.run()

        insights = result.key_insights
        assert len(insights) >= 3

        # Check for expected insight types
        insight_text = " ".join(insights).lower()
        assert "revenue" in insight_text or "profit" in insight_text
        assert "collaboration" in insight_text or "coordination" in insight_text

    def test_market_events_generated(self):
        """Test that appropriate market events are generated."""
        scenario = DemandSpikeScenario()
        result = scenario.run()

        events = result.market_events
        assert len(events) > 0

        # Should have demand spike event
        event_types = [e.event_type.value for e in events]
        assert "demand_spike" in event_types


class TestPriceWarScenario(TestScenarioBase):
    """Test the price war scenario."""

    def test_scenario_execution(self):
        """Test that the price war scenario executes successfully."""
        scenario = PriceWarScenario()
        result = scenario.run()

        self.assert_scenario_result_valid(result)

        # Price wars typically increase volume but may pressure margins
        assert result.final_metrics.total_revenue > result.initial_metrics.total_revenue
        assert result.final_metrics.price_optimization_score > 0.8  # Strategic pricing

    def test_competitor_response_handling(self):
        """Test that the scenario properly handles competitor price changes."""
        scenario = PriceWarScenario()
        result = scenario.run()

        # Should have competitor price change events
        event_types = [e.event_type.value for e in result.market_events]
        assert "competitor_price_change" in event_types

        # Should have price adjustment decisions
        decision_types = [d.action_type.value for d in result.decisions_made]
        assert "price_adjustment" in decision_types


class TestSeasonalInventoryScenario(TestScenarioBase):
    """Test the seasonal inventory management scenario."""

    def test_seasonal_preparation(self):
        """Test that the scenario properly prepares for seasonal demand."""
        scenario = SeasonalInventoryScenario()
        result = scenario.run()

        self.assert_scenario_result_valid(result)

        # Seasonal scenarios should show strong inventory and revenue improvements
        assert (
            result.final_metrics.inventory_turnover
            > result.initial_metrics.inventory_turnover
        )
        assert result.final_metrics.total_revenue > result.initial_metrics.total_revenue
        assert (
            result.final_metrics.waste_reduction_percentage
            > result.initial_metrics.waste_reduction_percentage
        )

    def test_seasonal_events(self):
        """Test that seasonal events are properly generated."""
        scenario = SeasonalInventoryScenario()
        result = scenario.run()

        event_types = [e.event_type.value for e in result.market_events]
        assert "seasonal_change" in event_types


class TestFlashSaleScenario(TestScenarioBase):
    """Test the flash sale coordination scenario."""

    def test_flash_sale_coordination(self):
        """Test that flash sale shows excellent coordination."""
        scenario = FlashSaleScenario()
        result = scenario.run()

        self.assert_scenario_result_valid(result)

        # Flash sales should show exceptional performance
        assert result.final_metrics.agent_collaboration_score > 0.95
        assert result.final_metrics.total_revenue > result.initial_metrics.total_revenue
        assert (
            result.final_metrics.inventory_turnover
            > result.initial_metrics.inventory_turnover
        )

    def test_promotional_effectiveness(self):
        """Test that promotional metrics improve significantly."""
        scenario = FlashSaleScenario()
        result = scenario.run()

        assert (
            result.final_metrics.promotion_effectiveness
            > result.initial_metrics.promotion_effectiveness
        )
        assert (
            result.final_metrics.price_optimization_score
            > result.initial_metrics.price_optimization_score
        )


class TestWasteReductionScenario(TestScenarioBase):
    """Test the waste reduction scenario."""

    def test_waste_reduction_outcomes(self):
        """Test that waste reduction scenario improves key metrics."""
        scenario = WasteReductionScenario()
        result = scenario.run()

        self.assert_scenario_result_valid(result)

        # Should show significant waste reduction
        assert (
            result.final_metrics.waste_reduction_percentage
            > result.initial_metrics.waste_reduction_percentage
        )
        assert (
            result.final_metrics.inventory_turnover
            > result.initial_metrics.inventory_turnover
        )

    def test_markdown_strategies(self):
        """Test that appropriate markdown decisions are made."""
        scenario = WasteReductionScenario()
        result = scenario.run()

        decision_types = [d.action_type.value for d in result.decisions_made]
        assert (
            "markdown_application" in decision_types
            or "price_adjustment" in decision_types
        )


class TestScenarioDataGenerator:
    """Test the scenario data generator."""

    def test_product_catalog_generation(self):
        """Test that product catalog is generated correctly."""
        generator = ScenarioDataGenerator(seed=42)
        products = generator.generate_product_catalog(10)

        assert len(products) == 10
        for product in products:
            assert product.id.startswith("SKU")
            assert product.current_price > 0
            assert product.inventory_level >= 0
            assert product.reorder_point >= 0

    def test_market_events_generation(self):
        """Test that market events are generated with proper structure."""
        generator = ScenarioDataGenerator(seed=42)
        events = generator.generate_market_events(5)

        assert len(events) == 5
        for event in events:
            assert len(event.affected_products) > 0
            assert 0 <= event.impact_magnitude <= 1
            assert event.metadata is not None

    def test_agent_decisions_generation(self):
        """Test that agent decisions are generated correctly."""
        generator = ScenarioDataGenerator(seed=42)
        decisions = generator.generate_agent_decisions(8)

        assert len(decisions) == 8
        for decision in decisions:
            assert decision.agent_id in [
                "pricing_agent",
                "inventory_agent",
                "promotion_agent",
                "orchestrator",
            ]
            assert decision.confidence_score >= 0
            assert decision.confidence_score <= 1
            assert len(decision.rationale) > 10

    def test_performance_metrics_generation(self):
        """Test that performance metrics are generated over time."""
        generator = ScenarioDataGenerator(seed=42)
        metrics = generator.generate_performance_metrics(7)

        assert len(metrics) == 7
        for i in range(1, len(metrics)):
            # Timestamps should be sequential
            assert metrics[i].timestamp > metrics[i - 1].timestamp

            # Business rules should be maintained
            assert metrics[i].total_profit <= metrics[i].total_revenue

    def test_data_persistence(self, tmp_path):
        """Test that generated data can be saved and loaded."""
        generator = ScenarioDataGenerator(seed=42)

        # Generate and save data
        test_dir = tmp_path / "test_data"
        generator.save_test_data(str(test_dir))

        # Load data back
        loaded_data = generator.load_test_data(str(test_dir))

        # Verify all data types are present
        expected_keys = [
            "products",
            "market_events",
            "agent_decisions",
            "performance_metrics",
            "collaboration_requests",
        ]
        for key in expected_keys:
            assert key in loaded_data
            assert len(loaded_data[key]) > 0


class TestScenarioIntegration:
    """Integration tests for multiple scenarios."""

    def test_all_scenarios_executable(self):
        """Test that all scenarios can be executed successfully."""
        scenarios = [
            DemandSpikeScenario(),
            PriceWarScenario(),
            SeasonalInventoryScenario(),
            FlashSaleScenario(),
            WasteReductionScenario(),
        ]

        for scenario in scenarios:
            result = scenario.run()
            assert result.success, (
                f"Scenario {scenario.name} failed: {result.error_message}"
            )
            assert result.duration_seconds > 0

    def test_scenario_metrics_consistency(self):
        """Test that all scenarios produce consistent metric improvements."""
        scenarios = [
            DemandSpikeScenario(),
            PriceWarScenario(),
            SeasonalInventoryScenario(),
            FlashSaleScenario(),
            WasteReductionScenario(),
        ]

        for scenario in scenarios:
            result = scenario.run()

            # All scenarios should maintain business rules
            assert (
                result.final_metrics.total_profit <= result.final_metrics.total_revenue
            )
            assert result.final_metrics.inventory_turnover >= 0
            assert 0 <= result.final_metrics.waste_reduction_percentage <= 100

            # All scores should be valid
            scores = [
                result.final_metrics.price_optimization_score,
                result.final_metrics.promotion_effectiveness,
                result.final_metrics.agent_collaboration_score,
            ]
            for score in scores:
                assert 0 <= score <= 1

    def test_scenario_decision_diversity(self):
        """Test that scenarios produce diverse agent decisions."""
        scenario = DemandSpikeScenario()
        result = scenario.run()

        # Should have decisions from multiple agents
        agent_ids = set(d.agent_id for d in result.decisions_made)
        assert len(agent_ids) >= 2  # At least 2 different agents

        # Should have different action types
        action_types = set(d.action_type.value for d in result.decisions_made)
        assert len(action_types) >= 3  # At least 3 different action types


if __name__ == "__main__":
    pytest.main([__file__])

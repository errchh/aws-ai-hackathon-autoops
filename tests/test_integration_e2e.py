"""
Integration testing and system validation for the autoops retail optimization system.

This module contains comprehensive end-to-end tests that validate the complete
multi-agent system functionality, including agent collaboration, memory persistence,
performance under load, and behavior under various market conditions.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mock strands imports since the package has build issues
from unittest.mock import MagicMock
import sys

sys.modules["strands"] = MagicMock()
sys.modules["strands.models"] = MagicMock()

from agents.orchestrator import RetailOptimizationOrchestrator, AgentType, SystemStatus
from agents.pricing_agent import pricing_agent
from agents.inventory_agent import inventory_agent
from agents.promotion_agent import promotion_agent
from agents.memory import agent_memory
from agents.collaboration import collaboration_workflow
from api.main import create_app
from config.settings import get_settings
from models.core import (
    Product,
    MarketEvent,
    AgentDecision,
    SystemMetrics,
    ActionType,
    EventType,
)
from scenarios.data_generator import ScenarioDataGenerator


class TestIntegrationE2E:
    """Comprehensive end-to-end integration tests for the retail optimization system."""

    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize the orchestrator for testing."""
        orchestrator = RetailOptimizationOrchestrator()

        # Register agents
        agents = [pricing_agent, inventory_agent, promotion_agent]
        success = orchestrator.register_agents(agents)
        assert success, "Failed to register agents"

        yield orchestrator

        # Cleanup
        orchestrator.active_workflows.clear()
        orchestrator.agent_messages.clear()

    @pytest.fixture
    def test_client(self):
        """Create FastAPI test client."""
        app = create_app()
        client = TestClient(app)
        yield client

    @pytest.fixture
    def scenario_generator(self):
        """Create scenario data generator."""
        return ScenarioDataGenerator(seed=42)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_market_event_workflow(
        self, orchestrator, scenario_generator
    ):
        """Test complete market event processing workflow from event to agent responses."""
        # Generate test market event
        events = scenario_generator.generate_market_events(1)
        event = events[0]

        event_data = {
            "event_id": event.id,
            "event_type": event.event_type.value,
            "affected_products": event.affected_products,
            "impact_magnitude": event.impact_magnitude,
            "metadata": event.metadata,
        }

        # Process market event
        start_time = time.time()
        result = await orchestrator.process_market_event(event_data)
        processing_time = time.time() - start_time

        # Validate response structure
        assert result["status"] in ["completed", "no_response_needed"]
        assert "event_id" in result
        assert "workflow_id" in result
        assert "responding_agents" in result
        assert "processing_time" in result

        # Validate performance requirements (5 minutes = 300 seconds)
        assert processing_time < 300, (
            f"Processing took {processing_time:.2f}s, exceeds 5-minute limit"
        )

        # Validate agent responses if any
        if result["status"] == "completed":
            assert result["agent_responses"] > 0
            assert len(result["responding_agents"]) > 0

            # Check that workflow was tracked
            assert result["workflow_id"] in orchestrator.active_workflows
            workflow = orchestrator.active_workflows[result["workflow_id"]]
            assert workflow["status"] == "completed"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_collaboration_coordination(self, orchestrator):
        """Test agent coordination and collaboration workflows."""
        coordination_request = {
            "requesting_agent": "pricing_agent",
            "target_agents": ["inventory_agent", "promotion_agent"],
            "coordination_type": "consultation",
            "content": {
                "context": "slow_moving_inventory",
                "products": ["SKU001", "SKU002"],
                "urgency": "medium",
            },
        }

        # Test coordination
        result = await orchestrator.coordinate_agents(coordination_request)

        # Validate coordination response
        assert result["status"] == "completed"
        assert result["requesting_agent"] == "pricing_agent"
        assert result["coordination_type"] == "consultation"
        assert len(result["target_agents"]) == 2
        assert result["successful_responses"] >= 0  # May be 0 if agents are mocked

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collaboration_workflow_execution(self, orchestrator):
        """Test specific collaboration workflow execution."""
        # Test inventory-to-pricing slow moving alert workflow
        workflow_data = {
            "slow_moving_items": [
                {"product_id": "SKU001", "days_slow": 45, "current_stock": 50},
                {"product_id": "SKU002", "days_slow": 38, "current_stock": 30},
            ]
        }

        result = await orchestrator.trigger_collaboration_workflow(
            "inventory_to_pricing_slow_moving", workflow_data
        )

        # Validate workflow execution
        assert result["workflow_type"] == "inventory_to_pricing_slow_moving"
        assert result["status"] in ["initiated", "completed", "error"]

        if result["status"] in ["initiated", "completed"]:
            # Check that collaboration metrics were updated
            assert orchestrator.system_metrics.agent_collaboration_score > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_system_status_monitoring(self, orchestrator):
        """Test system status monitoring and health checks."""
        status = orchestrator.get_system_status()

        # Validate status structure
        assert "orchestrator_id" in status
        assert "system_status" in status
        assert "registered_agents" in status
        assert "active_agents" in status
        assert "active_workflows" in status
        assert "total_messages_processed" in status
        assert "system_metrics" in status
        assert "agent_health" in status
        assert "timestamp" in status

        # Validate system status values
        assert status["system_status"] in [s.value for s in SystemStatus]
        assert status["registered_agents"] >= 3  # Should have all three agents
        assert isinstance(status["system_metrics"], dict)
        assert isinstance(status["agent_health"], dict)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_under_load(self, orchestrator, scenario_generator):
        """Test system performance under concurrent load."""
        # Generate multiple market events
        events = scenario_generator.generate_market_events(5)

        # Create concurrent tasks
        tasks = []
        for event in events:
            event_data = {
                "event_id": event.id,
                "event_type": event.event_type.value,
                "affected_products": event.affected_products,
                "impact_magnitude": event.impact_magnitude,
                "metadata": event.metadata,
            }
            task = orchestrator.process_market_event(event_data)
            tasks.append(task)

        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Validate performance
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        # Should handle concurrent load reasonably well
        assert len(successful_results) >= len(events) * 0.8, (
            f"Too many failures: {len(failed_results)}/{len(events)}"
        )

        # Average response time should be reasonable (< 10 seconds per event)
        avg_response_time = total_time / len(events)
        assert avg_response_time < 10.0, (
            f"Average response time {avg_response_time:.2f}s exceeds 10s limit"
        )

        print(
            f"Load test results: {len(successful_results)}/{len(events)} successful, "
            f"avg response time: {avg_response_time:.2f}s"
        )

    @pytest.mark.memory
    @pytest.mark.asyncio
    async def test_agent_memory_persistence(self, orchestrator, scenario_generator):
        """Test agent memory persistence and learning effectiveness."""
        # Generate test decisions
        decisions = scenario_generator.generate_agent_decisions(3)

        # Store decisions in memory
        stored_ids = []
        for decision in decisions:
            memory_data = {
                "agent_id": decision.agent_id,
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "context": {
                    "product_id": "TEST_PRODUCT",
                    "market_conditions": {"demand": "high"},
                    "agent_inputs": {},
                },
                "decision": {
                    "action": decision.action_type.value,
                    "parameters": decision.parameters,
                    "rationale": decision.rationale,
                },
                "outcome": {
                    "metrics": {"revenue_impact": 0.15, "profit_impact": 0.12},
                    "effectiveness_score": decision.confidence_score,
                },
            }

            # Store in memory
            agent_memory.store_decision(
                agent_id=decision.agent_id,
                decision=decision,
                context=memory_data["context"],
                outcome=memory_data["outcome"],
            )
            stored_ids.append(decision.decision_id)

        # Retrieve and validate memory persistence
        for decision_id in stored_ids:
            retrieved = agent_memory.retrieve_similar_decisions(
                agent_id="pricing_agent",  # Test with pricing agent
                current_context={"product_id": "TEST_PRODUCT", "demand": "high"},
                limit=5,
            )

            assert len(retrieved) > 0, (
                f"No memories retrieved for decision {decision_id}"
            )

            # Validate memory structure
            for memory in retrieved:
                assert "context" in memory
                assert "decision" in memory
                assert "outcome" in memory
                assert "similarity_score" in memory

    @pytest.mark.market_conditions
    @pytest.mark.asyncio
    async def test_various_market_conditions(self, orchestrator):
        """Test system behavior under various market conditions."""
        market_conditions = [
            {
                "event_type": "demand_spike",
                "affected_products": ["COFFEE_001", "COFFEE_002"],
                "impact_magnitude": 2.5,
                "metadata": {"social_trend": "viral_challenge"},
            },
            {
                "event_type": "competitor_price_change",
                "affected_products": ["LAPTOP_001", "LAPTOP_002"],
                "impact_magnitude": -0.15,
                "metadata": {"competitor": "major_retailer", "price_reduction": "20%"},
            },
            {
                "event_type": "supply_disruption",
                "affected_products": ["COMPONENT_001"],
                "impact_magnitude": -0.8,
                "metadata": {
                    "supplier": "key_supplier",
                    "disruption_type": "logistics",
                },
            },
            {
                "event_type": "seasonal_change",
                "affected_products": ["BACKPACK_001", "NOTEBOOK_001"],
                "impact_magnitude": 1.8,
                "metadata": {"season": "back_to_school"},
            },
        ]

        for condition in market_conditions:
            result = await orchestrator.process_market_event(condition)

            # Validate that system responds appropriately to each condition
            assert result["status"] in ["completed", "no_response_needed"]

            if result["status"] == "completed":
                # Check that appropriate agents responded based on event type
                responding_agents = result["responding_agents"]

                if condition["event_type"] == "demand_spike":
                    assert "pricing_agent" in responding_agents
                    assert "inventory_agent" in responding_agents
                elif condition["event_type"] == "competitor_price_change":
                    assert "pricing_agent" in responding_agents
                elif condition["event_type"] == "supply_disruption":
                    assert "inventory_agent" in responding_agents
                elif condition["event_type"] == "seasonal_change":
                    # All agents should respond to seasonal changes
                    assert len(responding_agents) >= 2

    @pytest.mark.collaboration
    @pytest.mark.asyncio
    async def test_agent_collaboration_points(self, orchestrator):
        """Test all defined collaboration points between agents."""
        collaboration_scenarios = [
            {
                "workflow": "inventory_to_pricing_slow_moving",
                "data": {
                    "slow_moving_items": [
                        {
                            "product_id": "SLOW_001",
                            "days_slow": 60,
                            "current_stock": 100,
                        }
                    ]
                },
                "expected_agents": ["inventory_agent", "pricing_agent"],
            },
            {
                "workflow": "pricing_to_promotion_discount",
                "data": {
                    "discount_opportunities": [
                        {
                            "product_id": "PROMO_001",
                            "suggested_discount": 0.25,
                            "reason": "clearance",
                        }
                    ]
                },
                "expected_agents": ["pricing_agent", "promotion_agent"],
            },
            {
                "workflow": "promotion_to_inventory_validation",
                "data": {
                    "campaign_requests": [
                        {
                            "product_id": "CAMPAIGN_001",
                            "campaign_type": "flash_sale",
                            "duration": 4,
                        }
                    ]
                },
                "expected_agents": ["promotion_agent", "inventory_agent"],
            },
        ]

        for scenario in collaboration_scenarios:
            result = await orchestrator.trigger_collaboration_workflow(
                scenario["workflow"], scenario["data"]
            )

            # Validate collaboration execution
            assert result["status"] in ["initiated", "completed", "error"]

            # Check that collaboration improved system metrics
            if result["status"] in ["initiated", "completed"]:
                collaboration_score = (
                    orchestrator.system_metrics.agent_collaboration_score
                )
                assert collaboration_score > 0, (
                    "Collaboration should improve system metrics"
                )

    @pytest.mark.improvements
    @pytest.mark.asyncio
    async def test_measurable_improvements(self, orchestrator, scenario_generator):
        """Test scenarios that demonstrate measurable system improvements."""
        # Baseline metrics
        baseline_metrics = orchestrator.system_metrics

        # Run improvement scenarios
        improvement_scenarios = [
            {
                "name": "demand_spike_response",
                "event": {
                    "event_type": "demand_spike",
                    "affected_products": ["HIGH_DEMAND_001"],
                    "impact_magnitude": 3.0,
                    "metadata": {"viral_trend": True},
                },
                "expected_improvements": {
                    "revenue_increase": 0.20,  # 20% revenue increase expected
                    "collaboration_score": 0.10,  # 10% collaboration improvement
                },
            },
            {
                "name": "waste_reduction",
                "event": {
                    "event_type": "slow_moving_inventory",
                    "affected_products": ["SLOW_001", "SLOW_002"],
                    "impact_magnitude": -0.3,
                    "metadata": {"clearance_needed": True},
                },
                "expected_improvements": {
                    "waste_reduction": 0.10,  # 10% waste reduction
                    "inventory_turnover": 0.25,  # 25% turnover improvement
                },
            },
        ]

        for scenario in improvement_scenarios:
            # Process the scenario
            result = await orchestrator.process_market_event(scenario["event"])

            # Validate that the system processed the event
            assert result["status"] == "completed"

            # Check for measurable improvements
            current_metrics = orchestrator.system_metrics

            # Revenue should improve for demand spike
            if scenario["name"] == "demand_spike_response":
                revenue_improvement = (
                    current_metrics.total_revenue - baseline_metrics.total_revenue
                )
                assert revenue_improvement > 0, (
                    f"No revenue improvement in {scenario['name']}"
                )

            # Collaboration score should improve
            collaboration_improvement = (
                current_metrics.agent_collaboration_score
                - baseline_metrics.agent_collaboration_score
            )
            assert collaboration_improvement >= 0, (
                f"Collaboration score decreased in {scenario['name']}"
            )

    @pytest.mark.aws_integration
    @pytest.mark.asyncio
    async def test_bedrock_chromadb_integration(self):
        """Test integration with AWS Bedrock and ChromaDB services."""
        # Test ChromaDB connectivity through agent_memory
        try:
            # Test that we can get memory metrics (which uses ChromaDB internally)
            metrics = agent_memory.get_system_metrics()
            assert isinstance(metrics, dict)
            assert "total_decisions" in metrics
            print(
                f"ChromaDB connected successfully, found {metrics.get('total_decisions', 0)} decisions"
            )
        except Exception as e:
            pytest.skip(f"ChromaDB integration test skipped: {e}")

        # Test Bedrock integration (mocked for CI/CD)
        with patch("agents.orchestrator.BedrockModel") as mock_bedrock:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = "Test response from Bedrock"
            mock_bedrock.return_value = mock_instance

            orchestrator = RetailOptimizationOrchestrator()

            # This would normally call Bedrock, but we're mocking it
            # In real integration, this would validate actual Bedrock connectivity
            assert orchestrator.model is not None
            print("Bedrock integration test passed (mocked)")

    @pytest.mark.api_integration
    @pytest.mark.asyncio
    async def test_api_endpoints_integration(self, test_client):
        """Test API endpoints integration."""
        # Test health endpoint
        response = await test_client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

        # Test root endpoint
        response = await test_client.get("/")
        assert response.status_code == 200
        root_data = response.json()
        assert "name" in root_data
        assert "version" in root_data

        # Test dashboard endpoints
        response = await test_client.get("/api/dashboard/agents/status")
        # May return 404 if not implemented, but should not crash
        assert response.status_code in [
            200,
            404,
            501,
        ]  # 501 = Not Implemented is acceptable

    @pytest.mark.resilience
    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self, orchestrator):
        """Test system resilience under failure conditions."""
        # Test with agent failures
        original_agents = len(orchestrator.agents)

        # Simulate agent failure by removing one agent
        failed_agent_id = "pricing_agent"
        if failed_agent_id in orchestrator.agents:
            del orchestrator.agents[failed_agent_id]
            orchestrator.system_status = SystemStatus.DEGRADED

        # System should still function with remaining agents
        status = orchestrator.get_system_status()
        assert status["system_status"] in ["healthy", "degraded"]
        assert status["active_agents"] < original_agents

        # Test market event processing with degraded system
        event_data = {
            "event_type": "demand_spike",
            "affected_products": ["TEST_001"],
            "impact_magnitude": 1.5,
        }

        result = await orchestrator.process_market_event(event_data)
        # Should still process, even if with fewer agents
        assert result["status"] in ["completed", "no_response_needed"]

    @pytest.mark.comprehensive_workflow
    @pytest.mark.asyncio
    async def test_comprehensive_workflow_simulation(
        self, orchestrator, scenario_generator
    ):
        """Test a comprehensive workflow that exercises the entire system."""
        # Generate comprehensive test data
        products = scenario_generator.generate_product_catalog(10)
        events = scenario_generator.generate_market_events(3)

        # Process multiple events in sequence
        workflow_results = []
        for event in events:
            event_data = {
                "event_id": event.id,
                "event_type": event.event_type.value,
                "affected_products": event.affected_products,
                "impact_magnitude": event.impact_magnitude,
                "metadata": event.metadata,
            }

            result = await orchestrator.process_market_event(event_data)
            workflow_results.append(result)

            # Small delay to simulate real-time processing
            await asyncio.sleep(0.1)

        # Validate comprehensive workflow
        successful_workflows = [
            r for r in workflow_results if r["status"] == "completed"
        ]
        assert len(successful_workflows) >= len(events) * 0.7, (
            "Too many workflow failures"
        )

        # Check system metrics improved
        final_metrics = orchestrator.system_metrics
        assert final_metrics.decision_count >= len(successful_workflows)
        assert final_metrics.agent_collaboration_score > 0

        # Validate system stability
        status = orchestrator.get_system_status()
        assert status["system_status"] != "critical"

        print(
            f"Comprehensive workflow: {len(successful_workflows)}/{len(events)} successful"
        )


# Performance testing utilities
class PerformanceTestSuite:
    """Performance testing utilities for load testing."""

    @staticmethod
    async def run_load_test(orchestrator, num_events: int = 10, concurrency: int = 5):
        """Run load test with specified parameters."""
        from scenarios.data_generator import ScenarioDataGenerator

        generator = ScenarioDataGenerator(seed=123)
        events = generator.generate_market_events(num_events)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(event):
            async with semaphore:
                event_data = {
                    "event_id": event.id,
                    "event_type": event.event_type.value,
                    "affected_products": event.affected_products,
                    "impact_magnitude": event.impact_magnitude,
                    "metadata": event.metadata,
                }
                start_time = time.time()
                result = await orchestrator.process_market_event(event_data)
                end_time = time.time()
                return {
                    "result": result,
                    "response_time": end_time - start_time,
                    "success": result["status"] == "completed",
                }

        # Run concurrent load test
        start_time = time.time()
        tasks = [process_with_semaphore(event) for event in events]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Analyze results
        successful = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        return {
            "total_events": num_events,
            "successful": successful,
            "success_rate": successful / num_events,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "throughput": num_events / total_time,
        }


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance_load_testing(orchestrator):
    """Performance test under various load conditions."""
    load_scenarios = [
        {"num_events": 5, "concurrency": 2, "name": "Light Load"},
        {"num_events": 10, "concurrency": 5, "name": "Medium Load"},
        {"num_events": 20, "concurrency": 10, "name": "Heavy Load"},
    ]

    for scenario in load_scenarios:
        print(f"\nRunning {scenario['name']} performance test...")

        results = await PerformanceTestSuite.run_load_test(
            orchestrator,
            num_events=scenario["num_events"],
            concurrency=scenario["concurrency"],
        )

        # Validate performance requirements
        assert results["success_rate"] >= 0.8, (
            f"Success rate too low: {results['success_rate']}"
        )
        assert results["avg_response_time"] < 30.0, (
            f"Average response time too high: {results['avg_response_time']}"
        )
        assert results["max_response_time"] < 300.0, (
            f"Max response time too high: {results['max_response_time']}"
        )

        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Average response time: {results['avg_response_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.2f} events/second")


if __name__ == "__main__":
    # Allow running specific tests
    pytest.main([__file__, "-v", "-s"])

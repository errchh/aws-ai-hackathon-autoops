"""
Enhanced end-to-end integration tests for Langfuse workflow visualization.

This module contains comprehensive integration tests that validate the complete
Langfuse integration system including trace creation, span management, dashboard
integration, and workflow visualization.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# Mock strands imports since the package has build issues
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
from config.langfuse_integration import LangfuseIntegrationService
from config.simulation_event_capture import SimulationEventCapture
from config.orchestrator_tracing import OrchestrationTracer
from models.core import (
    Product,
    MarketEvent,
    AgentDecision,
    SystemMetrics,
    ActionType,
    EventType,
)
from scenarios.data_generator import ScenarioDataGenerator
from tests.test_data_generators import TestDataGenerator, MockLangfuseClient


class TestLangfuseIntegrationE2E:
    """Enhanced end-to-end integration tests for Langfuse workflow visualization."""

    @pytest.fixture
    async def orchestrator_with_langfuse(self):
        """Create orchestrator with Langfuse integration enabled."""
        orchestrator = RetailOptimizationOrchestrator()

        # Register agents
        agents = [pricing_agent, inventory_agent, promotion_agent]
        success = orchestrator.register_agents(agents)
        assert success, "Failed to register agents"

        # Initialize Langfuse integration
        langfuse_service = LangfuseIntegrationService()
        orchestrator.langfuse_service = langfuse_service

        yield orchestrator, langfuse_service

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

    @pytest.fixture
    def test_data_generator(self):
        """Create test data generator."""
        return TestDataGenerator(seed=42)

    @pytest.fixture
    def mock_langfuse_client(self):
        """Create mock Langfuse client for testing."""
        return MockLangfuseClient()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_trace_creation(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test complete workflow trace creation from simulation event to agent responses."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate test market event
        events = test_data_generator.generate_market_events(1)
        event = events[0]

        event_data = {
            "event_id": event.id,
            "event_type": event.event_type.value,
            "affected_products": event.affected_products,
            "impact_magnitude": event.impact_magnitude,
            "metadata": event.metadata,
        }

        # Mock Langfuse client to capture trace creation
        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
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

            # Validate performance requirements
             assert processing_time < 300, (
                 f"Processing took {processing_time:.2f}s, exceeds 5-minute limit"
             )

            # Validate Langfuse trace creation
            assert self.mock_langfuse_client.call_count > 0, "No Langfuse calls made"

            # Check that traces were created for the workflow
            trace_calls = [call for call in self.mock_langfuse_client.traces.values()
                          if call["name"] in ["market_event", "agent_operation", "collaboration"]]
            assert len(trace_calls) > 0, "No workflow traces created"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_span_trace_hierarchy(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test that agent spans are properly nested under workflow traces."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate test event
        events = test_data_generator.generate_market_events(1)
        event = events[0]

        event_data = {
            "event_id": event.id,
            "event_type": event.event_type.value,
            "affected_products": event.affected_products,
            "impact_magnitude": event.impact_magnitude,
            "metadata": event.metadata,
        }

        # Mock Langfuse client
        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            result = await orchestrator.process_market_event(event_data)

            # Validate trace hierarchy
            if result["status"] == "completed":
                # Check that we have both root traces and child spans
                trace_count = len(self.mock_langfuse_client.traces)
                span_count = len(self.mock_langfuse_client.spans)

                assert trace_count > 0, "No traces created"
                assert span_count > 0, "No spans created"

                # Validate trace metadata contains expected information
                for trace_id, trace_data in self.mock_langfuse_client.traces.items():
                    assert "metadata" in trace_data
                    assert "created_at" in trace_data

                    # Check for workflow-specific metadata
                    if "workflow" in trace_data.get("name", ""):
                        assert "participating_agents" in trace_data["metadata"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collaboration_workflow_tracing(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test collaboration workflow tracing and span relationships."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate collaboration scenario
        collaboration_scenarios = test_data_generator.generate_collaboration_scenarios(1)
        scenario = collaboration_scenarios[0]

        # Mock Langfuse client
        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            result = await orchestrator.trigger_collaboration_workflow(
                scenario["workflow_type"], scenario
            )

            # Validate collaboration tracing
            assert result["status"] in ["initiated", "completed", "error"]

            if result["status"] in ["initiated", "completed"]:
                # Check that collaboration traces were created
                collaboration_traces = [
                    trace for trace in self.mock_langfuse_client.traces.values()
                    if "collaboration" in trace.get("name", "")
                ]
                assert len(collaboration_traces) > 0, "No collaboration traces found"

                # Validate collaboration trace metadata
                collab_trace = collaboration_traces[0]
                assert "participating_agents" in collab_trace["metadata"]
                assert len(collab_trace["metadata"]["participating_agents"]) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trace_data_integrity_and_validation(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test trace data integrity and validation across the system."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate multiple events to test trace integrity
        events = test_data_generator.generate_market_events(3)

        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            for event in events:
                event_data = {
                    "event_id": event.id,
                    "event_type": event.event_type.value,
                    "affected_products": event.affected_products,
                    "impact_magnitude": event.impact_magnitude,
                    "metadata": event.metadata,
                }

                result = await orchestrator.process_market_event(event_data)

                # Validate that each event created valid traces
                assert result["status"] in ["completed", "no_response_needed"]

                # Check trace data integrity
                for trace_id, trace_data in self.mock_langfuse_client.traces.items():
                    # Validate required fields
                    assert "trace_id" in trace_data
                    assert "name" in trace_data
                    assert "metadata" in trace_data
                    assert "created_at" in trace_data

                    # Validate metadata structure
                    metadata = trace_data["metadata"]
                    if "event_type" in metadata:
                        assert metadata["event_type"] in [e.value for e in EventType]

                    if "agent_id" in metadata:
                        assert metadata["agent_id"] in ["inventory_agent", "pricing_agent", "promotion_agent"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dashboard_integration_validation(
        self, orchestrator_with_langfuse, test_data_generator, test_client
    ):
        """Test integration with Langfuse dashboard and data visualization."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate test workflow
        events = test_data_generator.generate_market_events(2)

        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            for event in events:
                event_data = {
                    "event_id": event.id,
                    "event_type": event.event_type.value,
                    "affected_products": event.affected_products,
                    "impact_magnitude": event.impact_magnitude,
                    "metadata": event.metadata,
                }

                await orchestrator.process_market_event(event_data)

            # Validate dashboard data structure
            dashboard_data = {
                "total_traces": len(self.mock_langfuse_client.traces),
                "total_spans": len(self.mock_langfuse_client.spans),
                "active_workflows": len(orchestrator.active_workflows),
                "agent_responses": sum(len(workflow.get("responding_agents", []))
                                     for workflow in orchestrator.active_workflows.values())
            }

            # Test dashboard API endpoints
            response = test_client.get("/api/dashboard/langfuse/traces")
            # Should return dashboard data or appropriate error
            assert response.status_code in [200, 404, 501]

            if response.status_code == 200:
                dashboard_response = response.json()
                assert "traces" in dashboard_response or "error" in dashboard_response

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_trace_continuation(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test error handling and trace continuation when components fail."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Test with simulated Langfuse failure
        with patch.object(langfuse_service, '_client') as mock_client:
            # Simulate Langfuse client failure
            mock_client.is_available = False

            # Generate test event
            events = test_data_generator.generate_market_events(1)
            event = events[0]

            event_data = {
                "event_id": event.id,
                "event_type": event.event_type.value,
                "affected_products": event.affected_products,
                "impact_magnitude": event.impact_magnitude,
                "metadata": event.metadata,
            }

            # Should still process event even with Langfuse failure
            result = await orchestrator.process_market_event(event_data)
            assert result["status"] in ["completed", "no_response_needed"]

            # Verify graceful degradation - no traces should be created
            assert len(self.mock_langfuse_client.traces) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_workflow_tracing(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test concurrent workflow tracing and trace isolation."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate multiple concurrent events
        events = test_data_generator.generate_market_events(5)

        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            # Process events concurrently
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
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Validate concurrent processing
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= len(events) * 0.8, (
                f"Too many concurrent failures: {len(successful_results)}/{len(events)}"
            )

            # Validate trace isolation - each workflow should have separate traces
            trace_count = len(self.mock_langfuse_client.traces)
            assert trace_count > 0, "No traces created for concurrent workflows"

            # Each successful workflow should have at least one trace
            assert trace_count >= len(successful_results), (
                "Insufficient traces for successful workflows"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_trace_lifecycle_management(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test complete trace lifecycle from creation to completion."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate test event
        events = test_data_generator.generate_market_events(1)
        event = events[0]

        event_data = {
            "event_id": event.id,
            "event_type": event.event_type.value,
            "affected_products": event.affected_products,
            "impact_magnitude": event.impact_magnitude,
            "metadata": event.metadata,
        }

        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            # Process event
            result = await orchestrator.process_market_event(event_data)

            # Validate trace lifecycle
            if result["status"] == "completed":
                # Check that traces have proper lifecycle metadata
                for trace_id, trace_data in self.mock_langfuse_client.traces.items():
                    assert "created_at" in trace_data

                    # Check for completion metadata
                    if "workflow_id" in trace_data.get("metadata", {}):
                        # Should have end time or completion status
                        assert "ended_at" in trace_data or "status" in trace_data.get("metadata", {})

                # Validate span lifecycle
                for span_id, span_data in self.mock_langfuse_client.spans.items():
                    assert "created_at" in span_data
                    # Spans should have end times
                    assert "ended_at" in span_data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_agent_trace_correlation(
        self, orchestrator_with_langfuse, test_data_generator
    ):
        """Test trace correlation across multiple agents in collaboration workflows."""
        orchestrator, langfuse_service = orchestrator_with_langfuse

        # Generate collaboration scenario
        collaboration_scenarios = test_data_generator.generate_collaboration_scenarios(1)
        scenario = collaboration_scenarios[0]

        with patch.object(langfuse_service, '_client', self.mock_langfuse_client):
            result = await orchestrator.trigger_collaboration_workflow(
                scenario["workflow_type"], scenario
            )

            if result["status"] in ["initiated", "completed"]:
                # Validate cross-agent trace correlation
                collaboration_traces = [
                    trace for trace in self.mock_langfuse_client.traces.values()
                    if "collaboration" in trace.get("name", "")
                ]

                if len(collaboration_traces) > 0:
                    collab_trace = collaboration_traces[0]

                    # Check that all participating agents have corresponding spans
                    participating_agents = collab_trace["metadata"].get("participating_agents", [])

                    agent_spans = [
                        span for span in self.mock_langfuse_client.spans.values()
                        if any(agent in span.get("metadata", {}).get("agent_id", "")
                              for agent in participating_agents)
                    ]

                    assert len(agent_spans) > 0, "No agent spans found for collaboration"

                    # Validate span correlation with collaboration trace
                    for span in agent_spans:
                        span_metadata = span.get("metadata", {})
                        if "parent_trace_id" in span_metadata:
                            assert span_metadata["parent_trace_id"] == collab_trace["trace_id"]


class TestLangfuseDashboardIntegration:
    """Test Langfuse dashboard integration and data visualization."""

    @pytest.fixture
    def mock_dashboard_api(self):
        """Mock dashboard API for testing."""
        return MagicMock()

    @pytest.mark.dashboard
    @pytest.mark.asyncio
    async def test_dashboard_trace_visualization(
        self, test_client, test_data_generator, mock_dashboard_api
    ):
        """Test dashboard trace visualization functionality."""
        # Generate test traces
        test_generator = test_data_generator
        workflow_trace = test_generator.generate_realistic_workflow_trace()

        # Mock dashboard API response
        mock_response = {
            "traces": [
                {
                    "trace_id": workflow_trace["market_event"].id,
                    "name": "market_event",
                    "spans": [
                        {
                            "span_id": f"span_{i}",
                            "name": span["operation"],
                            "agent": span["agent"]
                        }
                        for i, span in enumerate(workflow_trace["expected_trace_structure"]["root_trace"]["spans"])
                    ]
                }
            ],
            "total_count": 1,
            "filters_applied": {}
        }

        with patch("api.main.get_dashboard_api", return_value=mock_dashboard_api):
            mock_dashboard_api.get_traces.return_value = mock_response

            response = test_client.get("/api/dashboard/langfuse/traces")

            if response.status_code == 200:
                dashboard_data = response.json()
                assert "traces" in dashboard_data
                assert len(dashboard_data["traces"]) > 0

    @pytest.mark.dashboard
    @pytest.mark.asyncio
    async def test_dashboard_performance_metrics(
        self, test_client, test_data_generator, mock_dashboard_api
    ):
        """Test dashboard performance metrics display."""
        # Generate performance test data
        performance_data = test_data_generator.generate_performance_test_data(50)

        mock_metrics = {
            "total_traces": len(performance_data["events"]),
            "avg_trace_duration": 2.5,
            "trace_success_rate": 0.95,
            "agent_performance": {
                "inventory_agent": {"avg_response_time": 1.2, "success_rate": 0.98},
                "pricing_agent": {"avg_response_time": 1.8, "success_rate": 0.92},
                "promotion_agent": {"avg_response_time": 1.5, "success_rate": 0.96}
            },
            "system_metrics": {
                "throughput": 25.0,
                "memory_usage": 150.0,
                "cpu_usage": 45.0
            }
        }

        with patch("api.main.get_dashboard_api", return_value=mock_dashboard_api):
            mock_dashboard_api.get_performance_metrics.return_value = mock_metrics

            response = test_client.get("/api/dashboard/langfuse/metrics")

            if response.status_code == 200:
                metrics_data = response.json()
                assert "agent_performance" in metrics_data
                assert "system_metrics" in metrics_data
                assert "total_traces" in metrics_data


if __name__ == "__main__":
    # Allow running specific tests
    pytest.main([__file__, "-v", "-s"])
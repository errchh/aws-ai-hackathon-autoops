"""
Tests for OrchestrationTracer and orchestrator workflow tracing functionality.

This module contains comprehensive tests for the OrchestrationTracer class,
including workflow tracing, agent coordination, conflict resolution, and
inter-agent communication tracing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List

from config.orchestrator_tracing import (
    OrchestrationTracer,
    OrchestrationEventType,
    OrchestrationContext,
    get_orchestration_tracer,
    initialize_orchestration_tracer,
)
from config.langfuse_integration import LangfuseIntegrationService


class TestOrchestrationTracer:
    """Test suite for OrchestrationTracer class."""

    @pytest.fixture
    def mock_integration_service(self):
        """Create a mock LangfuseIntegrationService."""
        mock_service = Mock(spec=LangfuseIntegrationService)
        mock_service.is_available = True
        mock_service.create_simulation_trace.return_value = "test_trace_id"
        mock_service.start_agent_span.return_value = "test_span_id"
        mock_service.end_agent_span = Mock()
        mock_service.finalize_trace = Mock()
        mock_service.log_agent_decision = Mock()
        mock_service.track_collaboration.return_value = "test_collab_trace_id"
        return mock_service

    @pytest.fixture
    def tracer(self, mock_integration_service):
        """Create an OrchestrationTracer instance with mocked service."""
        return OrchestrationTracer(mock_integration_service)

    def test_initialization(self, mock_integration_service):
        """Test tracer initialization."""
        tracer = OrchestrationTracer(mock_integration_service)

        assert tracer._integration_service == mock_integration_service
        assert tracer._active_workflows == {}
        assert tracer._active_spans == {}

    def test_trace_workflow_start_success(self, tracer, mock_integration_service):
        """Test successful workflow trace start."""
        workflow_id = "test_workflow_123"
        participating_agents = ["agent1", "agent2"]
        event_data = {"event_type": "demand_spike", "products": ["A", "B"]}

        trace_id = tracer.trace_workflow_start(
            workflow_id=workflow_id,
            participating_agents=participating_agents,
            event_data=event_data,
        )

        assert trace_id == "test_trace_id"
        mock_integration_service.create_simulation_trace.assert_called_once()
        call_args = mock_integration_service.create_simulation_trace.call_args[0][0]
        assert call_args["workflow_id"] == workflow_id
        assert call_args["participating_agents"] == participating_agents
        assert call_args["agent_count"] == 2
        assert call_args["event_data"] == event_data
        assert tracer._active_workflows[workflow_id] == "test_trace_id"

    def test_trace_workflow_start_unavailable(self, mock_integration_service):
        """Test workflow trace start when Langfuse is unavailable."""
        mock_integration_service.is_available = False
        tracer = OrchestrationTracer(mock_integration_service)

        trace_id = tracer.trace_workflow_start(
            workflow_id="test_workflow", participating_agents=["agent1"]
        )

        assert trace_id is None
        mock_integration_service.create_simulation_trace.assert_not_called()

    def test_trace_agent_coordination_success(self, tracer, mock_integration_service):
        """Test successful agent coordination tracing."""
        coordination_id = "coord_123"
        messages = [
            {"sender": "orchestrator", "recipient": "agent1", "content": "test"},
            {"sender": "orchestrator", "recipient": "agent2", "content": "test2"},
        ]
        workflow_id = "workflow_123"

        # Set up active workflow
        tracer._active_workflows[workflow_id] = "parent_trace_id"

        span_id = tracer.trace_agent_coordination(
            coordination_id=coordination_id, messages=messages, workflow_id=workflow_id
        )

        assert span_id == "test_span_id"
        mock_integration_service.start_agent_span.assert_called_once()
        call_args = mock_integration_service.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "orchestrator"
        assert call_args[1]["operation"] == "agent_coordination"
        assert call_args[1]["parent_trace_id"] == "parent_trace_id"
        assert call_args[1]["input_data"]["coordination_id"] == coordination_id
        assert call_args[1]["input_data"]["message_count"] == 2
        assert tracer._active_spans[coordination_id] == "test_span_id"

    def test_trace_conflict_resolution_success(self, tracer, mock_integration_service):
        """Test successful conflict resolution tracing."""
        conflict_data = {
            "conflict_id": "conflict_123",
            "conflict_type": "pricing_inventory_mismatch",
            "agents": ["pricing_agent", "inventory_agent"],
            "details": {"severity": "medium"},
        }
        workflow_id = "workflow_123"

        # Set up active workflow
        tracer._active_workflows[workflow_id] = "parent_trace_id"

        span_id = tracer.trace_conflict_resolution(
            conflict_data=conflict_data, workflow_id=workflow_id
        )

        assert span_id == "test_span_id"
        mock_integration_service.start_agent_span.assert_called_once()
        call_args = mock_integration_service.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "orchestrator"
        assert call_args[1]["operation"] == "conflict_resolution"
        assert call_args[1]["parent_trace_id"] == "parent_trace_id"
        assert call_args[1]["input_data"]["conflict_id"] == "conflict_123"
        assert (
            call_args[1]["input_data"]["conflict_type"] == "pricing_inventory_mismatch"
        )
        assert tracer._active_spans["conflict_123"] == "test_span_id"

    def test_trace_inter_agent_communication_success(
        self, tracer, mock_integration_service
    ):
        """Test successful inter-agent communication tracing."""
        sender = "orchestrator"
        recipient = "pricing_agent"
        message_data = {
            "message_type": "coordination",
            "content": {"action": "price_update"},
            "timestamp": datetime.now().isoformat(),
        }
        workflow_id = "workflow_123"

        # Set up active workflow
        tracer._active_workflows[workflow_id] = "parent_trace_id"

        span_id = tracer.trace_inter_agent_communication(
            sender_agent=sender,
            recipient_agent=recipient,
            message_data=message_data,
            workflow_id=workflow_id,
        )

        assert span_id == "test_span_id"
        mock_integration_service.start_agent_span.assert_called_once()
        call_args = mock_integration_service.start_agent_span.call_args
        assert call_args[1]["agent_id"] == "orchestrator"
        assert call_args[1]["operation"] == "inter_agent_communication"
        assert call_args[1]["parent_trace_id"] == "parent_trace_id"
        assert call_args[1]["input_data"]["sender"] == sender
        assert call_args[1]["input_data"]["recipient"] == recipient
        assert call_args[1]["input_data"]["message_type"] == "coordination"

    def test_end_coordination_span_success(self, tracer, mock_integration_service):
        """Test ending a coordination span successfully."""
        coordination_id = "coord_123"
        outcome = {"successful_responses": 2, "failed_responses": 0}

        # Set up active span
        tracer._active_spans[coordination_id] = "test_span_id"

        tracer.end_coordination_span(coordination_id, outcome)

        mock_integration_service.end_agent_span.assert_called_once_with(
            span_id="test_span_id", outcome=outcome, error=None
        )
        assert coordination_id not in tracer._active_spans

    def test_end_coordination_span_with_error(self, tracer, mock_integration_service):
        """Test ending a coordination span with an error."""
        coordination_id = "coord_123"
        outcome = {"successful_responses": 0, "failed_responses": 2}
        error = Exception("Timeout occurred")

        # Set up active span
        tracer._active_spans[coordination_id] = "test_span_id"

        tracer.end_coordination_span(coordination_id, outcome, error)

        mock_integration_service.end_agent_span.assert_called_once_with(
            span_id="test_span_id", outcome=outcome, error=error
        )
        assert coordination_id not in tracer._active_spans

    def test_end_conflict_resolution_span_success(
        self, tracer, mock_integration_service
    ):
        """Test ending a conflict resolution span successfully."""
        conflict_id = "conflict_123"
        resolution_outcome = {
            "conflicts_resolved": 1,
            "resolution_method": "orchestrator_mediation",
        }

        # Set up active span
        tracer._active_spans[conflict_id] = "test_span_id"

        tracer.end_conflict_resolution_span(conflict_id, resolution_outcome)

        mock_integration_service.end_agent_span.assert_called_once_with(
            span_id="test_span_id", outcome=resolution_outcome, error=None
        )
        assert conflict_id not in tracer._active_spans

    def test_end_communication_span_success(self, tracer, mock_integration_service):
        """Test ending a communication span successfully."""
        communication_id = "comm_123"
        response_data = {"status": "processed", "timestamp": datetime.now().isoformat()}

        # Set up active span
        tracer._active_spans[communication_id] = "test_span_id"

        tracer.end_communication_span(communication_id, response_data)

        mock_integration_service.end_agent_span.assert_called_once()
        call_args = mock_integration_service.end_agent_span.call_args
        assert call_args[1]["span_id"] == "test_span_id"
        assert call_args[1]["outcome"]["response_received"] is True
        assert call_args[1]["outcome"]["response_data"] == response_data
        assert call_args[1]["error"] is None
        assert communication_id not in tracer._active_spans

    def test_finalize_workflow_success(self, tracer, mock_integration_service):
        """Test finalizing a workflow successfully."""
        workflow_id = "workflow_123"
        final_outcome = {"status": "completed", "processing_time": 5.2}

        # Set up active workflow
        tracer._active_workflows[workflow_id] = "test_trace_id"

        tracer.finalize_workflow(workflow_id, final_outcome)

        mock_integration_service.finalize_trace.assert_called_once_with(
            "test_trace_id", final_outcome
        )
        assert workflow_id not in tracer._active_workflows

    def test_finalize_workflow_not_found(self, tracer, mock_integration_service):
        """Test finalizing a workflow that doesn't exist."""
        workflow_id = "nonexistent_workflow"

        # Should not raise an error
        tracer.finalize_workflow(workflow_id, {"status": "error"})

        mock_integration_service.finalize_trace.assert_not_called()

    def test_track_collaboration_success(self, tracer, mock_integration_service):
        """Test successful collaboration tracking."""
        workflow_id = "collab_123"
        participating_agents = ["agent1", "agent2", "agent3"]
        workflow_data = {"workflow_type": "inventory_to_pricing"}

        trace_id = tracer.track_collaboration(
            workflow_id=workflow_id,
            participating_agents=participating_agents,
            workflow_data=workflow_data,
        )

        assert trace_id == "test_collab_trace_id"
        mock_integration_service.track_collaboration.assert_called_once_with(
            workflow_id=workflow_id,
            participating_agents=participating_agents,
            workflow_data=workflow_data,
        )
        assert tracer._active_workflows[workflow_id] == "test_collab_trace_id"

    def test_log_orchestration_decision_with_workflow(
        self, tracer, mock_integration_service
    ):
        """Test logging orchestration decision with workflow context."""
        decision_data = {
            "decision_type": "agent_selection",
            "selected_agents": ["agent1", "agent2"],
            "confidence": 0.9,
        }
        workflow_id = "workflow_123"

        # Set up active workflow
        tracer._active_workflows[workflow_id] = "test_trace_id"

        tracer.log_orchestration_decision(decision_data, workflow_id)

        mock_integration_service.log_agent_decision.assert_called_once()
        call_args = mock_integration_service.log_agent_decision.call_args
        assert call_args[1]["agent_id"] == "orchestrator"
        assert call_args[1]["decision_data"] == decision_data
        assert call_args[1]["trace_context"]["workflow_id"] == workflow_id
        assert call_args[1]["trace_context"]["trace_id"] == "test_trace_id"

    def test_log_orchestration_decision_without_workflow(
        self, tracer, mock_integration_service
    ):
        """Test logging orchestration decision without workflow context."""
        decision_data = {
            "decision_type": "coordination_approach",
            "approach": "direct_messaging",
        }

        tracer.log_orchestration_decision(decision_data)

        mock_integration_service.log_agent_decision.assert_called_once()
        call_args = mock_integration_service.log_agent_decision.call_args
        assert call_args[1]["agent_id"] == "orchestrator"
        assert call_args[1]["decision_data"] == decision_data
        assert call_args[1]["trace_context"] == {}

    def test_get_active_workflows(self, tracer):
        """Test getting active workflows."""
        tracer._active_workflows = {"workflow1": "trace1", "workflow2": "trace2"}

        active = tracer.get_active_workflows()
        assert active == {"workflow1": "trace1", "workflow2": "trace2"}
        # Should return a copy, not the original
        assert active is not tracer._active_workflows

    def test_get_active_spans(self, tracer):
        """Test getting active spans."""
        tracer._active_spans = {"span1": "id1", "span2": "id2"}

        active = tracer.get_active_spans()
        assert active == {"span1": "id1", "span2": "id2"}
        # Should return a copy, not the original
        assert active is not tracer._active_spans

    def test_health_check(self, tracer, mock_integration_service):
        """Test health check functionality."""
        mock_integration_service.is_available = True

        health = tracer.health_check()

        assert health["service"] == "orchestration_tracer"
        assert health["status"] == "healthy"
        assert health["active_workflows"] == 0
        assert health["active_spans"] == 0
        assert health["langfuse_available"] is True

    def test_health_check_unavailable(self, tracer, mock_integration_service):
        """Test health check when Langfuse is unavailable."""
        mock_integration_service.is_available = False

        health = tracer.health_check()

        assert health["status"] == "degraded"
        assert health["langfuse_available"] is False


class TestOrchestrationEventType:
    """Test suite for OrchestrationEventType enum."""

    def test_event_types(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "market_event_processing",
            "agent_coordination",
            "conflict_detection",
            "conflict_resolution",
            "workflow_completion",
            "collaboration_trigger",
            "inter_agent_communication",
        ]

        for event_type in expected_types:
            assert hasattr(OrchestrationEventType, event_type.upper())
            assert OrchestrationEventType[event_type.upper()].value == event_type


class TestOrchestrationContext:
    """Test suite for OrchestrationContext dataclass."""

    def test_context_creation(self):
        """Test creating an OrchestrationContext instance."""
        context = OrchestrationContext(
            workflow_id="workflow_123",
            event_id="event_456",
            participating_agents=["agent1", "agent2"],
            operation_type=OrchestrationEventType.AGENT_COORDINATION,
            start_time=datetime.now(),
            metadata={"key": "value"},
        )

        assert context.workflow_id == "workflow_123"
        assert context.event_id == "event_456"
        assert context.participating_agents == ["agent1", "agent2"]
        assert context.operation_type == OrchestrationEventType.AGENT_COORDINATION
        assert "key" in context.metadata
        assert context.metadata["key"] == "value"

    def test_context_defaults(self):
        """Test OrchestrationContext with default values."""
        context = OrchestrationContext(workflow_id="workflow_123")

        assert context.workflow_id == "workflow_123"
        assert context.event_id is None
        assert context.participating_agents == []
        assert context.operation_type is None
        assert context.start_time is None
        assert context.metadata == {}


class TestGlobalFunctions:
    """Test suite for global functions."""

    @patch("config.orchestrator_tracing.OrchestrationTracer")
    def test_get_orchestration_tracer(self, mock_tracer_class):
        """Test getting the global orchestration tracer."""
        mock_instance = Mock()
        mock_tracer_class.return_value = mock_instance

        tracer = get_orchestration_tracer()

        assert tracer == mock_instance
        mock_tracer_class.assert_called_once()

    @patch("config.orchestrator_tracing._orchestration_tracer", None)
    @patch("config.orchestrator_tracing.OrchestrationTracer")
    def test_get_orchestration_tracer_singleton(self, mock_tracer_class):
        """Test that get_orchestration_tracer returns the same instance."""
        mock_instance = Mock()
        mock_tracer_class.return_value = mock_instance

        tracer1 = get_orchestration_tracer()
        tracer2 = get_orchestration_tracer()

        assert tracer1 == tracer2
        assert tracer1 == mock_instance
        mock_tracer_class.assert_called_once()

    def test_initialize_orchestration_tracer(self):
        """Test initializing orchestration tracer with custom service."""
        mock_service = Mock(spec=LangfuseIntegrationService)

        tracer = initialize_orchestration_tracer(mock_service)

        assert isinstance(tracer, OrchestrationTracer)
        assert tracer._integration_service == mock_service


if __name__ == "__main__":
    """Integration tests for orchestrator workflow tracing."""

    @pytest.fixture
    def mock_integration_service(self):
        """Create a mock LangfuseIntegrationService for integration tests."""
        mock_service = Mock(spec=LangfuseIntegrationService)
        mock_service.is_available = True
        mock_service.create_simulation_trace.return_value = "test_trace_id"
        mock_service.start_agent_span.return_value = "test_span_id"
        mock_service.end_agent_span = Mock()
        mock_service.finalize_trace = Mock()
        mock_service.log_agent_decision = Mock()
        mock_service.track_collaboration.return_value = "test_collab_trace_id"
        return mock_service

    @pytest.fixture
    def mock_orchestrator(self, mock_integration_service):
        """Create a mock RetailOptimizationOrchestrator with tracing."""
        with patch(
            "agents.orchestrator.RetailOptimizationOrchestrator"
        ) as mock_orch_class:
            mock_orch = Mock()
            mock_orch.orchestrator_id = "test_orchestrator"
            mock_orch.system_status = Mock()
            mock_orch.system_status.value = "healthy"
            mock_orch.agents = {"agent1": Mock(), "agent2": Mock()}
            mock_orch.agent_status = {
                "agent1": {"status": "active"},
                "agent2": {"status": "active"},
            }
            mock_orch.tracer = OrchestrationTracer(mock_integration_service)
            mock_orch._determine_responding_agents = Mock(
                return_value=["agent1", "agent2"]
            )
            mock_orch._send_coordination_message = Mock(
                return_value={
                    "agent_id": "agent1",
                    "response_type": "coordination_response",
                    "analysis": "Test response",
                }
            )
            mock_orch._detect_conflicts = Mock(return_value=[])
            mock_orch._resolve_conflicts = Mock(return_value={"status": "resolved"})
            mock_orch._synthesize_responses = Mock(
                return_value={"coordinated_plan": "test"}
            )
            mock_orch._update_system_metrics = Mock()
            mock_orch_class.return_value = mock_orch
            yield mock_orch

    @pytest.mark.asyncio
    async def test_process_market_event_tracing_integration(
        self, mock_orchestrator, mock_integration_service
    ):
        """Test that process_market_event properly integrates tracing."""
        event_data = {
            "event_id": "event_123",
            "event_type": "demand_spike",
            "affected_products": ["product_A", "product_B"],
            "impact_magnitude": 0.8,
        }

        # Mock the async method
        mock_orchestrator._send_coordination_message = Mock(
            side_effect=[
                {"agent_id": "agent1", "response": "response1"},
                {"agent_id": "agent2", "response": "response2"},
            ]
        )

        result = await mock_orchestrator.process_market_event(event_data)

        # Verify workflow trace was started
        mock_integration_service.create_simulation_trace.assert_called_once()
        trace_call = mock_integration_service.create_simulation_trace.call_args[0][0]
        assert trace_call["event_data"]["event_type"] == "demand_spike"
        assert trace_call["participating_agents"] == ["agent1", "agent2"]

        # Verify coordination span was started
        mock_integration_service.start_agent_span.assert_any_call(
            agent_id="orchestrator",
            operation="agent_coordination",
            parent_trace_id="test_trace_id",
            input_data=Mock(),
        )

        # Verify workflow was finalized
        mock_integration_service.finalize_trace.assert_called_once()
        finalize_call = mock_integration_service.finalize_trace.call_args
        assert finalize_call[0][0] == "test_trace_id"
        assert finalize_call[0][1]["workflow_status"] == "completed"

        # Verify result structure
        assert result["event_id"] == "event_123"
        assert result["status"] == "completed"
        assert "workflow_id" in result
        assert "processing_time" in result

    @pytest.mark.asyncio
    async def test_process_market_event_conflict_tracing(
        self, mock_orchestrator, mock_integration_service
    ):
        """Test tracing during conflict resolution in process_market_event."""
        event_data = {
            "event_id": "event_456",
            "event_type": "competitor_price_change",
            "affected_products": ["product_C"],
        }

        # Mock conflict detection
        mock_orchestrator._detect_conflicts = Mock(
            return_value=[
                {
                    "conflict_type": "pricing_inventory_mismatch",
                    "agents": ["pricing_agent", "inventory_agent"],
                    "products": ["product_C"],
                    "severity": "medium",
                }
            ]
        )

        # Mock conflict resolution
        mock_orchestrator._resolve_conflicts = Mock(
            return_value={
                "resolution_id": "res_123",
                "status": "resolved",
                "resolutions": [{"conflict_id": "conflict_123"}],
            }
        )

        result = await mock_orchestrator.process_market_event(event_data)

        # Verify conflict resolution span was started
        mock_integration_service.start_agent_span.assert_any_call(
            agent_id="orchestrator",
            operation="conflict_resolution",
            parent_trace_id="test_trace_id",
            input_data=Mock(),
        )

        # Verify conflict resolution span was ended
        mock_integration_service.end_agent_span.assert_any_call(
            span_id=Mock(), outcome=Mock(), error=None
        )

        # Verify orchestration decision was logged for conflict resolution
        mock_integration_service.log_agent_decision.assert_any_call(
            agent_id="orchestrator", decision_data=Mock(), trace_context=Mock()
        )

    @pytest.mark.asyncio
    async def test_process_market_event_timeout_tracing(
        self, mock_orchestrator, mock_integration_service
    ):
        """Test tracing when process_market_event times out."""
        event_data = {"event_id": "event_timeout", "event_type": "supply_disruption"}

        # Mock timeout by making _send_coordination_message raise TimeoutError
        import asyncio

        mock_orchestrator._send_coordination_message = Mock(
            side_effect=asyncio.TimeoutError()
        )

        result = await mock_orchestrator.process_market_event(event_data)

        # Verify coordination span was ended with error
        mock_integration_service.end_agent_span.assert_any_call(
            span_id=Mock(), outcome=Mock(), error=Mock()
        )

        # Verify workflow was finalized with timeout outcome
        mock_integration_service.finalize_trace.assert_called_once()
        finalize_call = mock_integration_service.finalize_trace.call_args
        assert finalize_call[0][1]["workflow_status"] == "timeout"

        # Verify result indicates timeout
        assert result["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_coordinate_agents_tracing_integration(
        self, mock_orchestrator, mock_integration_service
    ):
        """Test tracing in coordinate_agents method."""
        request = {
            "requesting_agent": "pricing_agent",
            "target_agents": ["inventory_agent", "promotion_agent"],
            "coordination_type": "consultation",
            "content": {"query": "stock_levels"},
        }

        # Mock agent responses
        mock_orchestrator._send_message_to_agent = Mock(
            side_effect=[
                {"agent_id": "inventory_agent", "response": "stock_data"},
                {"agent_id": "promotion_agent", "response": "promotion_data"},
            ]
        )

        result = await mock_orchestrator.coordinate_agents(request)

        # Verify coordination span was started
        mock_integration_service.start_agent_span.assert_any_call(
            agent_id="orchestrator",
            operation="agent_coordination",
            parent_trace_id=None,
            input_data=Mock(),
        )

        # Verify orchestration decision was logged
        mock_integration_service.log_agent_decision.assert_any_call(
            agent_id="orchestrator", decision_data=Mock(), trace_context={}
        )

        # Verify coordination span was ended
        mock_integration_service.end_agent_span.assert_any_call(
            span_id=Mock(), outcome=Mock(), error=None
        )

        # Verify result structure
        assert result["coordination_id"] is not None
        assert result["status"] == "completed"
        assert "coordination_results" in result

    def test_trigger_collaboration_workflow_tracing(
        self, mock_orchestrator, mock_integration_service
    ):
        """Test tracing in trigger_collaboration_workflow method."""
        workflow_type = "inventory_to_pricing_slow_moving"
        workflow_data = {
            "slow_moving_items": ["item1", "item2"],
            "participating_agents": ["inventory_agent", "pricing_agent"],
        }

        # Mock collaboration workflow
        with patch("agents.orchestrator.collaboration_workflow") as mock_collab:
            mock_collab.inventory_to_pricing_slow_moving_alert = Mock(
                return_value={"status": "initiated", "workflow_id": "collab_123"}
            )

            result = mock_orchestrator.trigger_collaboration_workflow(
                workflow_type, workflow_data
            )

            # Verify collaboration tracking was started
            mock_integration_service.track_collaboration.assert_called_once()
            collab_call = mock_integration_service.track_collaboration.call_args
            assert collab_call[1]["workflow_id"] == f"collab_{workflow_type}_"
            assert collab_call[1]["participating_agents"] == [
                "inventory_agent",
                "pricing_agent",
            ]

            # Verify workflow was finalized
            mock_integration_service.finalize_trace.assert_called_once()
            finalize_call = mock_integration_service.finalize_trace.call_args
            assert finalize_call[0][1]["collaboration_status"] == "initiated"

            # Verify result
            assert result["status"] == "initiated"


if __name__ == "__main__":
    pytest.main([__file__])

"""Tests for the MetricsCollector functionality."""

import pytest
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.metrics_collector import (
    MetricsCollector,
    AgentPerformanceMetrics,
    SystemWorkflowMetrics,
)


class TestMetricsCollector:
    """Test cases for MetricsCollector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(max_history_size=100)

    def test_initialization(self):
        """Test that MetricsCollector initializes correctly."""
        assert self.collector.max_history_size == 100
        assert len(self.collector._agent_metrics) == 0
        assert len(self.collector._active_workflows) == 0
        assert self.collector._system_metrics.total_events_processed == 0

    def test_start_and_end_operation(self):
        """Test starting and ending an operation."""
        operation_id = "test_operation_1"
        agent_id = "test_agent"

        # Start operation
        self.collector.start_operation(operation_id, agent_id, "test_operation")

        # Check that operation is tracked
        assert operation_id in self.collector._start_times
        assert agent_id in self.collector._agent_metrics
        assert self.collector._agent_metrics[agent_id].operation_count == 1

        # End operation
        self.collector.end_operation(operation_id, success=True)

        # Check that operation is completed
        assert operation_id not in self.collector._start_times
        metrics = self.collector._agent_metrics[agent_id]
        assert metrics.operation_count == 1
        assert metrics.success_count == 1
        assert metrics.total_response_time > 0

    def test_operation_with_error(self):
        """Test ending an operation with an error."""
        operation_id = "test_operation_2"
        agent_id = "test_agent"

        self.collector.start_operation(operation_id, agent_id, "test_operation")
        self.collector.end_operation(operation_id, success=False, error="Test error")

        metrics = self.collector._agent_metrics[agent_id]
        assert metrics.success_count == 0
        assert metrics.error_count == 1

    def test_workflow_tracking(self):
        """Test workflow start and end tracking."""
        workflow_id = "test_workflow_1"
        agents = ["agent1", "agent2"]

        # Start workflow
        self.collector.start_workflow(workflow_id, agents)

        assert workflow_id in self.collector._active_workflows
        assert self.collector._system_metrics.total_events_processed == 1

        # End workflow
        self.collector.end_workflow(workflow_id, success=True)

        assert workflow_id not in self.collector._active_workflows
        assert self.collector._system_metrics.total_workflows_completed == 1
        assert self.collector._system_metrics.average_workflow_duration > 0

    def test_agent_collaboration_recording(self):
        """Test recording agent collaboration."""
        agent_id = "test_agent"

        # Create agent metrics first
        self.collector._agent_metrics[agent_id] = AgentPerformanceMetrics(
            agent_id=agent_id
        )

        # Record collaboration
        self.collector.record_agent_collaboration(agent_id)

        # Check that collaboration is recorded
        assert self.collector._agent_metrics[agent_id].collaboration_count == 1

    def test_get_agent_metrics(self):
        """Test retrieving agent metrics."""
        agent_id = "test_agent"
        self.collector.start_operation("op1", agent_id, "test")
        self.collector.end_operation("op1", success=True)

        metrics = self.collector.get_agent_metrics(agent_id)
        assert metrics is not None
        assert metrics.agent_id == agent_id
        assert metrics.operation_count == 1

    def test_get_system_metrics(self):
        """Test retrieving system metrics."""
        self.collector.start_workflow("wf1", ["agent1"])
        self.collector.end_workflow("wf1", success=True)

        system_metrics = self.collector.get_system_metrics()
        assert system_metrics.total_workflows_completed == 1
        assert system_metrics.total_events_processed == 1

    def test_export_metrics_for_dashboard(self):
        """Test exporting metrics for dashboard."""
        agent_id = "test_agent"
        self.collector.start_operation("op1", agent_id, "test")
        self.collector.end_operation("op1", success=True)

        export_data = self.collector.export_metrics_for_dashboard()

        assert "timestamp" in export_data
        assert "agents" in export_data
        assert "system" in export_data
        assert agent_id in export_data["agents"]
        assert export_data["agents"][agent_id]["operation_count"] == 1

    def test_reset_metrics(self):
        """Test resetting all metrics."""
        self.collector.start_operation("op1", "agent1", "test")
        self.collector.end_operation("op1", success=True)
        self.collector.start_workflow("wf1", ["agent1"])
        self.collector.end_workflow("wf1", success=True)

        # Verify metrics exist
        assert len(self.collector._agent_metrics) > 0
        assert self.collector._system_metrics.total_workflows_completed > 0

        # Reset metrics
        self.collector.reset_metrics()

        # Verify metrics are reset
        assert len(self.collector._agent_metrics) == 0
        assert self.collector._system_metrics.total_workflows_completed == 0
        assert len(self.collector._active_workflows) == 0

    def test_metrics_summary(self):
        """Test getting metrics summary."""
        self.collector.start_operation("op1", "agent1", "test")
        self.collector.end_operation("op1", success=True)

        summary = self.collector.get_metrics_summary()

        assert "total_agents" in summary
        assert "total_operations" in summary
        assert summary["total_agents"] == 1
        assert summary["total_operations"] == 1

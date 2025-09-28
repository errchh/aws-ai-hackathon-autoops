"""Performance metrics collection for Langfuse workflow visualization."""

import logging
import time
from typing import Dict, List, Optional, Any, DefaultDict
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Lock

# Removed to avoid circular import

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformanceMetrics:
    """Data class for agent performance metrics."""

    agent_id: str
    operation_count: int = 0
    total_response_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    tool_usage_stats: Dict[str, int] = field(default_factory=dict)
    decision_confidence_scores: List[float] = field(default_factory=list)
    collaboration_count: int = 0
    last_activity: Optional[datetime] = None


@dataclass
class SystemWorkflowMetrics:
    """Data class for system-wide workflow metrics."""

    total_events_processed: int = 0
    total_workflows_completed: int = 0
    average_workflow_duration: float = 0.0
    agent_coordination_efficiency: float = 0.0
    conflict_resolution_rate: float = 0.0
    system_throughput: float = 0.0  # events per minute
    peak_throughput: float = 0.0
    error_rate: float = 0.0
    last_updated: Optional[datetime] = None


class MetricsCollector:
    """Collects and aggregates performance metrics for agents and workflows."""

    def __init__(self, max_history_size: int = 1000):
        """Initialize the metrics collector.

        Args:
            max_history_size: Maximum number of recent metrics to keep in memory
        """
        self.max_history_size = max_history_size
        self._lock = Lock()

        # Agent-specific metrics
        self._agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self._operation_history: deque = deque(maxlen=max_history_size)

        # System-wide metrics
        self._system_metrics = SystemWorkflowMetrics()

        # Workflow tracking
        self._active_workflows: Dict[str, Dict[str, Any]] = {}
        self._workflow_history: deque = deque(maxlen=max_history_size)

        # Performance tracking
        self._start_times: Dict[str, Dict[str, Any]] = {}
        self._event_timestamps: deque = deque(maxlen=max_history_size)

        logger.info("MetricsCollector initialized")

    def start_operation(
        self, operation_id: str, agent_id: str, operation_type: str
    ) -> None:
        """Start tracking an operation for performance metrics.

        Args:
            operation_id: Unique identifier for the operation
            agent_id: ID of the agent performing the operation
            operation_type: Type of operation being performed
        """
        with self._lock:
            self._start_times[operation_id] = {
                "start_time": time.time(),
                "agent_id": agent_id,
            }

            # Update agent metrics
            if agent_id not in self._agent_metrics:
                self._agent_metrics[agent_id] = AgentPerformanceMetrics(
                    agent_id=agent_id
                )

            metrics = self._agent_metrics[agent_id]
            metrics.operation_count += 1
            metrics.last_activity = datetime.now()

            # Record operation start
            self._operation_history.append(
                {
                    "operation_id": operation_id,
                    "agent_id": agent_id,
                    "operation_type": operation_type,
                    "start_time": datetime.now(),
                    "status": "started",
                }
            )

            logger.debug(
                f"Started tracking operation {operation_id} for agent {agent_id}"
            )

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error: Optional[str] = None,
        tool_usage: Optional[Dict[str, int]] = None,
        decision_confidence: Optional[float] = None,
        output_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End tracking an operation and update metrics.

        Args:
            operation_id: Unique identifier for the operation
            success: Whether the operation was successful
            error: Error message if operation failed
            tool_usage: Dictionary of tools used and their counts
            decision_confidence: Confidence score for decisions (0-1)
            output_data: Additional output data for analysis
        """
        with self._lock:
            if operation_id not in self._start_times:
                logger.warning(f"Operation {operation_id} not found in start times")
                return

            end_time = time.time()
            start_data = self._start_times[operation_id]
            start_time = start_data["start_time"]
            agent_id = start_data["agent_id"]
            duration = end_time - start_time

            if agent_id not in self._agent_metrics:
                self._agent_metrics[agent_id] = AgentPerformanceMetrics(
                    agent_id=agent_id
                )

            metrics = self._agent_metrics[agent_id]

            # Update metrics
            metrics.total_response_time += duration
            if success:
                metrics.success_count += 1
            else:
                metrics.error_count += 1

            # Update tool usage
            if tool_usage:
                for tool, count in tool_usage.items():
                    metrics.tool_usage_stats[tool] = (
                        metrics.tool_usage_stats.get(tool, 0) + count
                    )

            # Update decision confidence
            if decision_confidence is not None:
                metrics.decision_confidence_scores.append(decision_confidence)

            # Record operation end
            self._operation_history.append(
                {
                    "operation_id": operation_id,
                    "agent_id": agent_id,
                    "duration": duration,
                    "success": success,
                    "error": error,
                    "tool_usage": tool_usage,
                    "decision_confidence": decision_confidence,
                    "end_time": datetime.now(),
                    "status": "completed",
                }
            )

            # Clean up
            del self._start_times[operation_id]

            logger.debug(
                f"Ended tracking operation {operation_id} - Duration: {duration:.3f}s"
            )

    def start_workflow(self, workflow_id: str, participating_agents: List[str]) -> None:
        """Start tracking a workflow.

        Args:
            workflow_id: Unique identifier for the workflow
            participating_agents: List of agent IDs participating in the workflow
        """
        with self._lock:
            self._active_workflows[workflow_id] = {
                "workflow_id": workflow_id,
                "participating_agents": participating_agents,
                "start_time": time.time(),
                "events": [],
            }

            # Update system metrics
            self._system_metrics.total_events_processed += 1
            self._event_timestamps.append(time.time())

            logger.debug(f"Started tracking workflow {workflow_id}")

    def end_workflow(
        self,
        workflow_id: str,
        success: bool = True,
        conflicts_resolved: int = 0,
        coordination_events: int = 0,
    ) -> None:
        """End tracking a workflow and update system metrics.

        Args:
            workflow_id: Unique identifier for the workflow
            success: Whether the workflow was successful
            conflicts_resolved: Number of conflicts resolved
            coordination_events: Number of coordination events
        """
        with self._lock:
            if workflow_id not in self._active_workflows:
                logger.warning(f"Workflow {workflow_id} not found in active workflows")
                return

            workflow_data = self._active_workflows[workflow_id]
            end_time = time.time()
            start_time = workflow_data["start_time"]
            duration = end_time - start_time

            # Update system metrics
            self._system_metrics.total_workflows_completed += 1
            self._system_metrics.average_workflow_duration = (
                self._system_metrics.average_workflow_duration
                * (self._system_metrics.total_workflows_completed - 1)
                + duration
            ) / self._system_metrics.total_workflows_completed

            # Calculate efficiency metrics
            agent_count = len(workflow_data["participating_agents"])
            if agent_count > 1:
                coordination_efficiency = coordination_events / (
                    agent_count * (agent_count - 1)
                )
                self._system_metrics.agent_coordination_efficiency = (
                    coordination_efficiency
                )

            if conflicts_resolved > 0:
                self._system_metrics.conflict_resolution_rate = (
                    conflicts_resolved / agent_count
                )

            # Calculate throughput
            self._calculate_throughput()

            # Record workflow end
            self._workflow_history.append(
                {
                    "workflow_id": workflow_id,
                    "duration": duration,
                    "success": success,
                    "participating_agents": workflow_data["participating_agents"],
                    "conflicts_resolved": conflicts_resolved,
                    "coordination_events": coordination_events,
                    "end_time": datetime.now(),
                }
            )

            # Clean up
            del self._active_workflows[workflow_id]

            logger.debug(
                f"Ended tracking workflow {workflow_id} - Duration: {duration:.3f}s"
            )

    def record_agent_collaboration(self, agent_id: str) -> None:
        """Record a collaboration event for an agent.

        Args:
            agent_id: ID of the agent involved in collaboration
        """
        with self._lock:
            if agent_id in self._agent_metrics:
                self._agent_metrics[agent_id].collaboration_count += 1

    def _calculate_throughput(self) -> None:
        """Calculate current system throughput."""
        now = time.time()
        one_minute_ago = now - 60

        # Count events in the last minute
        recent_events = [ts for ts in self._event_timestamps if ts > one_minute_ago]
        current_throughput = len(recent_events) / 60.0  # events per second

        self._system_metrics.system_throughput = current_throughput

        # Update peak throughput
        if current_throughput > self._system_metrics.peak_throughput:
            self._system_metrics.peak_throughput = current_throughput

        # Calculate error rate
        total_operations = sum(
            metrics.operation_count for metrics in self._agent_metrics.values()
        )
        total_errors = sum(
            metrics.error_count for metrics in self._agent_metrics.values()
        )
        if total_operations > 0:
            self._system_metrics.error_rate = total_errors / total_operations

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentPerformanceMetrics]:
        """Get performance metrics for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            AgentPerformanceMetrics if agent exists, None otherwise
        """
        with self._lock:
            return self._agent_metrics.get(agent_id)

    def get_all_agent_metrics(self) -> Dict[str, AgentPerformanceMetrics]:
        """Get performance metrics for all agents.

        Returns:
            Dictionary mapping agent IDs to their metrics
        """
        with self._lock:
            return self._agent_metrics.copy()

    def get_system_metrics(self) -> SystemWorkflowMetrics:
        """Get current system-wide workflow metrics.

        Returns:
            SystemWorkflowMetrics instance
        """
        with self._lock:
            self._calculate_throughput()
            self._system_metrics.last_updated = datetime.now()
            return self._system_metrics

    def export_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Export metrics in a format suitable for dashboard display.

        Returns:
            Dictionary containing dashboard-ready metrics
        """
        with self._lock:
            # Agent performance summary
            agent_summaries = {}
            for agent_id, metrics in self._agent_metrics.items():
                avg_response_time = (
                    metrics.total_response_time / metrics.operation_count
                    if metrics.operation_count > 0
                    else 0.0
                )
                success_rate = (
                    metrics.success_count / metrics.operation_count
                    if metrics.operation_count > 0
                    else 0.0
                )
                avg_confidence = (
                    sum(metrics.decision_confidence_scores)
                    / len(metrics.decision_confidence_scores)
                    if metrics.decision_confidence_scores
                    else 0.0
                )

                agent_summaries[agent_id] = {
                    "operation_count": metrics.operation_count,
                    "average_response_time": round(avg_response_time, 3),
                    "success_rate": round(success_rate, 3),
                    "error_count": metrics.error_count,
                    "tool_usage": metrics.tool_usage_stats,
                    "average_confidence": round(avg_confidence, 3),
                    "collaboration_count": metrics.collaboration_count,
                    "last_activity": metrics.last_activity.isoformat()
                    if metrics.last_activity
                    else None,
                }

            # System summary
            system_summary = {
                "total_events_processed": self._system_metrics.total_events_processed,
                "total_workflows_completed": self._system_metrics.total_workflows_completed,
                "average_workflow_duration": round(
                    self._system_metrics.average_workflow_duration, 3
                ),
                "agent_coordination_efficiency": round(
                    self._system_metrics.agent_coordination_efficiency, 3
                ),
                "conflict_resolution_rate": round(
                    self._system_metrics.conflict_resolution_rate, 3
                ),
                "system_throughput": round(self._system_metrics.system_throughput, 3),
                "peak_throughput": round(self._system_metrics.peak_throughput, 3),
                "error_rate": round(self._system_metrics.error_rate, 3),
                "active_workflows": len(self._active_workflows),
                "last_updated": self._system_metrics.last_updated.isoformat()
                if self._system_metrics.last_updated
                else None,
            }

            return {
                "timestamp": datetime.now().isoformat(),
                "agents": agent_summaries,
                "system": system_summary,
                "export_version": "1.0",
            }

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._agent_metrics.clear()
            self._operation_history.clear()
            self._system_metrics = SystemWorkflowMetrics()
            self._active_workflows.clear()
            self._workflow_history.clear()
            self._start_times.clear()
            self._event_timestamps.clear()

            logger.info("All metrics have been reset")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for logging or monitoring.

        Returns:
            Dictionary containing key metrics summary
        """
        with self._lock:
            total_agents = len(self._agent_metrics)
            total_operations = sum(
                metrics.operation_count for metrics in self._agent_metrics.values()
            )
            total_errors = sum(
                metrics.error_count for metrics in self._agent_metrics.values()
            )

            return {
                "total_agents": total_agents,
                "total_operations": total_operations,
                "total_errors": total_errors,
                "active_workflows": len(self._active_workflows),
                "system_throughput": round(self._system_metrics.system_throughput, 3),
                "average_workflow_duration": round(
                    self._system_metrics.average_workflow_duration, 3
                ),
                "last_updated": datetime.now().isoformat(),
            }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics_collector(max_history_size: int = 1000) -> MetricsCollector:
    """Initialize the global metrics collector.

    Args:
        max_history_size: Maximum number of recent metrics to keep

    Returns:
        Initialized MetricsCollector instance
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector(max_history_size=max_history_size)
    return _metrics_collector

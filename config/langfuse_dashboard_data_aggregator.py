"""Dashboard data aggregator for combining metrics and views."""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock, Timer

from config.langfuse_integration import get_langfuse_integration
from config.metrics_collector import get_metrics_collector
from config.langfuse_alerting import get_alert_manager

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all dashboard components."""

    timestamp: datetime

    # System metrics
    total_events_processed: int = 0
    system_throughput: float = 0.0
    average_workflow_duration: float = 0.0
    error_rate: float = 0.0

    # Agent metrics
    total_agents: int = 0
    active_agents: int = 0
    agent_performance_scores: Dict[str, float] = field(default_factory=dict)

    # Workflow metrics
    total_workflows: int = 0
    active_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    collaboration_rate: float = 0.0

    # Alert metrics
    total_alerts: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0
    resolved_alerts: int = 0

    # Performance indicators
    system_health_score: float = 100.0
    average_agent_response_time: float = 0.0
    peak_throughput: float = 0.0


class DashboardDataAggregator:
    """Aggregates data from all dashboard components for comprehensive insights."""

    def __init__(self, update_interval_seconds: int = 60):
        """Initialize the data aggregator.

        Args:
            update_interval_seconds: Interval for automatic data aggregation
        """
        self._integration_service = get_langfuse_integration()
        self._metrics_collector = get_metrics_collector()
        self._alert_manager = get_alert_manager()

        # Aggregation settings
        self._update_interval = update_interval_seconds
        self._aggregation_timer: Optional[Timer] = None
        self._is_running = False

        # Data storage
        self._aggregated_data: Optional[AggregatedMetrics] = None
        self._data_lock = Lock()
        self._aggregation_callbacks: List[Callable[[AggregatedMetrics], None]] = []

        # Historical data for trends
        self._history_size = 100
        self._metrics_history: List[AggregatedMetrics] = []

        logger.info(
            f"Initialized dashboard data aggregator with {update_interval_seconds}s interval"
        )

    def start_auto_aggregation(self) -> None:
        """Start automatic data aggregation."""
        if self._is_running:
            logger.warning("Auto aggregation already running")
            return

        self._is_running = True
        self._schedule_next_aggregation()
        logger.info("Started automatic dashboard data aggregation")

    def stop_auto_aggregation(self) -> None:
        """Stop automatic data aggregation."""
        if not self._is_running:
            return

        self._is_running = False
        if self._aggregation_timer:
            self._aggregation_timer.cancel()
            self._aggregation_timer = None
        logger.info("Stopped automatic dashboard data aggregation")

    def _schedule_next_aggregation(self) -> None:
        """Schedule the next data aggregation."""
        if not self._is_running:
            return

        self._aggregation_timer = Timer(
            self._update_interval, self._perform_aggregation
        )
        self._aggregation_timer.daemon = True
        self._aggregation_timer.start()

    def _perform_aggregation(self) -> None:
        """Perform data aggregation and notify callbacks."""
        try:
            aggregated_metrics = self.aggregate_current_data()

            with self._data_lock:
                self._aggregated_data = aggregated_metrics
                self._add_to_history(aggregated_metrics)

            # Notify callbacks
            for callback in self._aggregation_callbacks:
                try:
                    callback(aggregated_metrics)
                except Exception as e:
                    logger.error(f"Error in aggregation callback: {e}")

            logger.debug("Completed data aggregation")

        except Exception as e:
            logger.error(f"Error during data aggregation: {e}")
        finally:
            self._schedule_next_aggregation()

    def aggregate_current_data(self) -> AggregatedMetrics:
        """Aggregate current data from all sources."""
        timestamp = datetime.now()

        # Get system metrics
        system_metrics = self._integration_service.get_system_metrics()

        # Get agent metrics
        agent_metrics = self._integration_service.export_metrics_for_dashboard()

        # Get alert statistics
        alert_stats = self._alert_manager.alerting_engine.get_alert_stats()

        # Get workflow metrics
        workflow_metrics = self._integration_service.get_system_metrics()

        # Calculate aggregated values
        total_events = system_metrics.get("total_events_processed", 0)
        throughput = system_metrics.get("system_throughput", 0.0)
        avg_workflow_duration = system_metrics.get("average_workflow_duration", 0.0)
        error_rate = system_metrics.get("error_rate", 0.0)

        # Agent aggregation
        agents_data = agent_metrics.get("agents", {})
        total_agents = len(agents_data)
        active_agents = sum(
            1
            for agent in agents_data.values()
            if agent.get("current", {}).get("status") == "active"
        )

        # Calculate agent performance scores
        agent_performance_scores = {}
        for agent_id, agent_data in agents_data.items():
            current = agent_data.get("current", {})
            success_rate = current.get("success_rate", 0.0)
            response_time = current.get("average_response_time", 0.0)
            # Simple performance score based on success rate and response time
            score = success_rate * (
                1.0 / (1.0 + response_time / 1000.0)
            )  # Normalize response time
            agent_performance_scores[agent_id] = min(score, 1.0) * 100.0

        # Workflow aggregation
        total_workflows = workflow_metrics.get(
            "total_workflows_completed", 0
        ) + workflow_metrics.get("active_workflows", 0)
        active_workflows = workflow_metrics.get("active_workflows", 0)
        completed_workflows = workflow_metrics.get("total_workflows_completed", 0)
        failed_workflows = workflow_metrics.get("failed_workflows", 0)
        collaboration_rate = workflow_metrics.get("collaboration_rate", 0.0)

        # Alert aggregation
        total_alerts = alert_stats.get("total_active", 0)
        critical_alerts = alert_stats.get("by_severity", {}).get("critical", 0)
        warning_alerts = alert_stats.get("by_severity", {}).get("warning", 0)
        resolved_alerts = len(self._alert_manager.alerting_engine.get_alert_history(50))

        # Calculate system health score
        health_score = self._calculate_system_health_score(
            error_rate, throughput, total_alerts, agent_performance_scores
        )

        # Calculate average agent response time
        avg_response_time = sum(agent_performance_scores.values()) / max(
            len(agent_performance_scores), 1
        )

        # Get peak throughput from metrics
        peak_throughput = system_metrics.get("peak_throughput", throughput)

        return AggregatedMetrics(
            timestamp=timestamp,
            total_events_processed=total_events,
            system_throughput=throughput,
            average_workflow_duration=avg_workflow_duration,
            error_rate=error_rate,
            total_agents=total_agents,
            active_agents=active_agents,
            agent_performance_scores=agent_performance_scores,
            total_workflows=total_workflows,
            active_workflows=active_workflows,
            completed_workflows=completed_workflows,
            failed_workflows=failed_workflows,
            collaboration_rate=collaboration_rate,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            warning_alerts=warning_alerts,
            resolved_alerts=resolved_alerts,
            system_health_score=health_score,
            average_agent_response_time=avg_response_time,
            peak_throughput=peak_throughput,
        )

    def _calculate_system_health_score(
        self,
        error_rate: float,
        throughput: float,
        total_alerts: int,
        agent_scores: Dict[str, float],
    ) -> float:
        """Calculate overall system health score (0-100)."""
        # Base score starts at 100
        health_score = 100.0

        # Penalize for high error rate
        if error_rate > 0.05:  # 5% error rate
            health_score -= min(error_rate * 1000, 30.0)  # Up to 30 point penalty

        # Penalize for low throughput (assuming expected throughput > 1.0)
        if throughput < 1.0:
            health_score -= min((1.0 - throughput) * 20, 20.0)  # Up to 20 point penalty

        # Penalize for too many alerts
        if total_alerts > 5:
            health_score -= min(total_alerts * 2, 20.0)  # Up to 20 point penalty

        # Factor in average agent performance
        if agent_scores:
            avg_agent_score = sum(agent_scores.values()) / len(agent_scores)
            health_score = (health_score + avg_agent_score) / 2.0

        return max(health_score, 0.0)

    def _add_to_history(self, metrics: AggregatedMetrics) -> None:
        """Add metrics to history, maintaining size limit."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._history_size:
            self._metrics_history.pop(0)

    def get_latest_aggregated_data(self) -> Optional[AggregatedMetrics]:
        """Get the latest aggregated metrics data."""
        with self._data_lock:
            return self._aggregated_data

    def get_metrics_history(self, limit: int = 50) -> List[AggregatedMetrics]:
        """Get historical aggregated metrics data."""
        return self._metrics_history[-limit:] if self._metrics_history else []

    def get_trend_analysis(
        self, metric_name: str, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get trend analysis for a specific metric.

        Args:
            metric_name: Name of the metric to analyze
            time_window_minutes: Time window for trend analysis

        Returns:
            Dictionary containing trend analysis
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        relevant_metrics = [
            m for m in self._metrics_history if m.timestamp >= cutoff_time
        ]

        if len(relevant_metrics) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # Extract metric values
        values = []
        for metric in relevant_metrics:
            if hasattr(metric, metric_name):
                values.append(getattr(metric, metric_name))
            else:
                # Handle nested metrics
                if metric_name.startswith("agent_performance_"):
                    agent_id = metric_name.replace("agent_performance_", "")
                    values.append(metric.agent_performance_scores.get(agent_id, 0.0))
                else:
                    return {"trend": "unknown_metric", "confidence": 0.0}

        if len(values) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # Simple trend calculation
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.05:  # 5% increase
            trend = "increasing"
        elif second_avg < first_avg * 0.95:  # 5% decrease
            trend = "decreasing"
        else:
            trend = "stable"

        # Calculate confidence based on consistency
        differences = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        avg_difference = sum(differences) / len(differences)
        confidence = min(avg_difference / max(values), 1.0) if values else 0.0

        return {
            "trend": trend,
            "confidence": confidence,
            "current_value": values[-1],
            "previous_value": values[0],
            "change_percentage": ((values[-1] - values[0]) / values[0]) * 100
            if values[0] != 0
            else 0,
            "data_points": len(values),
        }

    def get_system_insights(self) -> Dict[str, Any]:
        """Get system insights based on aggregated data."""
        latest = self.get_latest_aggregated_data()
        if not latest:
            return {"insights": [], "last_updated": None}

        insights = []

        # Performance insights
        if latest.system_health_score < 70:
            insights.append(
                {
                    "type": "performance_warning",
                    "severity": "high",
                    "title": "System Health Degraded",
                    "message": f"System health score is {latest.system_health_score:.1f}%",
                    "recommendations": [
                        "Check for resource bottlenecks",
                        "Review recent error patterns",
                        "Consider scaling system resources",
                    ],
                }
            )

        # Throughput insights
        if latest.system_throughput < 1.0:
            insights.append(
                {
                    "type": "throughput_warning",
                    "severity": "medium",
                    "title": "Low System Throughput",
                    "message": f"Current throughput is {latest.system_throughput:.2f} events/sec",
                    "recommendations": [
                        "Review workflow efficiency",
                        "Check for processing bottlenecks",
                        "Consider optimization strategies",
                    ],
                }
            )

        # Agent performance insights
        underperforming_agents = [
            agent_id
            for agent_id, score in latest.agent_performance_scores.items()
            if score < 70
        ]
        if underperforming_agents:
            insights.append(
                {
                    "type": "agent_performance",
                    "severity": "medium",
                    "title": "Agent Performance Issues",
                    "message": f"{len(underperforming_agents)} agents have performance scores below 70%",
                    "affected_agents": underperforming_agents,
                    "recommendations": [
                        "Review agent configurations",
                        "Check for resource constraints",
                        "Consider retraining or optimization",
                    ],
                }
            )

        # Alert insights
        if latest.critical_alerts > 0:
            insights.append(
                {
                    "type": "critical_alerts",
                    "severity": "critical",
                    "title": "Critical Alerts Active",
                    "message": f"{latest.critical_alerts} critical alerts require immediate attention",
                    "recommendations": [
                        "Review critical alert details",
                        "Take immediate corrective action",
                        "Investigate root causes",
                    ],
                }
            )

        return {
            "insights": insights,
            "total_insights": len(insights),
            "last_updated": latest.timestamp.isoformat(),
        }

    def add_aggregation_callback(
        self, callback: Callable[[AggregatedMetrics], None]
    ) -> None:
        """Add a callback for aggregation updates."""
        self._aggregation_callbacks.append(callback)
        logger.debug(
            f"Added aggregation callback. Total callbacks: {len(self._aggregation_callbacks)}"
        )

    def remove_aggregation_callback(
        self, callback: Callable[[AggregatedMetrics], None]
    ) -> bool:
        """Remove an aggregation callback."""
        try:
            self._aggregation_callbacks.remove(callback)
            logger.debug(
                f"Removed aggregation callback. Total callbacks: {len(self._aggregation_callbacks)}"
            )
            return True
        except ValueError:
            return False

    def export_aggregated_data(self, format: str = "json") -> str:
        """Export aggregated data in specified format.

        Args:
            format: Export format ('json', 'csv')

        Returns:
            Exported data as string
        """
        latest = self.get_latest_aggregated_data()
        if not latest:
            return ""

        if format.lower() == "json":
            return json.dumps(
                {
                    "timestamp": latest.timestamp.isoformat(),
                    "metrics": {
                        "total_events_processed": latest.total_events_processed,
                        "system_throughput": latest.system_throughput,
                        "average_workflow_duration": latest.average_workflow_duration,
                        "error_rate": latest.error_rate,
                        "total_agents": latest.total_agents,
                        "active_agents": latest.active_agents,
                        "total_workflows": latest.total_workflows,
                        "active_workflows": latest.active_workflows,
                        "completed_workflows": latest.completed_workflows,
                        "failed_workflows": latest.failed_workflows,
                        "collaboration_rate": latest.collaboration_rate,
                        "total_alerts": latest.total_alerts,
                        "critical_alerts": latest.critical_alerts,
                        "warning_alerts": latest.warning_alerts,
                        "system_health_score": latest.system_health_score,
                        "average_agent_response_time": latest.average_agent_response_time,
                        "peak_throughput": latest.peak_throughput,
                    },
                    "agent_performance_scores": latest.agent_performance_scores,
                },
                indent=2,
            )
        else:
            # CSV-like format for metrics
            lines = [
                "Metric,Value",
                f"Total Events Processed,{latest.total_events_processed}",
                f"System Throughput,{latest.system_throughput:.2f}",
                f"Average Workflow Duration,{latest.average_workflow_duration:.2f}",
                f"Error Rate,{latest.error_rate:.3f}",
                f"Total Agents,{latest.total_agents}",
                f"Active Agents,{latest.active_agents}",
                f"Total Workflows,{latest.total_workflows}",
                f"Active Workflows,{latest.active_workflows}",
                f"Completed Workflows,{latest.completed_workflows}",
                f"Failed Workflows,{latest.failed_workflows}",
                f"Collaboration Rate,{latest.collaboration_rate:.3f}",
                f"Total Alerts,{latest.total_alerts}",
                f"Critical Alerts,{latest.critical_alerts}",
                f"Warning Alerts,{latest.warning_alerts}",
                f"System Health Score,{latest.system_health_score:.1f}",
                f"Average Agent Response Time,{latest.average_agent_response_time:.2f}",
                f"Peak Throughput,{latest.peak_throughput:.2f}",
            ]
            return "\n".join(lines)


# Global data aggregator instance
_data_aggregator: Optional[DashboardDataAggregator] = None


def get_dashboard_data_aggregator() -> DashboardDataAggregator:
    """Get the global dashboard data aggregator instance."""
    global _data_aggregator
    if _data_aggregator is None:
        _data_aggregator = DashboardDataAggregator()
    return _data_aggregator


def initialize_dashboard_data_aggregator(
    update_interval_seconds: int = 60,
) -> DashboardDataAggregator:
    """Initialize the global dashboard data aggregator.

    Args:
        update_interval_seconds: Interval for automatic data aggregation

    Returns:
        Initialized DashboardDataAggregator instance
    """
    global _data_aggregator
    _data_aggregator = DashboardDataAggregator(update_interval_seconds)
    return _data_aggregator

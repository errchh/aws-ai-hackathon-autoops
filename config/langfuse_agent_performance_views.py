"""Agent performance monitoring views for Langfuse dashboard integration."""

import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AgentPerformanceSnapshot:
    """Snapshot of agent performance at a specific point in time."""

    timestamp: datetime
    agent_id: str
    operation_count: int
    success_rate: float
    average_response_time: float
    error_count: int
    tool_usage: Dict[str, int]
    collaboration_count: int
    confidence_scores: List[float]
    status: str = "active"


@dataclass
class AgentPerformanceTrend:
    """Trend data for agent performance over time."""

    agent_id: str
    metric_name: str
    time_period: str  # '1h', '24h', '7d', '30d'
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    trend_direction: str = "stable"  # 'increasing', 'decreasing', 'stable'
    trend_strength: float = 0.0  # 0-1, higher means stronger trend


class AgentPerformanceAnalyzer:
    """Analyzes agent performance data for dashboard visualization."""

    def __init__(self):
        self.performance_history: Dict[str, List[AgentPerformanceSnapshot]] = (
            defaultdict(list)
        )
        self.max_history_size = 1000

    def add_performance_snapshot(self, snapshot: AgentPerformanceSnapshot) -> None:
        """Add a performance snapshot to the history."""
        agent_history = self.performance_history[snapshot.agent_id]

        # Add new snapshot
        agent_history.append(snapshot)

        # Maintain history size limit
        if len(agent_history) > self.max_history_size:
            agent_history.pop(0)

    def get_agent_performance_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of agent performance."""
        if agent_id not in self.performance_history:
            return None

        snapshots = self.performance_history[agent_id]
        if not snapshots:
            return None

        latest = snapshots[-1]

        # Calculate trends
        response_time_trend = self._calculate_trend(agent_id, "average_response_time")
        success_rate_trend = self._calculate_trend(agent_id, "success_rate")
        operation_count_trend = self._calculate_trend(agent_id, "operation_count")

        return {
            "agent_id": agent_id,
            "current": {
                "operation_count": latest.operation_count,
                "success_rate": round(latest.success_rate, 3),
                "average_response_time": round(latest.average_response_time, 3),
                "error_count": latest.error_count,
                "collaboration_count": latest.collaboration_count,
                "status": latest.status,
                "last_updated": latest.timestamp.isoformat(),
            },
            "trends": {
                "response_time": response_time_trend,
                "success_rate": success_rate_trend,
                "operation_count": operation_count_trend,
            },
            "tool_usage": latest.tool_usage,
            "confidence_stats": {
                "average": round(
                    sum(latest.confidence_scores) / len(latest.confidence_scores), 3
                )
                if latest.confidence_scores
                else 0.0,
                "min": min(latest.confidence_scores)
                if latest.confidence_scores
                else 0.0,
                "max": max(latest.confidence_scores)
                if latest.confidence_scores
                else 0.0,
            },
        }

    def get_all_agents_summary(self) -> Dict[str, Any]:
        """Get performance summaries for all agents."""
        summaries = {}
        for agent_id in self.performance_history.keys():
            summary = self.get_agent_performance_summary(agent_id)
            if summary:
                summaries[agent_id] = summary

        return {
            "agents": summaries,
            "total_agents": len(summaries),
            "last_updated": datetime.now().isoformat(),
        }

    def get_agent_comparison(self) -> Dict[str, Any]:
        """Get comparison data between agents."""
        if not self.performance_history:
            return {"comparison": [], "last_updated": datetime.now().isoformat()}

        agents_data = []
        for agent_id, snapshots in self.performance_history.items():
            if snapshots:
                latest = snapshots[-1]
                agents_data.append(
                    {
                        "agent_id": agent_id,
                        "operation_count": latest.operation_count,
                        "success_rate": latest.success_rate,
                        "average_response_time": latest.average_response_time,
                        "error_rate": latest.error_count
                        / max(latest.operation_count, 1),
                        "collaboration_count": latest.collaboration_count,
                    }
                )

        # Sort by success rate (descending)
        agents_data.sort(key=lambda x: x["success_rate"], reverse=True)

        return {
            "comparison": agents_data,
            "best_performer": agents_data[0]["agent_id"] if agents_data else None,
            "worst_performer": agents_data[-1]["agent_id"] if agents_data else None,
            "last_updated": datetime.now().isoformat(),
        }

    def get_performance_alerts(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get performance alerts for an agent."""
        alerts = []

        if agent_id not in self.performance_history:
            return alerts

        snapshots = self.performance_history[agent_id]
        if len(snapshots) < 2:
            return alerts

        latest = snapshots[-1]
        previous = snapshots[-2]

        # Check for performance degradation
        if latest.average_response_time > previous.average_response_time * 1.5:
            alerts.append(
                {
                    "type": "performance_degradation",
                    "severity": "warning",
                    "title": "Response Time Increased",
                    "message": f"Response time increased by {((latest.average_response_time - previous.average_response_time) / previous.average_response_time * 100):.1f}%",
                    "timestamp": latest.timestamp.isoformat(),
                }
            )

        if latest.success_rate < previous.success_rate * 0.95:
            alerts.append(
                {
                    "type": "performance_degradation",
                    "severity": "error",
                    "title": "Success Rate Decreased",
                    "message": f"Success rate decreased by {((previous.success_rate - latest.success_rate) / previous.success_rate * 100):.1f}%",
                    "timestamp": latest.timestamp.isoformat(),
                }
            )

        if latest.error_count > previous.error_count * 2:
            alerts.append(
                {
                    "type": "error_spike",
                    "severity": "critical",
                    "title": "Error Count Spike",
                    "message": f"Error count increased from {previous.error_count} to {latest.error_count}",
                    "timestamp": latest.timestamp.isoformat(),
                }
            )

        return alerts

    def _calculate_trend(self, agent_id: str, metric_name: str) -> Dict[str, Any]:
        """Calculate trend for a specific metric."""
        if agent_id not in self.performance_history:
            return {"direction": "stable", "strength": 0.0}

        snapshots = self.performance_history[agent_id]
        if len(snapshots) < 3:
            return {"direction": "stable", "strength": 0.0}

        # Get recent values for trend calculation
        recent_snapshots = snapshots[-10:]  # Last 10 snapshots
        values = []

        for snapshot in recent_snapshots:
            if metric_name == "average_response_time":
                values.append(snapshot.average_response_time)
            elif metric_name == "success_rate":
                values.append(snapshot.success_rate)
            elif metric_name == "operation_count":
                values.append(snapshot.operation_count)
            else:
                return {"direction": "stable", "strength": 0.0}

        if len(values) < 3:
            return {"direction": "stable", "strength": 0.0}

        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Determine direction and strength
        if abs(slope) < 0.01:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) / max(y), 1.0)  # Normalize by max value
        else:
            direction = "decreasing"
            strength = min(abs(slope) / max(y), 1.0)

        return {
            "direction": direction,
            "strength": round(strength, 3),
            "slope": round(slope, 3),
        }

    def export_agent_performance_data(self, agent_id: str) -> Optional[str]:
        """Export agent performance data as JSON."""
        summary = self.get_agent_performance_summary(agent_id)
        if not summary:
            return None

        return json.dumps(summary, indent=2)


class LangfuseAgentPerformanceViewManager:
    """Manages agent performance views for the dashboard."""

    def __init__(self):
        self.analyzer = AgentPerformanceAnalyzer()
        self.performance_callbacks: List[
            Callable[[str, AgentPerformanceSnapshot], None]
        ] = []

    def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Update agent performance data."""
        snapshot = AgentPerformanceSnapshot(
            timestamp=datetime.now(),
            agent_id=agent_id,
            operation_count=metrics.get("operation_count", 0),
            success_rate=metrics.get("success_rate", 0.0),
            average_response_time=metrics.get("average_response_time", 0.0),
            error_count=metrics.get("error_count", 0),
            tool_usage=metrics.get("tool_usage", {}),
            collaboration_count=metrics.get("collaboration_count", 0),
            confidence_scores=metrics.get("confidence_scores", []),
            status=metrics.get("status", "active"),
        )

        self.analyzer.add_performance_snapshot(snapshot)

        # Trigger callbacks
        for callback in self.performance_callbacks:
            try:
                callback(agent_id, snapshot)
            except Exception:
                pass  # Ignore callback errors

    def get_dashboard_agent_data(self) -> Dict[str, Any]:
        """Get agent performance data formatted for dashboard display."""
        all_summaries = self.analyzer.get_all_agents_summary()
        comparison = self.analyzer.get_agent_comparison()

        # Get alerts for all agents
        all_alerts = []
        for agent_id in self.analyzer.performance_history.keys():
            alerts = self.analyzer.get_performance_alerts(agent_id)
            all_alerts.extend(alerts)

        return {
            "agents": all_summaries["agents"],
            "comparison": comparison["comparison"],
            "alerts": all_alerts,
            "summary": {
                "total_agents": all_summaries["total_agents"],
                "best_performer": comparison.get("best_performer"),
                "worst_performer": comparison.get("worst_performer"),
                "total_alerts": len(all_alerts),
            },
            "last_updated": datetime.now().isoformat(),
        }

    def get_agent_detailed_view(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view for a specific agent."""
        summary = self.analyzer.get_agent_performance_summary(agent_id)
        if not summary:
            return None

        alerts = self.analyzer.get_performance_alerts(agent_id)

        # Get historical data for charts
        snapshots = self.analyzer.performance_history.get(agent_id, [])
        historical_data = []

        for snapshot in snapshots[-50:]:  # Last 50 snapshots
            historical_data.append(
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "operation_count": snapshot.operation_count,
                    "success_rate": snapshot.success_rate,
                    "average_response_time": snapshot.average_response_time,
                    "error_count": snapshot.error_count,
                }
            )

        return {
            "agent_id": agent_id,
            "summary": summary,
            "alerts": alerts,
            "historical_data": historical_data,
            "last_updated": datetime.now().isoformat(),
        }

    def add_performance_callback(
        self, callback: Callable[[str, AgentPerformanceSnapshot], None]
    ) -> None:
        """Add a callback for performance updates."""
        self.performance_callbacks.append(callback)


# Global agent performance view manager instance
_agent_performance_manager: Optional[LangfuseAgentPerformanceViewManager] = None


def get_agent_performance_view_manager() -> LangfuseAgentPerformanceViewManager:
    """Get the global agent performance view manager instance."""
    global _agent_performance_manager
    if _agent_performance_manager is None:
        _agent_performance_manager = LangfuseAgentPerformanceViewManager()
    return _agent_performance_manager


def initialize_agent_performance_view_manager() -> LangfuseAgentPerformanceViewManager:
    """Initialize the global agent performance view manager."""
    global _agent_performance_manager
    _agent_performance_manager = LangfuseAgentPerformanceViewManager()
    return _agent_performance_manager

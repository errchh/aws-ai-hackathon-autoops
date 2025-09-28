"""Langfuse dashboard configuration for retail optimization workflows."""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class DashboardView:
    """Configuration for a dashboard view."""

    id: str
    name: str
    description: str
    type: str  # 'overview', 'agent_performance', 'workflow', 'alerts'
    filters: Dict[str, Any] = field(default_factory=dict)
    time_range: str = "1h"  # '1h', '24h', '7d', '30d'
    refresh_interval: int = 30  # seconds
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Main dashboard configuration."""

    name: str
    description: str
    version: str = "1.0"
    views: List[DashboardView] = field(default_factory=list)
    global_filters: Dict[str, Any] = field(default_factory=dict)
    alerting_config: Dict[str, Any] = field(default_factory=dict)


class LangfuseDashboardConfig:
    """Manages Langfuse dashboard configurations for retail optimization."""

    def __init__(self):
        self.config = self._create_default_config()

    def _create_default_config(self) -> DashboardConfig:
        """Create the default dashboard configuration."""

        views = [
            # Overview Dashboard
            DashboardView(
                id="overview",
                name="System Overview",
                description="High-level view of system performance and agent status",
                type="overview",
                time_range="1h",
                refresh_interval=30,
                layout={
                    "widgets": [
                        {
                            "type": "metric_card",
                            "title": "Total Events Processed",
                            "position": {"x": 0, "y": 0, "width": 3, "height": 2},
                            "metric": "total_events_processed",
                        },
                        {
                            "type": "metric_card",
                            "title": "System Throughput",
                            "position": {"x": 3, "y": 0, "width": 3, "height": 2},
                            "metric": "system_throughput",
                            "unit": "events/sec",
                        },
                        {
                            "type": "metric_card",
                            "title": "Average Workflow Duration",
                            "position": {"x": 6, "y": 0, "width": 3, "height": 2},
                            "metric": "average_workflow_duration",
                            "unit": "seconds",
                        },
                        {
                            "type": "metric_card",
                            "title": "Error Rate",
                            "position": {"x": 9, "y": 0, "width": 3, "height": 2},
                            "metric": "error_rate",
                            "format": "percentage",
                        },
                        {
                            "type": "agent_status_grid",
                            "title": "Agent Status",
                            "position": {"x": 0, "y": 2, "width": 12, "height": 3},
                        },
                        {
                            "type": "throughput_chart",
                            "title": "System Throughput Over Time",
                            "position": {"x": 0, "y": 5, "width": 6, "height": 4},
                        },
                        {
                            "type": "error_rate_chart",
                            "title": "Error Rate Trends",
                            "position": {"x": 6, "y": 5, "width": 6, "height": 4},
                        },
                    ]
                },
            ),
            # Agent Performance Dashboard
            DashboardView(
                id="agent_performance",
                name="Agent Performance",
                description="Detailed view of individual agent performance metrics",
                type="agent_performance",
                time_range="24h",
                refresh_interval=60,
                layout={
                    "widgets": [
                        {
                            "type": "agent_metrics_table",
                            "title": "Agent Performance Metrics",
                            "position": {"x": 0, "y": 0, "width": 12, "height": 4},
                            "columns": [
                                "agent_id",
                                "operation_count",
                                "success_rate",
                                "average_response_time",
                                "error_count",
                                "collaboration_count",
                            ],
                        },
                        {
                            "type": "response_time_chart",
                            "title": "Agent Response Times",
                            "position": {"x": 0, "y": 4, "width": 6, "height": 4},
                        },
                        {
                            "type": "success_rate_chart",
                            "title": "Agent Success Rates",
                            "position": {"x": 6, "y": 4, "width": 6, "height": 4},
                        },
                        {
                            "type": "tool_usage_heatmap",
                            "title": "Tool Usage by Agent",
                            "position": {"x": 0, "y": 8, "width": 12, "height": 4},
                        },
                    ]
                },
            ),
            # Workflow Visualization Dashboard
            DashboardView(
                id="workflow_visualization",
                name="Workflow Visualization",
                description="Visualization of simulation-to-decision workflows",
                type="workflow",
                time_range="1h",
                refresh_interval=15,
                layout={
                    "widgets": [
                        {
                            "type": "workflow_flow_diagram",
                            "title": "Active Workflows",
                            "position": {"x": 0, "y": 0, "width": 12, "height": 6},
                        },
                        {
                            "type": "trace_timeline",
                            "title": "Recent Traces",
                            "position": {"x": 0, "y": 6, "width": 8, "height": 4},
                        },
                        {
                            "type": "workflow_metrics",
                            "title": "Workflow Metrics",
                            "position": {"x": 8, "y": 6, "width": 4, "height": 4},
                        },
                    ]
                },
            ),
            # Alerts Dashboard
            DashboardView(
                id="alerts",
                name="Alerts & Monitoring",
                description="System alerts and performance monitoring",
                type="alerts",
                time_range="24h",
                refresh_interval=30,
                layout={
                    "widgets": [
                        {
                            "type": "active_alerts_list",
                            "title": "Active Alerts",
                            "position": {"x": 0, "y": 0, "width": 12, "height": 3},
                        },
                        {
                            "type": "alert_trends_chart",
                            "title": "Alert Trends",
                            "position": {"x": 0, "y": 3, "width": 6, "height": 4},
                        },
                        {
                            "type": "performance_alerts",
                            "title": "Performance Alerts",
                            "position": {"x": 6, "y": 3, "width": 6, "height": 4},
                        },
                    ]
                },
            ),
        ]

        alerting_config = {
            "performance_alerts": {
                "high_latency": {
                    "enabled": True,
                    "threshold_ms": 100,
                    "window_minutes": 5,
                    "severity": "warning",
                },
                "low_success_rate": {
                    "enabled": True,
                    "threshold_percentage": 95,
                    "window_minutes": 10,
                    "severity": "error",
                },
                "high_error_rate": {
                    "enabled": True,
                    "threshold_percentage": 5,
                    "window_minutes": 5,
                    "severity": "critical",
                },
            },
            "system_alerts": {
                "langfuse_unavailable": {
                    "enabled": True,
                    "check_interval_seconds": 30,
                    "severity": "critical",
                },
                "high_memory_usage": {
                    "enabled": True,
                    "threshold_percentage": 80,
                    "check_interval_seconds": 60,
                    "severity": "warning",
                },
            },
        }

        return DashboardConfig(
            name="AutoOps Retail Optimization Dashboard",
            description="Comprehensive dashboard for monitoring retail optimization workflows and agent performance",
            version="1.0",
            views=views,
            global_filters={
                "project": "autoops_retail_optimization",
                "environment": "production",
            },
            alerting_config=alerting_config,
        )

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get the complete dashboard configuration as a dictionary."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "views": [
                {
                    "id": view.id,
                    "name": view.name,
                    "description": view.description,
                    "type": view.type,
                    "filters": view.filters,
                    "time_range": view.time_range,
                    "refresh_interval": view.refresh_interval,
                    "layout": view.layout,
                }
                for view in self.config.views
            ],
            "global_filters": self.config.global_filters,
            "alerting_config": self.config.alerting_config,
        }

    def get_view_config(self, view_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific view."""
        for view in self.config.views:
            if view.id == view_id:
                return {
                    "id": view.id,
                    "name": view.name,
                    "description": view.description,
                    "type": view.type,
                    "filters": view.filters,
                    "time_range": view.time_range,
                    "refresh_interval": view.refresh_interval,
                    "layout": view.layout,
                }
        return None

    def export_config_json(self, filepath: str) -> None:
        """Export the dashboard configuration to a JSON file."""
        config_dict = self.get_dashboard_config()
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    def get_alerting_config(self) -> Dict[str, Any]:
        """Get the alerting configuration."""
        return self.config.alerting_config


# Global dashboard configuration instance
_dashboard_config: Optional[LangfuseDashboardConfig] = None


def get_dashboard_config() -> LangfuseDashboardConfig:
    """Get the global dashboard configuration instance."""
    global _dashboard_config
    if _dashboard_config is None:
        _dashboard_config = LangfuseDashboardConfig()
    return _dashboard_config


def initialize_dashboard_config() -> LangfuseDashboardConfig:
    """Initialize the global dashboard configuration."""
    global _dashboard_config
    _dashboard_config = LangfuseDashboardConfig()
    return _dashboard_config

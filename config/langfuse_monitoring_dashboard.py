"""Dashboard integration for Langfuse monitoring and alerting data.

This module provides API endpoints and data formatting for dashboard display
of monitoring metrics, health status, and alerting information.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from .langfuse_monitoring_alerting import (
    get_monitoring_system,
    LangfuseMonitoringAlertingSystem,
)
from .langfuse_health_checker import get_health_checker, LangfuseHealthChecker
from .metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard-ready metrics data."""

    timestamp: str
    health_status: str
    uptime_seconds: float
    active_alerts: int
    system_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    health_score: float
    recommendations: List[str]


@dataclass
class AlertSummary:
    """Summary of alerting status."""

    total_active: int
    by_severity: Dict[str, int]
    recent_count: int
    resolution_rate: float


@dataclass
class PerformanceSummary:
    """Summary of performance metrics."""

    average_response_time: float
    throughput: float
    error_rate: float
    queue_size: int
    trend: str  # "improving", "stable", "degrading"


class LangfuseMonitoringDashboard:
    """Dashboard integration for monitoring and alerting data."""

    def __init__(
        self,
        monitoring_system: Optional[LangfuseMonitoringAlertingSystem] = None,
        health_checker: Optional[LangfuseHealthChecker] = None,
    ):
        """Initialize the dashboard integration.

        Args:
            monitoring_system: Optional monitoring system instance
            health_checker: Optional health checker instance
        """
        self.monitoring_system = monitoring_system or get_monitoring_system()
        self.health_checker = health_checker or get_health_checker()
        self.metrics_collector = get_metrics_collector()

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics."""
        try:
            # Get monitoring summary
            summary = self.monitoring_system.get_monitoring_summary()

            # Get health diagnostics
            health_diag = self.health_checker.perform_comprehensive_health_check()

            # Calculate health score (0-100)
            health_score = self._calculate_health_score(summary, health_diag)

            # Get recommendations
            recommendations = self._generate_dashboard_recommendations(
                summary, health_diag
            )

            return DashboardMetrics(
                timestamp=summary.timestamp.isoformat(),
                health_status=summary.health_status.value,
                uptime_seconds=summary.uptime_seconds,
                active_alerts=summary.active_alerts,
                system_metrics=summary.system_metrics,
                performance_metrics=summary.performance_metrics,
                recent_alerts=summary.recent_alerts,
                health_score=health_score,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            # Return default metrics on error
            return DashboardMetrics(
                timestamp=datetime.now().isoformat(),
                health_status="unknown",
                uptime_seconds=0,
                active_alerts=0,
                system_metrics={},
                performance_metrics={},
                recent_alerts=[],
                health_score=0,
                recommendations=["Unable to retrieve monitoring data"],
            )

    def get_alert_summary(self) -> AlertSummary:
        """Get alerting summary for dashboard."""
        try:
            if not self.monitoring_system.alert_manager:
                return AlertSummary(0, {}, 0, 0.0)

            dashboard_data = (
                self.monitoring_system.alert_manager.get_dashboard_alert_data()
            )
            stats = dashboard_data.get("stats", {})

            # Calculate resolution rate (mock calculation)
            recent_alerts = dashboard_data.get("recent_alerts", [])
            resolved_count = len([a for a in recent_alerts if a.get("resolved", False)])
            total_recent = len(recent_alerts)
            resolution_rate = (
                (resolved_count / total_recent) if total_recent > 0 else 0.0
            )

            return AlertSummary(
                total_active=stats.get("total_active", 0),
                by_severity=stats.get("by_severity", {}),
                recent_count=total_recent,
                resolution_rate=resolution_rate,
            )

        except Exception as e:
            logger.error(f"Failed to get alert summary: {e}")
            return AlertSummary(0, {}, 0, 0.0)

    def get_performance_summary(self) -> PerformanceSummary:
        """Get performance summary for dashboard."""
        try:
            summary = self.monitoring_system.get_monitoring_summary()
            perf_metrics = summary.performance_metrics

            # Determine trend (simplified logic)
            trend = "stable"
            if perf_metrics.get("langfuse_latency", 0) > 1000:
                trend = "degrading"
            elif perf_metrics.get("trace_throughput", 0) > 50:
                trend = "improving"

            return PerformanceSummary(
                average_response_time=perf_metrics.get("langfuse_latency", 0),
                throughput=perf_metrics.get("trace_throughput", 0),
                error_rate=perf_metrics.get("error_rate", 0),
                queue_size=perf_metrics.get("queue_size", 0),
                trend=trend,
            )

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return PerformanceSummary(0, 0, 0, 0, "unknown")

    def get_health_diagnostics(self) -> Dict[str, Any]:
        """Get detailed health diagnostics."""
        try:
            diagnostics = self.health_checker.perform_comprehensive_health_check()

            return {
                "timestamp": diagnostics.timestamp.isoformat(),
                "overall_status": diagnostics.overall_status,
                "connectivity_tests": [
                    {
                        "name": test.test_name,
                        "success": test.success,
                        "duration_ms": test.duration_ms,
                        "error": test.error,
                        "details": test.details,
                    }
                    for test in diagnostics.connectivity_tests
                ],
                "performance_benchmarks": diagnostics.performance_benchmarks,
                "configuration_issues": diagnostics.configuration_issues,
                "security_issues": diagnostics.security_issues,
                "recommendations": diagnostics.recommendations,
            }

        except Exception as e:
            logger.error(f"Failed to get health diagnostics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e),
                "connectivity_tests": [],
                "performance_benchmarks": {},
                "configuration_issues": [],
                "security_issues": [],
                "recommendations": ["Unable to retrieve health diagnostics"],
            }

    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview combining all monitoring data."""
        try:
            dashboard_metrics = self.get_dashboard_metrics()
            alert_summary = self.get_alert_summary()
            performance_summary = self.get_performance_summary()
            health_diagnostics = self.get_health_diagnostics()

            return {
                "timestamp": dashboard_metrics.timestamp,
                "overview": {
                    "health_status": dashboard_metrics.health_status,
                    "health_score": dashboard_metrics.health_score,
                    "uptime_seconds": dashboard_metrics.uptime_seconds,
                    "active_alerts": dashboard_metrics.active_alerts,
                },
                "alerts": {
                    "total_active": alert_summary.total_active,
                    "by_severity": alert_summary.by_severity,
                    "recent_count": alert_summary.recent_count,
                    "resolution_rate": alert_summary.resolution_rate,
                },
                "performance": {
                    "average_response_time": performance_summary.average_response_time,
                    "throughput": performance_summary.throughput,
                    "error_rate": performance_summary.error_rate,
                    "queue_size": performance_summary.queue_size,
                    "trend": performance_summary.trend,
                },
                "health_diagnostics": health_diagnostics,
                "recommendations": dashboard_metrics.recommendations,
            }

        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overview": {
                    "health_status": "error",
                    "health_score": 0,
                    "uptime_seconds": 0,
                    "active_alerts": 0,
                },
                "error": str(e),
            }

    def _calculate_health_score(self, summary, health_diag) -> float:
        """Calculate overall health score (0-100)."""
        try:
            score = 100.0

            # Penalize based on health status
            if summary.health_status.value == "unhealthy":
                score -= 50
            elif summary.health_status.value == "degraded":
                score -= 25

            # Penalize based on active alerts
            alert_penalty = min(summary.active_alerts * 5, 30)
            score -= alert_penalty

            # Penalize based on performance issues
            perf_metrics = summary.performance_metrics
            if perf_metrics.get("error_rate", 0) > 0.1:
                score -= 20
            if perf_metrics.get("langfuse_latency", 0) > 1000:
                score -= 15

            # Check connectivity test failures
            failed_tests = len(
                [t for t in health_diag.connectivity_tests if not t.success]
            )
            test_penalty = min(failed_tests * 10, 25)
            score -= test_penalty

            return max(0, score)

        except Exception as e:
            logger.debug(f"Failed to calculate health score: {e}")
            return 0.0

    def _generate_dashboard_recommendations(self, summary, health_diag) -> List[str]:
        """Generate dashboard-specific recommendations."""
        recommendations = []

        try:
            # Health-based recommendations
            if summary.health_status.value == "unhealthy":
                recommendations.append(
                    "System health is critical - immediate attention required"
                )
            elif summary.health_status.value == "degraded":
                recommendations.append("System health is degraded - monitor closely")

            # Alert-based recommendations
            if summary.active_alerts > 5:
                recommendations.append(
                    "High number of active alerts - review and resolve critical issues"
                )

            # Performance-based recommendations
            perf_metrics = summary.performance_metrics
            if perf_metrics.get("error_rate", 0) > 0.05:
                recommendations.append(
                    "High error rate detected - investigate error patterns"
                )

            if perf_metrics.get("langfuse_latency", 0) > 500:
                recommendations.append(
                    "High latency detected - check network connectivity"
                )

            # Configuration recommendations
            if health_diag.configuration_issues:
                recommendations.append(
                    "Configuration issues detected - review settings"
                )

            # Security recommendations
            if health_diag.security_issues:
                recommendations.append(
                    "Security issues detected - address for production deployment"
                )

        except Exception as e:
            logger.debug(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations")

        return recommendations

    def export_metrics_json(self) -> str:
        """Export dashboard metrics as JSON string."""
        try:
            metrics = self.get_dashboard_metrics()
            return json.dumps(asdict(metrics), indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to export metrics JSON: {e}")
            return json.dumps({"error": str(e)})

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        # This would typically query a time-series database
        # For now, return current metrics as a placeholder
        try:
            current = self.get_dashboard_metrics()
            return [asdict(current)]
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []


# Global dashboard instance
_dashboard: Optional[LangfuseMonitoringDashboard] = None


def get_monitoring_dashboard() -> LangfuseMonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = LangfuseMonitoringDashboard()
    return _dashboard


def initialize_monitoring_dashboard(
    monitoring_system: Optional[LangfuseMonitoringAlertingSystem] = None,
    health_checker: Optional[LangfuseHealthChecker] = None,
) -> LangfuseMonitoringDashboard:
    """Initialize the global monitoring dashboard.

    Args:
        monitoring_system: Optional monitoring system instance
        health_checker: Optional health checker instance

    Returns:
        Initialized LangfuseMonitoringDashboard instance
    """
    global _dashboard
    _dashboard = LangfuseMonitoringDashboard(
        monitoring_system=monitoring_system,
        health_checker=health_checker,
    )
    return _dashboard

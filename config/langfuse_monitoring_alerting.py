"""Comprehensive monitoring and alerting system for Langfuse workflow visualization.

This module integrates all monitoring and alerting components to provide:
- Health checks for Langfuse connectivity
- Performance monitoring for tracing latency and throughput
- Alerting for trace success rates and system degradation
- Logging and debugging tools for troubleshooting
- Dashboard integration for monitoring data
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback

from .langfuse_performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    PerformanceThresholds,
    get_langfuse_performance_monitor,
    initialize_langfuse_performance_monitor,
)
from .langfuse_alerting import (
    AlertingEngine,
    LangfuseAlertManager,
    Alert,
    AlertSeverity,
    AlertRule,
    get_alert_manager,
    initialize_alert_manager,
)
from .langfuse_error_handler import (
    LangfuseErrorHandler,
    get_langfuse_error_handler,
)
from .metrics_collector import (
    MetricsCollector,
    get_metrics_collector,
)
from .langfuse_integration import (
    get_langfuse_integration,
)

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    timestamp: datetime
    checks: Dict[str, Any]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MonitoringSummary:
    """Summary of monitoring data."""

    timestamp: datetime
    health_status: HealthStatus
    active_alerts: int
    system_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recent_alerts: List[Dict[str, Any]]
    uptime_seconds: float


class LangfuseMonitoringAlertingSystem:
    """Unified monitoring and alerting system for Langfuse integration."""

    def __init__(
        self,
        performance_monitor: Optional[PerformanceMonitor] = None,
        alert_manager: Optional[LangfuseAlertManager] = None,
        error_handler: Optional[LangfuseErrorHandler] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        health_check_interval: int = 30,
        enable_debug_logging: bool = False,
    ):
        """Initialize the monitoring and alerting system.

        Args:
            performance_monitor: Optional performance monitor instance
            alert_manager: Optional alert manager instance
            error_handler: Optional error handler instance
            metrics_collector: Optional metrics collector instance
            health_check_interval: Seconds between health checks
            enable_debug_logging: Enable detailed debug logging
        """
        self.performance_monitor = (
            performance_monitor or get_langfuse_performance_monitor()
        )
        self.alert_manager = alert_manager or get_alert_manager()
        self.error_handler = error_handler or get_langfuse_error_handler()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.health_check_interval = health_check_interval
        self.enable_debug_logging = enable_debug_logging

        # Monitoring state
        self._monitoring_active = False
        self._health_check_thread: Optional[threading.Thread] = None
        self._last_health_check: Optional[datetime] = None
        self._system_start_time = datetime.now()

        # Enhanced alert rules for monitoring
        self._setup_monitoring_alert_rules()

        # Debug logging
        self._debug_log_file = None
        if enable_debug_logging:
            self._setup_debug_logging()

        logger.info("LangfuseMonitoringAlertingSystem initialized")

    def _setup_monitoring_alert_rules(self):
        """Set up additional alert rules specific to monitoring."""
        # Add trace success rate monitoring
        trace_success_rule = AlertRule(
            name="trace_success_rate",
            description="Trace success rate below threshold",
            severity=AlertSeverity.WARNING,
            enabled=True,
            check_interval_seconds=60,
            threshold=95.0,  # 95% success rate
            window_minutes=5,
            condition="less_than",
            metric_name="trace_success_rate",
            cooldown_minutes=10,
        )
        self.alert_manager.alerting_engine.add_rule(trace_success_rule)

        # Add system degradation monitoring
        degradation_rule = AlertRule(
            name="system_degradation",
            description="System in degraded state",
            severity=AlertSeverity.ERROR,
            enabled=True,
            check_interval_seconds=30,
            threshold=1.0,
            condition="greater_than",
            metric_name="degradation_level",
            cooldown_minutes=5,
        )
        self.alert_manager.alerting_engine.add_rule(degradation_rule)

        # Add buffer overflow monitoring
        buffer_overflow_rule = AlertRule(
            name="buffer_overflow",
            description="Trace buffer approaching capacity",
            severity=AlertSeverity.WARNING,
            enabled=True,
            check_interval_seconds=60,
            threshold=80.0,  # 80% of buffer capacity
            condition="greater_than",
            metric_name="buffer_utilization_percent",
            cooldown_minutes=15,
        )
        self.alert_manager.alerting_engine.add_rule(buffer_overflow_rule)

    def _setup_debug_logging(self):
        """Set up debug logging for troubleshooting."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"langfuse_monitoring_debug_{timestamp}.log"
            self._debug_log_file = open(log_filename, "w")

            # Add debug handler to logger
            debug_handler = logging.FileHandler(log_filename)
            debug_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            debug_handler.setFormatter(formatter)

            # Get the root logger and add our handler
            root_logger = logging.getLogger()
            root_logger.addHandler(debug_handler)
            root_logger.setLevel(logging.DEBUG)

            logger.info(f"Debug logging enabled, writing to {log_filename}")

        except Exception as e:
            logger.error(f"Failed to setup debug logging: {e}")

    def start_monitoring(self):
        """Start the monitoring system."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return

        self._monitoring_active = True
        self._start_health_check_thread()

        logger.info("Langfuse monitoring system started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False

        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)

        if self._debug_log_file:
            self._debug_log_file.close()

        logger.info("Langfuse monitoring system stopped")

    def _start_health_check_thread(self):
        """Start the background health check thread."""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True, name="LangfuseHealthCheck"
        )
        self._health_check_thread.start()
        logger.debug("Health check thread started")

    def _health_check_loop(self):
        """Background loop for health checks."""
        while self._monitoring_active:
            try:
                self.perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                if self.enable_debug_logging:
                    traceback.print_exc()
                time.sleep(self.health_check_interval)

    def perform_health_check(self) -> HealthCheckResult:
        """Perform comprehensive health check."""
        timestamp = datetime.now()
        checks = {}
        issues = []
        recommendations = []

        try:
            # Langfuse connectivity check
            langfuse_check = self._check_langfuse_connectivity()
            checks["langfuse_connectivity"] = langfuse_check

            if not langfuse_check["available"]:
                issues.append("Langfuse service is not available")
                recommendations.append(
                    "Check Langfuse credentials and network connectivity"
                )

            # Performance metrics check
            performance_check = self._check_performance_metrics()
            checks["performance"] = performance_check

            if performance_check["issues"]:
                issues.extend(performance_check["issues"])
                recommendations.extend(performance_check["recommendations"])

            # Error handling check
            error_check = self._check_error_handling()
            checks["error_handling"] = error_check

            if error_check["issues"]:
                issues.extend(error_check["issues"])
                recommendations.extend(error_check["recommendations"])

            # Metrics collection check
            metrics_check = self._check_metrics_collection()
            checks["metrics_collection"] = metrics_check

            if metrics_check["issues"]:
                issues.extend(metrics_check["issues"])
                recommendations.extend(metrics_check["recommendations"])

            # Alerting system check
            alerting_check = self._check_alerting_system()
            checks["alerting"] = alerting_check

            if alerting_check["issues"]:
                issues.extend(alerting_check["issues"])
                recommendations.extend(alerting_check["recommendations"])

            # Determine overall health status
            if len(issues) == 0:
                status = HealthStatus.HEALTHY
            elif len(issues) <= 2:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            self._last_health_check = timestamp

            if self.enable_debug_logging:
                self._log_debug_info(
                    "health_check",
                    {
                        "status": status.value,
                        "issues": issues,
                        "recommendations": recommendations,
                        "checks": checks,
                    },
                )

            logger.info(
                f"Health check completed: {status.value} with {len(issues)} issues"
            )

        except Exception as e:
            status = HealthStatus.UNKNOWN
            issues.append(f"Health check failed: {str(e)}")
            logger.error(f"Health check error: {e}")
            if self.enable_debug_logging:
                traceback.print_exc()

        return HealthCheckResult(
            status=status,
            timestamp=timestamp,
            checks=checks,
            issues=issues,
            recommendations=recommendations,
        )

    def _check_langfuse_connectivity(self) -> Dict[str, Any]:
        """Check Langfuse connectivity and basic functionality."""
        try:
            integration = get_langfuse_integration()
            health = integration.health_check()

            # Test basic operations
            can_create_trace = True
            try:
                test_trace_id = integration.create_simulation_trace(
                    {"test": "health_check"}
                )
                if test_trace_id:
                    integration.finalize_trace(test_trace_id)
                else:
                    can_create_trace = False
            except Exception as e:
                can_create_trace = False
                logger.debug(f"Trace creation test failed: {e}")

            return {
                "available": health.get("langfuse_available", False),
                "active_traces": health.get("active_traces", 0),
                "can_create_trace": can_create_trace,
                "integration_service": health.get("integration_service", "unknown"),
                "response_time": health.get("response_time", 0),
            }

        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "can_create_trace": False,
            }

    def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics and thresholds."""
        issues = []
        recommendations = []

        try:
            current = {}
            summary = {}
            if self.performance_monitor:
                summary = self.performance_monitor.get_performance_summary()
                current = summary.get("current", {})
            else:
                issues.append("Performance monitor not available")

            # Check key performance indicators
            cpu_usage = current.get("cpu_usage", 0)
            memory_usage = current.get("memory_usage", 0)
            langfuse_latency = current.get("langfuse_latency", 0)
            error_rate = current.get("error_rate", 0)
            queue_size = current.get("queue_size", 0)

            # Calculate trace success rate if available
            trace_success_rate = 95.0  # Default to healthy
            if "total_operations" in current and "error_count" in current:
                total_ops = current.get("total_operations", 0)
                error_count = current.get("error_count", 0)
                if total_ops > 0:
                    trace_success_rate = ((total_ops - error_count) / total_ops) * 100

            if cpu_usage > 80:
                issues.append(f"High CPU usage: {cpu_usage}%")
                recommendations.append(
                    "Consider reducing tracing frequency or optimizing system load"
                )

            if memory_usage > 85:
                issues.append(f"High memory usage: {memory_usage}%")
                recommendations.append(
                    "Consider increasing memory or reducing buffer sizes"
                )

            if langfuse_latency > 1000:  # 1 second
                issues.append(f"High Langfuse latency: {langfuse_latency}ms")
                recommendations.append(
                    "Check network connectivity or Langfuse service status"
                )

            if error_rate > 0.1:  # 10%
                issues.append(f"High error rate: {error_rate:.1%}")
                recommendations.append(
                    "Investigate error patterns and consider reducing system load"
                )

            if queue_size > 500:
                issues.append(f"Large trace queue: {queue_size} items")
                recommendations.append(
                    "Consider increasing flush frequency or reducing trace volume"
                )

            return {
                "issues": issues,
                "recommendations": recommendations,
                "metrics": current,
                "thresholds": summary.get("thresholds", {})
                if "summary" in locals()
                else {},
            }

        except Exception as e:
            return {
                "issues": [f"Performance check failed: {str(e)}"],
                "recommendations": ["Check performance monitor configuration"],
                "error": str(e),
            }

    def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling system status."""
        issues = []
        recommendations = []

        try:
            if not self.error_handler:
                issues.append("Error handler not initialized")
                recommendations.append("Initialize LangfuseErrorHandler")
                return {"issues": issues, "recommendations": recommendations}

            status = self.error_handler.get_status()

            # Check error metrics
            total_errors = status.get("total_errors", 0)
            consecutive_failures = status.get("consecutive_failures", 0)
            degradation_level = status.get("degradation_level", "normal")

            if total_errors > 100:
                issues.append(f"High error count: {total_errors}")
                recommendations.append(
                    "Review error patterns and consider system optimization"
                )

            if consecutive_failures > 5:
                issues.append(f"Consecutive failures: {consecutive_failures}")
                recommendations.append("Check Langfuse connectivity and configuration")

            if degradation_level != "normal":
                issues.append(f"System degraded: {degradation_level}")
                recommendations.append(
                    "Monitor system performance and consider scaling"
                )

            return {
                "issues": issues,
                "recommendations": recommendations,
                "status": status,
            }

        except Exception as e:
            return {
                "issues": [f"Error handling check failed: {str(e)}"],
                "recommendations": ["Check error handler configuration"],
                "error": str(e),
            }

    def _check_metrics_collection(self) -> Dict[str, Any]:
        """Check metrics collection system."""
        issues = []
        recommendations = []

        try:
            summary = self.metrics_collector.get_metrics_summary()

            # Check if metrics are being collected
            total_operations = summary.get("total_operations", 0)
            active_workflows = summary.get("active_workflows", 0)

            if (
                total_operations == 0
                and (datetime.now() - self._system_start_time).seconds > 300
            ):
                issues.append("No operations recorded in metrics")
                recommendations.append("Check if agents are generating metrics")

            if active_workflows > 50:
                issues.append(f"High number of active workflows: {active_workflows}")
                recommendations.append(
                    "Monitor system load and consider workflow optimization"
                )

            return {
                "issues": issues,
                "recommendations": recommendations,
                "summary": summary,
            }

        except Exception as e:
            return {
                "issues": [f"Metrics collection check failed: {str(e)}"],
                "recommendations": ["Check metrics collector configuration"],
                "error": str(e),
            }

    def _check_alerting_system(self) -> Dict[str, Any]:
        """Check alerting system status."""
        issues = []
        recommendations = []

        try:
            if self.alert_manager:
                dashboard_data = self.alert_manager.get_dashboard_alert_data()
                stats = dashboard_data.get("stats", {})
                active_alerts_count = len(dashboard_data.get("active_alerts", []))

                # Check for too many active alerts
                if stats.get("total_active", 0) > 10:
                    issues.append(
                        f"High number of active alerts: {stats['total_active']}"
                    )
                    recommendations.append("Review and resolve active alerts")
            else:
                issues.append("Alert manager not available")
                stats = {}
                active_alerts_count = 0

            # Check alert rule status
            enabled_rules = stats.get("enabled_rules", 0)
            total_rules = stats.get("total_rules", 0)

            if enabled_rules == 0:
                issues.append("No alert rules enabled")
                recommendations.append("Enable critical alert rules")

            return {
                "issues": issues,
                "recommendations": recommendations,
                "stats": stats,
                "active_alerts_count": active_alerts_count,
            }

        except Exception as e:
            return {
                "issues": [f"Alerting system check failed: {str(e)}"],
                "recommendations": ["Check alert manager configuration"],
                "error": str(e),
            }

    def get_monitoring_summary(self) -> MonitoringSummary:
        """Get comprehensive monitoring summary."""
        # Get latest health check
        health_result = self.perform_health_check()

        # Get system metrics
        system_metrics = {}
        try:
            integration = get_langfuse_integration()
            system_metrics = integration.get_system_metrics()
        except Exception as e:
            logger.debug(f"Failed to get system metrics: {e}")

        # Get performance metrics
        performance_metrics = {}
        try:
            performance_metrics = {}
            if self.performance_monitor:
                perf_summary = self.performance_monitor.get_performance_summary()
                performance_metrics = perf_summary.get("current", {})
        except Exception as e:
            logger.debug(f"Failed to get performance metrics: {e}")

        # Get recent alerts
        recent_alerts = []
        try:
            if self.alert_manager:
                dashboard_data = self.alert_manager.get_dashboard_alert_data()
                recent_alerts = dashboard_data.get("recent_alerts", [])
        except Exception as e:
            logger.debug(f"Failed to get recent alerts: {e}")

        # Calculate uptime
        uptime_seconds = (datetime.now() - self._system_start_time).total_seconds()

        active_alerts_count = 0
        if self.alert_manager:
            dashboard_data = self.alert_manager.get_dashboard_alert_data()
            active_alerts_count = len(dashboard_data.get("active_alerts", []))

        return MonitoringSummary(
            timestamp=datetime.now(),
            health_status=health_result.status,
            active_alerts=active_alerts_count,
            system_metrics=system_metrics,
            performance_metrics=performance_metrics,
            recent_alerts=recent_alerts,
            uptime_seconds=uptime_seconds,
        )

    def _log_debug_info(self, event_type: str, data: Dict[str, Any]):
        """Log debug information for troubleshooting."""
        if self._debug_log_file:
            try:
                debug_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "data": data,
                }
                self._debug_log_file.write(json.dumps(debug_entry) + "\n")
                self._debug_log_file.flush()
            except Exception as e:
                logger.error(f"Failed to write debug log: {e}")

    def add_custom_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.alert_manager.alerting_engine.add_rule(rule)
        logger.info(f"Added custom alert rule: {rule.name}")

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        summary = self.get_monitoring_summary()
        health_result = self.perform_health_check()

        # Get detailed component status
        components = {}

        try:
            if self.performance_monitor:
                components["performance_monitor"] = (
                    self.performance_monitor.get_performance_summary()
                )
            else:
                components["performance_monitor"] = {
                    "error": "Performance monitor not available"
                }
        except Exception as e:
            components["performance_monitor"] = {"error": str(e)}

        try:
            components["alert_manager"] = self.alert_manager.get_dashboard_alert_data()
        except Exception as e:
            components["alert_manager"] = {"error": str(e)}

        try:
            if self.error_handler:
                components["error_handler"] = self.error_handler.get_status()
        except Exception as e:
            components["error_handler"] = {"error": str(e)}

        try:
            components["metrics_collector"] = (
                self.metrics_collector.get_metrics_summary()
            )
        except Exception as e:
            components["metrics_collector"] = {"error": str(e)}

        return {
            "timestamp": summary.timestamp.isoformat(),
            "health_status": summary.health_status.value,
            "uptime_seconds": summary.uptime_seconds,
            "active_alerts": summary.active_alerts,
            "health_check": {
                "status": health_result.status.value,
                "issues": health_result.issues,
                "recommendations": health_result.recommendations,
                "last_check": self._last_health_check.isoformat()
                if self._last_health_check
                else None,
            },
            "components": components,
            "system_metrics": summary.system_metrics,
            "performance_metrics": summary.performance_metrics,
            "recent_alerts": summary.recent_alerts,
        }


# Global monitoring system instance
_monitoring_system: Optional[LangfuseMonitoringAlertingSystem] = None


def get_monitoring_system() -> LangfuseMonitoringAlertingSystem:
    """Get the global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = LangfuseMonitoringAlertingSystem()
    return _monitoring_system


def initialize_monitoring_system(
    performance_monitor: Optional[PerformanceMonitor] = None,
    alert_manager: Optional[LangfuseAlertManager] = None,
    error_handler: Optional[LangfuseErrorHandler] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    **kwargs,
) -> LangfuseMonitoringAlertingSystem:
    """Initialize the global monitoring system.

    Args:
        performance_monitor: Optional performance monitor instance
        alert_manager: Optional alert manager instance
        error_handler: Optional error handler instance
        metrics_collector: Optional metrics collector instance
        **kwargs: Additional configuration parameters

    Returns:
        Initialized LangfuseMonitoringAlertingSystem instance
    """
    global _monitoring_system
    _monitoring_system = LangfuseMonitoringAlertingSystem(
        performance_monitor=performance_monitor,
        alert_manager=alert_manager,
        error_handler=error_handler,
        metrics_collector=metrics_collector,
        **kwargs,
    )
    return _monitoring_system


def shutdown_monitoring_system():
    """Shutdown the global monitoring system."""
    global _monitoring_system
    if _monitoring_system:
        _monitoring_system.stop_monitoring()
        _monitoring_system = None

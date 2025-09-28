"""Langfuse alerting system for performance monitoring and error detection."""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    name: str
    description: str
    severity: AlertSeverity
    enabled: bool = True
    check_interval_seconds: int = 60
    threshold: float = 0.0
    window_minutes: int = 5
    condition: str = "greater_than"  # 'greater_than', 'less_than', 'equals'
    metric_name: str = ""
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 10


@dataclass
class Alert:
    """Represents an active alert."""

    id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metric_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AlertingEngine:
    """Engine for evaluating alert rules and generating alerts."""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules."""

        # Performance alert rules
        self.add_rule(
            AlertRule(
                name="high_latency",
                description="Agent response time exceeds threshold",
                severity=AlertSeverity.WARNING,
                enabled=True,
                check_interval_seconds=30,
                threshold=100.0,  # milliseconds
                window_minutes=5,
                condition="greater_than",
                metric_name="average_response_time",
                cooldown_minutes=10,
            )
        )

        self.add_rule(
            AlertRule(
                name="low_success_rate",
                description="Agent success rate below threshold",
                severity=AlertSeverity.ERROR,
                enabled=True,
                check_interval_seconds=60,
                threshold=95.0,  # percentage
                window_minutes=10,
                condition="less_than",
                metric_name="success_rate",
                cooldown_minutes=15,
            )
        )

        self.add_rule(
            AlertRule(
                name="high_error_rate",
                description="System error rate above threshold",
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                check_interval_seconds=30,
                threshold=5.0,  # percentage
                window_minutes=5,
                condition="greater_than",
                metric_name="error_rate",
                cooldown_minutes=5,
            )
        )

        # System health rules
        self.add_rule(
            AlertRule(
                name="langfuse_unavailable",
                description="Langfuse service unavailable",
                severity=AlertSeverity.CRITICAL,
                enabled=True,
                check_interval_seconds=30,
                threshold=1.0,
                condition="equals",
                metric_name="langfuse_health",
                cooldown_minutes=5,
            )
        )

        self.add_rule(
            AlertRule(
                name="high_memory_usage",
                description="System memory usage above threshold",
                severity=AlertSeverity.WARNING,
                enabled=True,
                check_interval_seconds=60,
                threshold=80.0,  # percentage
                condition="greater_than",
                metric_name="memory_usage",
                cooldown_minutes=10,
            )
        )

        # Workflow rules
        self.add_rule(
            AlertRule(
                name="long_workflow_duration",
                description="Workflow duration exceeds threshold",
                severity=AlertSeverity.WARNING,
                enabled=True,
                check_interval_seconds=60,
                threshold=30.0,  # seconds
                window_minutes=5,
                condition="greater_than",
                metric_name="average_workflow_duration",
                cooldown_minutes=10,
            )
        )

        self.add_rule(
            AlertRule(
                name="low_throughput",
                description="System throughput below expected levels",
                severity=AlertSeverity.WARNING,
                enabled=True,
                check_interval_seconds=60,
                threshold=1.0,  # events per second
                window_minutes=10,
                condition="less_than",
                metric_name="system_throughput",
                cooldown_minutes=15,
            )
        )

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        new_alerts = []
        current_time = datetime.now()

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # Check cooldown period
            if rule.last_triggered and current_time - rule.last_triggered < timedelta(
                minutes=rule.cooldown_minutes
            ):
                continue

            # Get metric value
            metric_value = self._get_metric_value(metrics, rule.metric_name)
            if metric_value is None:
                continue

            # Evaluate condition
            should_trigger = self._evaluate_condition(
                metric_value, rule.threshold, rule.condition
            )

            if should_trigger:
                alert = self._create_alert(rule, metric_value, current_time)
                new_alerts.append(alert)
                rule.last_triggered = current_time

                logger.warning(f"Alert triggered: {rule.name} - {alert.message}")

        return new_alerts

    def _get_metric_value(
        self, metrics: Dict[str, Any], metric_name: str
    ) -> Optional[float]:
        """Extract metric value from metrics dictionary."""
        # Handle nested metric paths like "agents.pricing_agent.average_response_time"
        keys = metric_name.split(".")
        value = metrics

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _evaluate_condition(
        self, value: float, threshold: float, condition: str
    ) -> bool:
        """Evaluate alert condition."""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001  # Handle floating point comparison
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        return False

    def _create_alert(
        self, rule: AlertRule, metric_value: float, timestamp: datetime
    ) -> Alert:
        """Create an alert from a triggered rule."""
        alert_id = f"{rule.name}_{int(timestamp.timestamp())}"

        if rule.name == "high_latency":
            title = "High Agent Latency Detected"
            message = f"Agent response time ({metric_value:.1f}ms) exceeds threshold ({rule.threshold}ms)"
        elif rule.name == "low_success_rate":
            title = "Low Success Rate Detected"
            message = f"Agent success rate ({metric_value:.1f}%) below threshold ({rule.threshold}%)"
        elif rule.name == "high_error_rate":
            title = "High Error Rate Detected"
            message = f"System error rate ({metric_value:.1f}%) above threshold ({rule.threshold}%)"
        elif rule.name == "langfuse_unavailable":
            title = "Langfuse Service Unavailable"
            message = "Langfuse integration service is not responding"
        elif rule.name == "high_memory_usage":
            title = "High Memory Usage"
            message = f"System memory usage ({metric_value:.1f}%) above threshold ({rule.threshold}%)"
        elif rule.name == "long_workflow_duration":
            title = "Long Workflow Duration"
            message = f"Workflow duration ({metric_value:.1f}s) exceeds threshold ({rule.threshold}s)"
        elif rule.name == "low_throughput":
            title = "Low System Throughput"
            message = f"System throughput ({metric_value:.2f} events/sec) below threshold ({rule.threshold})"
        else:
            title = f"Alert: {rule.name}"
            message = f"Metric {rule.metric_name} triggered alert condition"

        return Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            title=title,
            message=message,
            timestamp=timestamp,
            metric_value=metric_value,
            threshold=rule.threshold,
            metadata={
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "check_interval": rule.check_interval_seconds,
            },
        )

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            # Move to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history_size:
                self.alert_history.pop(0)

            # Remove from active alerts
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:] if self.alert_history else []

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [
            alert for alert in self.active_alerts.values() if alert.severity == severity
        ]

    def update_alerts(self, new_alerts: List[Alert]) -> None:
        """Update active alerts with new alerts."""
        for alert in new_alerts:
            self.active_alerts[alert.id] = alert

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = self.get_active_alerts()
        severity_counts = {
            severity.value: len(self.get_alerts_by_severity(severity))
            for severity in AlertSeverity
        }

        return {
            "total_active": len(active_alerts),
            "by_severity": severity_counts,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "last_updated": datetime.now().isoformat(),
        }


class LangfuseAlertManager:
    """Manages Langfuse-specific alerting and integration."""

    def __init__(self, alerting_engine: Optional[AlertingEngine] = None):
        self.alerting_engine = alerting_engine or AlertingEngine()
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)

    def check_and_trigger_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check metrics and trigger alerts if conditions are met."""
        new_alerts = self.alerting_engine.evaluate_rules(metrics)

        # Update active alerts
        self.alerting_engine.update_alerts(new_alerts)

        # Call callbacks for new alerts
        for alert in new_alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

        return new_alerts

    def get_dashboard_alert_data(self) -> Dict[str, Any]:
        """Get alert data formatted for dashboard display."""
        active_alerts = self.alerting_engine.get_active_alerts()
        alert_history = self.alerting_engine.get_alert_history(50)
        stats = self.alerting_engine.get_alert_stats()

        return {
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                }
                for alert in active_alerts
            ],
            "recent_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat()
                    if alert.resolved_at
                    else None,
                }
                for alert in alert_history
            ],
            "stats": stats,
        }


# Global alert manager instance
_alert_manager: Optional[LangfuseAlertManager] = None


def get_alert_manager() -> LangfuseAlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = LangfuseAlertManager()
    return _alert_manager


def initialize_alert_manager() -> LangfuseAlertManager:
    """Initialize the global alert manager."""
    global _alert_manager
    _alert_manager = LangfuseAlertManager()
    return _alert_manager

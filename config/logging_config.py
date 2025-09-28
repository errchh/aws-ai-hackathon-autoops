"""
Comprehensive logging and monitoring configuration for the retail optimization system.

This module provides structured logging, metrics collection, and monitoring
capabilities for system resilience and error tracking.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from agents.error_handling import ErrorContext, ErrorSeverity


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, default=str)


class ErrorContextFormatter(logging.Formatter):
    """Specialized formatter for ErrorContext objects."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format ErrorContext as structured log entry."""
        if hasattr(record, 'error_context') and isinstance(record.error_context, ErrorContext):
            error_context = record.error_context
            log_entry = {
                "timestamp": error_context.timestamp.isoformat(),
                "level": "ERROR",
                "error_id": error_context.error_id,
                "component": error_context.component,
                "error_type": error_context.error_type,
                "severity": error_context.severity.value,
                "message": error_context.message,
                "retry_count": error_context.retry_count,
                "resolved": error_context.resolved,
                "metadata": error_context.metadata
            }
            
            if error_context.stack_trace:
                log_entry["stack_trace"] = error_context.stack_trace
            
            return json.dumps(log_entry, default=str)
        
        return super().format(record)


class SystemMetricsLogger:
    """Logger for system metrics and performance data."""
    
    def __init__(self, log_file: str = "logs/metrics.log"):
        self.logger = logging.getLogger("system_metrics")
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler for metrics
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(StructuredFormatter())
        
        self.logger.addHandler(file_handler)
    
    def log_agent_performance(self, agent_id: str, operation: str, 
                            duration: float, success: bool, **kwargs) -> None:
        """Log agent performance metrics."""
        metrics = {
            "metric_type": "agent_performance",
            "agent_id": agent_id,
            "operation": operation,
            "duration_seconds": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.info("Agent performance metric", extra={"extra_fields": metrics})
    
    def log_system_health(self, health_data: Dict[str, Any]) -> None:
        """Log system health metrics."""
        metrics = {
            "metric_type": "system_health",
            "timestamp": datetime.now().isoformat(),
            **health_data
        }
        
        self.logger.info("System health metric", extra={"extra_fields": metrics})
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       duration: float, **kwargs) -> None:
        """Log API request metrics."""
        metrics = {
            "metric_type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.info("API request metric", extra={"extra_fields": metrics})


class ErrorLogger:
    """Specialized logger for error tracking and analysis."""
    
    def __init__(self, log_file: str = "logs/errors.log"):
        self.logger = logging.getLogger("error_tracking")
        self.logger.setLevel(logging.ERROR)
        
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler for errors
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(ErrorContextFormatter())
        
        self.logger.addHandler(file_handler)
    
    def log_error_context(self, error_context: ErrorContext) -> None:
        """Log an ErrorContext object."""
        self.logger.error(
            f"Error in {error_context.component}: {error_context.message}",
            extra={"error_context": error_context}
        )
    
    def log_circuit_breaker_event(self, component: str, event: str, **kwargs) -> None:
        """Log circuit breaker state changes."""
        event_data = {
            "event_type": "circuit_breaker",
            "component": component,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        self.logger.warning(
            f"Circuit breaker {event} for {component}",
            extra={"extra_fields": event_data}
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = True
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration for the system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Use structured JSON formatting
    
    Returns:
        Dictionary of configured loggers
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler for general logs
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "application.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create specialized loggers
    loggers = {
        "root": root_logger,
        "metrics": SystemMetricsLogger(str(log_path / "metrics.log")),
        "errors": ErrorLogger(str(log_path / "errors.log"))
    }
    
    # Agent-specific loggers
    for agent_name in ["pricing_agent", "inventory_agent", "promotion_agent", "orchestrator"]:
        agent_logger = logging.getLogger(agent_name)
        agent_logger.setLevel(getattr(logging, log_level.upper()))
        
        if enable_file:
            agent_handler = logging.handlers.RotatingFileHandler(
                log_path / f"{agent_name}.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            agent_handler.setFormatter(formatter)
            agent_logger.addHandler(agent_handler)
        
        loggers[agent_name] = agent_logger
    
    return loggers


class MonitoringIntegration:
    """Integration with monitoring systems and alerting."""
    
    def __init__(self):
        self.metrics_logger = SystemMetricsLogger()
        self.error_logger = ErrorLogger()
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate
            "response_time": 5.0,  # 5 seconds
            "health_percentage": 80.0  # 80% system health
        }
    
    def check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []
        
        # Check error rate
        if "error_rate" in metrics and metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate",
                "severity": "high",
                "message": f"Error rate {metrics['error_rate']:.2%} exceeds threshold",
                "threshold": self.alert_thresholds["error_rate"],
                "current_value": metrics["error_rate"]
            })
        
        # Check response time
        if "avg_response_time" in metrics and metrics["avg_response_time"] > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "response_time",
                "severity": "medium",
                "message": f"Average response time {metrics['avg_response_time']:.2f}s exceeds threshold",
                "threshold": self.alert_thresholds["response_time"],
                "current_value": metrics["avg_response_time"]
            })
        
        # Check system health
        if "health_percentage" in metrics and metrics["health_percentage"] < self.alert_thresholds["health_percentage"]:
            alerts.append({
                "type": "system_health",
                "severity": "critical" if metrics["health_percentage"] < 50 else "high",
                "message": f"System health {metrics['health_percentage']:.1f}% below threshold",
                "threshold": self.alert_thresholds["health_percentage"],
                "current_value": metrics["health_percentage"]
            })
        
        return alerts
    
    def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert notification (placeholder for actual implementation)."""
        # In a real implementation, this would integrate with:
        # - Email notifications
        # - Slack/Teams webhooks
        # - PagerDuty/OpsGenie
        # - AWS CloudWatch Alarms
        # - Prometheus AlertManager
        
        logger = logging.getLogger("alerts")
        logger.critical(f"ALERT: {alert['message']}", extra={"extra_fields": alert})
        
        print(f"🚨 ALERT [{alert['severity'].upper()}]: {alert['message']}")


# Global instances
metrics_logger = SystemMetricsLogger()
error_logger = ErrorLogger()
monitoring = MonitoringIntegration()


def log_performance_metric(agent_id: str, operation: str, duration: float, 
                          success: bool, **kwargs) -> None:
    """Convenience function for logging performance metrics."""
    metrics_logger.log_agent_performance(agent_id, operation, duration, success, **kwargs)


def log_error_context(error_context: ErrorContext) -> None:
    """Convenience function for logging error contexts."""
    error_logger.log_error_context(error_context)


def log_circuit_breaker_event(component: str, event: str, **kwargs) -> None:
    """Convenience function for logging circuit breaker events."""
    error_logger.log_circuit_breaker_event(component, event, **kwargs)


# Initialize logging on module import
if __name__ != "__main__":
    setup_logging()
"""Performance monitoring and automatic degradation for Langfuse integration."""

import logging
import psutil
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import threading
import os

from .langfuse_error_handler import LangfuseErrorHandler, DegradationLevel

try:
    from .langfuse_sampling import LangfuseSampler
except ImportError:
    LangfuseSampler = None

try:
    from .langfuse_buffer import LangfuseBuffer
except ImportError:
    LangfuseBuffer = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    langfuse_latency: float = 0.0
    trace_throughput: float = 0.0
    error_rate: float = 0.0
    queue_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceThresholds:
    """Thresholds for triggering performance actions."""

    max_cpu_usage: float = 80.0  # Percentage
    max_memory_usage: float = 85.0  # Percentage
    max_disk_usage: float = 90.0  # Percentage
    max_langfuse_latency: float = 1000.0  # Milliseconds
    min_trace_throughput: float = 10.0  # Traces per second
    max_error_rate: float = 0.1  # 10% error rate
    max_queue_size: int = 1000


class PerformanceMonitor:
    """Monitors system performance and triggers automatic degradation."""

    def __init__(
        self,
        error_handler: LangfuseErrorHandler,
        sampler: Optional[Any] = None,
        buffer: Optional[Any] = None,
        thresholds: Optional[PerformanceThresholds] = None,
        monitoring_interval: int = 10,
        history_size: int = 100,
    ):
        """Initialize the performance monitor.

        Args:
            error_handler: Error handler instance
            sampler: Optional sampler instance
            buffer: Optional buffer instance
            thresholds: Performance thresholds
            monitoring_interval: Seconds between monitoring checks
            history_size: Number of historical samples to keep
        """
        self.error_handler = error_handler
        self.sampler = sampler
        self.buffer = buffer
        self.thresholds = thresholds or PerformanceThresholds()
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size

        # Performance history
        self.performance_history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()

        # Control flags
        self._shutdown = False
        self._monitoring_thread: Optional[threading.Thread] = None

        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Start monitoring
        self._start_monitoring()

        logger.info("PerformanceMonitor initialized")

    def _start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="LangfusePerformanceMonitor"
        )
        self._monitoring_thread.start()
        logger.debug("Performance monitoring thread started")

    def _monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        while not self._shutdown:
            try:
                metrics = self._collect_metrics()
                self._analyze_performance(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system and Langfuse performance metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        disk = psutil.disk_usage("/")
        disk_usage = disk.percent

        # Network I/O (approximate)
        network_io = self._get_network_io()

        # Langfuse-specific metrics
        langfuse_latency = self._get_langfuse_latency()
        trace_throughput = self._get_trace_throughput()
        error_rate = self._get_error_rate()
        queue_size = self._get_queue_size()

        metrics = PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            langfuse_latency=langfuse_latency,
            trace_throughput=trace_throughput,
            error_rate=error_rate,
            queue_size=queue_size,
        )

        # Store in history
        with self._lock:
            self.performance_history.append(metrics)

        return metrics

    def _get_network_io(self) -> Dict[str, float]:
        """Get network I/O statistics."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except Exception:
            return {}

    def _get_langfuse_latency(self) -> float:
        """Get average Langfuse operation latency."""
        # This would integrate with the error handler's performance tracking
        if hasattr(self.error_handler, "error_metrics"):
            return (
                self.error_handler.error_metrics.average_response_time * 1000
            )  # Convert to ms
        return 0.0

    def _get_trace_throughput(self) -> float:
        """Get current trace throughput."""
        # This would integrate with the sampler's metrics
        if self.sampler and hasattr(self.sampler, "metrics"):
            total = self.sampler.metrics.total_traces
            if total > 0:
                # Calculate traces per second over the last minute
                cutoff = datetime.now() - timedelta(minutes=1)
                recent_traces = sum(
                    1 for m in self.performance_history if m.timestamp > cutoff
                )
                return recent_traces / 60.0
        return 0.0

    def _get_error_rate(self) -> float:
        """Get current error rate."""
        if hasattr(self.error_handler, "error_metrics"):
            total = self.error_handler.error_metrics.total_errors
            if total > 0:
                # Simple error rate calculation
                return min(1.0, total / max(1, self._get_total_operations()))
        return 0.0

    def _get_queue_size(self) -> int:
        """Get current queue/buffer size."""
        if self.buffer:
            stats = self.buffer.get_buffer_stats()
            return stats.get("memory_buffer_size", 0) + stats.get(
                "database_buffer_size", 0
            )
        return 0

    def _get_total_operations(self) -> int:
        """Get total number of operations for error rate calculation."""
        # This would need to be tracked separately
        return 1000  # Placeholder

    def _analyze_performance(self, metrics: PerformanceMetrics) -> None:
        """Analyze performance metrics and trigger actions if needed."""
        issues = []
        actions = []

        # Check CPU usage
        if metrics.cpu_usage > self.thresholds.max_cpu_usage:
            issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            actions.append("Consider reducing sampling rate")

        # Check memory usage
        if metrics.memory_usage > self.thresholds.max_memory_usage:
            issues.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            actions.append("Consider reducing buffer size or sampling rate")

        # Check disk usage
        if metrics.disk_usage > self.thresholds.max_disk_usage:
            issues.append(f"High disk usage: {metrics.disk_usage:.1f}%")
            actions.append("Consider cleaning up old traces")

        # Check Langfuse latency
        if metrics.langfuse_latency > self.thresholds.max_langfuse_latency:
            issues.append(f"High Langfuse latency: {metrics.langfuse_latency:.1f}ms")
            actions.append("Consider enabling buffering or reducing sampling")

        # Check throughput
        if metrics.trace_throughput < self.thresholds.min_trace_throughput:
            issues.append(
                f"Low trace throughput: {metrics.trace_throughput:.1f} traces/sec"
            )
            actions.append("System may be overloaded")

        # Check error rate
        if metrics.error_rate > self.thresholds.max_error_rate:
            issues.append(f"High error rate: {metrics.error_rate:.1%}")
            actions.append("Consider reducing system load")

        # Check queue size
        if metrics.queue_size > self.thresholds.max_queue_size:
            issues.append(f"Large queue size: {metrics.queue_size} traces")
            actions.append("Consider increasing flush frequency")

        # Trigger actions if issues found
        if issues:
            self._trigger_performance_actions(issues, actions, metrics)

        # Log performance summary
        if issues:
            logger.warning(f"Performance issues detected: {', '.join(issues)}")

    def _trigger_performance_actions(
        self, issues: List[str], actions: List[str], metrics: PerformanceMetrics
    ) -> None:
        """Trigger automatic performance actions based on issues."""
        # Determine degradation level based on severity
        degradation_level = self._determine_degradation_level(metrics)

        # Apply degradation if needed
        if degradation_level != self.error_handler.error_metrics.degradation_level:
            logger.warning(
                f"Triggering performance degradation to {degradation_level.value}"
            )
            self.error_handler._set_degradation_level(degradation_level)

        # Adjust sampling rate if sampler is available
        if self.sampler:
            self._adjust_sampling_for_performance(metrics)

        # Trigger alerts
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "actions": actions,
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "langfuse_latency": metrics.langfuse_latency,
                "error_rate": metrics.error_rate,
                "queue_size": metrics.queue_size,
            },
        }

        for callback in self.alert_callbacks:
            try:
                callback("performance_issue", alert_data)
            except Exception as e:
                logger.error(f"Error in performance alert callback: {e}")

    def _determine_degradation_level(
        self, metrics: PerformanceMetrics
    ) -> DegradationLevel:
        """Determine the appropriate degradation level based on metrics."""
        # Critical issues - disable tracing
        if (
            metrics.cpu_usage > 95.0
            or metrics.memory_usage > 95.0
            or metrics.error_rate > 0.5
        ):
            return DegradationLevel.DISABLED

        # High issues - buffering only
        elif (
            metrics.cpu_usage > 90.0
            or metrics.memory_usage > 90.0
            or metrics.langfuse_latency > 5000
            or metrics.error_rate > 0.3
        ):
            return DegradationLevel.BUFFERING_ONLY

        # Medium issues - reduced sampling
        elif (
            metrics.cpu_usage > self.thresholds.max_cpu_usage
            or metrics.memory_usage > self.thresholds.max_memory_usage
            or metrics.langfuse_latency > self.thresholds.max_langfuse_latency
            or metrics.error_rate > self.thresholds.max_error_rate
        ):
            return DegradationLevel.REDUCED_SAMPLING

        # Normal operation
        else:
            return DegradationLevel.NORMAL

    def _adjust_sampling_for_performance(self, metrics: PerformanceMetrics) -> None:
        """Adjust sampling rate based on performance metrics."""
        if not self.sampler:
            return

        # Calculate performance score (0-1, where 1 is best performance)
        performance_score = 1.0

        # Penalize for high resource usage
        if metrics.cpu_usage > 50:
            performance_score *= max(0.1, 1.0 - (metrics.cpu_usage - 50) / 50)
        if metrics.memory_usage > 60:
            performance_score *= max(0.1, 1.0 - (metrics.memory_usage - 60) / 40)

        # Penalize for high latency
        if metrics.langfuse_latency > 100:
            performance_score *= max(0.1, 1.0 - (metrics.langfuse_latency - 100) / 900)

        # Penalize for high error rate
        if metrics.error_rate > 0.05:
            performance_score *= max(0.1, 1.0 - metrics.error_rate * 10)

        # Adjust sampling rate based on performance score
        current_rate = self.sampler.metrics.current_sample_rate
        target_rate = current_rate * performance_score

        # Apply smoothing to avoid rapid changes
        smoothed_rate = current_rate * 0.7 + target_rate * 0.3
        self.sampler.update_sample_rate(smoothed_rate)

        logger.debug(
            f"Adjusted sampling rate to {smoothed_rate:.3f} based on performance score {performance_score:.3f}"
        )

    def add_alert_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Add a callback for performance alerts.

        Args:
            callback: Function to call with (alert_type, alert_data)
        """
        self.alert_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current and historical performance.

        Returns:
            Dictionary containing performance summary
        """
        with self._lock:
            if not self.performance_history:
                return {"message": "No performance data available"}

            latest = self.performance_history[-1]
            history = list(self.performance_history)

        return {
            "current": {
                "timestamp": latest.timestamp.isoformat(),
                "cpu_usage": round(latest.cpu_usage, 2),
                "memory_usage": round(latest.memory_usage, 2),
                "disk_usage": round(latest.disk_usage, 2),
                "langfuse_latency": round(latest.langfuse_latency, 2),
                "trace_throughput": round(latest.trace_throughput, 2),
                "error_rate": round(latest.error_rate, 4),
                "queue_size": latest.queue_size,
            },
            "averages": self._calculate_averages(history),
            "thresholds": {
                "max_cpu_usage": self.thresholds.max_cpu_usage,
                "max_memory_usage": self.thresholds.max_memory_usage,
                "max_langfuse_latency": self.thresholds.max_langfuse_latency,
                "max_error_rate": self.thresholds.max_error_rate,
            },
            "history_size": len(history),
        }

    def _calculate_averages(
        self, history: List[PerformanceMetrics]
    ) -> Dict[str, float]:
        """Calculate averages from performance history."""
        if not history:
            return {}

        return {
            "avg_cpu_usage": round(sum(m.cpu_usage for m in history) / len(history), 2),
            "avg_memory_usage": round(
                sum(m.memory_usage for m in history) / len(history), 2
            ),
            "avg_langfuse_latency": round(
                sum(m.langfuse_latency for m in history) / len(history), 2
            ),
            "avg_trace_throughput": round(
                sum(m.trace_throughput for m in history) / len(history), 2
            ),
            "avg_error_rate": round(
                sum(m.error_rate for m in history) / len(history), 4
            ),
        }

    def reset_history(self) -> None:
        """Reset the performance history."""
        with self._lock:
            self.performance_history.clear()
            logger.info("Performance history reset")

    def shutdown(self) -> None:
        """Shutdown the performance monitor."""
        self._shutdown = True

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("PerformanceMonitor shutdown completed")


# Global performance monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_langfuse_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _monitor


def initialize_langfuse_performance_monitor(
    error_handler: LangfuseErrorHandler,
    sampler: Optional[Any] = None,
    buffer: Optional[Any] = None,
    **kwargs,
) -> PerformanceMonitor:
    """Initialize the global performance monitor.

    Args:
        error_handler: Error handler instance
        sampler: Optional sampler instance
        buffer: Optional buffer instance
        **kwargs: Additional configuration parameters

    Returns:
        Initialized PerformanceMonitor instance
    """
    global _monitor
    _monitor = PerformanceMonitor(
        error_handler=error_handler, sampler=sampler, buffer=buffer, **kwargs
    )
    return _monitor


def shutdown_langfuse_performance_monitor() -> None:
    """Shutdown the global performance monitor."""
    global _monitor
    if _monitor:
        _monitor.shutdown()
        _monitor = None

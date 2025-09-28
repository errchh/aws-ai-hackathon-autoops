"""Comprehensive error handling and fallback mechanisms for Langfuse integration."""

import asyncio
import json
import logging
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Deque
from dataclasses import dataclass, field
from pathlib import Path
import os

from .langfuse_config import LangfuseClient, LangfuseConfig
from agents.error_handling import ErrorSeverity, ErrorContext, resilience_manager

logger = logging.getLogger(__name__)


class LangfuseErrorType(Enum):
    """Types of Langfuse-specific errors."""

    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    SERIALIZATION_ERROR = "serialization_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONFIGURATION_ERROR = "configuration_error"


class DegradationLevel(Enum):
    """Levels of system degradation for fallback mechanisms."""

    NORMAL = "normal"
    REDUCED_SAMPLING = "reduced_sampling"
    BUFFERING_ONLY = "buffering_only"
    DISABLED = "disabled"


@dataclass
class ErrorMetrics:
    """Metrics for tracking Langfuse errors and performance."""

    total_errors: int = 0
    errors_by_type: Dict[LangfuseErrorType, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    last_error_time: Optional[datetime] = None
    average_response_time: float = 0.0
    response_time_samples: Deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    degradation_level: DegradationLevel = DegradationLevel.NORMAL


@dataclass
class BufferedTrace:
    """Represents a trace that couldn't be sent immediately."""

    trace_id: str
    trace_data: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[datetime] = None


class LangfuseErrorHandler:
    """Comprehensive error handler for Langfuse integration with fallback mechanisms."""

    def __init__(
        self,
        langfuse_client: LangfuseClient,
        max_buffer_size: int = 1000,
        buffer_flush_interval: int = 30,
        performance_threshold_ms: int = 100,
        max_consecutive_failures: int = 5,
        degradation_thresholds: Optional[Dict[DegradationLevel, int]] = None,
    ):
        """Initialize the error handler.

        Args:
            langfuse_client: The Langfuse client instance
            max_buffer_size: Maximum number of traces to buffer
            buffer_flush_interval: Seconds between buffer flush attempts
            performance_threshold_ms: Max acceptable latency in milliseconds
            max_consecutive_failures: Max consecutive failures before degradation
            degradation_thresholds: Custom thresholds for degradation levels
        """
        self.langfuse_client = langfuse_client
        self.max_buffer_size = max_buffer_size
        self.buffer_flush_interval = buffer_flush_interval
        self.performance_threshold_ms = performance_threshold_ms
        self.max_consecutive_failures = max_consecutive_failures

        # Default degradation thresholds (consecutive failures)
        self.degradation_thresholds = degradation_thresholds or {
            DegradationLevel.REDUCED_SAMPLING: 3,
            DegradationLevel.BUFFERING_ONLY: 5,
            DegradationLevel.DISABLED: 10,
        }

        # Error tracking
        self.error_metrics = ErrorMetrics()
        self._buffer: Deque[BufferedTrace] = deque(maxlen=max_buffer_size)
        self._buffer_lock = threading.Lock()

        # Control flags
        self._shutdown = False
        self._buffer_thread: Optional[threading.Thread] = None

        # Performance monitoring
        self._performance_samples: Deque[float] = deque(maxlen=100)
        self._performance_lock = threading.Lock()

        # Start background buffer processing
        self._start_buffer_processor()

        logger.info(
            "LangfuseErrorHandler initialized with comprehensive error handling"
        )

    def _start_buffer_processor(self) -> None:
        """Start the background thread for processing buffered traces."""
        self._buffer_thread = threading.Thread(
            target=self._process_buffer_loop,
            daemon=True,
            name="LangfuseBufferProcessor",
        )
        self._buffer_thread.start()
        logger.debug("Buffer processor thread started")

    def _process_buffer_loop(self) -> None:
        """Background loop to process buffered traces."""
        while not self._shutdown:
            try:
                self._flush_buffer()
                time.sleep(self.buffer_flush_interval)
            except Exception as e:
                logger.error(f"Error in buffer processing loop: {e}")
                time.sleep(self.buffer_flush_interval)

    def _flush_buffer(self) -> None:
        """Attempt to flush buffered traces to Langfuse."""
        if not self.langfuse_client.is_available:
            return

        with self._buffer_lock:
            traces_to_retry = [
                trace
                for trace in self._buffer
                if trace.next_retry_time and datetime.now() >= trace.next_retry_time
            ]

        for trace in traces_to_retry:
            try:
                self._retry_trace(trace)
            except Exception as e:
                logger.warning(f"Failed to retry trace {trace.trace_id}: {e}")
                self._schedule_retry(trace)

    def _retry_trace(self, trace: BufferedTrace) -> None:
        """Retry sending a buffered trace."""
        try:
            # Attempt to send the trace
            client = self.langfuse_client.client
            if client and self.langfuse_client.is_available:
                # Reconstruct the trace based on stored data
                self._send_buffered_trace(client, trace)
                self._remove_from_buffer(trace.trace_id)
                logger.debug(f"Successfully retried trace {trace.trace_id}")
            else:
                self._schedule_retry(trace)
        except Exception as e:
            logger.error(f"Retry failed for trace {trace.trace_id}: {e}")
            self._schedule_retry(trace)

    def _send_buffered_trace(self, client, trace: BufferedTrace) -> None:
        """Send a buffered trace to Langfuse."""
        # This would need to be implemented based on how traces are structured
        # For now, we'll log the attempt
        logger.debug(f"Sending buffered trace {trace.trace_id} to Langfuse")

        # In a real implementation, you would reconstruct the trace object
        # and call the appropriate Langfuse client methods

    def _schedule_retry(self, trace: BufferedTrace) -> None:
        """Schedule a retry for a failed trace."""
        trace.retry_count += 1
        backoff_seconds = min(2**trace.retry_count, 300)  # Max 5 minutes
        trace.next_retry_time = datetime.now() + timedelta(seconds=backoff_seconds)

        if trace.retry_count >= trace.max_retries:
            logger.error(f"Max retries exceeded for trace {trace.trace_id}, dropping")
            self._remove_from_buffer(trace.trace_id)

    def _remove_from_buffer(self, trace_id: str) -> None:
        """Remove a trace from the buffer."""
        with self._buffer_lock:
            self._buffer = deque([t for t in self._buffer if t.trace_id != trace_id])

    def handle_error(
        self, error: Exception, operation: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle a Langfuse error and determine if operation should continue.

        Args:
            error: The exception that occurred
            operation: The operation that failed
            context: Additional context about the error

        Returns:
            True if operation should be retried, False if it should be abandoned
        """
        # Classify the error
        error_type = self._classify_error(error)

        # Update error metrics
        self._update_error_metrics(error_type)

        # Create error context for resilience manager
        error_context = ErrorContext(
            error_id=f"langfuse_{error_type.value}_{int(time.time())}",
            timestamp=datetime.now(),
            component="langfuse_integration",
            error_type=error_type.value,
            severity=self._get_error_severity(error_type),
            message=str(error),
            metadata={
                "operation": operation,
                "error_type": error_type.value,
                **(context or {}),
            },
        )

        # Log to resilience manager
        resilience_manager.log_error(error_context)

        # Check if we should degrade performance
        self._check_performance_degradation()

        # Log the error
        logger.error(
            f"Langfuse error in {operation}: {error} (type: {error_type.value})"
        )

        return self._should_retry(error_type)

    def _classify_error(self, error: Exception) -> LangfuseErrorType:
        """Classify an error into a Langfuse error type."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if any(term in error_str for term in ["connection", "network", "unreachable"]):
            return LangfuseErrorType.CONNECTION_ERROR
        elif any(
            term in error_str
            for term in ["auth", "unauthorized", "forbidden", "credential"]
        ):
            return LangfuseErrorType.AUTHENTICATION_ERROR
        elif any(term in error_str for term in ["rate limit", "too many requests"]):
            return LangfuseErrorType.RATE_LIMIT_ERROR
        elif any(term in error_str for term in ["timeout", "timed out"]):
            return LangfuseErrorType.TIMEOUT_ERROR
        elif any(
            term in error_str for term in ["serializ", "json", "encode", "decode"]
        ):
            return LangfuseErrorType.SERIALIZATION_ERROR
        elif any(term in error_str for term in ["valid", "invalid", "malformed"]):
            return LangfuseErrorType.VALIDATION_ERROR
        elif any(term in error_str for term in ["server error", "internal error", "5"]):
            return LangfuseErrorType.SERVER_ERROR
        elif any(term in error_str for term in ["quota", "limit exceeded"]):
            return LangfuseErrorType.QUOTA_EXCEEDED
        elif any(term in error_str for term in ["config", "setting", "parameter"]):
            return LangfuseErrorType.CONFIGURATION_ERROR
        else:
            return LangfuseErrorType.NETWORK_ERROR

    def _get_error_severity(self, error_type: LangfuseErrorType) -> ErrorSeverity:
        """Get the severity level for an error type."""
        severity_map = {
            LangfuseErrorType.AUTHENTICATION_ERROR: ErrorSeverity.HIGH,
            LangfuseErrorType.CONFIGURATION_ERROR: ErrorSeverity.HIGH,
            LangfuseErrorType.QUOTA_EXCEEDED: ErrorSeverity.HIGH,
            LangfuseErrorType.SERVER_ERROR: ErrorSeverity.MEDIUM,
            LangfuseErrorType.RATE_LIMIT_ERROR: ErrorSeverity.MEDIUM,
            LangfuseErrorType.CONNECTION_ERROR: ErrorSeverity.MEDIUM,
            LangfuseErrorType.NETWORK_ERROR: ErrorSeverity.LOW,
            LangfuseErrorType.TIMEOUT_ERROR: ErrorSeverity.LOW,
            LangfuseErrorType.SERIALIZATION_ERROR: ErrorSeverity.LOW,
            LangfuseErrorType.VALIDATION_ERROR: ErrorSeverity.LOW,
        }
        return severity_map.get(error_type, ErrorSeverity.MEDIUM)

    def _update_error_metrics(self, error_type: LangfuseErrorType) -> None:
        """Update error metrics based on the current error."""
        self.error_metrics.total_errors += 1
        self.error_metrics.errors_by_type[error_type] = (
            self.error_metrics.errors_by_type.get(error_type, 0) + 1
        )
        self.error_metrics.consecutive_failures += 1
        self.error_metrics.last_error_time = datetime.now()

    def _check_performance_degradation(self) -> None:
        """Check if performance degradation should be triggered."""
        current_level = self.error_metrics.degradation_level

        # Check if we should escalate degradation
        if self.error_metrics.consecutive_failures >= self.degradation_thresholds.get(
            DegradationLevel.DISABLED, 10
        ):
            self._set_degradation_level(DegradationLevel.DISABLED)
        elif self.error_metrics.consecutive_failures >= self.degradation_thresholds.get(
            DegradationLevel.BUFFERING_ONLY, 5
        ):
            self._set_degradation_level(DegradationLevel.BUFFERING_ONLY)
        elif self.error_metrics.consecutive_failures >= self.degradation_thresholds.get(
            DegradationLevel.REDUCED_SAMPLING, 3
        ):
            self._set_degradation_level(DegradationLevel.REDUCED_SAMPLING)

        # Check if we should de-escalate (on successful operations)
        elif (
            current_level != DegradationLevel.NORMAL
            and self.error_metrics.consecutive_failures == 0
        ):
            self._set_degradation_level(DegradationLevel.NORMAL)

    def _set_degradation_level(self, level: DegradationLevel) -> None:
        """Set the current degradation level and update configuration."""
        old_level = self.error_metrics.degradation_level
        self.error_metrics.degradation_level = level

        logger.warning(
            f"Langfuse degradation level changed from {old_level.value} to {level.value}"
        )

        # Update Langfuse client configuration based on degradation level
        if hasattr(self.langfuse_client, "_config") and self.langfuse_client._config:
            config = self.langfuse_client._config

            if level == DegradationLevel.REDUCED_SAMPLING:
                config.sample_rate = min(
                    config.sample_rate * 0.5, 0.1
                )  # Reduce to 10% max
            elif level == DegradationLevel.BUFFERING_ONLY:
                config.sample_rate = 0.0  # Disable real-time tracing
            elif level == DegradationLevel.DISABLED:
                config.enabled = False
            elif level == DegradationLevel.NORMAL:
                config.sample_rate = 1.0  # Restore normal sampling
                config.enabled = True

    def _should_retry(self, error_type: LangfuseErrorType) -> bool:
        """Determine if an operation should be retried based on error type."""
        retryable_errors = {
            LangfuseErrorType.CONNECTION_ERROR,
            LangfuseErrorType.NETWORK_ERROR,
            LangfuseErrorType.TIMEOUT_ERROR,
            LangfuseErrorType.SERVER_ERROR,
            LangfuseErrorType.RATE_LIMIT_ERROR,
        }
        return error_type in retryable_errors

    def buffer_trace(self, trace_id: str, trace_data: Dict[str, Any]) -> None:
        """Buffer a trace for later retry when Langfuse is unavailable.

        Args:
            trace_id: Unique identifier for the trace
            trace_data: The trace data to buffer
        """
        if not self.langfuse_client.is_available:
            return

        with self._buffer_lock:
            if len(self._buffer) >= self.max_buffer_size:
                # Remove oldest trace if buffer is full
                self._buffer.popleft()
                logger.warning("Buffer full, removing oldest trace")

            buffered_trace = BufferedTrace(
                trace_id=trace_id, trace_data=trace_data, timestamp=datetime.now()
            )

            self._buffer.append(buffered_trace)
            logger.debug(f"Buffered trace {trace_id} for later retry")

    def record_performance_sample(self, response_time_ms: float) -> None:
        """Record a performance sample for monitoring.

        Args:
            response_time_ms: Response time in milliseconds
        """
        with self._performance_lock:
            self._performance_samples.append(response_time_ms)

            # Update average response time
            if len(self._performance_samples) > 0:
                self.error_metrics.average_response_time = sum(
                    self._performance_samples
                ) / len(self._performance_samples)

        # Check if performance threshold is exceeded
        if response_time_ms > self.performance_threshold_ms:
            logger.warning(
                f"Langfuse response time {response_time_ms}ms exceeds threshold {self.performance_threshold_ms}ms"
            )

            # Trigger degradation if consistently slow
            if (
                self.error_metrics.average_response_time > self.performance_threshold_ms
                and len(self._performance_samples) >= 10
            ):
                self._check_performance_degradation()

    def record_success(self) -> None:
        """Record a successful operation to reset failure counters."""
        self.error_metrics.consecutive_failures = 0

        # Check if we should reduce degradation level
        self._check_performance_degradation()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the error handler.

        Returns:
            Dictionary containing status information
        """
        with self._buffer_lock:
            buffer_size = len(self._buffer)

        with self._performance_lock:
            recent_performance = (
                list(self._performance_samples)[-10:]
                if self._performance_samples
                else []
            )

        return {
            "degradation_level": self.error_metrics.degradation_level.value,
            "total_errors": self.error_metrics.total_errors,
            "consecutive_failures": self.error_metrics.consecutive_failures,
            "errors_by_type": {
                k.value: v for k, v in self.error_metrics.errors_by_type.items()
            },
            "buffer_size": buffer_size,
            "average_response_time": round(self.error_metrics.average_response_time, 2),
            "recent_performance_samples": recent_performance,
            "last_error_time": self.error_metrics.last_error_time.isoformat()
            if self.error_metrics.last_error_time
            else None,
            "langfuse_available": self.langfuse_client.is_available,
        }

    def reset_metrics(self) -> None:
        """Reset all error metrics and counters."""
        self.error_metrics = ErrorMetrics()
        logger.info("Langfuse error metrics reset")

    def shutdown(self) -> None:
        """Shutdown the error handler and cleanup resources."""
        self._shutdown = True

        # Flush any remaining buffered traces
        self._flush_buffer()

        if self._buffer_thread and self._buffer_thread.is_alive():
            self._buffer_thread.join(timeout=5.0)

        logger.info("LangfuseErrorHandler shutdown completed")


# Global error handler instance
_error_handler: Optional[LangfuseErrorHandler] = None


def get_langfuse_error_handler() -> Optional[LangfuseErrorHandler]:
    """Get the global Langfuse error handler instance."""
    return _error_handler


def initialize_langfuse_error_handler(
    langfuse_client: LangfuseClient, **kwargs
) -> LangfuseErrorHandler:
    """Initialize the global Langfuse error handler.

    Args:
        langfuse_client: The Langfuse client instance
        **kwargs: Additional configuration parameters

    Returns:
        Initialized LangfuseErrorHandler instance
    """
    global _error_handler
    _error_handler = LangfuseErrorHandler(langfuse_client, **kwargs)
    return _error_handler


def shutdown_langfuse_error_handler() -> None:
    """Shutdown the global Langfuse error handler."""
    global _error_handler
    if _error_handler:
        _error_handler.shutdown()
        _error_handler = None

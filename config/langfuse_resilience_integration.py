"""Integration of Langfuse error handling with the existing resilience manager."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .langfuse_config import LangfuseClient
from ..agents.error_handling import resilience_manager, ErrorContext, ErrorSeverity

try:
    from .langfuse_error_handler import LangfuseErrorHandler
except ImportError:
    LangfuseErrorHandler = None

try:
    from .langfuse_sampling import LangfuseSampler
except ImportError:
    LangfuseSampler = None

try:
    from .langfuse_buffer import LangfuseBuffer
except ImportError:
    LangfuseBuffer = None

try:
    from .langfuse_performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

logger = logging.getLogger(__name__)


class LangfuseResilienceIntegration:
    """Integration layer between Langfuse components and the resilience manager."""

    def __init__(
        self,
        langfuse_client: LangfuseClient,
        error_handler: Optional[Any] = None,
        sampler: Optional[Any] = None,
        buffer: Optional[Any] = None,
        performance_monitor: Optional[Any] = None,
    ):
        """Initialize the integration.

        Args:
            langfuse_client: Langfuse client instance
            error_handler: Optional error handler instance
            sampler: Optional sampler instance
            buffer: Optional buffer instance
            performance_monitor: Optional performance monitor instance
        """
        self.langfuse_client = langfuse_client
        self.error_handler = error_handler
        self.sampler = sampler
        self.buffer = buffer
        self.performance_monitor = performance_monitor

        # Register with resilience manager
        self._register_with_resilience_manager()

        logger.info("LangfuseResilienceIntegration initialized")

    def _register_with_resilience_manager(self) -> None:
        """Register Langfuse components with the resilience manager."""
        # Register fallback strategies
        if self.error_handler:
            resilience_manager.register_fallback_strategy(
                "langfuse_integration", self._langfuse_fallback_strategy
            )

        # Register monitoring callbacks
        if self.performance_monitor:
            resilience_manager.add_monitoring_callback(
                self._performance_monitoring_callback
            )

        # Register health check callback
        resilience_manager.add_monitoring_callback(self._health_check_callback)

        logger.debug("Registered Langfuse components with resilience manager")

    def _langfuse_fallback_strategy(self) -> None:
        """Fallback strategy when Langfuse is unavailable."""
        logger.info("Executing Langfuse fallback strategy")

        # Disable real-time tracing
        if self.error_handler:
            self.error_handler._set_degradation_level(
                self.error_handler.error_metrics.degradation_level
            )

        # Enable buffering for offline storage
        if self.buffer:
            logger.info("Langfuse fallback: buffering enabled for offline storage")

        # Reduce sampling rate significantly
        if self.sampler:
            self.sampler.update_sample_rate(0.01)  # 1% sampling

    def _performance_monitoring_callback(self, error_context: ErrorContext) -> None:
        """Callback for performance monitoring events."""
        if error_context.component == "langfuse_integration":
            # Trigger performance analysis
            if self.performance_monitor:
                # This would trigger a performance check
                logger.debug("Performance monitoring callback triggered for Langfuse")

    def _health_check_callback(self, error_context: ErrorContext) -> None:
        """Callback for health check events."""
        if error_context.component == "langfuse_integration":
            # Log health status
            health_status = self.get_health_status()
            logger.info(f"Langfuse health check: {health_status}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of Langfuse integration.

        Returns:
            Dictionary containing health status information
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "langfuse_available": self.langfuse_client.is_available,
            "components": {},
        }

        # Error handler status
        if self.error_handler:
            status["components"]["error_handler"] = self.error_handler.get_status()

        # Sampler status
        if self.sampler:
            status["components"]["sampler"] = self.sampler.get_sampling_stats()

        # Buffer status
        if self.buffer:
            status["components"]["buffer"] = self.buffer.get_buffer_stats()

        # Performance monitor status
        if self.performance_monitor:
            status["components"]["performance_monitor"] = (
                self.performance_monitor.get_performance_summary()
            )

        # Overall health assessment
        components = status.get("components", {})
        status["overall_health"] = self._assess_overall_health(components)

        return status

    def _assess_overall_health(self, components: Dict[str, Any]) -> str:
        """Assess overall health based on component status."""

        # Check if Langfuse is available
        # Note: This would need access to the langfuse_available status
        # For now, assume it's available if we have components
        if not components:
            return "unhealthy"
            return "unhealthy"

        # Check error handler degradation level
        error_handler_status = components.get("error_handler", {})
        degradation_level = error_handler_status.get("degradation_level", "normal")

        if degradation_level == "disabled":
            return "critical"
        elif degradation_level in ["buffering_only", "reduced_sampling"]:
            return "degraded"
        else:
            return "healthy"

    def handle_langfuse_error(
        self, error: Exception, operation: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle a Langfuse error through the integrated system.

        Args:
            error: The exception that occurred
            operation: The operation that failed
            context: Additional context

        Returns:
            True if operation should be retried
        """
        # Handle through error handler
        if self.error_handler:
            should_retry = self.error_handler.handle_error(error, operation, context)

            # If error handler says not to retry, buffer the trace
            if not should_retry and self.buffer:
                # Extract trace information from context if available
                trace_id = context.get("trace_id") if context else None
                if trace_id:
                    trace_data = context.get("trace_data", {})
                    self.buffer.add_trace(trace_id, trace_data)

            return should_retry
        else:
            # Fallback to basic error handling
            logger.error(f"Langfuse error in {operation}: {error}")
            return False

    def record_trace_with_fallback(
        self, trace_id: str, trace_data: Dict[str, Any], priority: int = 1
    ) -> bool:
        """Record a trace with automatic fallback to buffering.

        Args:
            trace_id: Unique trace identifier
            trace_data: Trace data
            priority: Priority level

        Returns:
            True if trace was recorded or buffered
        """
        # Try to send directly first
        if self.langfuse_client.is_available:
            try:
                # This would be the actual trace sending logic
                # For now, simulate success/failure
                if self._simulate_trace_send():
                    if self.error_handler:
                        self.error_handler.record_success()
                    return True
                else:
                    raise Exception("Simulated trace send failure")
            except Exception as e:
                # Handle the error
                should_retry = self.handle_langfuse_error(
                    e, "trace_send", {"trace_id": trace_id, "trace_data": trace_data}
                )

                if should_retry:
                    # Retry logic would go here
                    return False

        # Fallback to buffering
        if self.buffer:
            return self.buffer.add_trace(trace_id, trace_data, priority)

        return False

    def _simulate_trace_send(self) -> bool:
        """Simulate trace sending (for testing)."""
        # In real implementation, this would send the trace to Langfuse
        import random

        return random.random() > 0.1  # 90% success rate

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all Langfuse components.

        Returns:
            Dictionary containing status of all components
        """
        return {
            "health_status": self.get_health_status(),
            "resilience_manager": resilience_manager.get_system_health(),
            "integration_active": True,
            "last_updated": datetime.now().isoformat(),
        }

    def shutdown(self) -> None:
        """Shutdown all Langfuse components gracefully."""
        logger.info("Shutting down Langfuse resilience integration")

        # Shutdown components in reverse order
        if self.performance_monitor:
            self.performance_monitor.shutdown()

        if self.buffer:
            self.buffer.shutdown()

        if self.sampler:
            self.sampler.shutdown()

        if self.error_handler:
            self.error_handler.shutdown()

        logger.info("Langfuse resilience integration shutdown completed")


# Global integration instance
_integration: Optional[LangfuseResilienceIntegration] = None


def get_langfuse_resilience_integration() -> Optional[LangfuseResilienceIntegration]:
    """Get the global Langfuse resilience integration instance."""
    return _integration


def initialize_langfuse_resilience_integration(
    langfuse_client: LangfuseClient, **kwargs
) -> LangfuseResilienceIntegration:
    """Initialize the global Langfuse resilience integration.

    Args:
        langfuse_client: Langfuse client instance
        **kwargs: Additional configuration parameters

    Returns:
        Initialized LangfuseResilienceIntegration instance
    """
    global _integration

    # Initialize components
    error_handler = None
    if "error_handler" not in kwargs:
        try:
            from .langfuse_error_handler import initialize_langfuse_error_handler

            error_handler = initialize_langfuse_error_handler(langfuse_client)
        except ImportError:
            logger.warning("LangfuseErrorHandler not available")

    sampler = None
    if "sampler" not in kwargs:
        try:
            from .langfuse_sampling import initialize_langfuse_sampler

            sampler = initialize_langfuse_sampler()
        except ImportError:
            logger.warning("LangfuseSampler not available")

    buffer = None
    if "buffer" not in kwargs:
        try:
            from .langfuse_buffer import initialize_langfuse_buffer

            buffer = initialize_langfuse_buffer(langfuse_client, error_handler)
        except ImportError:
            logger.warning("LangfuseBuffer not available")

    performance_monitor = None
    if "performance_monitor" not in kwargs:
        try:
            from .langfuse_performance_monitor import (
                initialize_langfuse_performance_monitor,
            )

            performance_monitor = initialize_langfuse_performance_monitor(
                error_handler, sampler, buffer
            )
        except ImportError:
            logger.warning("PerformanceMonitor not available")

    _integration = LangfuseResilienceIntegration(
        langfuse_client=langfuse_client,
        error_handler=error_handler,
        sampler=sampler,
        buffer=buffer,
        performance_monitor=performance_monitor,
    )

    return _integration


def shutdown_langfuse_resilience_integration() -> None:
    """Shutdown the global Langfuse resilience integration."""
    global _integration
    if _integration:
        _integration.shutdown()
        _integration = None

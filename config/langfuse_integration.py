"""Langfuse integration service for workflow visualization."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

from langfuse import Langfuse, observe
from .langfuse_config import get_langfuse_client, LangfuseClient
from .langfuse_data_masking import (
    get_data_masker,
    get_secure_trace_manager,
    DataMasker,
    SecureTraceManager,
)
from .metrics_collector import get_metrics_collector, MetricsCollector

# Import performance optimization components with fallbacks
try:
    from config.langfuse_sampling import (
        get_langfuse_sampler,
        initialize_langfuse_sampler,
    )

    SAMPLING_AVAILABLE = True
except ImportError:
    SAMPLING_AVAILABLE = False
    logger.warning("Langfuse sampling not available")

try:
    from config.langfuse_async_processor import (
        get_async_trace_processor,
        initialize_async_trace_processor,
    )

    ASYNC_PROCESSING_AVAILABLE = True
except ImportError:
    ASYNC_PROCESSING_AVAILABLE = False
    logger.warning("Langfuse async processing not available")

try:
    from config.langfuse_performance_monitor import (
        get_langfuse_performance_monitor,
        initialize_langfuse_performance_monitor,
    )

    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    logger.warning("Langfuse performance monitoring not available")

try:
    from config.langfuse_buffer import get_langfuse_buffer, initialize_langfuse_buffer

    BUFFERING_AVAILABLE = True
except ImportError:
    BUFFERING_AVAILABLE = False
    logger.warning("Langfuse buffering not available")

try:
    from config.langfuse_compression import (
        get_trace_compressor,
        initialize_trace_compressor,
    )

    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    logger.warning("Langfuse compression not available")


class LangfuseIntegrationService:
    """Central service for managing Langfuse integration and trace coordination."""

    def __init__(
        self,
        client: Optional[LangfuseClient] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        data_masker: Optional[DataMasker] = None,
        secure_manager: Optional[SecureTraceManager] = None,
        enable_performance_optimizations: bool = True,
    ):
        """Initialize the integration service.

        Args:
            client: Optional LangfuseClient instance. If None, uses global client.
            metrics_collector: Optional MetricsCollector instance. If None, uses global instance.
            data_masker: Optional DataMasker instance. If None, uses global instance.
            secure_manager: Optional SecureTraceManager instance. If None, uses global instance.
        """
        self._client = client or get_langfuse_client()
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}
        self._active_workflows: Dict[str, Any] = {}
        self._metrics_collector = metrics_collector or get_metrics_collector()
        self._data_masker = data_masker or get_data_masker()
        self._secure_manager = secure_manager or get_secure_trace_manager()

    @property
    def is_available(self) -> bool:
        """Check if Langfuse integration is available."""
        return self._client.is_available

    @property
    def langfuse_client(self) -> Optional[Langfuse]:
        """Get the underlying Langfuse client."""
        return self._client.client

    def create_simulation_trace(
        self, event_data: Dict[str, Any], user_roles: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create a root trace for simulation events with security features.

        Args:
            event_data: Dictionary containing simulation event information
            user_roles: Optional list of user roles for access control

        Returns:
            Trace ID if successful, None if tracing is unavailable or access denied
        """
        if not self.is_available:
            logger.debug("Langfuse not available, skipping trace creation")
            return None

        try:
            # Apply security checks and masking
            secure_data = self._secure_manager.create_secure_trace(
                trace_type="simulation",
                data={
                    "name": "simulation_event",
                    "input": event_data,
                    "metadata": {
                        "event_type": event_data.get("type", "unknown"),
                        "trigger_source": event_data.get("source", "simulation"),
                        "timestamp": datetime.now().isoformat(),
                        "system": "autoops_retail_optimization",
                    },
                },
                user_roles=user_roles,
            )

            if not secure_data:
                logger.warning(
                    "Access denied or data masking failed for simulation trace"
                )
                return None

            trace_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            client = self.langfuse_client
            if not client:
                logger.debug("Langfuse client not available")
                return None

            trace = client.start_span(  # type: ignore
                name=secure_data["name"],
                input=secure_data["input"],
                metadata=secure_data["metadata"],
            )

            self._active_traces[trace_id] = trace

            # Audit trace access if enabled
            if self._client.config and self._client.config.audit_trace_access:
                self._secure_manager.audit_trace_access(trace_id, "system", "create")

            logger.debug(f"Created secure simulation trace: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to create simulation trace: {e}")
            return None

    def start_agent_span(
        self,
        agent_id: str,
        operation: str,
        parent_trace_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start a span for agent operations.

        Args:
            agent_id: Identifier for the agent
            operation: Name of the operation being performed
            parent_trace_id: Optional parent trace ID
            input_data: Optional input data for the operation

        Returns:
            Span ID if successful, None if tracing is unavailable
        """
        if not self.is_available:
            logger.debug("Langfuse not available, skipping span creation")
            return None

        try:
            span_id = (
                f"{agent_id}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            # Start metrics tracking
            self._metrics_collector.start_operation(span_id, agent_id, operation)

            # Get parent trace if specified
            parent_trace = None
            if parent_trace_id and parent_trace_id in self._active_traces:
                parent_trace = self._active_traces[parent_trace_id]

            if parent_trace:
                span = parent_trace.start_span(
                    name=f"{agent_id}_{operation}",
                    input=input_data or {},
                    metadata={
                        "agent_id": agent_id,
                        "operation": operation,
                        "start_time": datetime.now().isoformat(),
                        "parent_trace_id": parent_trace_id,
                    },
                )
            else:
                # Create standalone span if no parent trace
                span = self.langfuse_client.start_span(  # type: ignore
                    name=f"{agent_id}_{operation}",
                    input=input_data or {},
                    metadata={
                        "agent_id": agent_id,
                        "operation": operation,
                        "start_time": datetime.now().isoformat(),
                    },
                )

            self._active_spans[span_id] = span

            logger.debug(f"Started agent span: {span_id} for {agent_id}")
            return span_id

        except Exception as e:
            logger.error(f"Failed to start agent span: {e}")
            return None

    def end_agent_span(
        self,
        span_id: Optional[str],
        outcome: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """End an agent span with outcome data.

        Args:
            span_id: The span ID to end
            outcome: Optional outcome data
            error: Optional error if the operation failed
        """
        if not self.is_available or span_id not in self._active_spans:
            return

        try:
            span = self._active_spans[span_id]

            # End metrics tracking
            success = error is None
            self._metrics_collector.end_operation(
                operation_id=span_id,
                success=success,
                error=str(error) if error else None,
                output_data=outcome,
            )

            # Update span with outcome
            span.update(
                output=outcome or {},
                metadata={
                    "end_time": datetime.now().isoformat(),
                    "status": "error" if error else "success",
                },
            )

            # Add error information if present
            if error:
                span.update(level="ERROR", status_message=str(error))

            # End the span
            span.end()

            # Remove from active spans
            del self._active_spans[span_id]

            logger.debug(f"Ended agent span: {span_id}")

        except Exception as e:
            logger.error(f"Failed to end agent span {span_id}: {e}")

    def log_agent_decision(
        self,
        agent_id: str,
        decision_data: Dict[str, Any],
        trace_context: Optional[Dict[str, Any]] = None,
        user_roles: Optional[List[str]] = None,
    ) -> None:
        """Log agent decision-making process with security features.

        Args:
            agent_id: Identifier for the agent
            decision_data: Dictionary containing decision information
            trace_context: Optional trace context information
            user_roles: Optional list of user roles for access control
        """
        if not self.is_available:
            return

        try:
            # Apply data masking to decision data
            masked_inputs = self._data_masker.mask_data(decision_data.get("inputs", {}))
            masked_outputs = self._data_masker.mask_data(
                decision_data.get("outputs", {})
            )

            event_name = f"{agent_id}_decision"

            # Check access control for agent decision traces
            if not self._secure_manager._check_access("agent_decision", user_roles):
                logger.warning(f"Access denied for agent decision trace: {agent_id}")
                return

            self.langfuse_client.create_event(  # type: ignore
                name=event_name,
                input=masked_inputs,
                output=masked_outputs,
                metadata={
                    "agent_id": agent_id,
                    "decision_type": decision_data.get("type", "unknown"),
                    "confidence": decision_data.get("confidence"),
                    "reasoning": self._data_masker.mask_data(
                        decision_data.get("reasoning", "")
                    ),
                    "timestamp": datetime.now().isoformat(),
                    **(trace_context or {}),
                },
            )

            # Audit trace access if enabled
            if self._client.config and self._client.config.audit_trace_access:
                self._secure_manager.audit_trace_access(
                    f"{agent_id}_decision", "system", "create"
                )

            logger.debug(f"Logged secure decision for agent: {agent_id}")

        except Exception as e:
            logger.error(f"Failed to log agent decision: {e}")

    def track_collaboration(
        self,
        workflow_id: str,
        participating_agents: List[str],
        workflow_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Track cross-agent collaboration workflows.

        Args:
            workflow_id: Identifier for the collaboration workflow
            participating_agents: List of agent IDs participating
            workflow_data: Optional workflow data

        Returns:
            Trace ID if successful, None if tracing is unavailable
        """
        if not self.is_available:
            return None

        try:
            trace_id = (
                f"collab_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            trace = self.langfuse_client.start_span(  # type: ignore
                name="agent_collaboration",
                input=workflow_data or {},
                metadata={
                    "workflow_id": workflow_id,
                    "participating_agents": participating_agents,
                    "agent_count": len(participating_agents),
                    "timestamp": datetime.now().isoformat(),
                    "workflow_type": "collaboration",
                },
            )

            self._active_traces[trace_id] = trace

            logger.debug(f"Started collaboration tracking: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to track collaboration: {e}")
            return None

    def finalize_trace(
        self, trace_id: str, final_outcome: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize a trace with final outcome data.

        Args:
            trace_id: The trace ID to finalize
            final_outcome: Optional final outcome data
        """
        if not self.is_available or trace_id not in self._active_traces:
            return

        try:
            trace = self._active_traces[trace_id]

            trace.update(
                output=final_outcome or {},
                metadata={
                    "end_time": datetime.now().isoformat(),
                    "status": "completed",
                },
            )
            trace.end()

            # Remove from active traces
            del self._active_traces[trace_id]

            logger.debug(f"Finalized trace: {trace_id}")

        except Exception as e:
            logger.error(f"Failed to finalize trace {trace_id}: {e}")

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing operations.

        Args:
            operation_name: Name of the operation
            input_data: Optional input data
            metadata: Optional metadata

        Yields:
            Trace object if available, None otherwise
        """
        if not self.is_available:
            yield None
            return

        trace_id = None
        try:
            trace_id = (
                f"op_{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            trace = self.langfuse_client.start_span(  # type: ignore
                name=operation_name,
                input=input_data or {},
                metadata={
                    "operation": operation_name,
                    "start_time": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            )

            self._active_traces[trace_id] = trace
            yield trace

        except Exception as e:
            logger.error(f"Error in trace operation {operation_name}: {e}")
            yield None
        finally:
            if trace_id and trace_id in self._active_traces:
                self.finalize_trace(trace_id)

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self.is_available:
            self._client.flush()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        base_health = self._client.health_check()

        return {
            **base_health,
            "active_traces": len(self._active_traces),
            "active_spans": len(self._active_spans),
            "integration_service": "ready",
        }

    def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary containing agent metrics or None if not found
        """
        metrics = self._metrics_collector.get_agent_metrics(agent_id)
        if metrics:
            return {
                "agent_id": metrics.agent_id,
                "operation_count": metrics.operation_count,
                "average_response_time": (
                    metrics.total_response_time / metrics.operation_count
                    if metrics.operation_count > 0
                    else 0.0
                ),
                "success_rate": (
                    metrics.success_count / metrics.operation_count
                    if metrics.operation_count > 0
                    else 0.0
                ),
                "error_count": metrics.error_count,
                "tool_usage_stats": metrics.tool_usage_stats,
                "collaboration_count": metrics.collaboration_count,
                "last_activity": metrics.last_activity.isoformat()
                if metrics.last_activity
                else None,
            }
        return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide workflow metrics.

        Returns:
            Dictionary containing system metrics
        """
        system_metrics = self._metrics_collector.get_system_metrics()
        return {
            "total_events_processed": system_metrics.total_events_processed,
            "total_workflows_completed": system_metrics.total_workflows_completed,
            "average_workflow_duration": system_metrics.average_workflow_duration,
            "agent_coordination_efficiency": system_metrics.agent_coordination_efficiency,
            "conflict_resolution_rate": system_metrics.conflict_resolution_rate,
            "system_throughput": system_metrics.system_throughput,
            "peak_throughput": system_metrics.peak_throughput,
            "error_rate": system_metrics.error_rate,
            "active_workflows": len(getattr(self, "_active_workflows", {})),
            "last_updated": system_metrics.last_updated.isoformat()
            if system_metrics.last_updated
            else None,
        }

    def export_metrics_for_dashboard(self) -> Dict[str, Any]:
        """Export all metrics in dashboard-ready format.

        Returns:
            Dictionary containing all metrics for dashboard display
        """
        return self._metrics_collector.export_metrics_for_dashboard()

    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self._metrics_collector.reset_metrics()


# Global integration service instance
_integration_service: Optional[LangfuseIntegrationService] = None


def get_langfuse_integration() -> LangfuseIntegrationService:
    """Get the global Langfuse integration service instance."""
    global _integration_service
    if _integration_service is None:
        _integration_service = LangfuseIntegrationService()
    return _integration_service


def initialize_langfuse_integration(
    client: Optional[LangfuseClient] = None,
) -> LangfuseIntegrationService:
    """Initialize Langfuse integration service with optional client.

    Args:
        client: Optional LangfuseClient instance

    Returns:
        Initialized LangfuseIntegrationService instance
    """
    global _integration_service
    _integration_service = LangfuseIntegrationService(client)
    return _integration_service

"""Intelligent sampling strategies for Langfuse integration under high load."""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import random
import threading

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies."""

    FIXED_RATE = "fixed_rate"
    ADAPTIVE_LOAD = "adaptive_load"
    ERROR_BASED = "error_based"
    PRIORITY_BASED = "priority_based"
    TIME_BASED = "time_based"


class TracePriority(Enum):
    """Priority levels for traces."""

    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class SamplingMetrics:
    """Metrics for sampling decisions."""

    total_traces: int = 0
    sampled_traces: int = 0
    dropped_traces: int = 0
    current_sample_rate: float = 1.0
    average_system_load: float = 0.0
    load_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    last_adjustment: Optional[datetime] = None


@dataclass
class TraceContext:
    """Context information for sampling decisions."""

    trace_id: str
    priority: TracePriority
    estimated_cost: float  # Estimated computational cost
    timestamp: datetime
    agent_id: Optional[str] = None
    operation_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LangfuseSampler:
    """Intelligent sampler for Langfuse traces under varying load conditions."""

    def __init__(
        self,
        base_sample_rate: float = 1.0,
        strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE_LOAD,
        min_sample_rate: float = 0.01,
        max_sample_rate: float = 1.0,
        load_threshold: float = 0.8,
        adjustment_interval: int = 30,
        enable_priority_sampling: bool = True,
    ):
        """Initialize the sampler.

        Args:
            base_sample_rate: Base sampling rate (0.0 to 1.0)
            strategy: Sampling strategy to use
            min_sample_rate: Minimum allowed sampling rate
            max_sample_rate: Maximum allowed sampling rate
            load_threshold: System load threshold for rate adjustment
            adjustment_interval: Seconds between rate adjustments
            enable_priority_sampling: Whether to use priority-based sampling
        """
        self.base_sample_rate = base_sample_rate
        self.strategy = strategy
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.load_threshold = load_threshold
        self.adjustment_interval = adjustment_interval
        self.enable_priority_sampling = enable_priority_sampling

        # Metrics and state
        self.metrics = SamplingMetrics(current_sample_rate=base_sample_rate)
        self._lock = threading.Lock()
        self._shutdown = False
        self._adjustment_thread: Optional[threading.Thread] = None

        # Priority weights for priority-based sampling
        self.priority_weights = {
            TracePriority.CRITICAL: 1.0,
            TracePriority.HIGH: 0.8,
            TracePriority.MEDIUM: 0.5,
            TracePriority.LOW: 0.2,
        }

        # Start background adjustment thread
        self._start_adjustment_thread()

        logger.info(f"LangfuseSampler initialized with strategy {strategy.value}")

    def _start_adjustment_thread(self) -> None:
        """Start the background thread for periodic rate adjustments."""
        self._adjustment_thread = threading.Thread(
            target=self._adjustment_loop, daemon=True, name="LangfuseSamplingAdjustment"
        )
        self._adjustment_thread.start()
        logger.debug("Sampling adjustment thread started")

    def _adjustment_loop(self) -> None:
        """Background loop for periodic sampling rate adjustments."""
        while not self._shutdown:
            try:
                self._adjust_sampling_rate()
                time.sleep(self.adjustment_interval)
            except Exception as e:
                logger.error(f"Error in sampling adjustment loop: {e}")
                time.sleep(self.adjustment_interval)

    def _adjust_sampling_rate(self) -> None:
        """Adjust sampling rate based on current strategy and system conditions."""
        with self._lock:
            if self.strategy == SamplingStrategy.FIXED_RATE:
                return  # No adjustment needed

            elif self.strategy == SamplingStrategy.ADAPTIVE_LOAD:
                self._adjust_for_load()

            elif self.strategy == SamplingStrategy.ERROR_BASED:
                self._adjust_for_errors()

            elif self.strategy == SamplingStrategy.TIME_BASED:
                self._adjust_for_time()

            # Ensure rate stays within bounds
            self.metrics.current_sample_rate = max(
                self.min_sample_rate,
                min(self.max_sample_rate, self.metrics.current_sample_rate),
            )

            self.metrics.last_adjustment = datetime.now()

    def _adjust_for_load(self) -> None:
        """Adjust sampling rate based on system load."""
        # Get current system load (this would need to be implemented based on your monitoring)
        current_load = self._get_system_load()

        self.metrics.load_samples.append(current_load)
        if len(self.metrics.load_samples) > 0:
            self.metrics.average_system_load = sum(self.metrics.load_samples) / len(
                self.metrics.load_samples
            )

        if current_load > self.load_threshold:
            # Reduce sampling rate under high load
            reduction_factor = 1.0 - (
                (current_load - self.load_threshold) / (1.0 - self.load_threshold)
            )
            new_rate = self.metrics.current_sample_rate * reduction_factor
            self.metrics.current_sample_rate = max(self.min_sample_rate, new_rate)
            logger.debug(
                f"Reduced sampling rate to {self.metrics.current_sample_rate:.3f} due to high load ({current_load:.3f})"
            )
        else:
            # Gradually increase sampling rate when load is normal
            if self.metrics.current_sample_rate < self.base_sample_rate:
                self.metrics.current_sample_rate = min(
                    self.base_sample_rate, self.metrics.current_sample_rate * 1.1
                )

    def _adjust_for_errors(self) -> None:
        """Adjust sampling rate based on error rates."""
        # This would integrate with the error handler to get error rates
        # For now, implement a simple version
        error_rate = self._get_error_rate()

        if error_rate > 0.1:  # 10% error rate
            # Reduce sampling to reduce system stress
            self.metrics.current_sample_rate *= 0.8
            logger.debug(
                f"Reduced sampling rate to {self.metrics.current_sample_rate:.3f} due to high error rate ({error_rate:.3f})"
            )
        elif (
            error_rate < 0.01
            and self.metrics.current_sample_rate < self.base_sample_rate
        ):
            # Increase sampling when error rate is low
            self.metrics.current_sample_rate = min(
                self.base_sample_rate, self.metrics.current_sample_rate * 1.2
            )

    def _adjust_for_time(self) -> None:
        """Adjust sampling rate based on time of day or business hours."""
        current_hour = datetime.now().hour

        # Business hours: 9 AM - 6 PM
        if 9 <= current_hour <= 18:
            # Higher sampling during business hours
            target_rate = self.base_sample_rate
        else:
            # Lower sampling during off-hours
            target_rate = self.base_sample_rate * 0.3

        # Smooth transition to target rate
        if self.metrics.current_sample_rate < target_rate:
            self.metrics.current_sample_rate = min(
                target_rate, self.metrics.current_sample_rate * 1.1
            )
        elif self.metrics.current_sample_rate > target_rate:
            self.metrics.current_sample_rate = max(
                target_rate, self.metrics.current_sample_rate * 0.9
            )

    def _get_system_load(self) -> float:
        """Get current system load from performance monitor."""
        try:
            from .langfuse_performance_monitor import get_langfuse_performance_monitor

            monitor = get_langfuse_performance_monitor()
            if monitor:
                summary = monitor.get_performance_summary()
                if summary and "current" in summary:
                    # Calculate composite load from CPU, memory, and queue metrics
                    cpu_load = summary["current"].get("cpu_usage", 0) / 100.0
                    memory_load = summary["current"].get("memory_usage", 0) / 100.0
                    queue_load = min(
                        1.0, summary["current"].get("queue_size", 0) / 1000.0
                    )
                    return max(cpu_load, memory_load, queue_load)
        except ImportError:
            pass

        # Fallback to simulated load based on recent activity
        return min(1.0, len(self.metrics.load_samples) / 50.0)

    def _get_error_rate(self) -> float:
        """Get current error rate from error handler and async processor."""
        try:
            from .langfuse_error_handler import get_langfuse_error_handler

            error_handler = get_langfuse_error_handler()
            if error_handler and hasattr(error_handler, "error_metrics"):
                total_errors = error_handler.error_metrics.total_errors
                # Use a placeholder for total_operations since it's not available
                total_operations = max(
                    100, total_errors * 20
                )  # Assume 5% error rate baseline
                if total_operations > 0:
                    return min(1.0, total_errors / total_operations)

            # Also check async processor error rate
            try:
                # Import here to avoid circular imports
                import importlib

                async_module = importlib.import_module(
                    ".langfuse_async_processor", package=__name__
                )
                get_processor = getattr(async_module, "get_async_trace_processor", None)
                if get_processor:
                    async_processor = get_processor()
                    if async_processor:
                        stats = async_processor.get_processing_stats()
                        total_ops = stats.get("operations_processed", 0) + stats.get(
                            "operations_failed", 0
                        )
                        if total_ops > 0:
                            return stats["operations_failed"] / total_ops
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass

        # Fallback to simulated error rate
        return 0.05  # 5% error rate

    def should_sample(self, trace_context: TraceContext) -> bool:
        """Determine if a trace should be sampled.

        Args:
            trace_context: Context information for the trace

        Returns:
            True if the trace should be sampled
        """
        with self._lock:
            self.metrics.total_traces += 1

            if (
                self.strategy == SamplingStrategy.PRIORITY_BASED
                and self.enable_priority_sampling
            ):
                return self._should_sample_by_priority(trace_context)
            else:
                return self._should_sample_by_rate(trace_context)

    def _should_sample_by_rate(self, trace_context: TraceContext) -> bool:
        """Simple rate-based sampling decision."""
        sampled = random.random() < self.metrics.current_sample_rate

        if sampled:
            self.metrics.sampled_traces += 1
        else:
            self.metrics.dropped_traces += 1

        return sampled

    def _should_sample_by_priority(self, trace_context: TraceContext) -> bool:
        """Priority-based sampling with rate adjustment."""
        # Calculate effective sample rate based on priority
        priority_weight = self.priority_weights.get(trace_context.priority, 0.5)
        effective_rate = self.metrics.current_sample_rate * priority_weight

        # Ensure minimum sampling for critical traces
        if trace_context.priority == TracePriority.CRITICAL:
            effective_rate = max(effective_rate, 0.1)

        sampled = random.random() < effective_rate

        if sampled:
            self.metrics.sampled_traces += 1
        else:
            self.metrics.dropped_traces += 1

        return sampled

    def get_priority_for_trace(
        self,
        agent_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TracePriority:
        """Determine the priority of a trace based on context.

        Args:
            agent_id: ID of the agent creating the trace
            operation_type: Type of operation being traced
            metadata: Additional metadata

        Returns:
            Priority level for the trace
        """
        # Critical operations
        if operation_type in ["error", "failure", "critical", "system_failure"]:
            return TracePriority.CRITICAL

        # High priority agents/operations
        if agent_id in ["orchestrator", "system_monitor"] or operation_type in [
            "decision",
            "coordination",
        ]:
            return TracePriority.HIGH

        # Medium priority for regular agent operations
        if agent_id and operation_type:
            return TracePriority.MEDIUM

        # Low priority for routine operations
        return TracePriority.LOW

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get current sampling statistics.

        Returns:
            Dictionary containing sampling metrics
        """
        with self._lock:
            total = self.metrics.total_traces
            sampled = self.metrics.sampled_traces
            dropped = self.metrics.dropped_traces

            return {
                "current_sample_rate": round(self.metrics.current_sample_rate, 4),
                "total_traces": total,
                "sampled_traces": sampled,
                "dropped_traces": dropped,
                "sampling_ratio": round(sampled / total, 4) if total > 0 else 0.0,
                "strategy": self.strategy.value,
                "average_system_load": round(self.metrics.average_system_load, 4),
                "last_adjustment": self.metrics.last_adjustment.isoformat()
                if self.metrics.last_adjustment
                else None,
                "priority_sampling_enabled": self.enable_priority_sampling,
            }

    def force_sample(self, trace_context: TraceContext) -> bool:
        """Force sampling of a trace (bypass normal sampling logic).

        Args:
            trace_context: Context information for the trace

        Returns:
            Always True (trace will be sampled)
        """
        with self._lock:
            self.metrics.sampled_traces += 1
            self.metrics.total_traces += 1
            return True

    def update_sample_rate(self, new_rate: float) -> None:
        """Manually update the sampling rate.

        Args:
            new_rate: New sampling rate (0.0 to 1.0)
        """
        with self._lock:
            self.metrics.current_sample_rate = max(
                self.min_sample_rate, min(self.max_sample_rate, new_rate)
            )
            self.metrics.last_adjustment = datetime.now()
            logger.info(
                f"Sampling rate manually updated to {self.metrics.current_sample_rate}"
            )

    def reset_metrics(self) -> None:
        """Reset all sampling metrics."""
        with self._lock:
            self.metrics = SamplingMetrics(current_sample_rate=self.base_sample_rate)
            logger.info("Sampling metrics reset")

    def shutdown(self) -> None:
        """Shutdown the sampler and cleanup resources."""
        self._shutdown = True

        if self._adjustment_thread and self._adjustment_thread.is_alive():
            self._adjustment_thread.join(timeout=5.0)

        logger.info("LangfuseSampler shutdown completed")


# Global sampler instance
_sampler: Optional[LangfuseSampler] = None


def get_langfuse_sampler() -> Optional[LangfuseSampler]:
    """Get the global Langfuse sampler instance."""
    return _sampler


def initialize_langfuse_sampler(
    base_sample_rate: float = 1.0,
    strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE_LOAD,
    **kwargs,
) -> LangfuseSampler:
    """Initialize the global Langfuse sampler.

    Args:
        base_sample_rate: Base sampling rate
        strategy: Sampling strategy to use
        **kwargs: Additional configuration parameters

    Returns:
        Initialized LangfuseSampler instance
    """
    global _sampler
    _sampler = LangfuseSampler(
        base_sample_rate=base_sample_rate, strategy=strategy, **kwargs
    )
    return _sampler


def shutdown_langfuse_sampler() -> None:
    """Shutdown the global Langfuse sampler."""
    global _sampler
    if _sampler:
        _sampler.shutdown()
        _sampler = None

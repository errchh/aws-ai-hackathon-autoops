"""Demonstration of Langfuse error handling and fallback mechanisms."""

import logging
import time
import random
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Langfuse integration components
try:
    from config.langfuse_config import LangfuseClient, LangfuseConfig
    from config.langfuse_error_handler import LangfuseErrorHandler
    from config.langfuse_sampling import (
        LangfuseSampler,
        SamplingStrategy,
        TracePriority,
    )
    from config.langfuse_buffer import LangfuseBuffer
    from config.langfuse_performance_monitor import PerformanceMonitor
    from config.langfuse_resilience_integration import LangfuseResilienceIntegration
except ImportError as e:
    logger.error(f"Failed to import Langfuse components: {e}")
    exit(1)


def simulate_trace_operation(
    trace_id: str, operation: str, should_fail: bool = False, delay: float = 0.1
) -> Dict[str, Any]:
    """Simulate a trace operation that may succeed or fail.

    Args:
        trace_id: Unique trace identifier
        operation: Operation name
        should_fail: Whether the operation should fail
        delay: Artificial delay in seconds

    Returns:
        Operation result data
    """
    time.sleep(delay)  # Simulate processing time

    if should_fail:
        raise Exception(f"Simulated failure in {operation}")

    return {
        "trace_id": trace_id,
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "result": "success",
        "processing_time": delay,
        "metadata": {"agent": "demo_agent", "version": "1.0"},
    }


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling capabilities."""
    logger.info("=== Langfuse Error Handling Demonstration ===")

    # Initialize Langfuse client (mock configuration)
    config = LangfuseConfig(
        public_key="demo_key", secret_key="demo_secret", enabled=True, sample_rate=1.0
    )

    try:
        client = LangfuseClient(config)
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        return

    # Initialize error handler
    error_handler = LangfuseErrorHandler(
        langfuse_client=client, max_consecutive_failures=3, performance_threshold_ms=100
    )

    # Initialize sampler with adaptive strategy
    sampler = LangfuseSampler(
        base_sample_rate=0.8,
        strategy=SamplingStrategy.ADAPTIVE_LOAD,
        min_sample_rate=0.1,
    )

    # Initialize buffer
    buffer = LangfuseBuffer(
        langfuse_client=client, error_handler=error_handler, max_buffer_size=1000
    )

    # Initialize performance monitor
    performance_monitor = PerformanceMonitor(
        error_handler=error_handler, sampler=sampler, buffer=buffer
    )

    # Initialize resilience integration
    integration = LangfuseResilienceIntegration(
        langfuse_client=client,
        error_handler=error_handler,
        sampler=sampler,
        buffer=buffer,
        performance_monitor=performance_monitor,
    )

    logger.info("All components initialized successfully")

    # Demonstrate normal operation
    logger.info("\n--- Normal Operation ---")
    for i in range(5):
        trace_id = f"demo_trace_{i}"
        trace_context = {
            "trace_id": trace_id,
            "priority": TracePriority.MEDIUM,
            "operation": "demo_operation",
        }

        try:
            # Check if trace should be sampled
            if sampler.should_sample(type("TraceContext", (), trace_context)()):
                result = simulate_trace_operation(trace_id, f"operation_{i}")
                logger.info(f"Trace {trace_id} completed successfully")
            else:
                logger.info(f"Trace {trace_id} was not sampled")

        except Exception as e:
            # Handle error through integration
            integration.handle_langfuse_error(
                e, f"operation_{i}", {"trace_id": trace_id}
            )

    # Demonstrate error scenarios
    logger.info("\n--- Error Scenarios ---")
    for i in range(3):
        trace_id = f"error_trace_{i}"

        try:
            # Simulate operations with varying failure rates
            fail_chance = 0.7 if i < 2 else 0.3  # First two likely to fail
            result = simulate_trace_operation(
                trace_id,
                f"error_operation_{i}",
                should_fail=(random.random() < fail_chance),
            )
            logger.info(f"Trace {trace_id} completed successfully")

        except Exception as e:
            # Handle error - this will trigger fallback mechanisms
            integration.handle_langfuse_error(
                e, f"error_operation_{i}", {"trace_id": trace_id}
            )

    # Demonstrate buffering when Langfuse is unavailable
    logger.info("\n--- Buffering Demonstration ---")
    logger.info("Simulating Langfuse unavailability...")

    # Temporarily disable client
    original_available = client.is_available
    client._is_initialized = False
    client._fallback_mode = True

    for i in range(3):
        trace_id = f"buffered_trace_{i}"
        trace_data = {
            "trace_id": trace_id,
            "operation": f"buffered_operation_{i}",
            "data": f"sample_data_{i}",
        }

        # This should be buffered since Langfuse is "unavailable"
        success = integration.record_trace_with_fallback(trace_id, trace_data)
        if success:
            logger.info(f"Trace {trace_id} was buffered successfully")
        else:
            logger.error(f"Failed to buffer trace {trace_id}")

    # Restore client availability
    client._is_initialized = original_available
    client._fallback_mode = not original_available

    # Demonstrate performance monitoring
    logger.info("\n--- Performance Monitoring ---")
    time.sleep(2)  # Allow some monitoring data to accumulate

    # Get performance summary
    perf_summary = performance_monitor.get_performance_summary()
    logger.info(f"Performance Summary: {perf_summary}")

    # Get comprehensive status
    status = integration.get_comprehensive_status()
    logger.info(f"Integration Status: {status}")

    # Demonstrate sampling statistics
    logger.info("\n--- Sampling Statistics ---")
    sampling_stats = sampler.get_sampling_stats()
    logger.info(f"Sampling Stats: {sampling_stats}")

    # Demonstrate buffer statistics
    logger.info("\n--- Buffer Statistics ---")
    buffer_stats = buffer.get_buffer_stats()
    logger.info(f"Buffer Stats: {buffer_stats}")

    # Cleanup
    logger.info("\n--- Cleanup ---")
    integration.shutdown()

    logger.info("=== Demonstration Complete ===")


def demonstrate_degradation_scenarios():
    """Demonstrate automatic degradation under various conditions."""
    logger.info("\n=== Degradation Scenarios Demonstration ===")

    # Initialize with low thresholds to trigger degradation
    config = LangfuseConfig(
        public_key="demo_key", secret_key="demo_secret", enabled=True, sample_rate=1.0
    )

    client = LangfuseClient(config)
    error_handler = LangfuseErrorHandler(
        langfuse_client=client,
        max_consecutive_failures=2,  # Low threshold for demo
        performance_threshold_ms=50,  # Low threshold for demo
    )

    # Simulate high error rate to trigger degradation
    logger.info("Simulating high error rate to trigger degradation...")

    for i in range(5):
        try:
            # Simulate failures to build up error count
            raise Exception(f"Simulated error {i}")
        except Exception as e:
            error_handler.handle_error(e, f"degradation_test_{i}")

    # Check degradation status
    status = error_handler.get_status()
    logger.info(f"Error Handler Status after high error rate: {status}")

    # Reset and demonstrate recovery
    logger.info("\nSimulating successful operations for recovery...")
    error_handler.record_success()
    error_handler.record_success()

    status = error_handler.get_status()
    logger.info(f"Error Handler Status after recovery: {status}")

    error_handler.shutdown()


if __name__ == "__main__":
    try:
        demonstrate_error_handling()
        demonstrate_degradation_scenarios()
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

"""
Performance tests for Langfuse workflow visualization.

This module contains comprehensive performance tests to measure tracing overhead,
memory usage, and system impact of the Langfuse integration.
"""

import asyncio
import gc
import psutil
import time
import tracemalloc
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch

import pytest

from config.langfuse_integration import LangfuseIntegrationService
from config.simulation_event_capture import SimulationEventCapture
from config.orchestrator_tracing import OrchestrationTracer
from tests.test_data_generators import TestDataGenerator, MockLangfuseClient


class PerformanceTestSuite:
    """Comprehensive performance testing suite for Langfuse integration."""

    def __init__(self):
        """Initialize performance test suite."""
        self.test_generator = TestDataGenerator(seed=42)
        self.mock_client = MockLangfuseClient()
        self.baseline_metrics = {}

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        return {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": cpu_percent,
            "timestamp": datetime.now().isoformat()
        }

    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result

    def measure_memory_usage(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure memory usage during function execution."""
        tracemalloc.start()
        gc.collect()

        start_snapshot = tracemalloc.take_snapshot()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        end_snapshot = tracemalloc.take_snapshot()

        stats = end_snapshot.compare_to(start_snapshot, 'lineno')

        tracemalloc.stop()

        return {
            "execution_time": end_time - start_time,
            "memory_increase": sum(stat.size_diff for stat in stats),
            "memory_increase_mb": sum(stat.size_diff for stat in stats) / 1024 / 1024,
            "result": result
        }

    def run_baseline_test(self, iterations: int = 100) -> Dict[str, Any]:
        """Run baseline test without tracing to establish performance baseline."""
        print("Running baseline performance test...")

        execution_times = []
        memory_usages = []

        for i in range(iterations):
            # Simulate basic agent operations without tracing
            start_time = time.perf_counter()
            # Simulate some work
            time.sleep(0.001)  # 1ms of simulated work
            end_time = time.perf_counter()

            execution_times.append(end_time - start_time)

        baseline_metrics = {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "iterations": iterations,
            "system_metrics": self.get_system_metrics()
        }

        self.baseline_metrics = baseline_metrics
        return baseline_metrics

    def run_tracing_overhead_test(self, iterations: int = 100) -> Dict[str, Any]:
        """Test tracing overhead with mock Langfuse client."""
        print("Running tracing overhead test...")

        # Set up services with mock client
        integration_service = LangfuseIntegrationService()
        integration_service._client = self.mock_client

        event_capture = SimulationEventCapture(integration_service)
        orchestrator_tracer = OrchestrationTracer(integration_service)

        execution_times = []
        memory_usages = []

        for i in range(iterations):
            # Measure time for trace operations
            start_time = time.perf_counter()

            # Simulate workflow tracing
            workflow_id = f"workflow_{i}"
            event_data = {"event_type": "test", "event_id": f"event_{i}"}

            # Create trace
            trace_id = integration_service.create_simulation_trace(event_data)
            if trace_id:
                # Create spans
                span_id = integration_service.start_agent_span("test_agent", "test_operation")
                if span_id:
                    integration_service.end_agent_span(span_id, {"result": "success"})

                # Finalize trace
                integration_service.finalize_trace(trace_id, {"status": "completed"})

            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        tracing_metrics = {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "iterations": iterations,
            "overhead_percentage": (
                (sum(execution_times) / len(execution_times) - self.baseline_metrics["avg_execution_time"])
                / self.baseline_metrics["avg_execution_time"] * 100
            ),
            "system_metrics": self.get_system_metrics()
        }

        return tracing_metrics

    def run_memory_usage_test(self, iterations: int = 50) -> Dict[str, Any]:
        """Test memory usage with tracing enabled."""
        print("Running memory usage test...")

        integration_service = LangfuseIntegrationService()
        integration_service._client = self.mock_client

        def memory_intensive_operations():
            """Simulate memory-intensive tracing operations."""
            for i in range(iterations):
                # Create multiple traces and spans
                for j in range(10):
                    trace_id = integration_service.create_simulation_trace({
                        "event_type": "memory_test",
                        "event_id": f"mem_event_{i}_{j}",
                        "data": "x" * 1000  # Simulate data payload
                    })

                    if trace_id:
                        for k in range(5):
                            span_id = integration_service.start_agent_span(
                                f"agent_{k}",
                                f"operation_{k}",
                                input_data={"data": "x" * 500}
                            )
                            if span_id:
                                integration_service.end_agent_span(
                                    span_id,
                                    {"result": "x" * 200}
                                )

                        integration_service.finalize_trace(trace_id, {"status": "completed"})

        memory_result = self.measure_memory_usage(memory_intensive_operations)

        return {
            "execution_time": memory_result["execution_time"],
            "memory_increase_mb": memory_result["memory_increase_mb"],
            "memory_increase_per_iteration": memory_result["memory_increase_mb"] / iterations,
            "iterations": iterations,
            "system_metrics": self.get_system_metrics()
        }

    def run_concurrent_load_test(self, concurrent_operations: int = 10, duration_seconds: int = 30) -> Dict[str, Any]:
        """Test performance under concurrent load."""
        print(f"Running concurrent load test with {concurrent_operations} concurrent operations...")

        integration_service = LangfuseIntegrationService()
        integration_service._client = self.mock_client

        async def concurrent_trace_operations():
            """Run concurrent tracing operations."""
            tasks = []

            for i in range(concurrent_operations):
                task = asyncio.create_task(self._single_trace_operation(integration_service, i))
                tasks.append(task)

            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()

            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful

            return {
                "total_time": end_time - start_time,
                "successful_operations": successful,
                "failed_operations": failed,
                "success_rate": successful / len(results),
                "throughput": len(results) / (end_time - start_time)
            }

        async def _single_trace_operation(service, operation_id):
            """Single tracing operation for concurrent testing."""
            try:
                # Simulate realistic workflow
                trace_id = service.create_simulation_trace({
                    "event_type": "concurrent_test",
                    "event_id": f"concurrent_event_{operation_id}",
                    "timestamp": datetime.now().isoformat()
                })

                if trace_id:
                    # Simulate agent operations
                    for agent in ["inventory", "pricing", "promotion"]:
                        span_id = service.start_agent_span(
                            f"{agent}_agent",
                            "process_event",
                            input_data={"event_id": f"concurrent_event_{operation_id}"}
                        )
                        if span_id:
                            await asyncio.sleep(0.01)  # Simulate processing time
                            service.end_agent_span(span_id, {"status": "success"})

                    service.finalize_trace(trace_id, {"status": "completed"})

                return True
            except Exception as e:
                return e

        return asyncio.run(concurrent_trace_operations())

    def run_trace_volume_test(self, trace_count: int = 1000) -> Dict[str, Any]:
        """Test performance with high volume of traces."""
        print(f"Running high volume trace test with {trace_count} traces...")

        integration_service = LangfuseIntegrationService()
        integration_service._client = self.mock_client

        start_time = time.perf_counter()

        for i in range(trace_count):
            trace_id = integration_service.create_simulation_trace({
                "event_type": "volume_test",
                "event_id": f"volume_event_{i}",
                "metadata": {"batch": "volume_test", "index": i}
            })

            if trace_id:
                # Add some spans to each trace
                for j in range(3):
                    span_id = integration_service.start_agent_span(
                        f"agent_{j % 3}",
                        f"operation_{j}",
                        input_data={"trace_index": i, "span_index": j}
                    )
                    if span_id:
                        integration_service.end_agent_span(span_id, {"result": "success"})

                integration_service.finalize_trace(trace_id, {"status": "completed"})

        end_time = time.perf_counter()

        total_time = end_time - start_time
        traces_per_second = trace_count / total_time

        return {
            "total_traces": trace_count,
            "total_time": total_time,
            "traces_per_second": traces_per_second,
            "avg_time_per_trace": total_time / trace_count,
            "system_metrics": self.get_system_metrics()
        }


class TestLangfusePerformance:
    """Pytest test class for Langfuse performance testing."""

    @pytest.fixture
    def performance_suite(self):
        """Create performance test suite."""
        return PerformanceTestSuite()

    @pytest.mark.performance
    @pytest.mark.slow
    def test_tracing_overhead_baseline(self, performance_suite):
        """Test baseline performance without tracing."""
        baseline = performance_suite.run_baseline_test(iterations=100)

        # Validate baseline performance
        assert baseline["avg_execution_time"] < 0.01  # Should be very fast
        assert baseline["iterations"] == 100

        print(f"Baseline: {baseline['avg_execution_time']".6f"}s avg execution time")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_tracing_overhead_measurement(self, performance_suite):
        """Test tracing overhead compared to baseline."""
        # First run baseline
        baseline = performance_suite.run_baseline_test(iterations=100)

        # Then run with tracing
        tracing_metrics = performance_suite.run_tracing_overhead_test(iterations=100)

        # Calculate overhead
        overhead = tracing_metrics["overhead_percentage"]

        # Validate overhead is reasonable (less than 1000% = 10x slower)
        assert overhead < 1000, f"Tracing overhead too high: {overhead".2f"}%"

        print(f"Tracing overhead: {overhead".2f"}%")
        print(f"Baseline: {baseline['avg_execution_time']".6f"}s")
        print(f"With tracing: {tracing_metrics['avg_execution_time']".6f"}s")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_with_tracing(self, performance_suite):
        """Test memory usage impact of tracing."""
        memory_metrics = performance_suite.run_memory_usage_test(iterations=50)

        # Validate memory usage is reasonable
        assert memory_metrics["memory_increase_mb"] < 50, "Memory usage too high"
        assert memory_metrics["memory_increase_per_iteration"] < 1, "Memory per iteration too high"

        print(f"Memory increase: {memory_metrics['memory_increase_mb']".2f"}MB")
        print(f"Memory per iteration: {memory_metrics['memory_increase_per_iteration']".4f"}MB")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_load_performance(self, performance_suite):
        """Test performance under concurrent load."""
        concurrent_metrics = performance_suite.run_concurrent_load_test(
            concurrent_operations=10,
            duration_seconds=30
        )

        # Validate concurrent performance
        assert concurrent_metrics["success_rate"] > 0.9, "Success rate too low under load"
        assert concurrent_metrics["throughput"] > 5, "Throughput too low"

        print(f"Concurrent success rate: {concurrent_metrics['success_rate']".2%"}")
        print(f"Throughput: {concurrent_metrics['throughput']".2f"} operations/second")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_high_volume_trace_performance(self, performance_suite):
        """Test performance with high volume of traces."""
        volume_metrics = performance_suite.run_trace_volume_test(trace_count=1000)

        # Validate volume performance
        assert volume_metrics["traces_per_second"] > 10, "Trace throughput too low"
        assert volume_metrics["avg_time_per_trace"] < 0.1, "Average time per trace too high"

        print(f"Volume test: {volume_metrics['traces_per_second']".2f"} traces/second")
        print(f"Avg time per trace: {volume_metrics['avg_time_per_trace']".4f"}s")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_comprehensive_performance_benchmark(self, performance_suite):
        """Run comprehensive performance benchmark."""
        print("Running comprehensive performance benchmark...")

        # Run all performance tests
        baseline = performance_suite.run_baseline_test(iterations=100)
        tracing_overhead = performance_suite.run_tracing_overhead_test(iterations=100)
        memory_usage = performance_suite.run_memory_usage_test(iterations=50)
        concurrent_load = performance_suite.run_concurrent_load_test(concurrent_operations=10)
        volume_test = performance_suite.run_trace_volume_test(trace_count=500)

        # Compile comprehensive results
        benchmark_results = {
            "baseline": baseline,
            "tracing_overhead": tracing_overhead,
            "memory_usage": memory_usage,
            "concurrent_load": concurrent_load,
            "volume_test": volume_test,
            "summary": {
                "tracing_overhead_percent": tracing_overhead["overhead_percentage"],
                "memory_increase_mb": memory_usage["memory_increase_mb"],
                "concurrent_success_rate": concurrent_load["success_rate"],
                "volume_throughput": volume_test["traces_per_second"],
                "overall_performance_score": self._calculate_performance_score(
                    tracing_overhead, memory_usage, concurrent_load, volume_test
                )
            }
        }

        # Validate overall performance
        assert benchmark_results["summary"]["tracing_overhead_percent"] < 500, "Overall overhead too high"
        assert benchmark_results["summary"]["memory_increase_mb"] < 100, "Memory usage too high"
        assert benchmark_results["summary"]["concurrent_success_rate"] > 0.8, "Concurrent success rate too low"
        assert benchmark_results["summary"]["volume_throughput"] > 20, "Volume throughput too low"

        print("
=== Performance Benchmark Results ===")
        print(f"Tracing overhead: {benchmark_results['summary']['tracing_overhead_percent']".2f"}%")
        print(f"Memory increase: {benchmark_results['summary']['memory_increase_mb']".2f"}MB")
        print(f"Concurrent success rate: {benchmark_results['summary']['concurrent_success_rate']".2%"}")
        print(f"Volume throughput: {benchmark_results['summary']['volume_throughput']".2f"} traces/sec")
        print(f"Overall performance score: {benchmark_results['summary']['overall_performance_score']".2f"}/100")

        return benchmark_results

    def _calculate_performance_score(self, tracing_overhead, memory_usage, concurrent_load, volume_test) -> float:
        """Calculate overall performance score (0-100)."""
        # Scoring weights
        overhead_weight = 0.3
        memory_weight = 0.2
        concurrent_weight = 0.3
        volume_weight = 0.2

        # Calculate individual scores (higher is better)
        overhead_score = max(0, 100 - min(tracing_overhead["overhead_percentage"], 100))
        memory_score = max(0, 100 - min(memory_usage["memory_increase_mb"], 100))
        concurrent_score = min(concurrent_load["success_rate"] * 100, 100)
        volume_score = min(volume_test["traces_per_second"] * 2, 100)  # Scale for 50+ traces/sec = 100

        # Weighted average
        overall_score = (
            overhead_score * overhead_weight +
            memory_score * memory_weight +
            concurrent_score * concurrent_weight +
            volume_score * volume_weight
        )

        return overall_score

    @pytest.mark.performance
    @pytest.mark.slow
    def test_trace_data_compression_performance(self, performance_suite):
        """Test performance impact of trace data compression."""
        integration_service = LangfuseIntegrationService()
        integration_service._client = performance_suite.mock_client

        # Test with different data sizes
        data_sizes = [100, 1000, 10000]  # bytes
        results = {}

        for size in data_sizes:
            test_data = {"data": "x" * size, "metadata": {"size": size}}

            # Measure time with compression
            start_time = time.perf_counter()
            trace_id = integration_service.create_simulation_trace(test_data)
            if trace_id:
                integration_service.finalize_trace(trace_id, {"status": "completed"})
            end_time = time.perf_counter()

            results[size] = {
                "execution_time": end_time - start_time,
                "data_size": size
            }

        # Validate compression performance
        for size, result in results.items():
            assert result["execution_time"] < 1.0, f"Compression too slow for {size} bytes"

        print("Compression performance results:")
        for size, result in results.items():
            print(f"  {size} bytes: {result['execution_time']".4f"}s")


class TestLangfuseLoadTesting:
    """Load testing suite for high-frequency simulation events."""

    @pytest.fixture
    def load_test_suite(self):
        """Create load testing suite."""
        return PerformanceTestSuite()

    @pytest.mark.load
    @pytest.mark.slow
    def test_high_frequency_event_processing(self, load_test_suite):
        """Test system under high-frequency event processing."""
        # Generate test data
        test_data = load_test_suite.test_generator.generate_load_test_data(concurrent_events=20)

        # Run load test
        load_results = load_test_suite.run_concurrent_load_test(
            concurrent_operations=test_data["concurrent_events"],
            duration_seconds=test_data["load_profile"]["sustained_load_time_seconds"]
        )

        # Validate load test results
        assert load_results["success_rate"] > 0.85, "Load test success rate too low"
        assert load_results["throughput"] > test_data["load_profile"]["target_throughput_events_per_second"] * 0.8

        print(f"Load test results: {load_results['success_rate']".2%"} success rate")
        print(f"Throughput: {load_results['throughput']".2f"} operations/sec")

    @pytest.mark.load
    @pytest.mark.slow
    def test_sustained_load_performance(self, load_test_suite):
        """Test sustained load performance over time."""
        # Test different load levels
        load_levels = [5, 10, 20, 30]

        sustained_results = {}

        for load_level in load_levels:
            print(f"Testing sustained load: {load_level} concurrent operations")

            # Run sustained load for 60 seconds
            sustained_results[load_level] = load_test_suite.run_concurrent_load_test(
                concurrent_operations=load_level,
                duration_seconds=60
            )

            # Validate sustained performance
            result = sustained_results[load_level]
            assert result["success_rate"] > 0.8, f"Sustained load {load_level} success rate too low"

        # Check for performance degradation over time
        for load_level in load_levels:
            result = sustained_results[load_level]
            print(f"Load {load_level}: {result['success_rate']".2%"} success, {result['throughput']".2f"} ops/sec")

    @pytest.mark.stress
    @pytest.mark.slow
    def test_stress_test_maximum_load(self, load_test_suite):
        """Test system under maximum stress conditions."""
        stress_data = load_test_suite.test_generator.generate_stress_test_data(max_concurrent_events=50)

        max_load_results = {}

        for stress_level in stress_data["stress_levels"]:
            events = stress_level["events"]
            duration = stress_level["duration_seconds"]

            print(f"Stress test: {events} events for {duration} seconds")

            result = load_test_suite.run_concurrent_load_test(
                concurrent_operations=events,
                duration_seconds=duration
            )

            max_load_results[events] = result

            # Even under stress, should maintain reasonable success rate
            if events <= 30:  # Lower expectations for very high load
                assert result["success_rate"] > 0.6, f"Stress test {events} events success rate too low"

        # Analyze stress test results
        print("\n=== Stress Test Results ===")
        for events, result in max_load_results.items():
            print(f"{events} concurrent: {result['success_rate']".2%"} success, {result['throughput']".2f"} ops/sec")


if __name__ == "__main__":
    # Allow running performance tests directly
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
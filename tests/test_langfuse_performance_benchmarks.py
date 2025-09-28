"""
Comprehensive performance benchmarks for Langfuse workflow visualization.

This module contains detailed performance benchmarks to measure tracing overhead,
memory usage, CPU impact, and system performance under various load conditions.
"""

import asyncio
import gc
import psutil
import statistics
import time
import tracemalloc
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch

import pytest

from config.langfuse_integration import LangfuseIntegrationService
from config.simulation_event_capture import SimulationEventCapture
from config.orchestrator_tracing import OrchestrationTracer
from tests.test_data_generators import TestDataGenerator, MockLangfuseClient


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite for Langfuse integration."""

    def __init__(self):
        """Initialize performance benchmark suite."""
        self.test_generator = TestDataGenerator(seed=42)
        self.mock_client = MockLangfuseClient()
        self.baseline_metrics = {}
        self.benchmark_results = {}

    def get_detailed_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_times = process.cpu_times()
        io_counters = process.io_counters()

        return {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "cpu_times_user": cpu_times.user,
            "cpu_times_system": cpu_times.system,
            "io_read_bytes": io_counters.read_bytes,
            "io_write_bytes": io_counters.write_bytes,
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "timestamp": datetime.now().isoformat()
        }

    def measure_execution_with_memory_profiling(
        self, func, *args, **kwargs
    ) -> Dict[str, Any]:
        """Measure execution with detailed memory profiling."""
        tracemalloc.start()
        gc.collect()

        start_snapshot = tracemalloc.take_snapshot()
        start_metrics = self.get_detailed_system_metrics()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = None
            error = str(e)

        end_time = time.perf_counter()
        end_metrics = self.get_detailed_system_metrics()
        end_snapshot = tracemalloc.take_snapshot()

        stats = end_snapshot.compare_to(start_snapshot, 'lineno')

        tracemalloc.stop()

        return {
            "execution_time": end_time - start_time,
            "memory_increase": sum(stat.size_diff for stat in stats),
            "memory_increase_mb": sum(stat.size_diff for stat in stats) / 1024 / 1024,
            "memory_percent_increase": end_metrics["memory_percent"] - start_metrics["memory_percent"],
            "cpu_time_increase": (end_metrics["cpu_times_user"] + end_metrics["cpu_times_system"]) -
                               (start_metrics["cpu_times_user"] + start_metrics["cpu_times_system"]),
            "io_bytes_increase": (end_metrics["io_read_bytes"] + end_metrics["io_write_bytes"]) -
                               (start_metrics["io_read_bytes"] + start_metrics["io_write_bytes"]),
            "result": result,
            "error": error if 'error' in locals() else None,
            "start_metrics": start_metrics,
            "end_metrics": end_metrics,
            "memory_stats": [
                {
                    "file": stat.traceback.filename,
                    "line": stat.traceback.lineno,
                    "size_diff": stat.size_diff,
                    "size_diff_mb": stat.size_diff / 1024 / 1024
                }
                for stat in stats[:10]  # Top 10 memory changes
            ]
        }

    def run_tracing_overhead_benchmark(
        self,
        iterations: int = 100,
        concurrent_operations: int = 1
    ) -> Dict[str, Any]:
        """Run comprehensive tracing overhead benchmark."""
        print(f"Running tracing overhead benchmark: {iterations} iterations, {concurrent_operations} concurrent")

        # Baseline test (no tracing)
        baseline_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            time.sleep(0.001)  # Simulate basic operation
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)

        # Tracing test
        tracing_times = []
        memory_measurements = []

        async def single_tracing_operation():
            """Single tracing operation for benchmarking."""
            integration_service = LangfuseIntegrationService()

            with patch.object(integration_service, '_client', self.mock_client):
                start_time = time.perf_counter()

                # Simulate workflow with tracing
                trace_id = integration_service.create_simulation_trace({
                    "event_type": "benchmark_test",
                    "event_id": f"bench_{time.time()}",
                    "metadata": {"benchmark": True}
                })

                if trace_id:
                    span_id = integration_service.start_agent_span(
                        "benchmark_agent",
                        "benchmark_operation",
                        input_data={"test": "data"}
                    )

                    if span_id:
                        integration_service.end_agent_span(span_id, {"result": "success"})

                    integration_service.finalize_trace(trace_id)

                end_time = time.perf_counter()
                return end_time - start_time

        async def run_concurrent_benchmark():
            """Run concurrent tracing operations."""
            tasks = [single_tracing_operation() for _ in range(iterations)]
            results = await asyncio.gather(*tasks)
            return results

        if concurrent_operations == 1:
            # Sequential execution
            for _ in range(iterations):
                exec_result = self.measure_execution_with_memory_profiling(
                    lambda: asyncio.run(single_tracing_operation())
                )
                tracing_times.append(exec_result["execution_time"])
                memory_measurements.append(exec_result)
        else:
            # Concurrent execution
            tracing_times = asyncio.run(run_concurrent_benchmark())
            # Memory measurement for concurrent case
            exec_result = self.measure_execution_with_memory_profiling(
                lambda: asyncio.run(run_concurrent_benchmark())
            )
            memory_measurements.append(exec_result)

        # Calculate statistics
        baseline_stats = {
            "mean": statistics.mean(baseline_times),
            "median": statistics.median(baseline_times),
            "stdev": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0,
            "min": min(baseline_times),
            "max": max(baseline_times),
            "p95": statistics.quantiles(baseline_times, n=20)[18] if len(baseline_times) >= 20 else max(baseline_times)
        }

        tracing_stats = {
            "mean": statistics.mean(tracing_times),
            "median": statistics.median(tracing_times),
            "stdev": statistics.stdev(tracing_times) if len(tracing_times) > 1 else 0,
            "min": min(tracing_times),
            "max": max(tracing_times),
            "p95": statistics.quantiles(tracing_times, n=20)[18] if len(tracing_times) >= 20 else max(tracing_times)
        }

        # Calculate overhead
        overhead_mean = ((tracing_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]) * 100
        overhead_p95 = ((tracing_stats["p95"] - baseline_stats["p95"]) / baseline_stats["p95"]) * 100

        # Memory analysis
        memory_increases = [m["memory_increase_mb"] for m in memory_measurements]
        memory_stats = {
            "mean_increase_mb": statistics.mean(memory_increases),
            "max_increase_mb": max(memory_increases),
            "total_increase_mb": sum(memory_increases)
        }

        return {
            "benchmark_type": "tracing_overhead",
            "iterations": iterations,
            "concurrent_operations": concurrent_operations,
            "baseline_stats": baseline_stats,
            "tracing_stats": tracing_stats,
            "overhead_percentage": {
                "mean": overhead_mean,
                "p95": overhead_p95
            },
            "memory_stats": memory_stats,
            "system_impact": {
                "memory_increase_mb": memory_stats["mean_increase_mb"],
                "performance_overhead_percent": overhead_mean
            },
            "recommendations": self._generate_performance_recommendations(
                overhead_mean, overhead_p95, memory_stats
            )
        }

    def run_memory_usage_benchmark(
        self,
        trace_count: int = 1000,
        span_depth: int = 5
    ) -> Dict[str, Any]:
        """Run memory usage benchmark with varying trace complexity."""
        print(f"Running memory usage benchmark: {trace_count} traces, depth {span_depth}")

        integration_service = LangfuseIntegrationService()

        def create_complex_trace_hierarchy():
            """Create a complex trace hierarchy for memory testing."""
            with patch.object(integration_service, '_client', self.mock_client):
                # Create root trace
                root_trace_id = integration_service.create_simulation_trace({
                    "event_type": "memory_benchmark",
                    "complexity": "high",
                    "trace_count": trace_count
                })

                if not root_trace_id:
                    return

                # Create nested spans
                def create_nested_spans(parent_id: str, depth: int, current_depth: int = 0):
                    if current_depth >= depth:
                        return

                    for i in range(3):  # 3 child spans per level
                        span_id = integration_service.start_agent_span(
                            f"agent_{current_depth}_{i}",
                            f"operation_{current_depth}_{i}",
                            input_data={"depth": current_depth, "index": i}
                        )

                        if span_id:
                            # Create nested spans
                            create_nested_spans(span_id, depth, current_depth + 1)

                            # End the span
                            integration_service.end_agent_span(span_id, {
                                "result": f"completed_depth_{current_depth}"
                            })

                create_nested_spans(root_trace_id, span_depth)

                # Finalize root trace
                integration_service.finalize_trace(root_trace_id)

        # Measure memory usage
        exec_result = self.measure_execution_with_memory_profiling(create_complex_trace_hierarchy)

        # Force garbage collection and measure again
        gc.collect()
        post_gc_metrics = self.get_detailed_system_metrics()

        return {
            "benchmark_type": "memory_usage",
            "trace_count": trace_count,
            "span_depth": span_depth,
            "execution_metrics": exec_result,
            "post_gc_metrics": post_gc_metrics,
            "memory_efficiency": {
                "memory_per_trace_mb": exec_result["memory_increase_mb"] / trace_count,
                "memory_per_span_mb": exec_result["memory_increase_mb"] / (trace_count * (3 ** span_depth)),
                "gc_effectiveness": (
                    exec_result["end_metrics"]["memory_rss_mb"] - post_gc_metrics["memory_rss_mb"]
                )
            },
            "memory_leak_indicators": {
                "uncollectable_memory_mb": (
                    exec_result["end_metrics"]["memory_rss_mb"] - post_gc_metrics["memory_rss_mb"]
                ),
                "memory_growth_rate": exec_result["memory_increase_mb"] / exec_result["execution_time"]
            }
        }

    def run_concurrent_load_benchmark(
        self,
        concurrent_operations: int = 10,
        duration_seconds: int = 60,
        operation_rate: float = 5.0  # operations per second
    ) -> Dict[str, Any]:
        """Run concurrent load benchmark simulating high-frequency events."""
        print(f"Running concurrent load benchmark: {concurrent_operations} ops, {duration_seconds}s duration")

        integration_service = LangfuseIntegrationService()
        results = {
            "successful_operations": 0,
            "failed_operations": 0,
            "execution_times": [],
            "memory_samples": [],
            "cpu_samples": []
        }

        async def worker_operation(worker_id: int):
            """Individual worker operation."""
            operation_count = 0
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                operation_start = time.perf_counter()

                with patch.object(integration_service, '_client', self.mock_client):
                    # Simulate workflow processing
                    trace_id = integration_service.create_simulation_trace({
                        "event_type": "load_test",
                        "worker_id": worker_id,
                        "operation": operation_count
                    })

                    if trace_id:
                        span_id = integration_service.start_agent_span(
                            f"worker_{worker_id}_agent",
                            "process_operation",
                            input_data={"operation_id": operation_count}
                        )

                        if span_id:
                            integration_service.end_agent_span(span_id, {"status": "completed"})

                        integration_service.finalize_trace(trace_id)

                operation_end = time.perf_counter()
                results["execution_times"].append(operation_end - operation_start)
                results["successful_operations"] += 1
                operation_count += 1

                # Rate limiting
                await asyncio.sleep(1.0 / operation_rate)

        async def run_load_test():
            """Run the concurrent load test."""
            # Sample system metrics periodically
            async def metrics_sampler():
                while True:
                    results["memory_samples"].append(self.get_detailed_system_metrics()["memory_rss_mb"])
                    results["cpu_samples"].append(psutil.cpu_percent())
                    await asyncio.sleep(1.0)

            # Start metrics sampling
            sampler_task = asyncio.create_task(metrics_sampler())

            # Start worker tasks
            worker_tasks = [
                worker_operation(worker_id)
                for worker_id in range(concurrent_operations)
            ]

            # Wait for duration
            await asyncio.sleep(duration_seconds)

            # Cancel remaining tasks
            for task in worker_tasks:
                task.cancel()

            sampler_task.cancel()

        # Run the load test
        try:
            asyncio.run(run_load_test())
        except Exception as e:
            results["error"] = str(e)
            results["failed_operations"] += 1

        # Calculate statistics
        if results["execution_times"]:
            execution_stats = {
                "mean": statistics.mean(results["execution_times"]),
                "median": statistics.median(results["execution_times"]),
                "min": min(results["execution_times"]),
                "max": max(results["execution_times"]),
                "total_operations": len(results["execution_times"])
            }
        else:
            execution_stats = {"mean": 0, "median": 0, "min": 0, "max": 0, "total_operations": 0}

        memory_stats = {
            "mean_mb": statistics.mean(results["memory_samples"]) if results["memory_samples"] else 0,
            "max_mb": max(results["memory_samples"]) if results["memory_samples"] else 0,
            "samples": len(results["memory_samples"])
        }

        cpu_stats = {
            "mean_percent": statistics.mean(results["cpu_samples"]) if results["cpu_samples"] else 0,
            "max_percent": max(results["cpu_samples"]) if results["cpu_samples"] else 0,
            "samples": len(results["cpu_samples"])
        }

        return {
            "benchmark_type": "concurrent_load",
            "concurrent_operations": concurrent_operations,
            "duration_seconds": duration_seconds,
            "target_rate_ops_per_sec": operation_rate,
            "actual_rate_ops_per_sec": execution_stats["total_operations"] / duration_seconds,
            "success_rate": results["successful_operations"] / (results["successful_operations"] + results["failed_operations"]),
            "execution_stats": execution_stats,
            "memory_stats": memory_stats,
            "cpu_stats": cpu_stats,
            "performance_score": self._calculate_performance_score(
                execution_stats, memory_stats, cpu_stats, results.get("success_rate", 0)
            )
        }

    def run_trace_volume_benchmark(
        self,
        trace_counts: List[int] = [100, 500, 1000, 2000]
    ) -> Dict[str, Any]:
        """Run trace volume benchmark with increasing trace counts."""
        print(f"Running trace volume benchmark: {trace_counts}")

        volume_results = {}

        for trace_count in trace_counts:
            print(f"  Testing {trace_count} traces...")

            integration_service = LangfuseIntegrationService()

            def create_volume_test():
                """Create specified number of traces."""
                with patch.object(integration_service, '_client', self.mock_client):
                    for i in range(trace_count):
                        trace_id = integration_service.create_simulation_trace({
                            "event_type": "volume_test",
                            "batch": f"batch_{trace_count}",
                            "index": i,
                            "metadata": {"test": "volume_benchmark"}
                        })

                        if trace_id:
                            # Add some spans to each trace
                            for span_idx in range(3):
                                span_id = integration_service.start_agent_span(
                                    f"agent_{i}",
                                    f"operation_{span_idx}",
                                    input_data={"trace_index": i, "span_index": span_idx}
                                )

                                if span_id:
                                    integration_service.end_agent_span(span_id, {"status": "completed"})

                            integration_service.finalize_trace(trace_id)

            # Measure performance
            exec_result = self.measure_execution_with_memory_profiling(create_volume_test)

            volume_results[trace_count] = {
                "execution_time": exec_result["execution_time"],
                "memory_increase_mb": exec_result["memory_increase_mb"],
                "traces_per_second": trace_count / exec_result["execution_time"],
                "memory_per_trace_mb": exec_result["memory_increase_mb"] / trace_count,
                "success": exec_result["error"] is None
            }

        # Calculate scaling analysis
        trace_counts_list = list(volume_results.keys())
        times = [volume_results[count]["execution_time"] for count in trace_counts_list]
        memory_usage = [volume_results[count]["memory_increase_mb"] for count in trace_counts_list]

        # Linear regression for scaling analysis
        if len(trace_counts_list) >= 2:
            time_slope = (times[-1] - times[0]) / (trace_counts_list[-1] - trace_counts_list[0])
            memory_slope = (memory_usage[-1] - memory_usage[0]) / (trace_counts_list[-1] - trace_counts_list[0])
        else:
            time_slope = memory_slope = 0

        return {
            "benchmark_type": "trace_volume",
            "trace_counts": trace_counts,
            "volume_results": volume_results,
            "scaling_analysis": {
                "time_scaling_factor": time_slope,
                "memory_scaling_factor": memory_slope,
                "is_linear_time": abs(time_slope - (times[0] / trace_counts_list[0])) < 0.1,
                "is_linear_memory": abs(memory_slope - (memory_usage[0] / trace_counts_list[0])) < 0.1
            },
            "recommendations": {
                "max_efficient_trace_count": self._find_optimal_trace_count(volume_results),
                "memory_efficiency_threshold": 100,  # MB per 1000 traces
                "performance_efficiency_threshold": 100  # traces per second
            }
        }

    def _generate_performance_recommendations(
        self,
        overhead_mean: float,
        overhead_p95: float,
        memory_stats: Dict[str, float]
    ) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []

        if overhead_mean > 50:
            recommendations.append(
                "High tracing overhead detected. Consider reducing trace frequency or implementing sampling."
            )
        elif overhead_mean > 20:
            recommendations.append(
                "Moderate tracing overhead. Monitor performance and consider optimization if needed."
            )
        else:
            recommendations.append("Tracing overhead is within acceptable limits.")

        if overhead_p95 > 100:
            recommendations.append(
                "High percentile latency impact. Consider async tracing or batch processing."
            )

        if memory_stats["mean_increase_mb"] > 50:
            recommendations.append(
                "Significant memory usage detected. Implement trace cleanup and memory management."
            )
        elif memory_stats["mean_increase_mb"] > 20:
            recommendations.append(
                "Moderate memory usage. Monitor memory growth over time."
            )

        return recommendations

    def _calculate_performance_score(
        self,
        execution_stats: Dict[str, float],
        memory_stats: Dict[str, float],
        cpu_stats: Dict[str, float],
        success_rate: float
    ) -> float:
        """Calculate overall performance score (0-100)."""
        # Weight different factors
        execution_score = max(0, 100 - (execution_stats["mean"] * 1000))  # Lower is better
        memory_score = max(0, 100 - (memory_stats["mean_mb"] * 2))  # Lower is better
        cpu_score = max(0, 100 - cpu_stats["mean_percent"])  # Lower is better
        reliability_score = success_rate * 100

        # Weighted average
        overall_score = (
            execution_score * 0.4 +
            memory_score * 0.3 +
            cpu_score * 0.2 +
            reliability_score * 0.1
        )

        return min(100, max(0, overall_score))

    def _find_optimal_trace_count(self, volume_results: Dict[int, Dict]) -> int:
        """Find optimal trace count based on efficiency metrics."""
        best_count = 100
        best_efficiency = 0

        for count, results in volume_results.items():
            if results["success"]:
                efficiency = results["traces_per_second"] / (results["memory_per_trace_mb"] + 1)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_count = count

        return best_count


class TestLangfusePerformanceBenchmarks:
    """Pytest test class for Langfuse performance benchmarks."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create performance benchmark suite."""
        return PerformanceBenchmarkSuite()

    @pytest.mark.performance
    @pytest.mark.slow
    def test_tracing_overhead_benchmark(self, benchmark_suite):
        """Test tracing overhead benchmark."""
        results = benchmark_suite.run_tracing_overhead_benchmark(
            iterations=50, concurrent_operations=1
        )

        assert results["benchmark_type"] == "tracing_overhead"
        assert results["overhead_percentage"]["mean"] >= 0
        assert results["memory_stats"]["mean_increase_mb"] >= 0
        assert len(results["recommendations"]) > 0

        print(f"Tracing overhead: {results['overhead_percentage']['mean']".2f"}% mean")
        print(f"Memory increase: {results['memory_stats']['mean_increase_mb']".2f"} MB")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_benchmark(self, benchmark_suite):
        """Test memory usage benchmark."""
        results = benchmark_suite.run_memory_usage_benchmark(
            trace_count=100, span_depth=3
        )

        assert results["benchmark_type"] == "memory_usage"
        assert results["memory_efficiency"]["memory_per_trace_mb"] >= 0
        assert results["memory_leak_indicators"]["memory_growth_rate"] >= 0

        print(f"Memory per trace: {results['memory_efficiency']['memory_per_trace_mb']".4f"} MB")
        print(f"Memory growth rate: {results['memory_leak_indicators']['memory_growth_rate']".4f"} MB/s")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_load_benchmark(self, benchmark_suite):
        """Test concurrent load benchmark."""
        results = benchmark_suite.run_concurrent_load_benchmark(
            concurrent_operations=5, duration_seconds=10, operation_rate=3.0
        )

        assert results["benchmark_type"] == "concurrent_load"
        assert results["success_rate"] >= 0
        assert results["performance_score"] >= 0

        print(f"Success rate: {results['success_rate']".2%"}")
        print(f"Performance score: {results['performance_score']".1f"}/100")
        print(f"Actual throughput: {results['actual_rate_ops_per_sec']".2f"} ops/sec")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_trace_volume_benchmark(self, benchmark_suite):
        """Test trace volume benchmark."""
        results = benchmark_suite.run_trace_volume_benchmark(
            trace_counts=[50, 100, 200]
        )

        assert results["benchmark_type"] == "trace_volume"
        assert len(results["volume_results"]) == 3

        # Check scaling analysis
        scaling = results["scaling_analysis"]
        assert "time_scaling_factor" in scaling
        assert "memory_scaling_factor" in scaling

        print(f"Time scaling factor: {scaling['time_scaling_factor']".4f"}")
        print(f"Memory scaling factor: {scaling['memory_scaling_factor']".4f"}")

    @pytest.mark.performance
    @pytest.mark.slow
    def test_comprehensive_performance_suite(self, benchmark_suite):
        """Run comprehensive performance test suite."""
        print("Running comprehensive performance benchmark suite...")

        # Run all benchmarks
        overhead_results = benchmark_suite.run_tracing_overhead_benchmark(
            iterations=30, concurrent_operations=1
        )

        memory_results = benchmark_suite.run_memory_usage_benchmark(
            trace_count=50, span_depth=2
        )

        load_results = benchmark_suite.run_concurrent_load_benchmark(
            concurrent_operations=3, duration_seconds=5, operation_rate=2.0
        )

        volume_results = benchmark_suite.run_trace_volume_benchmark(
            trace_counts=[25, 50, 75]
        )

        # Compile comprehensive results
        comprehensive_results = {
            "overhead_benchmark": overhead_results,
            "memory_benchmark": memory_results,
            "load_benchmark": load_results,
            "volume_benchmark": volume_results,
            "overall_assessment": {
                "performance_score": (
                    overhead_results["system_impact"]["performance_overhead_percent"] * 0.4 +
                    (memory_results["memory_efficiency"]["memory_per_trace_mb"] * 10) * 0.3 +
                    (100 - load_results["performance_score"]) * 0.3
                ),
                "memory_efficiency_score": 100 - (memory_results["memory_efficiency"]["memory_per_trace_mb"] * 20),
                "scalability_score": 100 if volume_results["scaling_analysis"]["is_linear_time"] else 50
            }
        }

        # Validate comprehensive results
        overall = comprehensive_results["overall_assessment"]
        assert overall["performance_score"] >= 0
        assert overall["memory_efficiency_score"] >= 0
        assert overall["scalability_score"] >= 0

        print(f"Overall performance score: {overall['performance_score']".1f"}/100")
        print(f"Memory efficiency score: {overall['memory_efficiency_score']".1f"}/100")
        print(f"Scalability score: {overall['scalability_score']".1f"}/100")

        return comprehensive_results


if __name__ == "__main__":
    # Allow running benchmarks directly
    suite = PerformanceBenchmarkSuite()

    print("=== Langfuse Performance Benchmark Suite ===")

    # Run individual benchmarks
    print("\n1. Tracing Overhead Benchmark:")
    overhead = suite.run_tracing_overhead_benchmark(iterations=20)
    print(f"   Mean overhead: {overhead['overhead_percentage']['mean']".2f"}%")

    print("\n2. Memory Usage Benchmark:")
    memory = suite.run_memory_usage_benchmark(trace_count=30, span_depth=2)
    print(f"   Memory per trace: {memory['memory_efficiency']['memory_per_trace_mb']".4f"} MB")

    print("\n3. Concurrent Load Benchmark:")
    load_test = suite.run_concurrent_load_benchmark(concurrent_operations=2, duration_seconds=3)
    print(f"   Success rate: {load_test['success_rate']".2%"}")

    print("\n4. Trace Volume Benchmark:")
    volume = suite.run_trace_volume_benchmark(trace_counts=[10, 20, 30])
    print(f"   Optimal trace count: {suite._find_optimal_trace_count(volume['volume_results'])}")

    print("\n=== Benchmark Suite Complete ===")